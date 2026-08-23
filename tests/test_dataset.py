"""Tests for the dataset tooling and tiled-inference maths.

Nothing here needs the dataset or a GPU: labels are synthesised in tmp_path
and the NMS/tiling logic is tested on plain arrays.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcb_detection.dataset import (  # noqa: E402
    CLASS_NAMES,
    DUPLICATE_MERGES,
    NUM_CLASSES,
    apply_class_merges,
    find_label_files,
    validate_labels,
    write_data_yaml,
)
from pcb_detection.tiling import non_max_suppression  # noqa: E402


def write_labels(tmp_path, name, lines):
    folder = tmp_path / "train" / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestClassList:
    def test_count_matches_the_annotations(self):
        assert NUM_CLASSES == 50
        assert len(CLASS_NAMES) == 50

    def test_merge_targets_are_valid_indices(self):
        for source, target in DUPLICATE_MERGES.items():
            assert 0 <= source < NUM_CLASSES
            assert 0 <= target < NUM_CLASSES

    def test_merges_point_at_the_canonical_spelling(self):
        # Every merge should collapse a variant onto a name that still exists.
        for source, target in DUPLICATE_MERGES.items():
            assert CLASS_NAMES[source] != CLASS_NAMES[target]
            assert CLASS_NAMES[target] in CLASS_NAMES

    def test_no_merge_target_is_itself_merged_away(self):
        # Otherwise a single pass would leave dangling indices.
        for target in DUPLICATE_MERGES.values():
            assert target not in DUPLICATE_MERGES


class TestDataYaml:
    def test_declares_every_class(self, tmp_path):
        path = write_data_yaml(tmp_path / "data.yaml")
        content = open(path, encoding="utf-8").read()
        assert f"nc: {NUM_CLASSES}" in content
        for index in (0, 21, 33, 49):
            assert f"  {index}: '{CLASS_NAMES[index]}'" in content

    def test_declares_all_three_splits(self, tmp_path):
        content = open(write_data_yaml(tmp_path / "data.yaml"), encoding="utf-8").read()
        for split in ("train:", "val:", "test:"):
            assert split in content


class TestValidation:
    def test_accepts_well_formed_labels(self, tmp_path):
        write_labels(tmp_path, "a.txt", ["9 0.5 0.5 0.1 0.2", "21 0.1 0.9 0.05 0.05"])
        report = validate_labels(find_label_files(tmp_path))
        assert report["total"] == 2
        assert report["clean"]

    def test_flags_coordinates_outside_the_unit_range(self, tmp_path):
        write_labels(tmp_path, "a.txt", ["9 1.5 0.5 0.1 0.2"])
        report = validate_labels(find_label_files(tmp_path))
        assert report["issues"]["out_of_range"] == 1
        assert not report["clean"]

    def test_flags_negative_coordinates(self, tmp_path):
        write_labels(tmp_path, "a.txt", ["9 -0.1 0.5 0.1 0.2"])
        assert validate_labels(find_label_files(tmp_path))["issues"]["out_of_range"] == 1

    def test_flags_a_class_beyond_the_class_list(self, tmp_path):
        write_labels(tmp_path, "a.txt", [f"{NUM_CLASSES} 0.5 0.5 0.1 0.2"])
        assert validate_labels(find_label_files(tmp_path))["issues"]["class_out_of_bounds"] == 1

    def test_flags_truncated_rows(self, tmp_path):
        write_labels(tmp_path, "a.txt", ["9 0.5 0.5"])
        assert validate_labels(find_label_files(tmp_path))["issues"]["format"] == 1

    def test_blank_lines_are_ignored(self, tmp_path):
        write_labels(tmp_path, "a.txt", ["9 0.5 0.5 0.1 0.2", "", "  ", "21 0.2 0.2 0.1 0.1"])
        report = validate_labels(find_label_files(tmp_path))
        assert report["total"] == 2
        assert report["clean"]


class TestClassMerging:
    def test_relabels_duplicate_classes(self, tmp_path):
        path = write_labels(tmp_path, "a.txt", ["47 0.5 0.5 0.1 0.2", "49 0.2 0.2 0.1 0.1"])
        changed = apply_class_merges(find_label_files(tmp_path))

        assert changed == 2
        classes = [line.split()[0] for line in path.read_text().strip().splitlines()]
        assert classes == ["4", "21"]  # capacitor -> Capacitor, iC -> IC

    def test_leaves_canonical_classes_untouched(self, tmp_path):
        path = write_labels(tmp_path, "a.txt", ["4 0.5 0.5 0.1 0.2", "21 0.2 0.2 0.1 0.1"])
        before = path.read_text()

        assert apply_class_merges(find_label_files(tmp_path)) == 0
        assert path.read_text() == before

    def test_preserves_the_bounding_box_geometry(self, tmp_path):
        path = write_labels(tmp_path, "a.txt", ["47 0.5 0.5 0.1 0.2"])
        apply_class_merges(find_label_files(tmp_path))
        assert path.read_text().strip().split()[1:] == ["0.5", "0.5", "0.1", "0.2"]

    def test_is_idempotent(self, tmp_path):
        path = write_labels(tmp_path, "a.txt", ["47 0.5 0.5 0.1 0.2"])
        apply_class_merges(find_label_files(tmp_path))
        after_first = path.read_text()

        assert apply_class_merges(find_label_files(tmp_path)) == 0
        assert path.read_text() == after_first


class TestNonMaxSuppression:
    def test_empty_input(self):
        assert non_max_suppression([]) == []

    def test_single_box_survives(self):
        assert len(non_max_suppression([(0, 0, 10, 10, 0.9, 0)])) == 1

    def test_heavily_overlapping_boxes_collapse(self):
        boxes = [(0, 0, 10, 10, 0.9, 0), (1, 1, 11, 11, 0.8, 0)]
        assert len(non_max_suppression(boxes, iou_threshold=0.45)) == 1

    def test_the_highest_scoring_box_is_the_one_kept(self):
        boxes = [(0, 0, 10, 10, 0.6, 0), (1, 1, 11, 11, 0.95, 0)]
        kept = non_max_suppression(boxes, iou_threshold=0.45)
        assert len(kept) == 1
        assert kept[0][4] == pytest.approx(0.95)

    def test_distant_boxes_both_survive(self):
        boxes = [(0, 0, 10, 10, 0.9, 0), (100, 100, 110, 110, 0.8, 0)]
        assert len(non_max_suppression(boxes)) == 2

    def test_overlap_across_different_classes_is_kept(self):
        # A resistor box and an IC box at the same place are different claims,
        # not duplicates - suppression must be class-aware.
        boxes = [(0, 0, 10, 10, 0.9, 0), (1, 1, 11, 11, 0.8, 5)]
        assert len(non_max_suppression(boxes, iou_threshold=0.45)) == 2

    def test_threshold_controls_strictness(self):
        boxes = [(0, 0, 10, 10, 0.9, 0), (5, 5, 15, 15, 0.8, 0)]
        assert len(non_max_suppression(boxes, iou_threshold=0.9)) == 2
        assert len(non_max_suppression(boxes, iou_threshold=0.1)) == 1
