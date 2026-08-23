"""Dataset download and label validation.

The dataset is hosted on Roboflow Universe rather than committed here — it is
724 MB across 3,170 annotated boards, which does not belong in a git history.
"""

import math
import os
from pathlib import Path

WORKSPACE = "parsa-karkooti"
PROJECT = "pcb-wik5v-zonih"
VERSION = 1
UNIVERSE_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"

#: Class names, in label-index order, exactly as the annotations use them.
CLASS_NAMES = [
    "-", "Battery", "Button", "Buzzer", "Capacitor", "Capacitor Jumper",
    "Capacitor Network", "Clock", "Conductor", "Connector", "Diode", "Display",
    "EM", "Electrolytic Capacitor", "Electrolytic capacitor", "Ferrite Bead",
    "Fiducial", "Filter", "Flex Cable", "Fuse", "Heatsink", "IC", "Indicator",
    "Inductor", "Jumper", "Led", "PS", "Pads", "Pins", "Potentiometer", "RC",
    "RP", "Resistance", "Resistor", "Resistor Jumper", "Resistor Network",
    "Resistor1", "Resistort", "Ressistor", "SK", "Switch", "Test Point",
    "Transducer", "Transformer", "Transistor", "Unknown Unlabeled",
    "Zener Diode", "capacitor", "coil", "iC",
]

NUM_CLASSES = len(CLASS_NAMES)

#: Classes that are the same physical part under a different spelling or case.
#: Merging these is the single highest-value fix available to this dataset —
#: see docs/dataset-analysis.md.
DUPLICATE_MERGES = {
    14: 13,  # "Electrolytic capacitor" -> "Electrolytic Capacitor"
    47: 4,   # "capacitor"              -> "Capacitor"
    49: 21,  # "iC"                     -> "IC"
    36: 33,  # "Resistor1"              -> "Resistor"
    37: 33,  # "Resistort"              -> "Resistor"
    38: 33,  # "Ressistor"              -> "Resistor"
    32: 33,  # "Resistance"             -> "Resistor"
}


def download_dataset(api_key=None, version=VERSION, fmt="yolov8"):
    """Download the dataset from Roboflow. Returns the local dataset directory.

    The API key is read from ``ROBOFLOW_API_KEY`` if not passed explicitly.
    """
    from roboflow import Roboflow

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Roboflow API key.\n"
            "  export ROBOFLOW_API_KEY='your_key'\n"
            "Get a free key at https://app.roboflow.com/settings/api\n"
            f"Dataset: {UNIVERSE_URL}"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    return project.version(version).download(fmt).location


def write_data_yaml(path, root="."):
    """Write the ``data.yaml`` that Ultralytics reads."""
    names = "\n".join(f"  {i}: '{name}'" for i, name in enumerate(CLASS_NAMES))
    content = (
        f"train: {root}/train/images\n"
        f"val:   {root}/valid/images\n"
        f"test:  {root}/test/images\n\n"
        f"nc: {NUM_CLASSES}\n"
        f"names:\n{names}\n"
    )
    Path(path).write_text(content, encoding="utf-8")
    return path


def find_label_files(root="."):
    """Every YOLO label file under ``root``."""
    return sorted(Path(root).glob("**/labels/*.txt"))


def validate_labels(label_paths, num_classes=NUM_CLASSES):
    """Check that every annotation is well-formed.

    YOLO format is ``class x_center y_center width height``, with the four
    geometry values normalised to [0, 1]. Returns a dict of issue counts.
    """
    issues = {"format": 0, "out_of_range": 0, "class_out_of_bounds": 0}
    total = 0

    for path in label_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            total += 1
            parts = line.split()
            if len(parts) < 5:
                issues["format"] += 1
                continue

            if not 0 <= int(float(parts[0])) < num_classes:
                issues["class_out_of_bounds"] += 1

            values = [float(v) for v in parts[1:5]]
            if any(math.isnan(v) or v < 0.0 or v > 1.0 for v in values):
                issues["out_of_range"] += 1

    return {"total": total, "issues": issues, "clean": not any(issues.values())}


def apply_class_merges(label_paths, mapping=None):
    """Rewrite label files in place, merging duplicate classes.

    Destructive: it edits the label files. Returns the number of annotations
    that changed class.
    """
    mapping = DUPLICATE_MERGES if mapping is None else mapping
    changed = 0

    for path in label_paths:
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()

        out, modified = [], False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append("")
                continue

            parts = stripped.split()
            cls = int(float(parts[0]))
            if cls in mapping:
                parts[0] = str(mapping[cls])
                modified = True
                changed += 1
            out.append(" ".join(parts))

        if modified:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")

    return changed
