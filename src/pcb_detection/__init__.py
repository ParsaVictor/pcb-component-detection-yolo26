"""YOLOv8 detection of electronic components on populated PCBs.

A baseline detector over a 50-class, 3,170-board dataset, with the dataset
tooling and analysis needed to understand — and improve on — its numbers.
"""

from .dataset import (
    CLASS_NAMES,
    DUPLICATE_MERGES,
    NUM_CLASSES,
    UNIVERSE_URL,
    apply_class_merges,
    download_dataset,
    find_label_files,
    validate_labels,
    write_data_yaml,
)
from .tiling import non_max_suppression, tile_inference
from .training import best_weights, evaluate, train

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "DUPLICATE_MERGES",
    "NUM_CLASSES",
    "UNIVERSE_URL",
    "apply_class_merges",
    "best_weights",
    "download_dataset",
    "evaluate",
    "find_label_files",
    "non_max_suppression",
    "tile_inference",
    "train",
    "validate_labels",
    "write_data_yaml",
]
