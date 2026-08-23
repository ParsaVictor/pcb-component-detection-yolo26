#!/usr/bin/env python
"""Download the PCB component dataset from Roboflow Universe.

    export ROBOFLOW_API_KEY="your_key"
    python scripts/download_dataset.py

Free key: https://app.roboflow.com/settings/api
Dataset:  https://universe.roboflow.com/parsa-karkooti/pcb-wik5v-zonih
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcb_detection import (  # noqa: E402
    UNIVERSE_URL,
    download_dataset,
    find_label_files,
    validate_labels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--format", default="yolov8")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    print(f"Dataset: {UNIVERSE_URL}")
    location = download_dataset(version=args.version, fmt=args.format)
    print(f"Downloaded to: {location}")

    if args.skip_validation:
        return

    print("\nValidating labels...")
    labels = find_label_files(location)
    report = validate_labels(labels)
    print(f"  {len(labels)} label files, {report['total']} annotations")
    if report["clean"]:
        print("  All annotations are well-formed.")
    else:
        print(f"  Issues found: {report['issues']}")


if __name__ == "__main__":
    main()
