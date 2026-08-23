#!/usr/bin/env python
"""Fine-tune YOLOv8 on the PCB component dataset.

    python scripts/train.py --data PCB-1/data.yaml --epochs 50 --device 0

Pass --merge-duplicates to collapse the duplicate class labels first; see
docs/dataset-analysis.md for why that matters.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcb_detection import (  # noqa: E402
    apply_class_merges,
    evaluate,
    find_label_files,
    train,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    parser.add_argument("--name", default="pcb_yolov8")
    parser.add_argument(
        "--merge-duplicates",
        action="store_true",
        help="collapse duplicate class labels before training (edits label files)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" else int(args.device)

    if args.merge_duplicates:
        root = os.path.dirname(os.path.abspath(args.data))
        changed = apply_class_merges(find_label_files(root))
        print(f"Merged duplicate classes: {changed} annotations relabelled\n")

    model, _ = train(
        args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
    )

    print("\nEvaluating on the validation split...")
    metrics = evaluate(model, args.data, device=device)
    for key in ("precision", "recall", "mAP50", "mAP50-95"):
        print(f"  {key:10s}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
