#!/usr/bin/env python
"""Run detection on an image or a folder.

    python scripts/predict.py --weights runs/train/pcb_yolov8/weights/best.pt \
        --source data/samples --save

Add --tile for high-resolution boards where the components are small relative
to the frame.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcb_detection import CLASS_NAMES, tile_inference  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image file or folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save", action="store_true", help="write annotated images")
    parser.add_argument("--tile", action="store_true", help="use tiled inference")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    return parser.parse_args()


def main():
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)

    if args.tile:
        if os.path.isdir(args.source):
            raise SystemExit("--tile takes a single image, not a folder")

        boxes = tile_inference(
            model,
            args.source,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou=args.iou,
        )
        print(f"{len(boxes)} detections in {args.source}")

        counts = {}
        for *_, cls in boxes:
            name = CLASS_NAMES[int(cls)]
            counts[name] = counts.get(name, 0) + 1
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:24s}: {count}")
        return

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=args.save,
    )
    total = sum(len(r.boxes) for r in results)
    print(f"{total} detections across {len(results)} image(s)")
    if args.save:
        print("Annotated images written under runs/detect/")


if __name__ == "__main__":
    main()
