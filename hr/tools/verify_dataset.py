"""Check a built dataset before training (runs locally and on Colab, takes about a minute).

Checks: every image has a label file; label values in range; class ids valid; images are in colour;
no board appears in two splits (needs split_manifest.csv); boxes per split/class; tiny-box share.
Exit code 1 when a hard error is found, so a pipeline can stop before wasting GPU time.

  python tools/verify_dataset.py --root final_dataset
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def check_view(root, view):
    d = root / view
    cfg = yaml.safe_load(open(d / "data.yaml", encoding="utf8"))
    names = list(cfg["names"].values()) if isinstance(cfg["names"], dict) else cfg["names"]
    out = {"errors": [], "warnings": [], "splits": {}}
    img_root = d / "images" if (d / "images").exists() else root / "fine" / "images"   # HR zip stores pictures once
    for sp in ("train", "val", "test"):
        imgs = sorted((img_root / sp).glob("*.jpg"))
        cnt, tiny, n_box, sat_low, empty = Counter(), 0, 0, 0, 0
        for im in imgs:
            lbl = d / "labels" / sp / (im.stem + ".txt")
            if not lbl.exists():
                out["errors"].append(f"{view}/{sp}: missing label for {im.name}")
                continue
            lines = [l.split() for l in lbl.read_text(encoding="utf8").splitlines() if l.strip()]
            if not lines:
                empty += 1
            for p in lines:
                try:
                    c, x, y, w, h = int(p[0]), *map(float, p[1:5])
                except Exception:
                    out["errors"].append(f"{view}/{sp}: bad line in {lbl.name}: {p}")
                    continue
                if not (0 <= c < len(names)) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    out["errors"].append(f"{view}/{sp}: out-of-range value in {lbl.name}: {p}")
                    continue
                cnt[names[c]] += 1; n_box += 1
                tiny += min(w, h) * 640 < 4
            if len(imgs) <= 120 or im.name.endswith("_r90.jpg") is False:
                img = cv2.imread(str(im))
                if float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) < 3:
                    sat_low += 1
        out["splits"][sp] = {"images": len(imgs), "boxes": n_box, "empty_label_files": empty,
                             "boxes_under_4px_at_640": tiny, "grayscale_images": sat_low, "per_class": dict(cnt)}
        if sat_low:
            out["errors"].append(f"{view}/{sp}: {sat_low} grayscale images")
        if not imgs:
            out["errors"].append(f"{view}/{sp}: no images")
        for n in names:
            if sp != "train" and cnt[n] < 10:
                out["warnings"].append(f"{view}/{sp}: class '{n}' has only {cnt[n]} boxes (its metric will be noisy)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="final_dataset")
    a = ap.parse_args()
    root = Path(a.root)
    report = {v: check_view(root, v) for v in ("any", "fine")}

    man = root / "split_manifest.csv"
    if man.exists():
        boards = defaultdict(set)
        for r in csv.DictReader(open(man, encoding="utf8")):
            boards[r["board"]].add(r["split"])
        leaks = [b for b, s in boards.items() if len(s) > 1]
        report["boards"] = len(boards)
        report["boards_in_several_splits"] = len(leaks)
        if leaks:
            report["fine"]["errors"].append(f"{len(leaks)} boards appear in more than one split")

    errors = report["any"]["errors"] + report["fine"]["errors"]
    (root / "qa_report.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    for v in ("any", "fine"):
        print(f"[{v}]", {sp: (s["images"], s["boxes"]) for sp, s in report[v]["splits"].items()}, "(images, boxes)")
    print("boards:", report.get("boards"), " boards in several splits:", report.get("boards_in_several_splits"))
    tiny = sum(s["boxes_under_4px_at_640"] for s in report["fine"]["splits"].values())
    tot = sum(s["boxes"] for s in report["fine"]["splits"].values())
    print(f"boxes thinner than 4 px at 640: {tiny}/{tot} ({100 * tiny / max(tot, 1):.0f}%)")
    for w in report["fine"]["warnings"][:10]:
        print("WARN", w)
    if errors:
        print(f"\n{len(errors)} ERRORS, first ones:")
        for e in errors[:10]:
            print("  ", e)
        sys.exit(1)
    print("DATASET OK")


if __name__ == "__main__":
    main()
