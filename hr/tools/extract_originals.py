"""Pick the least-augmented copy of every source image from Roboflow exports.

Roboflow names every exported file `<name>_<ext>.rf.<hash>.<ext>`, so all augmented
copies of one source image share the part before `.rf.`. Within a group we keep the
copy with the least augmentation evidence:
  * black rotation fill (fraction of near-black pixels)
  * salt-and-pepper noise (difference to a 3x3 median)
  * label box area (rotation / shear inflates the boxes)
Exact pixel duplicates are dropped. Source folders are never modified.

Usage: python extract_originals.py <dataset_dir> <out_dir>
"""
import csv
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from yololib import parse_line
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
RF_RE = re.compile(r"\.rf\.[0-9a-f]+$")


def base_name(stem):
    return RF_RE.sub("", stem)


def mean_box_area(label_path):
    if not label_path.exists():
        return 0.0
    areas = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        b = parse_line(ln)
        if b:
            areas.append(b[3] * b[4])
    return float(np.mean(areas)) if areas else 0.0


def metrics(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    black = float((img.max(axis=2) <= 3).mean())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = float(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)).mean())
    digest = hashlib.md5(img.tobytes()).hexdigest()
    return {"black": black, "noise": noise, "area": mean_box_area(label_path), "md5": digest}


def ranks(values):
    order = np.argsort(values, kind="stable")
    r = np.empty(len(values))
    r[order] = np.arange(len(values))
    return r


def main(src, dst):
    src, dst = Path(src), Path(dst)
    out_img, out_lbl = dst / "images", dst / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                groups[base_name(p.stem)].append((split, p, src / split / "labels" / (p.stem + ".txt")))

    rows = []
    for name, items in sorted(groups.items()):
        seen, cands = set(), []
        for split, img, lbl in items:
            m = metrics(img, lbl)
            if m is None or m["md5"] in seen:
                continue
            seen.add(m["md5"])
            cands.append((split, img, lbl, m))
        if not cands:
            continue
        score = sum(ranks([c[3][k] for c in cands]) for k in ("black", "noise", "area"))
        split, img, lbl, m = cands[int(np.argmin(score))]
        shutil.copy2(img, out_img / (name + img.suffix))
        if lbl.exists():
            shutil.copy2(lbl, out_lbl / (name + ".txt"))
        rows.append({
            "source_name": name, "kept_file": img.name, "original_split": split,
            "copies_in_group": len(items), "unique_copies": len(cands),
            "splits_in_group": "+".join(sorted({c[0] for c in cands})),
            "black_frac": round(m["black"], 4), "noise": round(m["noise"], 3),
            "mean_box_area": round(m["area"], 5),
        })

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    yaml = src / "data.yaml"
    if yaml.exists():
        shutil.copy2(yaml, dst / "source_data.yaml")
    leaks = sum(1 for r in rows if "+" in r["splits_in_group"])
    print(f"{src.name}: {sum(len(v) for v in groups.values())} files -> {len(rows)} source images; "
          f"{leaks} groups were spread over several splits (leakage)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
