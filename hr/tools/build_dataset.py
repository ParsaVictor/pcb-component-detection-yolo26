"""Build the training datasets (fine classes + class-agnostic "component") from data/all_data.

Steps
  1. keep only the chosen sources
  2. find the same board photographed/labelled in several sources (twins) and keep one copy
  3. split by BOARD (never by file): test comes from the client's own boards (source parsa)
  4. map classes with configs/class_map.yaml, drop duplicate boxes
  5. optional offline rot90/rot270 copies of the TRAIN images (colour is kept)

Output: final_dataset/{fine,any}/{images,labels}/{train,val,test}, data.yaml, split_manifest.csv, report.txt
Usage:  python tools/build_dataset.py --sources parsa carddata dhbw
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PRIORITY = ["parsa", "carddata", "dhbw", "tesis", "thai34"]   # which twin survives (all colour sources)
IMG_EXT = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- image quality
def image_quality(path):
    """(mean saturation 0-255, share of near-black pixels): grayscale exports and heavy rotation fill are rejected."""
    im = cv2.imread(str(path))
    return float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean()), float((im.max(axis=2) <= 3).mean())


# ---------------------------------------------------------------- twins
def image_feature(path, size=32):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(cv2.equalizeHist(g), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean()
    return (g / (np.linalg.norm(g) + 1e-6))


def find_twin_clusters(files, sources, thr, names=None, twins_csv=None, min_inliers=35):
    """Union images of DIFFERENT sources that show the same board:
    SIFT matches from data/all_data/twins.csv (tools/find_twins.py) and thumbnail correlation (any 90-degree rotation)."""
    feats = [image_feature(f) for f in files]
    flat = np.array([f.ravel() for f in feats])
    rots = np.array([[np.rot90(f, k).ravel() for k in range(4)] for f in feats])
    sim = np.zeros((len(files), len(files)), np.float32)
    for k in range(4):
        sim = np.maximum(sim, flat @ rots[:, k, :].T)
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    src = np.array(sources)
    for i, j in zip(*np.where(np.triu(sim >= thr, 1))):
        if src[i] != src[j]:
            parent[find(i)] = find(j)
    if twins_csv and Path(twins_csv).exists():
        idx = {n: i for i, n in enumerate(names)}
        for r in csv.DictReader(open(twins_csv, encoding="utf8")):
            if int(r["inliers"]) >= min_inliers and r["file_a"] in idx and r["file_b"] in idx:
                parent[find(idx[r["file_a"]])] = find(idx[r["file_b"]])
    return [find(i) for i in range(len(files))]


# ---------------------------------------------------------------- labels
def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax1, bx1)-max(ax0, bx0)), max(0, min(ay1, by1)-max(ay0, by0))
    inter = iw*ih
    return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-12)


def dedupe(boxes, thr=0.85):
    """boxes: [(cls, cx, cy, w, h)] -> remove same-class near-identical boxes."""
    kept, removed = [], 0
    for b in boxes:
        if any(k[0] == b[0] and iou(k[1:], b[1:]) > thr for k in kept):
            removed += 1
        else:
            kept.append(b)
    return kept, removed


def rot90_boxes(boxes, k):
    """k=1: image rotated 90 deg counter-clockwise (cv2.ROTATE_90_COUNTERCLOCKWISE), k=3: clockwise."""
    out = []
    for c, x, y, w, h in boxes:
        out.append((c, y, 1-x, h, w) if k == 1 else (c, 1-y, x, h, w))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-data", default="data/all_data")
    ap.add_argument("--class-map", default="configs/class_map.yaml")
    ap.add_argument("--out", default="final_dataset")
    ap.add_argument("--sources", nargs="+", default=["parsa", "carddata", "dhbw"])
    ap.add_argument("--twin-thr", type=float, default=0.90)
    ap.add_argument("--twin-inliers", type=int, default=100)
    ap.add_argument("--split-tries", type=int, default=2000, help="random board splits tried; the most class-balanced wins")
    ap.add_argument("--keep-twins", action="store_true", help="do not drop twin copies from other sources")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--no-rot90", action="store_true", help="skip offline rot90/rot270 copies of train images")
    ap.add_argument("--min-sat", type=float, default=3.0, help="reject grayscale images (colour matters)")
    ap.add_argument("--max-black", type=float, default=0.60, help="reject images that are mostly black fill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-from", default="", help="split_manifest.csv of an earlier build: reuse its boards and "
                    "train/val/test split exactly (used for the full-resolution rebuild, so results stay comparable)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    A, OUT = Path(a.all_data), Path(a.out)

    raw_names = yaml.safe_load(open(A/"data.yaml", encoding="utf8"))["names"]
    cmap = yaml.safe_load(open(a.class_map, encoding="utf8"))
    fine_names = list(cmap["fine"].keys())
    raw2fine = {raw: fine for fine, raws in cmap["fine"].items() for raw in raws}
    comp_groups = set(cmap["component_groups"])
    dropped_raw = set(cmap.get("drop", []))

    rows = [r for r in csv.DictReader(open(A/"manifest.csv", encoding="utf8")) if r["source"] in a.sources]
    quality, bad = {}, []
    for r in rows:
        sat, blk = image_quality(A/"images"/r["file"])
        quality[r["file"]] = (sat, blk)
        if sat < a.min_sat or blk > a.max_black:
            bad.append((r["file"], round(sat, 1), round(blk, 2)))
    bad_names = {b[0] for b in bad}
    rows = [r for r in rows if r["file"] not in bad_names]
    files = [A/"images"/r["file"] for r in rows]
    print(f"{len(rows)} images from {a.sources}; rejected {len(bad)} (grayscale or mostly black): {bad[:5]}")

    fixed_split = {}
    if a.split_from:
        prev = {r["file"]: r for r in csv.DictReader(open(a.split_from, encoding="utf8"))}
        missing = [r["file"] for r in rows if r["file"] not in prev]
        rows = [r for r in rows if r["file"] in prev]
        files = [A/"images"/r["file"] for r in rows]
        cluster = [int(prev[r["file"]]["board"]) for r in rows]
        fixed_split = {int(prev[r["file"]]["board"]): prev[r["file"]]["split"] for r in rows}
        print(f"split taken from {a.split_from}: {len(rows)} images, {len(missing)} not in it (skipped)")
    else:
        cluster = find_twin_clusters(files, [r["source"] for r in rows], a.twin_thr, [r["file"] for r in rows],
                                     A/"twins.csv", a.twin_inliers)
    members = defaultdict(list)
    for i, c in enumerate(cluster):
        members[c].append(i)
    keep, twin_dropped = [], []
    for c, idx in members.items():
        idx.sort(key=lambda i: PRIORITY.index(rows[i]["source"]))
        keep.append(idx[0])
        (keep.extend(idx[1:]) if a.keep_twins else twin_dropped.extend(idx[1:]))
    print(f"{len(members)} distinct boards; {len(twin_dropped)} twin copies dropped" if not a.keep_twins
          else f"{len(members)} distinct boards (twins kept)")

    # ---- labels for every kept image (needed to balance the split)
    unmapped, dup_removed, labels_of = Counter(), 0, {}
    for i in keep:
        r = rows[i]
        raw_boxes = []
        for ln in (A/"labels"/(Path(r["file"]).stem + ".txt")).read_text(encoding="utf8").splitlines():
            p = ln.split()
            if len(p) == 5:
                raw_boxes.append((f'{r["source"]}:{raw_names[int(p[0])].split(":", 1)[1]}', *map(float, p[1:])))
        per_view = {"fine": [], "any": []}
        for raw, cx, cy, w, h in raw_boxes:
            if raw in dropped_raw:
                continue
            fine = raw2fine.get(raw)
            if fine is None:
                unmapped[raw] += 1
                continue
            per_view["fine"].append((fine_names.index(fine), cx, cy, w, h))
            if fine in comp_groups:
                per_view["any"].append((0, cx, cy, w, h))
        for view in per_view:
            per_view[view], n = dedupe(per_view[view])
            dup_removed += n if view == "fine" else 0
        labels_of[i] = per_view

    # ---- split by board (cluster); test boards come from the client's source (parsa)
    boards = sorted(members.keys())
    board_cls = {c: np.zeros(len(fine_names)) for c in boards}
    for i in keep:
        for b in labels_of[i]["fine"]:
            board_cls[cluster[i]][b[0]] += 1
    total = sum(board_cls.values()) + 1e-9
    n_test, n_val = round(len(boards)*a.test_frac), round(len(boards)*a.val_frac)
    parsa_boards = [c for c in boards if any(rows[i]["source"] == "parsa" for i in members[c])]
    pool_test = parsa_boards if len(parsa_boards) >= n_test else boards
    best = None
    for _ in range(0 if fixed_split else a.split_tries):
        t = set(rnd.sample(pool_test, n_test))
        v = set(rnd.sample([c for c in boards if c not in t], n_val))
        tv = sum(board_cls[c] for c in v) / total; tt = sum(board_cls[c] for c in t) / total
        score = float((((tv - a.val_frac) / a.val_frac) ** 2 + ((tt - a.test_frac) / a.test_frac) ** 2).sum())
        if best is None or score < best[0]:
            best = (score, t, v)
    test_boards, val_boards = (best[1], best[2]) if best else (set(), set())
    split_of = {c: "test" if c in test_boards else "val" if c in val_boards else "train" for c in boards}
    if fixed_split:
        split_of = {c: fixed_split[c] for c in boards}
        test_boards = {c for c in boards if split_of[c] == "test"}
        val_boards = {c for c in boards if split_of[c] == "val"}
        print("split: reused unchanged")
    else:
        print(f"split balance score {best[0]:.2f} (0 = every class split exactly {a.val_frac:.0%}/{a.test_frac:.0%})")

    for view in ("fine", "any"):
        for sp in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = OUT/view/sub/sp
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)

    stats = {v: {s: Counter() for s in ("train", "val", "test")} for v in ("fine", "any")}
    manifest = []
    for i in keep:
        r = rows[i]
        sp = split_of[cluster[i]]
        img = cv2.imread(str(A/"images"/r["file"]))
        per_view = labels_of[i]

        variants = [("", img, None)]
        if sp == "train" and not a.no_rot90:
            variants += [("_r90", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),
                         ("_r270", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 3)]
        stem = Path(r["file"]).stem
        for suffix, im, k in variants:
            name = stem + suffix
            for view, boxes in per_view.items():
                bx = boxes if k is None else rot90_boxes(boxes, k)
                cv2.imwrite(str(OUT/view/"images"/sp/(name + ".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (OUT/view/"labels"/sp/(name + ".txt")).write_text(
                    "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in bx), encoding="utf8")
                for b in bx:
                    stats[view][sp][b[0]] += 1
        manifest.append({"file": r["file"], "source": r["source"], "board": cluster[i], "split": sp,
                         "boxes_fine": len(per_view["fine"]), "boxes_any": len(per_view["any"])})

    for view, names in (("fine", fine_names), ("any", ["component"])):
        yaml.safe_dump({"path": str((OUT/view).resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "nc": len(names), "names": {i: n for i, n in enumerate(names)}},
                       open(OUT/view/"data.yaml", "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
    with open(OUT/"split_manifest.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)

    lines = [f"sources: {a.sources}", f"boards: {len(boards)}  train/val/test boards: "
             f"{sum(v=='train' for v in split_of.values())}/{len(val_boards)}/{len(test_boards)}",
             f"twin copies dropped: {len(twin_dropped)}   duplicate boxes removed: {dup_removed}", ""]
    for view, names in (("fine", fine_names), ("any", ["component"])):
        lines.append(f"[{view}] boxes per class (train / val / test)")
        for ci, n in enumerate(names):
            lines.append(f"  {n:24s}{stats[view]['train'][ci]:8d}{stats[view]['val'][ci]:8d}{stats[view]['test'][ci]:8d}")
        lines.append("")
    if unmapped:
        lines.append("UNMAPPED raw classes (dropped, add them to configs/class_map.yaml if needed):")
        lines += [f"  {k}: {v}" for k, v in unmapped.most_common()]
    (OUT/"report.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
