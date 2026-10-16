"""Full-resolution dataset: the same boards and the same train/val/test split as final_dataset, but built from
the original photos (tools/fetch_originals.py) and cut into overlapping tiles so every part keeps its real size.

  1 collect   data/clean_hr -> data/all_data_hr                    (tools/build_all_data.py)
  2 build     class map + split reused from final_dataset           (tools/build_dataset.py --split-from, no rot90 copies)
                -> data/hr_full/{fine,any}
  3 tile      tiles of --tile px with --overlap, images written ONCE (fine/images); any/ only has labels
                (the Colab notebook links any/images to fine/images after unzipping)   -> final_dataset_hr
  4 boards    the full test boards for board-level previews        -> final_dataset_hr/boards_test
  5 zip       upload_to_drive/pcb_data_hr.zip

  python tools/build_hr_dataset.py
"""
import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd):
    print("   $", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)


def tiles_of(length, tile, step):
    pos = list(range(0, max(length - tile, 0) + 1, step))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def read_boxes(p):
    if not p.exists():
        return []
    return [tuple(map(float, l.split())) for l in p.read_text(encoding="utf8").splitlines() if len(l.split()) == 5]


def clip_to_tile(boxes, W, H, tx, ty, tw, th, min_visible, min_px):
    out = []
    for c, cx, cy, w, h in boxes:
        x0, x1, y0, y1 = (cx - w / 2) * W, (cx + w / 2) * W, (cy - h / 2) * H, (cy + h / 2) * H
        ix0, ix1, iy0, iy1 = max(x0, tx), min(x1, tx + tw), max(y0, ty), min(y1, ty + th)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) < min_visible * (x1 - x0) * (y1 - y0) or min(ix1 - ix0, iy1 - iy0) < min_px:
            continue
        out.append((int(c), ((ix0 + ix1) / 2 - tx) / tw, ((iy0 + iy1) / 2 - ty) / th, (ix1 - ix0) / tw, (iy1 - iy0) / th))
    return out


def fmt(boxes):
    return "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes)


def tile_all(src, dst, a):
    """src/{fine,any}/{images,labels}/{split} -> dst: tiles once in fine/images, labels for both views."""
    stats = {v: {s: Counter() for s in ("train", "val", "test")} for v in ("fine", "any")}
    n_tiles, scales = Counter(), []
    for sp in ("train", "val", "test"):
        for v in ("fine", "any"):
            (dst / v / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (dst / "fine" / "images" / sp).mkdir(parents=True, exist_ok=True)
        for img_p in sorted((src / "fine" / "images" / sp).glob("*.jpg")):
            img = cv2.imread(str(img_p))
            fine = read_boxes(src / "fine" / "labels" / sp / (img_p.stem + ".txt"))
            anyb = read_boxes(src / "any" / "labels" / sp / (img_p.stem + ".txt"))
            H, W = img.shape[:2]
            # very large photos: shrink so the median part is at most --max-part px (keeps the part scale
            # similar across sources and avoids hundreds of near-empty tiles); never enlarge
            s = 1.0
            if fine and a.max_part:
                med = float(np.median([min(b[3] * W, b[4] * H) for b in fine]))
                if med > a.max_part:
                    s = a.max_part / med
            if s < 1.0:
                img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
                H, W = img.shape[:2]
            scales.append(s)
            step = int(a.tile * (1 - a.overlap))
            for ty in tiles_of(H, a.tile, step):
                for tx in tiles_of(W, a.tile, step):
                    th, tw = min(a.tile, H), min(a.tile, W)
                    f = clip_to_tile(fine, W, H, tx, ty, tw, th, a.min_visible, a.min_px)
                    g = clip_to_tile(anyb, W, H, tx, ty, tw, th, a.min_visible, a.min_px)
                    if not f and not g:
                        # background tiles: a few in train only (deterministic choice)
                        if sp != "train" or zlib.crc32(f"{img_p.stem}_{tx}_{ty}".encode()) % 100 >= a.keep_empty * 100:
                            continue
                    name = f"{img_p.stem}_x{tx}_y{ty}"
                    cv2.imwrite(str(dst / "fine" / "images" / sp / (name + ".jpg")), img[ty:ty + th, tx:tx + tw],
                                [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    (dst / "fine" / "labels" / sp / (name + ".txt")).write_text(fmt(f), encoding="utf8")
                    (dst / "any" / "labels" / sp / (name + ".txt")).write_text(fmt(g), encoding="utf8")
                    n_tiles[sp] += 1
                    for b in f:
                        stats["fine"][sp][b[0]] += 1
                    for b in g:
                        stats["any"][sp][b[0]] += 1
    for v in ("fine", "any"):
        y = yaml.safe_load(open(src / v / "data.yaml", encoding="utf8"))
        y["path"] = "."   # the training script points this at the real folder
        yaml.safe_dump(y, open(dst / v / "data.yaml", "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
    return stats, n_tiles, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-visible", type=float, default=0.5, help="keep a box cut by the tile edge if this share is inside")
    ap.add_argument("--min-px", type=int, default=3)
    ap.add_argument("--keep-empty", type=float, default=0.10, help="share of part-free train tiles kept as background")
    ap.add_argument("--max-part", type=float, default=64, help="shrink photos whose median part is bigger than this (px); 0 = off")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--skip-collect", action="store_true")
    a = ap.parse_args()

    if not a.skip_collect:
        shutil.rmtree(ROOT / "data" / "all_data_hr", ignore_errors=True)
        run("tools/build_all_data.py", "--clean", "data/clean_hr", "--out", "data/all_data_hr")
        shutil.rmtree(ROOT / "data" / "hr_full", ignore_errors=True)
        run("tools/build_dataset.py", "--all-data", "data/all_data_hr", "--out", "data/hr_full",
            "--sources", "parsa", "carddata", "dhbw", "--split-from", "final_dataset/split_manifest.csv", "--no-rot90",
            "--min-sat", "0", "--max-black", "1.0")

    src, dst = ROOT / "data" / "hr_full", ROOT / "final_dataset_hr"
    shutil.rmtree(dst, ignore_errors=True)
    print("tiling ...", flush=True)
    stats, n_tiles, scales = tile_all(src, dst, a)

    # full test boards, for board-level previews with tiled inference
    bt = dst / "boards_test"
    for sub in ("images", "labels_fine"):
        (bt / sub).mkdir(parents=True, exist_ok=True)
    for p in (src / "fine" / "images" / "test").glob("*.jpg"):
        shutil.copy2(p, bt / "images" / p.name)
        shutil.copy2(src / "fine" / "labels" / "test" / (p.stem + ".txt"), bt / "labels_fine" / (p.stem + ".txt"))
    shutil.copy2(src / "split_manifest.csv", dst / "split_manifest.csv")

    names = {v: yaml.safe_load(open(dst / v / "data.yaml", encoding="utf8"))["names"] for v in ("fine", "any")}
    lines = [(src / "report.txt").read_text(encoding="utf8").split("\n\n")[0], "",
             f"FULL RESOLUTION, tiles {a.tile}px overlap {a.overlap:.0%} (images stored once in fine/images)",
             f"tiles train/val/test: {n_tiles['train']}/{n_tiles['val']}/{n_tiles['test']}",
             f"photos shrunk because their parts were larger than {a.max_part:.0f}px: {sum(s < 1 for s in scales)} of {len(scales)}", ""]
    for v in ("fine", "any"):
        nm = names[v]
        nm = list(nm.values()) if isinstance(nm, dict) else nm
        lines.append(f"[{v}] boxes per class (train / val / test)")
        for ci, n in enumerate(nm):
            lines.append(f"  {n:24s}{stats[v]['train'][ci]:8d}{stats[v]['val'][ci]:8d}{stats[v]['test'][ci]:8d}")
        lines.append("")
    rep = (src / "report.txt").read_text(encoding="utf8")
    if "UNMAPPED" in rep:
        lines.append(rep[rep.index("UNMAPPED"):])
    (dst / "report.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))

    out = ROOT / "upload_to_drive"
    out.mkdir(exist_ok=True)
    z = out / "pcb_data_hr.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(dst.rglob("*")):
            if f.is_file():
                zf.write(f, Path("final_dataset_hr") / f.relative_to(dst))
    print(f"zip: {z} ({z.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
