"""Cut big board photos into overlapping tiles (images + YOLO labels).

Use this when the Roboflow export is full resolution (export a version WITHOUT the 640x640 resize):
the parts are only a few pixels wide in the 640 stretch, tiles keep their real size.

  python tools/tile_dataset.py --src final_dataset/fine --dst final_dataset_tiled/fine --tile 800 --overlap 0.25
"""
import argparse
import shutil
from pathlib import Path

import cv2
import yaml


def tiles_of(length, tile, step):
    pos = list(range(0, max(length - tile, 0) + 1, step))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--tile", type=int, default=800); ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-visible", type=float, default=0.5, help="keep a box if this fraction of it is inside the tile")
    ap.add_argument("--min-px", type=int, default=3, help="drop boxes smaller than this after clipping")
    ap.add_argument("--keep-empty", type=float, default=0.1, help="share of label-free tiles to keep as background")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    step = int(a.tile * (1 - a.overlap))
    n_img = n_box = 0
    for sp in ("train", "val", "test"):
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True); (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for img_p in sorted((src / "images" / sp).glob("*.*")):
            img = cv2.imread(str(img_p)); H, W = img.shape[:2]
            lbl = src / "labels" / sp / (img_p.stem + ".txt")
            boxes = [tuple(map(float, l.split())) for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            for ty in tiles_of(H, a.tile, step):
                for tx in tiles_of(W, a.tile, step):
                    th, tw = min(a.tile, H), min(a.tile, W)
                    out = []
                    for c, cx, cy, w, h in boxes:
                        x0, x1, y0, y1 = (cx - w / 2) * W, (cx + w / 2) * W, (cy - h / 2) * H, (cy + h / 2) * H
                        ix0, ix1, iy0, iy1 = max(x0, tx), min(x1, tx + tw), max(y0, ty), min(y1, ty + th)
                        if ix1 <= ix0 or iy1 <= iy0:
                            continue
                        if (ix1 - ix0) * (iy1 - iy0) < a.min_visible * (x1 - x0) * (y1 - y0) or min(ix1 - ix0, iy1 - iy0) < a.min_px:
                            continue
                        out.append((int(c), ((ix0 + ix1) / 2 - tx) / tw, ((iy0 + iy1) / 2 - ty) / th, (ix1 - ix0) / tw, (iy1 - iy0) / th))
                    if not out and (sp != "train" or hash((img_p.stem, tx, ty)) % 100 >= a.keep_empty * 100):
                        continue
                    name = f"{img_p.stem}_x{tx}_y{ty}"
                    cv2.imwrite(str(dst / "images" / sp / (name + ".jpg")), img[ty:ty + th, tx:tx + tw], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (dst / "labels" / sp / (name + ".txt")).write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in out))
                    n_img += 1; n_box += len(out)
    y = yaml.safe_load(open(src / "data.yaml", encoding="utf8"))
    y["path"] = str(dst.resolve())
    yaml.safe_dump(y, open(dst / "data.yaml", "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
    print(f"{n_img} tiles, {n_box} boxes -> {dst}")


if __name__ == "__main__":
    main()
