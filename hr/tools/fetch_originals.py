"""Download the ORIGINAL (full-resolution) board photos behind the 640x640 Roboflow exports.

Roboflow keeps every uploaded image at its real size; only the exported versions were resized.
For every board kept in final_dataset/split_manifest.csv this script finds the original image in the
source project, downloads it and its annotation (boxes in original pixel coordinates).

  list   : page through each project's images (id, name, size, url)       -> data/hr/<source>/index.json
  fetch  : one board after the other, resumable, with retries             -> data/hr/<source>/originals/<id>.jpg + .json
  build  : clean folders in the same layout as data/clean, labels from the original annotation
           (no augmented copy, no rotation/shear)                          -> data/clean_hr/<folder>/

Needs ROBOFLOW_API_KEY. Slow internet is fine: files are downloaded one at a time and finished ones are skipped.

  python tools/fetch_originals.py list
  python tools/fetch_originals.py fetch
  python tools/fetch_originals.py build
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
HR = ROOT / "data" / "hr"
API = "https://api.roboflow.com"
# source -> (workspace/project, cleaned folder name used by the 640 pipeline)
PROJECTS = {
    "parsa": ("parsa-karkooti/pcb-wik5v-zonih", "parsa_50class"),
    "carddata": ("carddata-3mujr/pcb-electronic-components", "carddata_v3"),
    "dhbw": ("dhbw-mosbach-studienarbeit/circuit-board", "dhbw"),
}


def key():
    k = os.environ.get("ROBOFLOW_API_KEY")
    if not k:
        sys.exit("set ROBOFLOW_API_KEY")
    return k


class Resp:
    def __init__(self, content):
        self.content = content

    def json(self):
        return json.loads(self.content)


def call(method, url, tries=8, params=None, json_body=None, timeout=90, **kw):
    """HTTP through curl with retries (Python's TLS through the local proxy keeps getting reset; curl does not)."""
    json_body = kw.pop("json", json_body)
    if params:
        url = url + ("&" if "?" in url else "?") + urlencode(params)
    cmd = ["curl", "-sS", "-L", "--max-time", str(timeout), "-X", method, "-w", "\n%{http_code}", url]
    if json_body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
    # api.roboflow.com answers directly (faster, fewer drops); the image storage only through the proxy
    direct = url.startswith(API)
    for t in range(tries):
        use = cmd[:1] + (["--noproxy", "*"] if direct and t % 2 == 0 else []) + cmd[1:]
        p = subprocess.run(use, input=json.dumps(json_body).encode() if json_body is not None else None,
                           capture_output=True)
        body, _, code = p.stdout.rpartition(b"\n")
        code = code.decode(errors="replace").strip()
        if p.returncode == 0 and code == "200":
            return Resp(body)
        if code in ("400", "401", "403", "404"):
            raise RuntimeError(f"{code} {url.split('?')[0]}: {body[:200]!r}")
        err = p.stderr.decode(errors="replace").strip() or f"HTTP {code}"
        wait = min(60, 3 * 2 ** t)
        print(f"      retry {t + 1}/{tries} in {wait}s ({err[:80]})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"gave up on {url.split('?')[0]}")


def norm_name(name):
    """'pcb167.jpg' / 'pcb167_jpg' / 'pcb167_jpg.rf.<hash>' -> 'pcb167_jpg' (the 640 pipeline's source_name)."""
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"\.(jpe?g|png|bmp)$", lambda m: "_" + m.group(1), name, flags=re.I)
    return name


# ---------------------------------------------------------------- list
def cmd_list(a):
    k = key()
    for src, (proj, _) in PROJECTS.items():
        out = HR / src / "index.json"
        if out.exists() and not a.force:
            print(f"[{src}] index exists ({len(json.load(open(out)))} images) - skipped")
            continue
        rows, off, total = {}, 0, None
        while total is None or off < total:
            r = call("POST", f"{API}/{proj}/search", params={"api_key": k},
                     json={"limit": 250, "offset": off, "fields": ["id", "name", "width", "height", "split", "url"]})
            d = r.json()
            total = d["total"]
            for x in d["results"]:
                rows[x["id"]] = x
            print(f"[{src}] {len(rows)}/{total}", flush=True)
            off += 250
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(list(rows.values()), open(out, "w"), indent=0)


# ---------------------------------------------------------------- fetch
def kept_boards():
    """(source, source_name, file) of every board in the current dataset."""
    out = []
    for r in csv.DictReader(open(ROOT / "final_dataset" / "split_manifest.csv", encoding="utf8")):
        src = r["source"]
        out.append((src, r["file"].split("__", 1)[1].rsplit(".", 1)[0], r["file"]))
    return out


def candidates():
    boards = kept_boards()
    index = {s: json.load(open(HR / s / "index.json")) for s in PROJECTS}
    by_name = {s: defaultdict(list) for s in PROJECTS}
    for s, rows in index.items():
        for x in rows:
            by_name[s][norm_name(x["name"])].append(x)
    return [(src, name, file, by_name[src].get(name, [])) for src, name, file in boards]


def old_box_count(file):
    p = ROOT / "data" / "all_data" / "labels" / (Path(file).stem + ".txt")
    return sum(1 for l in p.read_text(encoding="utf8").splitlines() if l.strip()) if p.exists() else 0


# ---------------------------------------------------------------- geometry: original <-> 640 copy
# The 640 copy is the original, maybe turned by 90 deg steps / flipped (export augmentation), then stretched
# (Roboflow "Stretch") or letterboxed ("Fit (black edges)", dhbw) to 640x640.
def to_640(boxes, W, H, k, flip, fit):
    """Original boxes (pixels, cx cy w h) -> normalised boxes of the 640 copy."""
    out = []
    for cx, cy, w, h in boxes:
        x, y, ww, hh, Wc, Hc = cx, cy, w, h, W, H
        for _ in range(k):                     # np.rot90 (counter-clockwise): (x, y) -> (y, W - x)
            x, y, ww, hh, Wc, Hc = y, Wc - x, hh, ww, Hc, Wc
        if flip:
            y = Hc - y
        if fit:
            s = 640 / max(Wc, Hc); px, py = (640 - Wc * s) / 2, (640 - Hc * s) / 2
            out.append(((x * s + px) / 640, (y * s + py) / 640, ww * s / 640, hh * s / 640))
        else:
            out.append((x / Wc, y / Hc, ww / Wc, hh / Hc))
    return np.array(out).reshape(-1, 4)


def from_640(boxes, W, H, k, flip, fit):
    """Inverse of to_640: normalised boxes of the 640 copy -> pixels of the original (W x H)."""
    Wc, Hc = (H, W) if k % 2 else (W, H)
    out = []
    for cx, cy, w, h in boxes:
        if fit:
            s = 640 / max(Wc, Hc); px, py = (640 - Wc * s) / 2, (640 - Hc * s) / 2
            x, y, ww, hh = (cx * 640 - px) / s, (cy * 640 - py) / s, w * 640 / s, h * 640 / s
        else:
            x, y, ww, hh = cx * Wc, cy * Hc, w * Wc, h * Hc
        if flip:
            y = Hc - y
        cw, ch = Wc, Hc
        for _ in range(k):                     # undo one counter-clockwise turn
            x, y, ww, hh, cw, ch = ch - y, x, hh, ww, ch, cw
        out.append((x, y, ww, hh))
    return out


def _iou(A, B):
    a = np.c_[A[:, 0] - A[:, 2] / 2, A[:, 1] - A[:, 3] / 2, A[:, 0] + A[:, 2] / 2, A[:, 1] + A[:, 3] / 2]
    b = np.c_[B[:, 0] - B[:, 2] / 2, B[:, 1] - B[:, 3] / 2, B[:, 0] + B[:, 2] / 2, B[:, 1] + B[:, 3] / 2]
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    return inter / ((A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter + 1e-12)


def match(api_boxes, W, H, old, fit):
    """Which turn/flip maps the original's boxes onto the checked 640 labels: (share of 640 boxes found, k, flip)."""
    if not api_boxes or not old:
        return 0.0, 0, False
    ob = np.array([o[1:] for o in old])
    best = (0.0, 0, False)
    for k in range(4):
        for flip in (False, True):
            m = _iou(to_640([b[1:] for b in api_boxes], W, H, k, flip, fit), ob)
            sc = float((m.max(axis=0) > 0.5).mean())
            if sc > best[0]:
                best = (sc, k, flip)
    return best


def pick(cands, file):
    """Several images in a project can share a name (crops of the same board, re-uploads, augmented copies).
    Prefer the image that IS the source of the checked 640 copy (its boxes line up after a turn/flip): then the
    checked labels can be carried over exactly. Otherwise the most complete annotation with the biggest parts."""
    old = old_labels(file)
    src = cands[0]["src"] if cands else ""
    best, fallback = None, None
    for x in cands:
        ann = json.load(open(HR / x["src"] / "originals" / f"{x['id']}.json", encoding="utf8"))
        boxes, aw, ah = boxes_from_annotation(ann)
        W, H = float(aw or x["width"]), float(ah or x["height"])
        sc, k, flip = match(boxes, W, H, old, fit=(src == "dhbw"))
        key = (sc >= 0.9, round(sc, 1), x["width"] * x["height"])
        if sc >= 0.7 and (best is None or key > best[0]):
            best = (key, x, {"mode": "transfer", "score": round(sc, 3), "k": k, "flip": flip, "ann_w": W, "ann_h": H})
        if len(boxes) >= 0.6 * max(1, len(old)):
            side = float(np.median([min(b[3], b[4]) for b in boxes])) if boxes else 0
            fk = (side, x["width"] * x["height"])
            if fallback is None or fk > fallback[0]:
                fallback = (fk, x, {"mode": "api", "score": round(sc, 3)})
    return best or fallback


def cmd_fetch(a):
    k = key()
    allc = candidates()
    n_ann = sum(len(c) for *_, c in allc)
    print(f"{len(allc)} boards, {n_ann} candidate images; step 1: annotations (small JSON)", flush=True)
    t0, i = time.time(), 0
    for src, name, file, cands in allc:
        d = HR / src / "originals"
        d.mkdir(parents=True, exist_ok=True)
        for x in cands:
            i += 1
            ann = d / f"{x['id']}.json"
            if not ann.exists():
                r = call("GET", f"{API}/{PROJECTS[src][0]}/images/{x['id']}", params={"api_key": k})
                ann.write_text(json.dumps({k: v for k, v in r.json()["image"].items() if k != "embedding"}), encoding="utf8")
            if i % 50 == 0:
                print(f"   annotations {i}/{n_ann} | {time.time() - t0:.0f}s", flush=True)
    picks, missing = {}, []
    for src, name, file, cands in allc:
        for x in cands:
            x["src"] = src
        b = pick(cands, file)
        if b is None:
            missing.append(file)
        else:
            picks[file] = {"src": src, "name": name, "id": b[1]["id"], "url": b[1]["url"],
                           "width": b[1]["width"], "height": b[1]["height"], **b[2]}
    json.dump(picks, open(HR / "picks.json", "w"), indent=1)
    (HR / "missing.txt").write_text("\n".join(missing), encoding="utf8")
    print(f"step 2: images, one at a time ({len(picks)} picked, {len(missing)} without a usable original)", flush=True)
    done = 0
    for n, (file, p) in enumerate(picks.items(), 1):
        img = HR / p["src"] / "originals" / f"{p['id']}.jpg"
        if not img.exists():
            r = call("GET", p["url"], timeout=300)
            tmp = img.with_suffix(".part")
            tmp.write_bytes(r.content)
            if cv2.imread(str(tmp)) is None:
                tmp.unlink()
                raise RuntimeError(f"broken image {p['url']}")
            tmp.replace(img)
            done += 1
        if n % 10 == 0 or n == len(picks):
            print(f"   images {n}/{len(picks)} | {done} new | {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- build
def boxes_from_annotation(ann):
    """Roboflow image annotation -> [(label, cx, cy, w, h)] in pixels (polygons -> tight box)."""
    a = ann.get("annotation") or {}
    out = []
    for b in a.get("boxes", []):
        if b.get("points"):   # polygon: points are [x, y] pairs or {"x", "y"} dicts
            pts = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in b["points"]]
            xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
            out.append((b["label"], (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)))
        else:
            out.append((b["label"], float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])))
    return out, a.get("width"), a.get("height")


def old_labels(file):
    """Raw labels of the 640 copy: [(label, cx, cy, w, h)] normalised, label without the source prefix."""
    names = yaml.safe_load(open(ROOT / "data" / "all_data" / "data.yaml", encoding="utf8"))["names"]
    p = ROOT / "data" / "all_data" / "labels" / (Path(file).stem + ".txt")
    out = []
    for ln in p.read_text(encoding="utf8").splitlines():
        q = ln.split()
        if len(q) == 5:
            out.append((names[int(q[0])].split(":", 1)[1], *map(float, q[1:])))
    return out


_RAW2FINE = None


def to_fine(src, label):
    """Raw label -> final class of configs/class_map.yaml (several raw names are one class, e.g. Capacitor Jumper)."""
    global _RAW2FINE
    if _RAW2FINE is None:
        cmap = yaml.safe_load(open(ROOT / "configs" / "class_map.yaml", encoding="utf8"))
        _RAW2FINE = {raw: fine for fine, raws in cmap["fine"].items() for raw in raws}
    return _RAW2FINE.get(f"{src}:{label}", f"raw:{label}")


def hist_similarity(a_labels, b_labels):
    """Cosine similarity of the class histograms: the same board has (almost) the same parts list."""
    ca, cb = Counter(a_labels), Counter(b_labels)
    keys = set(ca) | set(cb)
    va, vb = np.array([ca[k] for k in keys], float), np.array([cb[k] for k in keys], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def cmd_build(a):
    picks = json.load(open(HR / "picks.json"))
    out_root = ROOT / "data" / "clean_hr"
    shutil.rmtree(out_root, ignore_errors=True)
    stats, rows_out, names_of, report = Counter(), defaultdict(list), defaultdict(list), []
    sizes = []
    for src, name, file in kept_boards():
        folder = PROJECTS[src][1]
        d = out_root / folder
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
        old = old_labels(file)
        p = picks.get(file)
        img = HR / src / "originals" / f"{p['id']}.jpg" if p else None
        if p and img.exists():
            hr = cv2.imread(str(img))
            H, W = hr.shape[:2]
            if p["mode"] == "transfer":
                # the checked 640 labels (names AND boxes), mapped back onto the original photo: class names in the
                # Roboflow project were renamed since the export (e.g. 'capacitor' cans -> 'Capacitor'), the export is right
                aw, ah = p["ann_w"], p["ann_h"]
                pix = from_640([o[1:] for o in old], aw, ah, p["k"], p["flip"], fit=(src == "dhbw"))
                sx, sy = W / aw, H / ah
                boxes = [(o[0], cx * sx, cy * sy, w * sx, h * sy) for o, (cx, cy, w, h) in zip(old, pix)]
                hs = p["score"]
                status = "original, checked labels carried over"
            else:
                boxes, aw, ah = boxes_from_annotation(json.load(open(img.with_suffix(".json"), encoding="utf8")))
                if (aw and ah) and (int(aw), int(ah)) != (W, H):   # annotation made on another size: rescale
                    sx, sy = W / float(aw), H / float(ah)
                    boxes = [(l, cx * sx, cy * sy, w * sx, h * sy) for l, cx, cy, w, h in boxes]
                hs = hist_similarity([to_fine(src, b[0]) for b in boxes], [to_fine(src, b[0]) for b in old])
                if hs < a.min_hist:
                    p = None
                    status = f"parts list differs (similarity {hs:.2f}) - kept the 640 copy"
                else:
                    status = "original, labels of the original"
        else:
            p = None
            status = "no original - kept the 640 copy"
        if not p:
            img = ROOT / "data" / "all_data" / "images" / file
            H, W = cv2.imread(str(img)).shape[:2]
            boxes = [(l, cx * W, cy * H, w * W, h * H) for l, cx, cy, w, h in old]
            hs = 1.0
        lines = []
        for lab, cx, cy, w, h in boxes:
            if lab not in names_of[folder]:
                names_of[folder].append(lab)
            x0, x1 = max(0.0, cx - w / 2), min(float(W), cx + w / 2)
            y0, y1 = max(0.0, cy - h / 2), min(float(H), cy + h / 2)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            lines.append(f"{names_of[folder].index(lab)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            sizes.append(min(x1 - x0, y1 - y0))
        shutil.copy2(img, d / "images" / f"{name}.jpg")   # copied as is, no re-encoding
        (d / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf8")
        rows_out[folder].append({"source_name": name, "kept_file": img.name, "original_split": "",
                                 "copies_in_group": 1, "unique_copies": 1, "splits_in_group": "",
                                 "black_frac": 0, "noise": 0, "mean_box_area": 0})
        stats[status.split(" (")[0].split(" -")[0]] += 1
        report.append((file, status, f"{hs:.3f}", p["id"] if p else "", f"{W}x{H}", len(old), len(lines)))
    for folder, rows in rows_out.items():
        d = out_root / folder
        with open(d / "manifest.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        yaml.safe_dump({"names": names_of[folder], "nc": len(names_of[folder])}, open(d / "source_data.yaml", "w", encoding="utf8"),
                       allow_unicode=True, sort_keys=False)
    with open(out_root / "match_report.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["file", "status", "parts_similarity", "roboflow_id", "size", "boxes_640", "boxes_now"])
        w.writerows(report)
    print(dict(stats))
    print("part size (short side, px): p10/p25/p50/p75 =", np.percentile(sizes, [10, 25, 50, 75]).round(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch", "build"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-hist", type=float, default=0.9, help="parts-list similarity below this: keep the 640 copy instead")
    a = ap.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
