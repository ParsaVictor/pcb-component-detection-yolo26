"""Quick test of trained models on board photos, with a clean (non-cluttered) drawing.

Modes
  any   : the "component" detector only        -> every element, type unknown
  fine  : the multi-class detector only
  both  : "any" finds the elements, "fine" names them; elements the fine model cannot
          name get the label "?" (these are the ones to hand to the random forest)

Examples
  python predict.py --any final/any_best.pt --source board.jpg --mode any
  python predict.py --any final/any_best.pt --fine final/fine_best.pt --source boards/ --mode both --tile 640
  python predict.py --fine final/fine_best.pt --source board.jpg --mode fine --conf 0.15 --labels names

Outputs (in --out): <name>_elements.jpg (where: every part in one colour), <name>_types.jpg (what: colour per type),
                    <name>_detections.csv, <name>_summary.json
"""
import argparse
import colorsys
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------- detection
def detect(model, img, conf, imgsz, tile, overlap, iou, device, max_det=1000):
    """Return tensors boxes[N,4] (xyxy), scores[N], cls[N]; tiled when tile > 0."""
    H, W = img.shape[:2]
    if not tile or max(H, W) <= tile:
        r = model.predict(img, conf=conf, imgsz=imgsz, device=device, max_det=max_det, verbose=False)[0]
        return r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long()
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, W - tile) + 1, step)); ys = list(range(0, max(1, H - tile) + 1, step))
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    B, S, C = [], [], []
    for y in ys:
        for x in xs:
            r = model.predict(img[y:y + tile, x:x + tile], conf=conf, imgsz=imgsz, device=device, max_det=max_det, verbose=False)[0]
            b = r.boxes.xyxy.cpu().clone(); b[:, [0, 2]] += x; b[:, [1, 3]] += y
            B.append(b); S.append(r.boxes.conf.cpu()); C.append(r.boxes.cls.cpu().long())
    B, S, C = torch.cat(B), torch.cat(S), torch.cat(C)
    keep = batched_nms(B, S, C, iou)
    return B[keep], S[keep], C[keep]


def agnostic_nms(B, S, C, iou):
    keep = batched_nms(B, S, torch.zeros_like(C), iou)
    return B[keep], S[keep], C[keep]


def combine(any_det, fine_det, names, match_iou=0.3):
    """Name every 'any' box with the best-overlapping fine box; otherwise '?' (-> random forest)."""
    Ba, Sa, _ = any_det
    Bf, Sf, Cf = fine_det
    labels, conf_cls = [], []
    iou_m = box_iou(Ba, Bf) if len(Ba) and len(Bf) else torch.zeros(len(Ba), len(Bf))
    for i in range(len(Ba)):
        if iou_m.shape[1] and iou_m[i].max() >= match_iou:
            j = int(iou_m[i].argmax()); labels.append(names[int(Cf[j])]); conf_cls.append(float(Sf[j]))
        else:
            labels.append("?"); conf_cls.append(0.0)
    return labels, conf_cls


# --------------------------------------------------------------- drawing
SHORT = {"Resistor": "R", "Capacitor": "C", "Electrolytic Capacitor": "EC", "IC": "IC", "Inductor": "L", "Diode": "D",
         "LED": "LED", "Transistor": "Q", "Oscillator": "X", "Connector": "CN", "Switch/Button": "SW", "Other": "O",
         "Marker": "M", "component": "", "?": "?"}


def palette(labels):
    uniq = sorted(set(labels), key=lambda s: (s == "?", s))
    out = {}
    for i, k in enumerate(uniq):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.9, 1.0)
        out[k] = (int(b * 255), int(g * 255), int(r * 255))
    if "?" in out:
        out["?"] = (210, 210, 210)
    return out


ELEMENT_COLOUR = (0, 255, 120)   # one bright green for "where are the parts"


def _canvas(img, target):
    """Upscale small photos so thin parts stay visible; big photos keep their own resolution (capped)."""
    H0, W0 = img.shape[:2]
    up = target / max(H0, W0)
    up = max(up, 1.0) if max(H0, W0) < target else min(1.0, 4000 / max(H0, W0))
    out = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA) if up != 1 else img.copy()
    return out, up


def _style(H, W):
    """Line width and font size that grow with the picture, so they look the same on any photo."""
    side = max(H, W)
    return max(2, round(side / 900)), max(0.7, side / 2600)


def _legend(rows, total, col, H, title):
    ph = max(H, 54 * (len(rows) + 3))
    panel = np.full((ph, 420, 3), 28, np.uint8)
    cv2.putText(panel, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Total: {total}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for r, (k, n) in enumerate(rows):
        y = 150 + r * 48
        cv2.rectangle(panel, (18, y - 26), (52, y + 8), col[k], -1)
        cv2.putText(panel, f"{k}: {n}", (66, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (235, 235, 235), 2, cv2.LINE_AA)
    return panel


def _join(out, panel):
    H, W = out.shape[:2]
    ph = panel.shape[0]
    left = np.vstack([out, np.zeros((ph - H, W, 3), np.uint8)]) if ph > H else out
    return np.hstack([left, panel])


def draw_elements(img, boxes, target=2400, fill=0.28):
    """Step 1 picture: WHERE the parts are. Every part in one colour, thick outline and a light fill."""
    out, up = _canvas(img, target)
    H, W = out.shape[:2]
    lw, _ = _style(H, W)
    bx = (boxes * up).numpy() if len(boxes) else np.zeros((0, 4))
    over = out.copy()
    for x0, y0, x1, y1 in bx.round().astype(int):
        cv2.rectangle(over, (x0, y0), (x1, y1), ELEMENT_COLOUR, -1)
    out = cv2.addWeighted(over, fill, out, 1 - fill, 0)
    for x0, y0, x1, y1 in bx.round().astype(int):
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), lw + 2, cv2.LINE_AA)       # dark rim: readable on any board colour
        cv2.rectangle(out, (x0, y0), (x1, y1), ELEMENT_COLOUR, lw, cv2.LINE_AA)
    return _join(out, _legend([("element", len(bx))], len(bx), {"element": ELEMENT_COLOUR}, H, "Elements"))


def draw(img, boxes, labels, scores, mode, min_label_px, target=2400):
    """Step 2 picture: WHAT each part is. Colour per type, thick boxes, class codes on boxes big enough to carry
    them (never overlapping), and a legend panel with the counts.
    mode: none | auto | ids | names"""
    out, up = _canvas(img, target)
    H, W = out.shape[:2]
    lw, fs = _style(H, W)
    col = palette(labels)
    bx = (boxes * up).numpy() if len(boxes) else np.zeros((0, 4))
    for i in range(len(bx)):
        x0, y0, x1, y1 = [int(round(v)) for v in bx[i]]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), col[labels[i]], lw, cv2.LINE_AA)
    if mode != "none":
        font, th_txt = cv2.FONT_HERSHEY_SIMPLEX, max(1, lw - 1)
        taken = []
        area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) if len(bx) else []
        for i in (np.argsort(-np.asarray(area)) if len(bx) else []):
            x0, y0, x1, y1 = [int(round(v)) for v in bx[i]]
            if mode == "auto" and min(x1 - x0, y1 - y0) < min_label_px * up:
                continue
            text = {"ids": str(i + 1), "names": f"{labels[i]} {scores[i]:.2f}"}.get(mode, SHORT.get(labels[i], labels[i][:3]))
            if not text:
                continue
            (tw, th), _ = cv2.getTextSize(text, font, fs, th_txt)
            for tx, ty in ((x0, y0 - 4), (x0, y1 + th + 4), (x1 - tw, y0 - 4), (x1 - tw, y1 + th + 4)):
                rect = (tx - 2, ty - th - 2, tx + tw + 2, ty + 4)
                if rect[0] < 0 or rect[1] < 0 or rect[2] > W or rect[3] > H:
                    continue
                if all(rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3] for r in taken):
                    taken.append(rect)
                    cv2.putText(out, text, (tx, ty), font, fs, (0, 0, 0), th_txt + 3, cv2.LINE_AA)
                    cv2.putText(out, text, (tx, ty), font, fs, col[labels[i]], th_txt, cv2.LINE_AA)
                    break
    counts = {k: labels.count(k) for k in col}
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    return _join(out, _legend(rows, len(labels), col, H, "Types")), counts


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image or folder")
    ap.add_argument("--mode", choices=["any", "fine", "both"], default="both")
    ap.add_argument("--any", dest="any_w", help="weights of the 'component' detector")
    ap.add_argument("--fine", dest="fine_w", help="weights of the multi-class detector")
    ap.add_argument("--conf", type=float, default=0.20, help="lower = more elements (and more false ones)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU when merging tiles / agnostic NMS")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=0, help="tile size in px for big photos (0 = off), e.g. 640")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=1000, help="max detections per image / tile")
    ap.add_argument("--scale", type=float, default=1.0, help="resize the photo first (e.g. 0.5 when the parts look much "
                    "bigger than in the training tiles); boxes are reported in original pixels")
    ap.add_argument("--labels", choices=["none", "ids", "names", "auto"], default="auto")
    ap.add_argument("--min-label-px", type=int, default=24, help="a box needs this many px (at 640) on its short side to get a text label")
    ap.add_argument("--refine", help="python file with classify(list_of_bgr_crops)->list[(label, conf)] for the '?' boxes")
    ap.add_argument("--out", default="infer_out")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    need_any = a.mode in ("any", "both"); need_fine = a.mode in ("fine", "both")
    if need_any and not a.any_w: ap.error("--any weights are required for this mode")
    if need_fine and not a.fine_w: ap.error("--fine weights are required for this mode")
    m_any = YOLO(a.any_w) if need_any else None
    m_fine = YOLO(a.fine_w) if need_fine else None
    fine_names = list(m_fine.names.values()) if m_fine else []

    refine = None
    if a.refine:
        import importlib.util
        spec = importlib.util.spec_from_file_location("refine_mod", a.refine)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); refine = mod.classify

    src = Path(a.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name}: cannot be read - skipped")
            continue
        orig = img
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA if a.scale < 1 else cv2.INTER_LINEAR)
        if a.mode == "any":
            B, S, C = agnostic_nms(*detect(m_any, img, a.conf, a.imgsz, a.tile, a.overlap, a.iou, a.device, a.max_det), a.iou)
            labels, cls_conf = ["component"] * len(B), [float(s) for s in S]
        elif a.mode == "fine":
            B, S, C = detect(m_fine, img, a.conf, a.imgsz, a.tile, a.overlap, a.iou, a.device, a.max_det)
            labels, cls_conf = [fine_names[int(c)] for c in C], [float(s) for s in S]
        else:
            da = agnostic_nms(*detect(m_any, img, a.conf, a.imgsz, a.tile, a.overlap, a.iou, a.device, a.max_det), a.iou)
            df = detect(m_fine, img, a.conf, a.imgsz, a.tile, a.overlap, a.iou, a.device, a.max_det)
            B, S, _ = da
            labels, cls_conf = combine(da, df, fine_names)
        # back to the original photo: boxes in its pixels, crops for --refine at full resolution
        if a.scale != 1.0:
            B = B / a.scale
            img = orig
        if refine is not None:
            idx = [i for i, l in enumerate(labels) if l == "?"]
            if idx:
                crops = [img[max(int(B[i][1]), 0):int(B[i][3]), max(int(B[i][0]), 0):int(B[i][2])] for i in idx]
                for i, (lab, c) in zip(idx, refine(crops)):
                    labels[i], cls_conf[i] = lab, c

        stem = f.stem
        # 1) where the parts are (one colour)  2) what they are (colour per type)
        cv2.imwrite(str(out / f"{stem}_elements.jpg"), draw_elements(img, B), [cv2.IMWRITE_JPEG_QUALITY, 92])
        vis, counts = draw(img, B, labels, [float(s) for s in S], a.labels, a.min_label_px)
        if a.mode != "any":
            cv2.imwrite(str(out / f"{stem}_types.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out / f"{stem}_detections.csv", "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh); w.writerow(["id", "label", "box_conf", "label_conf", "x0", "y0", "x1", "y1"])
            for i in range(len(B)):
                w.writerow([i + 1, labels[i], f"{float(S[i]):.3f}", f"{cls_conf[i]:.3f}", *[int(v) for v in B[i]]])
        (out / f"{stem}_summary.json").write_text(json.dumps({"image": f.name, "total": len(B), "counts": counts}, indent=2))
        print(f"{f.name}: {len(B)} elements  {counts}")


if __name__ == "__main__":
    main()
