"""Test-only engine of the PCB detector: no training here.

  detect      tiled YOLO pass (+ one whole-board pass for big parts), raw results CACHED at a low confidence,
              so changing CONF / IOU / merge settings re-filters in milliseconds without running the model again
  postprocess the fix for big parts detected several times (see merge_big_parts)
  fuse        'any' (where) + 'fine' (what) + random forest (for the parts YOLO could not name)
  evaluate    precision / recall / F1 against the board's labels (dataset boards), duplicates counted
  draw        1) elements in one colour  2) colour per type  3) error map (correct / missed / false / duplicate)
"""
import colorsys
import hashlib
import os
import time
import warnings

warnings.filterwarnings("ignore")          # numpy / sklearn / torch notices would repeat for every part
os.environ.setdefault("YOLO_VERBOSE", "False")

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

RAW_CONF = 0.05          # raw detections are kept down to this score; CONF can then be changed freely above it
TRAIN_MAX_PART = 64      # training photos were shrunk until their median part was at most this many px


# ================================================================== timing
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Clock:
    """with Clock() as c: ...  -> c.ms"""
    def __enter__(self):
        _sync(); self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        _sync(); self.ms = (time.perf_counter() - self.t) * 1000


def load_yolo(path, device, imgsz=1024, half=False):
    from ultralytics import YOLO
    m = YOLO(path)
    with Clock() as c:   # warm-up: the first call builds CUDA kernels and would spoil the timing
        m.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, half=half, verbose=False)
    m.warmup_ms = c.ms
    return m


# ================================================================== raw detection (slow part, cached)
def tiles_of(length, tile, step):
    """Same tiling as the training set (tools/build_hr_dataset.py)."""
    pos = list(range(0, max(length - tile, 0) + 1, step))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def _predict(model, imgs, cfg):
    out = []
    for i in range(0, len(imgs), cfg["batch"]):
        out += model.predict(imgs[i:i + cfg["batch"]], conf=RAW_CONF, imgsz=cfg["imgsz"], device=cfg["device"],
                             half=cfg["half"], max_det=cfg["max_det"], verbose=False)
    return out


def raw_pass(model, img, cfg):
    """Every tile + (big photos) the whole board once. Boxes in the pixels of `img`.
    cut[i] = the box touches an INNER tile border, i.e. it is probably only a piece of a bigger part."""
    H, W = img.shape[:2]
    tile, edge = cfg["tile"], cfg["edge_px"]
    step = max(1, int(tile * (1 - cfg["overlap"])))
    boxes, scores, cls, cut, src = [], [], [], [], []
    if tile and max(H, W) > tile:
        pos = [(x, y) for y in tiles_of(H, tile, step) for x in tiles_of(W, tile, step)]
        crops = [img[y:y + tile, x:x + tile] for x, y in pos]
        for (x, y), r in zip(pos, _predict(model, crops, cfg)):
            b = r.boxes.xyxy.cpu().numpy().copy()
            tw, th = min(tile, W - x), min(tile, H - y)
            c = np.zeros(len(b), bool)
            if x > 0:          c |= b[:, 0] <= edge
            if y > 0:          c |= b[:, 1] <= edge
            if x + tw < W:     c |= b[:, 2] >= tw - edge
            if y + th < H:     c |= b[:, 3] >= th - edge
            b[:, [0, 2]] += x; b[:, [1, 3]] += y
            boxes.append(b); scores.append(r.boxes.conf.cpu().numpy()); cls.append(r.boxes.cls.cpu().numpy().astype(int))
            cut.append(c); src.append(np.zeros(len(b), int))
        n_tiles = len(pos)
        if cfg["global_pass"]:   # the whole board in one look: big parts are seen complete here
            r = _predict(model, [img], cfg)[0]
            b = r.boxes.xyxy.cpu().numpy()
            boxes.append(b); scores.append(r.boxes.conf.cpu().numpy()); cls.append(r.boxes.cls.cpu().numpy().astype(int))
            cut.append(np.zeros(len(b), bool)); src.append(np.ones(len(b), int))
    else:
        r = _predict(model, [img], cfg)[0]
        boxes.append(r.boxes.xyxy.cpu().numpy()); scores.append(r.boxes.conf.cpu().numpy())
        cls.append(r.boxes.cls.cpu().numpy().astype(int)); cut.append(np.zeros(len(boxes[-1]), bool))
        src.append(np.zeros(len(boxes[-1]), int))
        n_tiles = 1
    cat = lambda v, d, s: np.concatenate(v) if v else np.zeros(s, d)
    return dict(boxes=cat(boxes, float, (0, 4)).reshape(-1, 4), scores=cat(scores, float, 0), cls=cat(cls, int, 0),
                cut=cat(cut, bool, 0), src=cat(src, int, 0), n_tiles=n_tiles, size=(W, H))


def _resize(img, s):
    return img if s == 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


_CACHE = {}


def image_key(img):
    return hashlib.md5(img[::7, ::7].tobytes()).hexdigest() + f"{img.shape}"


def detect_raw(model, img, cfg, scale=None, tag=""):
    """Raw detections of one model, in ORIGINAL photo pixels. Cached per (model, photo, scale, tile settings).
    scale None = automatic: if the parts look bigger than in training, the photo is shrunk like in training."""
    keys = ("tile", "overlap", "imgsz", "global_pass", "edge_px", "half", "max_det")
    base = (tag or id(model), image_key(img)) + tuple(cfg[k] for k in keys)
    probe_ms = 0.0
    if scale is None:
        if cfg["auto_scale"]:
            r1 = detect_raw(model, img, cfg, 1.0, tag)
            sel = r1["scores"] >= 0.25
            b = r1["boxes"][sel]
            med = float(np.median(np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))) if len(b) >= 3 else 0.0
            scale = 1.0
            if med > TRAIN_MAX_PART * cfg["auto_scale_tol"]:
                scale = round(TRAIN_MAX_PART / med, 3)
                probe_ms = r1["ms"]
            if scale == 1.0:
                return dict(r1, probe_ms=0.0, median_part=med)
        else:
            scale = cfg["scale"]
    key = base + (scale,)
    if key not in _CACHE:
        with Clock() as c:
            r = raw_pass(model, _resize(img, scale), cfg)
        r["boxes"] = r["boxes"] / scale
        r.update(scale=scale, ms=c.ms, cached=False)
        _CACHE[key] = r
    else:
        _CACHE[key]["cached"] = True
    return dict(_CACHE[key], probe_ms=probe_ms)


def clear_cache():
    _CACHE.clear()


# ================================================================== post-processing (fast)
def _area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _inter(a, b):
    """Intersection area matrix [len(a), len(b)]."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def merge_big_parts(B, S, C, cut, src, p, agnostic):
    """The fix for 'one big part -> several boxes', without deleting small parts that sit under a big one.

    Why it happens: a part bigger than the tile overlap is seen only in pieces, and every tile reports its piece
    (the training labels also kept pieces >= 50 % visible). Plain NMS cannot remove them: a piece covers only a
    part of the full box, so their IoU is low (and pieces of neighbouring tiles may not overlap at all).

    1  pieces inside a bigger COMPLETE box (from a tile where it fits, or from the whole-board pass) are dropped
       -> only boxes that touch an inner tile border are candidates, so a small part lying under a big one,
          seen whole in its tile, is never removed
    2  remaining pieces of the same part in neighbouring tiles are joined into one box (union)
    3  near-same-size duplicates (one box almost inside another, similar area) are removed; a small part under
       a big one has a much smaller area, so it is kept
    """
    keep = np.ones(len(B), bool)
    A = _area(B)
    if len(B) == 0:
        return B, S, C, np.zeros(0, int)
    # global-pass boxes only bring the big parts
    small_global = (src == 1) & (np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]) < p["global_min_px"])
    keep &= ~small_global
    I = _inter(B, B)
    ios = I / np.maximum(A[:, None], 1e-6)            # ios[i, j] = share of box i inside box j
    same = np.ones_like(I, bool) if agnostic else (C[:, None] == C[None, :])
    # 1) piece inside a bigger complete box
    if p["contain"] > 0:
        whole = keep & ~cut
        cond = (ios >= p["contain"]) & whole[None, :] & (A[None, :] > 1.2 * A[:, None])
        keep &= ~(cut & cond.any(1))
    # 2) join pieces across tile borders
    group = np.arange(len(B))
    if p["join"]:
        idx = np.where(keep & cut)[0]
        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]; i = group[i]
            return i
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                if not same[i, j] or I[i, j] <= 0:
                    continue
                # pieces of one part line up: they share most of their extent across the border
                ox = min(B[i, 2], B[j, 2]) - max(B[i, 0], B[j, 0]); oy = min(B[i, 3], B[j, 3]) - max(B[i, 1], B[j, 1])
                ax = ox / max(min(B[i, 2] - B[i, 0], B[j, 2] - B[j, 0]), 1e-6)
                ay = oy / max(min(B[i, 3] - B[i, 1], B[j, 3] - B[j, 1]), 1e-6)
                if max(ax, ay) >= p["join_align"] and min(ax, ay) > 0:
                    group[root(i)] = root(j)
        roots = np.array([root(i) for i in range(len(B))]) if len(idx) else group
        for r in np.unique(roots[idx]) if len(idx) else []:
            members = idx[roots[idx] == r]
            if len(members) > 1:
                best = members[np.argmax(S[members])]
                B[best] = [B[members, 0].min(), B[members, 1].min(), B[members, 2].max(), B[members, 3].max()]
                keep[members] = False; keep[best] = True
        A = _area(B); I = _inter(B, B); ios = I / np.maximum(A[:, None], 1e-6)
    n_merged = np.zeros(len(B), int)
    # 3) same-size duplicates
    if p["dup_ios"] > 0:
        order = np.argsort(-S)
        for k, i in enumerate(order):
            if not keep[i]:
                continue
            for j in order[k + 1:]:
                if keep[j] and same[i, j] and ios[j, i] >= p["dup_ios"] and A[j] >= p["dup_area"] * A[i] and A[j] <= A[i] / max(p["dup_area"], 1e-6):
                    keep[j] = False; n_merged[i] += 1
    return B[keep], S[keep], C[keep], n_merged[keep]


def postprocess(raw, conf, p, agnostic):
    """raw -> final boxes (original pixels). p['mode']: 'smart' (big-part fix) or 'simple' (NMS only, as before)."""
    sel = raw["scores"] >= conf
    B, S, C = raw["boxes"][sel].copy(), raw["scores"][sel], raw["cls"][sel]
    cut, src = raw["cut"][sel], raw["src"][sel]
    if agnostic:
        C = np.zeros_like(C)
    if p["mode"] == "simple":
        keep = (src == 0)
        B, S, C = B[keep], S[keep], C[keep]
    else:
        B, S, C, _ = merge_big_parts(B, S, C, cut, src, p, agnostic)
    if len(B):
        tB, tS, tC = torch.tensor(B, dtype=torch.float32), torch.tensor(S, dtype=torch.float32), torch.tensor(C)
        k = batched_nms(tB, tS, tC, p["iou"]).numpy()
        B, S, C = B[k], S[k], C[k]
        if not agnostic and p["cross_iou"] < 1:      # one part, two names: keep the more confident one
            k = batched_nms(torch.tensor(B, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                            torch.zeros(len(B), dtype=torch.long), p["cross_iou"]).numpy()
            B, S, C = B[k], S[k], C[k]
    o = np.argsort(-S)
    return B[o], S[o], C[o]


# ================================================================== fusion of the three models
def fuse(any_det, fine_det, fine_names, p):
    """Each 'any' box gets the name of the best-overlapping 'fine' box; unmatched -> '?'.
    Confident fine boxes that 'any' missed are added. Returns dict of lists."""
    Ba, Sa = any_det[0], any_det[1]
    Bf, Sf, Cf = fine_det
    labels, lconf, srcs = ["?"] * len(Ba), [0.0] * len(Ba), ["any"] * len(Ba)
    used = np.zeros(len(Bf), bool)
    if len(Ba) and len(Bf):
        iou = box_iou(torch.tensor(Ba, dtype=torch.float32), torch.tensor(Bf, dtype=torch.float32)).numpy()
        for i, j in sorted(zip(*np.where(iou >= p["match_iou"])), key=lambda ij: -iou[ij]):
            if labels[i] == "?" and not used[j]:
                labels[i], lconf[i], srcs[i] = fine_names[int(Cf[j])], float(Sf[j]), "fine"; used[j] = True
    B, S = list(Ba), list(Sa)
    for j in np.where(~used & (Sf >= p["add_fine_conf"]))[0]:
        if len(Ba) and box_iou(torch.tensor(Bf[j:j + 1], dtype=torch.float32), torch.tensor(Ba, dtype=torch.float32)).max() >= 0.3:
            continue
        B.append(Bf[j]); S.append(Sf[j]); labels.append(fine_names[int(Cf[j])]); lconf.append(float(Sf[j])); srcs.append("fine+")
    return dict(boxes=np.array(B).reshape(-1, 4), scores=np.array(S), labels=labels, label_conf=lconf, src=srcs)


def crop(img, b, pad=0.08):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = b
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img[int(max(y0 - py, 0)):int(min(y1 + py, H)), int(max(x0 - px, 0)):int(min(x1 + px, W))]


def tree_refine(res, img, rf, p, to_name):
    """Random forest for the boxes YOLO could not name ('?') and, optionally, for weak YOLO names.
    The forest only overrides when it is confident (prob >= tree_min_prob)."""
    import tree_features as tf
    idx = [i for i, (l, c) in enumerate(zip(res["labels"], res["label_conf"]))
           if l == "?" or (p["tree_if_fine_below"] > 0 and c < p["tree_if_fine_below"])]
    idx = [i for i in idx if min(res["boxes"][i][2] - res["boxes"][i][0], res["boxes"][i][3] - res["boxes"][i][1]) >= p["tree_min_px"]]
    with Clock() as c:
        out = tf.classify(rf, [crop(img, res["boxes"][i]) for i in idx]) if idx else []
    changed = 0
    for i, (cl, pr) in zip(idx, out):
        if pr >= p["tree_min_prob"] and pr > res["label_conf"][i]:
            res["labels"][i], res["label_conf"][i], res["src"][i] = to_name(cl), pr, "tree"; changed += 1
    return dict(n_asked=len(idx), n_changed=changed, ms=c.ms)


# ================================================================== accuracy against labels
def read_labels(txt, W, H, names, drop=()):
    B, L = [], []
    if txt and os.path.exists(txt):
        for l in open(txt, encoding="utf8"):
            v = l.split()
            if len(v) == 5 and names[int(v[0])] not in drop:
                c, x, y, w, h = int(v[0]), *map(float, v[1:])
                B.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]); L.append(names[c])
    return np.array(B).reshape(-1, 4), L


def evaluate(B, S, labels, GB, GL, iou_thr=0.5, typed=True):
    """Greedy matching by score at IoU >= iou_thr. Returns detection P/R/F1, typed P/R/F1, duplicates and
    per-box status (for the error map)."""
    n, g = len(B), len(GB)
    st_pred, st_gt = ["fp"] * n, ["miss"] * g
    tp = tp_typed = 0
    if n and g:
        iou = box_iou(torch.tensor(B, dtype=torch.float32), torch.tensor(GB, dtype=torch.float32)).numpy()
        I = _inter(np.asarray(B, float), np.asarray(GB, float)); ios = I / np.maximum(_area(np.asarray(B, float))[:, None], 1e-6)
        taken = np.zeros(g, bool)
        for i in np.argsort(-np.asarray(S)):
            cand = np.where(~taken & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                taken[j] = True; tp += 1
                ok = (not typed) or labels[i] == GL[j]
                tp_typed += ok
                st_pred[i] = "ok" if ok else "wrong_type"; st_gt[j] = "found"
        for i in range(n):
            if st_pred[i] != "fp":
                continue
            if ((ios[i] >= 0.6) & taken).any():      # lies on a part that was already found = duplicate piece
                st_pred[i] = "dup"
            elif iou[i].max() >= 0.1:                 # touches a label but IoU < threshold = box drawn too big/small/shifted
                st_pred[i] = "loc"
            # else stays "fp": no label anywhere near. On these boards many such boxes are REAL parts that the
            # people who labelled the board skipped (check them on the error map), so precision is a lower bound
    f1 = lambda p_, r_: 2 * p_ * r_ / max(p_ + r_, 1e-9)
    P, R = tp / max(n, 1), tp / max(g, 1)
    Pt, Rt = tp_typed / max(n, 1), tp_typed / max(g, 1)
    return dict(n_pred=n, n_gt=g, tp=tp, P=P, R=R, F1=f1(P, R), dup=st_pred.count("dup"),
                loc=st_pred.count("loc"), unl=st_pred.count("fp"),
                type_acc=tp_typed / max(tp, 1), Pt=Pt, Rt=Rt, F1t=f1(Pt, Rt), st_pred=st_pred, st_gt=st_gt)


# ================================================================== drawing (same look as predict.py)
SHORT = {"Resistor": "R", "Capacitor": "C", "Electrolytic Capacitor": "EC", "IC": "IC", "Inductor": "L", "Diode": "D",
         "LED": "LED", "Transistor": "Q", "Oscillator": "X", "Connector": "CN", "Switch/Button": "SW", "Other": "O",
         "Marker": "M", "component": "", "?": "?"}
ELEMENT_COLOUR = (0, 255, 120)
FIXED = {}


def palette(labels):
    out = {}
    for k in sorted(set(labels), key=lambda s: (s == "?", s)):
        if k not in FIXED:   # one colour per type that stays the same from picture to picture
            i = len(FIXED)
            r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.9, 1.0)
            FIXED[k] = (int(b * 255), int(g * 255), int(r * 255))
        out[k] = FIXED[k]
    if "?" in out:
        out["?"] = (210, 210, 210)
    return out


def _canvas(img, target):
    H0, W0 = img.shape[:2]
    up = target / max(H0, W0)
    up = max(up, 1.0) if max(H0, W0) < target else min(1.0, 4000 / max(H0, W0))
    out = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA) if up != 1 else img.copy()
    return out, up


def _style(H, W):
    side = max(H, W)
    return max(2, round(side / 900)), max(0.7, side / 2600)


def _legend(rows, col, H, title, lines=()):
    ph = max(H, 48 * (len(rows) + len(lines) + 4))
    panel = np.full((ph, 470, 3), 28, np.uint8)
    cv2.putText(panel, title, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    y = 90
    for t in lines:
        cv2.putText(panel, t, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 230, 255), 2, cv2.LINE_AA); y += 40
    y += 20
    for k, n in rows:
        cv2.rectangle(panel, (18, y - 26), (52, y + 8), col[k], -1)
        cv2.putText(panel, f"{k}: {n}", (66, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 235, 235), 2, cv2.LINE_AA)
        y += 46
    return panel


def _join(out, panel):
    H, W = out.shape[:2]
    ph = panel.shape[0]
    left = np.vstack([out, np.zeros((ph - H, W, 3), np.uint8)]) if ph > H else out
    return np.hstack([left, panel])


def draw_elements(img, boxes, lines=(), target=2400, fill=0.28):
    out, up = _canvas(img, target)
    H, W = out.shape[:2]
    lw, _ = _style(H, W)
    bx = (np.asarray(boxes) * up).round().astype(int).reshape(-1, 4)
    over = out.copy()
    for x0, y0, x1, y1 in bx:
        cv2.rectangle(over, (x0, y0), (x1, y1), ELEMENT_COLOUR, -1)
    out = cv2.addWeighted(over, fill, out, 1 - fill, 0)
    for x0, y0, x1, y1 in bx:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), ELEMENT_COLOUR, lw, cv2.LINE_AA)
    return _join(out, _legend([("element", len(bx))], {"element": ELEMENT_COLOUR}, H, "Elements", lines))


def draw_types(img, boxes, labels, scores, src=None, text="auto", min_label_px=24, lines=(), target=2400):
    """text: none | auto (short code on big boxes) | names (name + confidence %) | conf (confidence % only)
    Boxes named by the random forest get a white corner mark and count as '(tree)' in the legend."""
    out, up = _canvas(img, target)
    H, W = out.shape[:2]
    lw, fs = _style(H, W)
    col = palette(labels)
    bx = (np.asarray(boxes) * up).reshape(-1, 4)
    src = src or [""] * len(bx)
    for i in range(len(bx)):
        x0, y0, x1, y1 = [int(round(v)) for v in bx[i]]
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), col[labels[i]], lw, cv2.LINE_AA)
        if src[i] == "tree":
            d = max(6, lw * 4)
            cv2.line(out, (x0, y0), (x0 + d, y0), (255, 255, 255), lw + 1); cv2.line(out, (x0, y0), (x0, y0 + d), (255, 255, 255), lw + 1)
    if text != "none" and len(bx):
        font, tt = cv2.FONT_HERSHEY_SIMPLEX, max(1, lw - 1)
        taken = []
        area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        for i in np.argsort(-area):
            x0, y0, x1, y1 = [int(round(v)) for v in bx[i]]
            if text == "auto" and min(x1 - x0, y1 - y0) < min_label_px * up:
                continue
            s = {"names": f"{labels[i]} {scores[i] * 100:.0f}%", "conf": f"{scores[i] * 100:.0f}%"}.get(text, SHORT.get(labels[i], labels[i][:3]))
            if not s:
                continue
            (tw, th), _ = cv2.getTextSize(s, font, fs, tt)
            for tx, ty in ((x0, y0 - 4), (x0, y1 + th + 4), (x1 - tw, y0 - 4), (x1 - tw, y1 + th + 4)):
                r = (tx - 2, ty - th - 2, tx + tw + 2, ty + 4)
                if r[0] < 0 or r[1] < 0 or r[2] > W or r[3] > H:
                    continue
                if all(r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3] for q in taken):
                    taken.append(r)
                    cv2.putText(out, s, (tx, ty), font, fs, (0, 0, 0), tt + 3, cv2.LINE_AA)
                    cv2.putText(out, s, (tx, ty), font, fs, col[labels[i]], tt, cv2.LINE_AA)
                    break
    counts = {k: labels.count(k) for k in col}
    tree_n = {k: sum(1 for l, s_ in zip(labels, src) if l == k and s_ == "tree") for k in col}
    rows = [(f"{k}" + (f" ({tree_n[k]} tree)" if tree_n[k] else ""), n) for k, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    col2 = {f"{k}" + (f" ({tree_n[k]} tree)" if tree_n[k] else ""): c for k, c in col.items()}
    return _join(out, _legend(rows, col2, H, "Types", lines)), counts


ERR_COL = {"ok": (0, 220, 0), "wrong_type": (0, 215, 255), "loc": (255, 200, 0), "fp": (0, 140, 255),
           "dup": (255, 0, 255), "miss": (0, 0, 255)}
ERR_NAME = {"ok": "correct", "wrong_type": "found, wrong type", "loc": "box size/position off",
            "fp": "no label here (check!)", "dup": "duplicate piece", "miss": "missed (label)"}


def draw_errors(img, B, ev, GB, lines=(), target=2400):
    """Error map against the labels: green correct, yellow wrong type, orange false, magenta duplicate, red missed."""
    out, up = _canvas(img, target)
    H, W = out.shape[:2]
    lw, _ = _style(H, W)
    items = [(b, s) for b, s in zip(np.asarray(B).reshape(-1, 4), ev["st_pred"])] + \
            [(b, "miss") for b, s in zip(np.asarray(GB).reshape(-1, 4), ev["st_gt"]) if s == "miss"]
    for b, s in items:
        x0, y0, x1, y1 = (np.asarray(b) * up).round().astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), lw + 2, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), ERR_COL[s], lw, cv2.LINE_AA)
    cnt = {k: 0 for k in ERR_COL}
    for _, s in items:
        cnt[s] += 1
    rows = [(ERR_NAME[k], cnt[k]) for k in ERR_COL]
    return _join(out, _legend(rows, {ERR_NAME[k]: v for k, v in ERR_COL.items()}, H, "Errors vs labels", lines))


# ================================================================== one board, all stages
def model_tag(path):
    return f"{path}|{os.path.getmtime(path) if os.path.exists(path) else 0}"


_TREE_CACHE = {}
_TREE_MS = []   # measured ms per part, used to report the real forest time of cached answers


def tree_probs(M, img, boxes, p, ask=None):
    """Forest probabilities {type: prob} for each box (None = not asked: too small or ask[i] False).
    Results are cached per photo + box, so moving a slider does not run the forest again."""
    import tree_features as tf
    key0 = image_key(img) + M.get("tree_kind", "")
    out, todo = [None] * len(boxes), []
    for i, b in enumerate(boxes):
        if ask is not None and not ask[i]:
            continue
        if min(b[2] - b[0], b[3] - b[1]) < p["tree_min_px"]:
            continue
        k = (key0, tuple(int(round(v)) for v in b))
        if k in _TREE_CACHE:
            out[i] = _TREE_CACHE[k]
        else:
            todo.append((i, k))
    with Clock() as c:
        if todo:
            X = tf.features_many([crop(img, boxes[i]) for i, _ in todo])
            P = M["rf"].predict_proba(X)
            names = [M["to_name"](cl) for cl in M["rf"].classes_]
            for (i, k), pr in zip(todo, P):
                _TREE_CACHE[k] = out[i] = dict(zip(names, map(float, pr)))
    n = sum(o is not None for o in out)
    if todo:
        _TREE_MS.append(c.ms / len(todo))
    per = float(np.median(_TREE_MS)) if _TREE_MS else 0.0
    return out, dict(n_asked=n, n_new=len(todo), ms=c.ms + (n - len(todo)) * per)


def vote(fu, probs, p):
    """Type of each part from BOTH the 'fine' model and the forest:
    score(type) = (1 - w) * fine_confidence * [fine said this type] + w * forest_probability(type)
    w = p['tree_weight']. No fine name -> the forest decides alone (if its probability >= tree_min_prob)."""
    w = p["tree_weight"]
    labels, conf, src = [], [], []
    for lf, cf, pt, s0 in zip(fu["labels"], fu["label_conf"], probs, fu["src"]):
        if pt is None:
            labels.append(lf); conf.append(cf); src.append(s0)
            continue
        if lf == "?":
            c = max(pt, key=pt.get)
            ok = pt[c] >= p["tree_min_prob"]
            labels.append(c if ok else "?"); conf.append(pt[c] if ok else 0.0); src.append("tree" if ok else s0)
            continue
        score = {k: w * v for k, v in pt.items()}
        score[lf] = score.get(lf, 0.0) + (1 - w) * cf
        c = max(score, key=score.get)
        labels.append(c)
        conf.append(score[c] / ((1 - w) * cf + w) if ((1 - w) * cf + w) > 0 else 0.0)
        src.append("fine+tree" if c == lf and max(pt, key=pt.get) == lf else ("fine" if c == lf else "tree"))
    return labels, conf, src


def detect_multi(model, img, cfg, scale, tag):
    """detect_raw + (cfg['zoom'] > 1) one more pass on the photo ENLARGED by that factor, for boards whose parts
    are tiny (median part < cfg['zoom_below'] px): small parts that were missed at normal size are found there.
    Both passes are merged by the normal post-processing (duplicates removed)."""
    r = detect_raw(model, img, cfg, scale, tag)
    z = cfg.get("zoom", 1.0)
    if z <= 1.0:
        return r
    b = r["boxes"][r["scores"] >= 0.25] * r["scale"]
    med = float(np.median(np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))) if len(b) >= 3 else 0.0
    big = max(img.shape[:2]) * r["scale"] * z
    if med >= cfg.get("zoom_below", 32) or big > cfg.get("zoom_max_side", 8000):
        return r
    r2 = detect_raw(model, img, cfg, round(r["scale"] * z, 3), tag)
    out = dict(r)
    for k in ("boxes", "scores", "cls", "cut", "src"):
        out[k] = np.concatenate([r[k], r2[k]])
    out.update(ms=r["ms"] + r2["ms"], n_tiles=r["n_tiles"] + r2["n_tiles"], cached=r["cached"] and r2["cached"],
               zoomed=True)
    return out


def run_board(M, img, cfg, p, stages=("any", "fine", "tree")):
    """M: dict(any=YOLO, fine=YOLO, rf=forest, tags=..., fine_names=[...], to_name=fn).
    Returns per-stage results with timings in ms (detection timings of cached results are the ones measured
    when the model actually ran)."""
    R, scale = {}, None
    if M.get("any") is not None:
        raw = detect_raw(M["any"], img, cfg, None, M["tags"]["any"])
        scale = raw["scale"]
        raw = detect_multi(M["any"], img, cfg, scale, M["tags"]["any"])
        with Clock() as c:
            B, S, _ = postprocess(raw, p["conf_any"], p, agnostic=True)
        R["any"] = dict(boxes=B, scores=S, labels=["component"] * len(B), t_det=raw["ms"] + raw["probe_ms"],
                        t_post=c.ms, cached=raw["cached"], scale=raw["scale"], n_tiles=raw["n_tiles"], n_raw=int((raw["scores"] >= p["conf_any"]).sum()),
                        zoomed=raw.get("zoomed", False))
    if "fine" in stages and M.get("fine") is not None:
        raw = detect_raw(M["fine"], img, cfg, scale, M["tags"]["fine"]) if scale is None else             detect_multi(M["fine"], img, cfg, scale, M["tags"]["fine"])
        with Clock() as c:
            B, S, C = postprocess(raw, p["conf_fine"], p, agnostic=False)
        R["fine"] = dict(boxes=B, scores=S, labels=[M["fine_names"][int(k)] for k in C], cls=C,
                         t_det=raw["ms"] + raw["probe_ms"], t_post=c.ms, cached=raw["cached"], scale=raw["scale"],
                         n_tiles=raw["n_tiles"], n_raw=int((raw["scores"] >= p["conf_fine"]).sum()))
    if "any" in R and "fine" in R:
        with Clock() as c:
            fu = fuse((R["any"]["boxes"], R["any"]["scores"]), (R["fine"]["boxes"], R["fine"]["scores"], R["fine"]["cls"]),
                      M["fine_names"], p)
        fu["t_fuse"] = c.ms
        R["both"] = fu
        if "vote" in stages and M.get("rf") is not None:
            # model 2 names what it can; every part it could not name, or named with confidence < fine_min,
            # goes to the forest, which names it when its probability >= tree_min_prob
            ask = [l == "?" or c < p["fine_min"] for l, c in zip(fu["labels"], fu["label_conf"])]
            probs, tinfo = tree_probs(M, img, fu["boxes"], p, ask)
            L, C, S_ = list(fu["labels"]), list(fu["label_conf"]), list(fu["src"])
            for i, pt in enumerate(probs):
                if pt:
                    c = max(pt, key=pt.get)
                    if pt[c] >= p["tree_min_prob"]:
                        S_[i] = "tree" if L[i] != c else "fine+tree"
                        L[i], C[i] = c, pt[c]
            tinfo["n_changed"] = sum(1 for a_, b_ in zip(L, fu["labels"]) if a_ != b_)
            R["vote"] = dict(boxes=fu["boxes"], scores=fu["scores"], labels=L, label_conf=C, src=S_, tree=tinfo)
        if "tree" in stages and M.get("rf") is not None:
            fu3 = dict(boxes=fu["boxes"], scores=fu["scores"], labels=list(fu["labels"]),
                       label_conf=list(fu["label_conf"]), src=list(fu["src"]))
            fu3["tree"] = tree_refine(fu3, img, M["rf"], p, M["to_name"])
            R["all3"] = fu3
    if "tree_all" in stages and M.get("rf") is not None and "any" in R:
        a = R["any"]
        probs, tinfo = tree_probs(M, img, a["boxes"], p)
        L, C, S_ = [], [], []
        for pr in probs:
            c = max(pr, key=pr.get) if pr else None
            ok = c is not None and pr[c] >= p["tree_min_prob"]
            L.append(c if ok else "?"); C.append(pr[c] if ok else 0.0); S_.append("tree" if ok else "any")
        tinfo["n_changed"] = S_.count("tree")
        R["any_tree"] = dict(boxes=a["boxes"], scores=a["scores"], labels=L, label_conf=C, src=S_, tree=tinfo)
    return R


# ================================================================== notebook views
def show(img, width=1400):
    try:
        from IPython.display import Image, display
        display(Image(data=cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), width=width))
    except ImportError:
        pass


def _ms(v):
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


STAGES = {1: ("any",), 2: ("any", "tree_all"), 3: ("any", "fine", "tree"), 4: ("any", "fine", "vote")}
TITLE = {1: "1 - model 1 only (where)", 2: "2 - model 1 + forest (forest names every part)",
         3: "3 - model 1 + model 2, forest only for '?'", 4: "4 - all parts: model 2 names, forest for '?' and weak names"}
RESULT = {2: "any_tree", 3: "all3", 4: "vote"}


def stage_result(R, stage):
    """Final boxes of a stage + its time split."""
    a = R["any"]
    t = {"any: detect": a["t_det"], "any: merge": a["t_post"]}
    if stage == 1:
        return a["boxes"], a["scores"], a["labels"], ["any"] * len(a["boxes"]), [float(s) for s in a["scores"]], t
    if stage != 2:
        f = R["fine"]
        t.update({"fine: detect": f["t_det"], "fine: merge": f["t_post"], "fuse": R["both"]["t_fuse"]})
    r = R[RESULT[stage]]
    t["tree"] = r["tree"]["ms"]
    return r["boxes"], r["scores"], r["labels"], r["src"], r["label_conf"], t


def view(M, img, gt, cfg, p, stage, text="auto", show_errors=True, show_img=True, name=""):
    """Run (cached) + draw + print speed and accuracy of one stage. gt = (boxes, labels) or None."""
    R = run_board(M, img, cfg, p, STAGES[stage])
    B, S, L, src, lconf, t = stage_result(R, stage)
    total = sum(t.values())
    a = R["any"]
    lines = [f"time {_ms(total)}" + (" (cached run)" if a["cached"] else ""),
             f"photo scale {a['scale']:g}, {a['n_tiles']} tiles"]
    ev = None
    if gt is not None and len(gt[0]):
        GB, GL = gt
        if stage == 1:
            keep = [i for i, l in enumerate(GL) if l != "Marker"]
            GB, GL = GB[keep], [GL[i] for i in keep]
        ev = evaluate(B, S, L, GB, GL, typed=stage > 1)
        lines.append(f"found P {ev['P'] * 100:.0f}%  R {ev['R'] * 100:.0f}%  F1 {ev['F1'] * 100:.0f}%")
        if stage > 1:
            lines.append(f"type correct {ev['type_acc'] * 100:.0f}% of found")
        lines.append(f"duplicates {ev['dup']}")
    conf_txt = f"mean confidence {np.mean(lconf) * 100:.0f}%" if len(lconf) else "no element"
    lines.append(conf_txt)

    print(f"\n=== {TITLE[stage]}  {name}")
    print(f"elements: {len(B)}" + (f"   labels on the board: {len(ev['st_gt'])}" if ev else "   (no labels for this photo: accuracy = model confidence only)"))
    print("time:", "  |  ".join(f"{k} {_ms(v)}" for k, v in t.items()), f"  ||  TOTAL {_ms(total)}")
    if a["cached"]:
        print("     (the models did not run again: same photo and detection settings -> cached result; the times are from the real run)")
    if ev:
        print(f"accuracy (IoU>=0.5): precision {ev['P'] * 100:.1f}%  recall {ev['R'] * 100:.1f}%  F1 {ev['F1'] * 100:.1f}%"
              f"  | duplicates {ev['dup']}  missed {ev['st_gt'].count('miss')}")
        print(f"     'not correct' boxes: {ev['loc']} box size/position off (part is right), "
              f"{ev['unl']} with NO label near them -> look at the orange boxes on the error map: "
              f"if they are real parts, the board's labels are incomplete and the real precision is higher")
        if stage > 1:
            print(f"type: correct for {ev['type_acc'] * 100:.1f}% of the found parts  | typed F1 {ev['F1t'] * 100:.1f}%"
                  f"  | '?' (no name) {L.count('?')}")
    print(conf_txt)
    if stage > 1:
        tr = R[RESULT[stage]]["tree"]
        new = tr.get("n_new", tr["n_asked"])
        print(f"random forest: asked about {tr['n_asked']} parts ({new} new, {_ms(tr['ms'])}"
              + (f", {tr['ms'] / new:.0f} ms per part" if new else ", all from cache") + f"), changed/gave the name of {tr['n_changed']}")
        from collections import Counter
        print("names from:", dict(Counter(src)), "  (fine = model 2, fine+ = found only by model 2, tree = forest, "
              "fine+tree = both agree, any = no name '?')")
    if show_img:
        if stage == 1:
            show(draw_elements(img, B, lines))
        else:
            show(draw_types(img, B, L, lconf, src, text, lines=lines)[0])
        if ev and show_errors:
            try:
                from IPython.display import HTML, display
                display(HTML('<div style="height:48px"></div><div style="direction:rtl;text-align:right;font-family:Tahoma;'
                             'font-size:15px;border-top:2px solid #888;padding-top:10px">'
                             '<b>نقشه‌ی خطا در مقایسه با برچسب‌های این برد:</b> سبز = درست، زرد = پیدا شده ولی نوع اشتباه، '
                             'آبی روشن = جای جعبه کمی فرق دارد، نارنجی = این‌جا برچسبی نیست (اگر روی عکس المان است، '
                             'برچسب‌زن جا انداخته و خطای مدل نیست)، بنفش = تکراری، قرمز = المان برچسب‌خورده‌ای که پیدا نشد</div>'))
            except ImportError:
                pass
            show(draw_errors(img, B, ev, gt[0] if stage > 1 else GB, lines[:1]))
    return dict(R=R, boxes=B, labels=L, ev=ev, t=t, total=total)


def benchmark(M, boards, cfg, p, read_gt, progress=True):
    """Speed and accuracy over many labelled boards. boards: list of image paths; read_gt(path, W, H) -> (B, L).
    Compares the old merge ('simple' = NMS only) with the big-part fix ('smart')."""
    rows = {k: dict(tp=0, n=0, g=0, tpt=0, dup=0, q=0) for k in
            ("1 any (old NMS)", "1 any (big-part fix)", "  fine alone", "2 any+fine", "3 any+fine+forest")}
    times = {k: [] for k in ("any", "fine", "tree", "stage 1", "stage 2", "stage 3")}
    px = []
    for n, f in enumerate(boards, 1):
        img = cv2.imread(f)
        H, W = img.shape[:2]
        GB, GL = read_gt(f, W, H)
        ga = [i for i, l in enumerate(GL) if l != "Marker"]
        GBa, GLa = GB[ga], [GL[i] for i in ga]
        R = run_board(M, img, cfg, p, STAGES[3])
        old = postprocess(detect_raw(M["any"], img, cfg, R["any"]["scale"], M["tags"]["any"]), p["conf_any"], dict(p, mode="simple"), True)
        cases = {"1 any (old NMS)": (old[0], old[1], ["component"] * len(old[0]), GBa, GLa, False),
                 "1 any (big-part fix)": (R["any"]["boxes"], R["any"]["scores"], R["any"]["labels"], GBa, GLa, False),
                 "  fine alone": (R["fine"]["boxes"], R["fine"]["scores"], R["fine"]["labels"], GB, GL, True),
                 "2 any+fine": (R["both"]["boxes"], R["both"]["scores"], R["both"]["labels"], GB, GL, True),
                 "3 any+fine+forest": (R["all3"]["boxes"], R["all3"]["scores"], R["all3"]["labels"], GB, GL, True)}
        for k, (B, S, L, gb, gl, typed) in cases.items():
            ev = evaluate(B, S, L, gb, gl, typed=typed)
            r = rows[k]
            r["tp"] += ev["tp"]; r["n"] += ev["n_pred"]; r["g"] += ev["n_gt"]; r["dup"] += ev["dup"]
            r["tpt"] += round(ev["type_acc"] * ev["tp"]); r["q"] += list(L).count("?")
        ta = R["any"]["t_det"] + R["any"]["t_post"]
        tf_ = R["fine"]["t_det"] + R["fine"]["t_post"] + R["both"]["t_fuse"]
        tt = R["all3"]["tree"]["ms"]
        for k, v in (("any", ta), ("fine", tf_), ("tree", tt), ("stage 1", ta), ("stage 2", ta + tf_), ("stage 3", ta + tf_ + tt)):
            times[k].append(v)
        px.append(W * H / 1e6)
        if progress:
            print(f"\r{n}/{len(boards)}  {os.path.basename(f)[:40]:40s}", end="", flush=True)
    print()
    f1 = lambda a, b: 2 * a * b / max(a + b, 1e-9)
    print(f"{'':24s}{'found P':>9}{'R':>7}{'F1':>7}{'type ok':>9}{'typed F1':>10}{'dups':>7}{'no name':>9}")
    for k, r in rows.items():
        P, Rr = r["tp"] / max(r["n"], 1), r["tp"] / max(r["g"], 1)
        typed = not k.startswith("1")
        Pt, Rt = r["tpt"] / max(r["n"], 1), r["tpt"] / max(r["g"], 1)
        print(f"{k:24s}{P * 100:8.1f}%{Rr * 100:6.1f}%{f1(P, Rr) * 100:6.1f}%"
              + (f"{r['tpt'] / max(r['tp'], 1) * 100:8.1f}%{f1(Pt, Rt) * 100:9.1f}%" if typed else f"{'-':>9}{'-':>10}")
              + f"{r['dup']:7d}{r['q'] if typed else 0:9d}")
    print(f"\nspeed per board (mean of {len(boards)} boards, {np.mean(px):.1f} megapixel on average; times of the real model runs):")
    for k, v in times.items():
        print(f"  {k:10s} mean {_ms(np.mean(v)):>9}   median {_ms(np.median(v)):>9}   max {_ms(np.max(v)):>9}")
    return rows, times


def final_view(M, img, cfg, p, name="", out_dir=None, text="auto", gt=None):
    """The quickest complete answer: one picture with every part and its type (any + fine, the forest only for
    the parts the fine model could not name), plus a CSV of all parts."""
    R = run_board(M, img, cfg, p, STAGES[4] if M.get("rf") is not None else ("any", "fine"))
    st = 4 if "vote" in R else None
    if st is None:   # no forest: model 1 + model 2 only
        R["vote"] = dict(R["both"], tree=dict(n_asked=0, n_changed=0, ms=0.0))
        st = 4
    B, S, L, src, lconf, t = stage_result(R, st)
    total = sum(t.values())
    from collections import Counter
    n_tree = src.count("tree")
    lines = [f"{len(B)} parts in {_ms(total)}" + (" (cached run)" if R["any"]["cached"] else ""),
             f"mean confidence {np.mean(lconf) * 100:.0f}%" if len(lconf) else "no part found",
             f"named by forest: {n_tree}, unknown: {L.count('?')}"]
    vis, counts = draw_types(img, B, L, lconf, src, text, lines=lines)
    show(vis)
    print(f"{name}: {len(B)} parts | time {_ms(total)}  (" + ", ".join(f"{k} {_ms(v)}" for k, v in t.items()) + ")")
    print("per type:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if R["any"].get("zoomed"):
        print("tiny parts: the photo was also checked enlarged")
    if gt is not None and len(gt[0]):
        ev = evaluate(B, S, L, gt[0], gt[1], typed=True)
        print(f"labels of this board: found {ev['tp']} of {ev['n_gt']} parts ({ev['R'] * 100:.0f}%), "
              f"type right for {ev['type_acc'] * 100:.0f}% of them")
    if out_dir:
        import csv
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(name or "board")[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_result.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(out_dir, f"{stem}_parts.csv"), "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh); w.writerow(["id", "type", "type_conf", "found_conf", "named_by", "x0", "y0", "x1", "y1"])
            for i in range(len(B)):
                w.writerow([i + 1, L[i], f"{lconf[i]:.3f}", f"{S[i]:.3f}", src[i], *[int(v) for v in B[i]]])
        print("saved:", os.path.join(out_dir, f"{stem}_result.jpg"), "+ _parts.csv")
    return R


def settings_for(cfg, p, mode, conf):
    """Per-cell settings. mode 'complete' = do not miss parts: more tile overlap, an extra enlarged pass on boards
    with tiny parts, and parts that only the 'fine' model saw are added at a lower confidence."""
    c, q = dict(cfg), dict(p)
    q["conf_any"] = q["conf_fine"] = float(conf)
    if mode == "complete":
        c.update(overlap=max(c["overlap"], 0.35), zoom=1.5, zoom_below=32)
        q["add_fine_conf"] = min(q["add_fine_conf"], 0.25)
    return c, q
