"""Same random forest and same hand-made features as final_33.ipynb, but trained on parts cut from the
TRAIN boards of the HR dataset (the original forest learned from close-up photos and reaches only ~40 % on
board crops). Test = parts of the 46 TEST boards (never seen).

  py -I tools/test_nb/train_tree_boards.py
-> upload_to_drive/PCB_Test/tree/tree_rf_boards.joblib + tree_features_boards.csv (+ report)
"""
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tree_features as tf  # noqa: E402

ROOT = HERE.parent.parent
DATA = ROOT / "data" / "hr_full" / "fine"
OUT = ROOT / "upload_to_drive" / "PCB_Test" / "tree"
PER_CLASS = 1500
MIN_PX = 6
SKIP = {"Marker"}


def boxes(split):
    names = yaml.safe_load(open(DATA / "data.yaml", encoding="utf8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    out = []
    for lp in sorted((DATA / "labels" / split).glob("*.txt")):
        for l in lp.read_text(encoding="utf8").splitlines():
            if l.strip():
                c, *v = l.split()
                n = names[int(c)]
                if n not in SKIP:
                    out.append((lp.stem, n, *map(float, v)))
    return out


def crops_of(job):
    import cv2
    split, stem, items = job
    img = cv2.imdecode(np.fromfile(str(DATA / "images" / split / f"{stem}.jpg"), np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    rows = []
    for n, x, y, w, h in items:
        x, y, w, h = x * W, y * H, w * W, h * H
        if min(w, h) < MIN_PX:
            continue
        cr = img[int(max(y - h / 2, 0)):int(y + h / 2), int(max(x - w / 2, 0)):int(x + w / 2)]
        if cr.size:
            rows.append([stem, n] + tf.features_row(cr))
    return rows


def table(split, per_class):
    bx = boxes(split)
    random.Random(0).shuffle(bx)
    take, cnt = {}, Counter()
    for stem, n, *v in bx:
        if per_class is None or cnt[n] < per_class:
            take.setdefault(stem, []).append((n, *v)); cnt[n] += 1
    jobs = [(split, s, it) for s, it in take.items()]
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(crops_of, jobs) for r in rs]
    return pd.DataFrame(rows, columns=["filename", "Category"] + tf.FEATURE_COLS)


if __name__ == "__main__":
    t = time.time()
    tr = table("train", PER_CLASS)
    te = table("test", 400)
    print(f"train crops {len(tr)}  test crops {len(te)}  ({time.time() - t:.0f}s)")
    print("train per class:", dict(Counter(tr.Category)))
    tr.to_csv(OUT / "tree_features_boards.csv", index=False)
    rf = tf.train_forest(OUT / "tree_features_boards.csv")
    pred = rf.predict(te[tf.FEATURE_COLS].fillna(0).values)
    acc = accuracy_score(te.Category, pred)
    P = rf.predict_proba(te[tf.FEATURE_COLS].fillna(0).values).max(1)
    ok = pred == te.Category.values
    conf = "\n".join(f"  prob >= {th}: keeps {np.mean(P >= th) * 100:.0f}% of parts, accuracy {ok[P >= th].mean() * 100:.1f}%"
                     for th in (0.3, 0.4, 0.5, 0.6, 0.7))
    rep = (f"RandomForest(100 trees, depth 10), final_33 features, trained on {len(tr)} parts of the TRAIN boards\n"
           f"accuracy on {len(te)} parts of the TEST boards: {acc * 100:.1f}%\n{conf}\n\n"
           f"{classification_report(te.Category, pred)}\nclasses {list(rf.classes_)}\n")
    joblib.dump(rf, OUT / "tree_rf_boards.joblib", compress=3)
    (OUT / "tree_report_boards.txt").write_text(rep, encoding="utf8")
    print(rep)
