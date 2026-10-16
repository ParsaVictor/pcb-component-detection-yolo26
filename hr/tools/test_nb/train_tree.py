"""Rebuild the random forest of درخت/درخت/final_33.ipynb and SAVE it (the notebook never saved it).

  py -I tools/test_nb/train_tree.py
-> upload_to_drive/PCB_Test/tree/tree_rf.joblib + tree_features.csv + tree_report.txt
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tree_features as tf  # noqa: E402

ROOT = HERE.parent.parent
SRC = ROOT / "درخت" / "درخت" / "Befor"
OUT = ROOT / "upload_to_drive" / "PCB_Test" / "tree"


def one(p):
    import cv2
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # unicode path safe
    if img is None:
        return None
    return [p.name, tf.infer_category(p.name)] + tf.features_row(img)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in SRC.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    t = time.time()
    with ProcessPoolExecutor() as ex:
        rows = [r for r in ex.map(one, files, chunksize=8) if r]
    print(f"{len(rows)} crops, features in {time.time() - t:.0f}s")
    df = pd.DataFrame(rows, columns=["filename", "Category"] + tf.FEATURE_COLS)
    df = df[df.Category != "unknown"]
    df.to_csv(OUT / "tree_features.csv", index=False)

    X, y = df[tf.FEATURE_COLS].fillna(0).values, df.Category.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    rep = classification_report(yte, rf.predict(Xte))
    final = tf.train_forest(OUT / "tree_features.csv")          # saved model: all 1195 crops
    joblib.dump(final, OUT / "tree_rf.joblib", compress=3)
    txt = (f"RandomForest(100 trees, depth 10) - same settings as final_33.ipynb\n"
           f"held-out 25% accuracy: {acc * 100:.2f}%  ({len(yte)} crops)\n\n{rep}\n"
           f"saved model trained on all {len(y)} crops; classes {list(final.classes_)}\n")
    (OUT / "tree_report.txt").write_text(txt, encoding="utf8")
    print(txt)
