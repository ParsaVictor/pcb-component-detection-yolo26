"""Find the same board photographed/labelled in different sources (rotation/scale/colour invariant).

SIFT keypoints + ratio test + RANSAC affine; a pair of images from DIFFERENT sources with enough
inliers is a twin.  Result: data/all_data/twins.csv, used by build_dataset.py to keep twins in one split
and to drop duplicate copies.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def describe(path, side=480, n=500):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA))
    kp, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, None)
    return (np.float32([k.pt for k in kp]), d) if d is not None and len(kp) >= 8 else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-data", default="data/all_data")
    ap.add_argument("--sources", nargs="+", default=["parsa", "carddata", "dhbw", "tesis", "thai34"])
    ap.add_argument("--min-inliers", type=int, default=18)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    A = Path(a.all_data)
    rows = [r for r in csv.DictReader(open(A / "manifest.csv", encoding="utf8")) if r["source"] in a.sources]
    with ThreadPoolExecutor(a.workers) as ex:
        feats = list(ex.map(lambda r: describe(A / "images" / r["file"]), rows))
    print(f"described {len(rows)} images")

    bf = cv2.BFMatcher(cv2.NORM_L2)

    def compare(i):
        out = []
        pi, di = feats[i]
        if di is None:
            return out
        for j in range(i + 1, len(rows)):
            pj, dj = feats[j]
            if dj is None or rows[i]["source"] == rows[j]["source"]:
                continue
            m = bf.knnMatch(di, dj, k=2)
            good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(good) < a.min_inliers:
                continue
            src = pi[[g.queryIdx for g in good]]; dst = pj[[g.trainIdx for g in good]]
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
            n = int(inl.sum()) if inl is not None else 0
            if n >= a.min_inliers:
                out.append((rows[i]["file"], rows[j]["file"], n))
        return out

    with ThreadPoolExecutor(a.workers) as ex:
        found = [t for res in ex.map(compare, range(len(rows))) for t in res]
    with open(A / "twins.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["file_a", "file_b", "inliers"]); w.writerows(found)
    print(f"{len(found)} twin pairs -> {A / 'twins.csv'}")


if __name__ == "__main__":
    main()
