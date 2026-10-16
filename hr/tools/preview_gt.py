"""Draw ground-truth labels with the same renderer as predict.py (what a good model's output looks like)."""
import sys
from pathlib import Path
import cv2, torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infer"))
from predict import draw

root, view, split, name, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
names = yaml.safe_load(open(root / view / "data.yaml", encoding="utf8"))["names"]
img = cv2.imread(str(root / view / "images" / split / (name + ".jpg"))); H, W = img.shape[:2]
B, L = [], []
for ln in (root / view / "labels" / split / (name + ".txt")).read_text().split("\n"):
    if ln.strip():
        c, x, y, w, h = ln.split(); x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        B.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2]); L.append(names[int(c)])
vis, counts = draw(img, torch.tensor(B), L, [1.0] * len(L), "auto", 24)
cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 92]); print(counts)
