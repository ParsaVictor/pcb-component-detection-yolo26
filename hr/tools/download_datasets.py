"""Download Roboflow Universe datasets. The API key is read from ROBOFLOW_API_KEY only."""
import os, shutil, sys
from roboflow import Roboflow

DATASETS = [
    ("dhbw-mosbach-studienarbeit", "circuit-board", 5, "dhbw"),
    # v3 = colour, no augmentation (v5 was grayscale + augmented)
    ("carddata-3mujr", "pcb-electronic-components", 3, "carddata_v3"),
    ("miuler-blas", "tesis_v001", 2, "tesis"),
    ("thi-vn-s-workspace", "thai34", 3, "thai34"),
]

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
for ws, proj, ver, folder in DATASETS:
    out = os.path.join("data/raw", folder)
    if os.path.exists(os.path.join(out, "data.yaml")):
        print("SKIP (already downloaded)", folder, flush=True)
        continue
    for fmt in ("yolo26", "yolov8"):
        shutil.rmtree(out, ignore_errors=True)
        try:
            rf.workspace(ws).project(proj).version(ver).download(fmt, location=out)
            print("OK", folder, fmt, flush=True)
            break
        except Exception as e:
            print("FAIL", folder, fmt, str(e)[:300], flush=True)
