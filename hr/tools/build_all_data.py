"""Collect every cleaned dataset into one folder without merging classes.

Every class keeps its original name, prefixed with its source ("carddata:Resistor"),
so the merge step later is a pure mapping and nothing is lost.
Output: data/all_data/images, data/all_data/labels, data/all_data/data.yaml, raw_classes.csv, manifest.csv
  python tools/build_all_data.py --clean data/clean_hr --out data/all_data_hr
"""
import argparse, csv, shutil
from collections import Counter
from pathlib import Path
import yaml
from yololib import parse_line

ap = argparse.ArgumentParser()
ap.add_argument("--clean", default="data/clean", help="data/clean_hr for the full-resolution originals")
ap.add_argument("--out", default="data/all_data")
args = ap.parse_args()
CLEAN = Path(args.clean)
SOURCES = {"parsa": "parsa_50class", "carddata": "carddata_v3", "dhbw": "dhbw", "tesis": "tesis", "thai34": "thai34"}
SOURCES = {s: f for s, f in SOURCES.items() if (CLEAN / f / "manifest.csv").exists()}
OUT = Path(args.out)
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

key_to_id, counts, manifest = {}, Counter(), []
for src, folder in SOURCES.items():
    d = CLEAN / folder
    names = yaml.safe_load(open(d / "source_data.yaml", encoding="utf8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    for row in csv.DictReader(open(d / "manifest.csv", encoding="utf8")):
        stem = row["source_name"]
        img = next((p for p in (d / "images").glob(stem + ".*")), None)
        if img is None:
            continue
        new = f"{src}__{stem}"
        lines = []
        lbl = d / "labels" / (stem + ".txt")
        if lbl.exists():
            for ln in lbl.read_text(encoding="utf8").splitlines():
                b = parse_line(ln)
                if b is None:
                    continue
                key = f"{src}:{names[b[0]]}"
                gid = key_to_id.setdefault(key, len(key_to_id))
                counts[key] += 1
                lines.append(f"{gid} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
        shutil.copy2(img, OUT / "images" / (new + img.suffix))
        (OUT / "labels" / (new + ".txt")).write_text("\n".join(lines), encoding="utf8")
        manifest.append({"file": new + img.suffix, "source": src, "source_name": stem,
                         "original_split": row["original_split"], "copies_in_group": row["copies_in_group"],
                         "boxes": len(lines)})

with open(OUT / "raw_classes.csv", "w", newline="", encoding="utf8") as f:
    w = csv.writer(f); w.writerow(["id", "source", "name", "boxes"])
    for k, i in key_to_id.items():
        s, n = k.split(":", 1); w.writerow([i, s, n, counts[k]])
with open(OUT / "manifest.csv", "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
yaml.safe_dump({"path": str(OUT.resolve()), "train": "images", "val": "images",
                "nc": len(key_to_id), "names": {i: k for k, i in key_to_id.items()}},
               open(OUT / "data.yaml", "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
print(len(manifest), "images,", len(key_to_id), "raw classes")
