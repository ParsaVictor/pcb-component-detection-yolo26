"""Dry-run the Colab notebook on this computer (tiny CPU run) to catch mistakes before you upload it.

Colab's /content is mapped to a scratch folder, Drive is a copy of upload_to_drive/, installs are skipped,
the model/epochs/image size are shrunk.  Only the glue code is exercised, not the training quality.

  python tools/test_notebook.py <scratch_dir>                        # PCB_Colab.ipynb
  python tools/test_notebook.py <scratch_dir> PCB_Colab_HR.ipynb     # full-resolution variant
"""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = Path(sys.argv[1]).resolve()
content = (base / "content").as_posix()
drive = base / "content/drive/MyDrive/pcb"
drive.mkdir(parents=True, exist_ok=True)
for f in (ROOT / "upload_to_drive").iterdir():
    if not (drive / f.name).exists():
        shutil.copy2(f, drive / f.name)

nb_name = sys.argv[2] if len(sys.argv) > 2 else "PCB_Colab.ipynb"
nb = json.load(open(ROOT / "upload_to_drive" / nb_name, encoding="utf8"))
g = {"__name__": "__main__"}
test_dir = ROOT / ("final_dataset_hr/boards_test/images" if "HR" in nb_name else "final_dataset/fine/images/test")
test_image = test_dir.glob("*.jpg").__next__().as_posix()
done = 0
for n, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]).replace("/content/", content + "/")
    print(f"\n===== cell {n} " + "=" * 50)
    if src.startswith("%%writefile"):
        first, body = src.split("\n", 1)
        path = Path(first.split(None, 1)[1].strip()); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf8"); print("wrote", path, len(body), "chars"); done += 1
        compile(body, str(path), "exec")
        continue
    if "IMAGE = \"\"" in src:
        src = src.replace('IMAGE = ""', f'IMAGE = "{test_image}"').replace("CONF  = 0.20", "CONF  = 0.01")
    try:
        exec(compile(src, f"cell{n}", "exec"), g)
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc(); print("CELL FAILED", n); sys.exit(1)
    done += 1
    if "SKIP_INSTALL" in src and "DRIVE_DIR" in src and "WORK" not in src.split("=")[0]:   # the settings cell
        g.update(SKIP_INSTALL=True, MODEL="yolo26n.pt", IMGSZ=320, EPOCHS=1,
                 EXTRA="--fraction 0.01 --device cpu --workers 0 --cache False --batch 8")
print(f"\nNOTEBOOK DRY RUN OK ({done} code cells)")
