"""Build the single self-contained Colab notebook  upload_to_drive/PCB_Colab.ipynb.

The notebook carries ALL code (verify / train / predict are embedded from train/, infer/, tools/),
installs and upgrades what it needs, and every block skips itself when it is already done.
Edit the code in the repo, run  python tools/build_notebook.py  (or run_pipeline.py) to regenerate it.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--hr", action="store_true", help="notebook for the full-resolution tiled dataset (pcb_data_hr.zip, runs_hr/)")
HR = _ap.parse_args().hr
# HR = the shared project notebook; it lives next to its data and checkpoints in one Drive folder (PCB_Detector)
OUT = ROOT / "upload_to_drive" / ("PCB_Detector/PCB_Detector_Colab.ipynb" if HR else "PCB_Colab.ipynb")
# variant settings, filled into the cells below
V = dict(ZIP="pcb_data_hr.zip", DATA="final_dataset_hr", RUNS="runs_hr", BATCH="8", EXTRA="--cache disk", EPOCHS="80", FILEID="1RZFp4TYgWJdIV3RBGg7Pw7tt9ilmM3CB",
         PREVIEW='f"{DATA}/boards_test/images"', TILE="1024", RUNS_DIR="", RUNS_ID="1yq4gdsJe_sAKt0c10tzWhBHi40pEiiop") if HR else \
    dict(ZIP="pcb_data.zip", DATA="final_dataset", RUNS="runs", BATCH="8", EXTRA="", EPOCHS="100", FILEID="",
         PREVIEW='f"{DATA}/fine/images/test"', TILE="0", RUNS_DIR="/content/drive/MyDrive/pcb/runs", RUNS_ID="")
ULTRALYTICS = "8.4.174"   # pinned: the version the training code was tested with; everyone gets the same behaviour


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def embed(path):
    return (ROOT / path).read_text(encoding="utf8").rstrip("\n")


cells = [
    md("""
# تشخیص المان‌های روی برد با YOLO26 (Colab)

همه چیز در **یک پوشه‌ی Drive** است (`@RUNS@`): همین نوت‌بوک، داده (`@ZIP@`) و آموزش. همه‌ی کدها داخل همین نوت‌بوک است.
لینک پوشه: https://drive.google.com/drive/folders/@RUNS_ID@

**برای هر حساب (فقط یک بار):** لینک بالا را باز کن ▸ روی اسم پوشه راست‌کلیک ▸ `Organize ▸ Add shortcut` ▸ `My Drive`. (حساب صاحب پوشه لازم ندارد.) نوت‌بوک پوشه را با شناسه‌اش پیدا می‌کند، در هر حسابی.

1. `Runtime ▸ Change runtime type ▸ T4 GPU` (اگر GPU نبود روی CPU هم اجرا می‌شود، ولی خیلی کند).
2. `Runtime ▸ Run all`.
3. اگر Colab قطع شد: دوباره Run all. هر بلوکی که کارش تمام شده خودش را می‌پرد و آموزش از آخرین epoch ادامه می‌دهد.
4. **هم‌زمان فقط یک نفر آموزش را اجرا کند** (همه در یک پوشه ذخیره می‌کنند).

| بلوک | کار |
|---|---|
| ⚙️ تنظیمات | تنها جایی که لازم است عوض کنی |
| 1 نصب | نصب و به‌روزرسانی کتابخانه‌ها، اتصال به Drive، دیدن GPU |
| 2 کد | سه فایل کد (بررسی داده، آموزش، تست مدل) |
| 3 داده | باز کردن `@ZIP@` |
| 4 بررسی داده | اگر مشکلی باشد قبل از آموزش متوقف می‌شود |
| 5 وضعیت | آموزش تا کجا رفته (بدون اجرای آموزش) |
| 6 آموزش | مرحله‌ی `any` (هر المانی را پیدا کن)، سپس `fine` (نامش را بگو) - با خروجی و نمودار زنده |
| 7 نتیجه | دقت روی بردهای تست و نمونه‌ی خروجی |
| 8 تست روی عکس خودت | هر عکس برد را بده و خروجی را ببین |
"""),
    code('''
# ⚙️ تنظیمات - فقط همین بلوک را در صورت نیاز تغییر بده
RUNS_DIR    = "@RUNS_DIR@"                             # خالی = پوشه‌ی آموزش با شناسه‌ی Drive خودکار پیدا می‌شود (پیشنهاد)
RUNS_FOLDER_ID = "@RUNS_ID@"   # شناسه‌ی پوشه‌ی آموزش در Drive (همه‌ی حساب‌ها همین یک پوشه را می‌بینند)
RUNS_NAME   = "@RUNS@"                       # اسم همان پوشه (وزن‌ها، نمودارها، نتیجه‌ها، داده و نوت‌بوک)
MODEL       = "yolo26s.pt"                   # yolo26n.pt (سبک) | yolo26s.pt | yolo26m.pt (دقیق‌تر، کندتر)
IMGSZ       = 1024                           # اگر حافظه‌ی GPU کم آمد 800 بگذار
EPOCHS      = @EPOCHS@                           # هر epoch در دیتاست تکه‌تکه داده‌ی بیشتری دارد؛ اگر 30 epoch بهتر نشود خودش می‌ایستد
STAGES      = "any fine"                     # "any" = فقط پیدا کردن، "fine" = نام‌گذاری، یا هر دو
SESSION_HOURS = 0                            # مثلا 3.5: قبل از این همه ساعت، بعد از ذخیره‌ی یک epoch، آموزش تمیز متوقف می‌شود (0 = خاموش)
BATCH       = "@BATCH@"                            # اگر خطای حافظه (out of memory) آمد "4" بگذار؛ "-1" = خودکار
EXTRA       = "@EXTRA@"                             # آرگومان اضافه برای آموزش، مثلا "--patience 50" یا "--cos-lr" یا "--export onnx"
SKIP_INSTALL = False                         # True اگر مطمئنی نصب است
ZIP_FILE_ID = "@FILEID@"   # اگر zip در Drive نبود از این لینک دانلود می‌شود (gdown)
ALLOW_NEW_RUN = False      # False: اگر آموزش قبلی (last.pt) در RUNS پیدا نشود، آموزش از صفر شروع نمی‌شود
'''),
    md("## 1) نصب و به‌روزرسانی"),
    code('''
import os, sys, subprocess
WORK = "/content/work"   # دیسک سریع Colab (با قطع شدن پاک می‌شود؛ نتایج روی Drive می‌مانند)

def run(cmd, check=True, watch=None):
    """Run a command and show its output LIVE in this cell (plain subprocess.run shows nothing in Colab).
    watch: glob of an image (e.g. progress.png) that is shown and refreshed after every epoch."""
    import codecs, glob as _glob
    print("$", " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    dec = codecs.getincrementaldecoder("utf8")("replace")
    shown = {}
    def refresh():
        try:
            from IPython.display import Image, display
        except ImportError:
            return
        for f in sorted(_glob.glob(watch)):
            if f in shown:
                shown[f].update(Image(f, width=1100))
            else:
                shown[f] = display(Image(f, width=1100), display_id=True)
    try:
        for chunk in iter(lambda: p.stdout.read1(4096), b""):
            text = dec.decode(chunk)
            sys.stdout.write(text); sys.stdout.flush()
            if watch and "| saved" in text:
                refresh()
    except KeyboardInterrupt:
        p.terminate(); p.wait(); print("\\nstopped (run the block again to resume)"); raise
    p.wait()
    if check and p.returncode:
        raise RuntimeError(f"command failed (exit code {p.returncode}) - read the output above")
    return p.returncode

def sh(*cmd):
    run(cmd)

# اتصال به Drive (خارج از Colab بی‌خطر رد می‌شود)
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Colab نیست - از پوشه‌ی محلی استفاده می‌شود")

def drive_id(path):
    try:
        return os.getxattr(path, "user.drive.id").decode()
    except Exception:
        return None

def find_runs():
    """The training folder is the same Drive folder for every account: the owner sees it in My Drive,
    others through a shortcut. It is recognised by its Drive ID, so the right folder is used everywhere."""
    import glob as _g
    if RUNS_DIR:
        return RUNS_DIR
    pats = [f"/content/drive/.shortcut-targets-by-id/{RUNS_FOLDER_ID}/{RUNS_NAME}",
            f"/content/drive/MyDrive/{RUNS_NAME}", f"/content/drive/MyDrive/*/{RUNS_NAME}", f"/content/drive/MyDrive/*/*/{RUNS_NAME}",
            f"/content/drive/.shortcut-targets-by-id/*/{RUNS_NAME}", f"/content/drive/.shortcut-targets-by-id/*/*/{RUNS_NAME}"]
    cands = [c for pat in pats for c in _g.glob(pat) if os.path.isdir(c)]
    for c in cands:
        if RUNS_FOLDER_ID in c or drive_id(c) == RUNS_FOLDER_ID:
            return c
    with_ck = [c for c in cands if _g.glob(f"{c}/*/weights/last.pt")]
    if with_ck:   # Drive IDs not readable on this runtime: fall back to the folder that holds the training
        print("⚠️ شناسه‌ی Drive خوانده نشد؛ پوشه‌ای که آموزش در آن است انتخاب شد:", with_ck)
        return max(with_ck, key=lambda c: len(_g.glob(f"{c}/*/weights/*.pt")))
    raise FileNotFoundError(
        f"پوشه‌ی آموزش ({RUNS_NAME}) در این حساب دیده نمی‌شود. این لینک را باز کن: https://drive.google.com/drive/folders/{RUNS_FOLDER_ID} "
        "▸ راست‌کلیک روی اسم پوشه ▸ Organize ▸ Add shortcut ▸ My Drive. بعد Runtime ▸ Disconnect and delete runtime و دوباره Run all.")

RUNS = find_runs()
DRIVE_DIR = RUNS            # data zip and notebook live in the same folder
print("پوشه‌ی آموزش و داده:", RUNS)

if not SKIP_INSTALL:
    # نسخه‌ی ثابت: همان نسخه‌ای که کد آموزش با آن آزمایش شده؛ برای همه یکسان است
    sh(sys.executable, "-m", "pip", "install", "-q", "ultralytics==@ULTRALYTICS@", "pyyaml")

import ultralytics, torch, cv2, yaml
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| opencv", cv2.__version__)
ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
assert ver >= (8, 4), "ultralytics قدیمی است (برای YOLO26 نسخه‌ی 8.4 یا بالاتر لازم است): این بلوک را دوباره اجرا کن یا Runtime را Restart کن"
GPU = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if GPU else "ندارد -> آموزش روی CPU انجام می‌شود (خیلی کندتر؛ اگر شد Runtime را روی T4 GPU بگذار)")
os.makedirs(WORK, exist_ok=True); os.makedirs(RUNS, exist_ok=True)
'''),
    md("## 2) کد (سه فایل؛ هر کدام را می‌شود همین‌جا ویرایش کرد)"),
    code("%%writefile /content/work/verify_dataset.py\n" + embed("tools/verify_dataset.py")),
    code("%%writefile /content/work/train_yolo26.py\n" + embed("train/train_yolo26.py")),
    code("%%writefile /content/work/predict.py\n" + embed("infer/predict.py")),
    md("## 3) داده\nاگر دیتاست قبلاً باز شده باشد این بلوک چیزی را دوباره کپی نمی‌کند."),
    code('''
import zipfile, shutil, glob
DATA = f"{WORK}/@DATA@"
ZIP_NAME = "@ZIP@"
if os.path.exists(f"{DATA}/fine/data.yaml"):
    print("دیتاست از قبل آماده است - رد شد")
else:
    local = f"{WORK}/{ZIP_NAME}"
    src = f"{DRIVE_DIR}/{ZIP_NAME}"
    if not os.path.exists(src):
        found = [f for pat in (f"/content/drive/MyDrive/{ZIP_NAME}", f"/content/drive/MyDrive/*/{ZIP_NAME}",
                               f"/content/drive/.shortcut-targets-by-id/*/*/{ZIP_NAME}") for f in glob.glob(pat)]
        src = found[0] if found else None
    if src:
        print(f"کپی {src} روی دیسک سریع Colab ...")
        shutil.copy2(src, local)
    elif ZIP_FILE_ID:
        print(f"{ZIP_NAME} در Drive نبود - دانلود از لینک ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"], check=True)
        import gdown
        gdown.download(id=ZIP_FILE_ID, output=local, quiet=False)
    else:
        raise FileNotFoundError(f"{ZIP_NAME} در Google Drive پیدا نشد - آن را در {DRIVE_DIR} بگذار")
    if not zipfile.is_zipfile(local):
        raise RuntimeError("فایل zip معتبر نیست - دسترسی لینک یا آپلود را چک کن")
    with zipfile.ZipFile(local) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"فایل zip خراب است ({bad}) - دوباره آپلودش کن")
        print("باز کردن ...")
        z.extractall(WORK)
    os.remove(local)
    print("دیتاست باز شد:", DATA)
# the any/ view uses the same pictures as fine/: link them (no extra disk space) when the zip stores them once
for sp in ("train", "val", "test"):
    src_dir, dst_dir = f"{DATA}/fine/images/{sp}", f"{DATA}/any/images/{sp}"
    if os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        for f in os.listdir(src_dir):
            try:
                os.link(f"{src_dir}/{f}", f"{dst_dir}/{f}")
            except OSError:
                shutil.copy2(f"{src_dir}/{f}", f"{dst_dir}/{f}")
        print(f"any/images/{sp}: {len(os.listdir(dst_dir))} عکس")
if os.path.exists(f"{DATA}/report.txt"):
    print(open(f"{DATA}/report.txt", encoding="utf8").read()[:2500])
'''),
    md("## 4) بررسی داده (قبل از مصرف وقت GPU)\nبرچسب‌ها، رنگی بودن همه‌ی عکس‌ها، و اینکه یک برد در دو بخش train/val/test نباشد. اگر خطا باشد اینجا متوقف می‌شود."),
    code('''
rc = run([sys.executable, f"{WORK}/verify_dataset.py", "--root", DATA], check=False)
assert rc == 0, "دیتاست مشکل دارد - گزارش بالا را بخوان و آموزش را شروع نکن"
'''),
    md("## 5) وضعیت آموزش\nبعد از قطع شدن، قبل از ادامه ببین کار تا کجا رفته (چیزی را اجرا یا تغییر نمی‌دهد)."),
    code('''
import csv, json, yaml
from IPython.display import Image, display
for stage in STAGES.split():
    d = f"{RUNS}/{stage}_{os.path.splitext(MODEL)[0]}"
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f"{d}/results.csv"))] if os.path.exists(f"{d}/results.csv") else []
    rows = list({int(float(r["epoch"])): r for r in rows}.values())   # repeated epochs after a fallback: keep the latest
    total = yaml.safe_load(open(f"{d}/args.yaml"))["epochs"] if os.path.exists(f"{d}/args.yaml") else EPOCHS
    if os.path.exists(f"{d}/DONE"):
        state = "تمام شده"
        if os.path.exists(f"{d}/test_metrics.json"):
            m = json.load(open(f"{d}/test_metrics.json", encoding="utf8"))
            state += f" | TEST mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50-95']:.3f}"
    elif rows:
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        state = (f"در حال آموزش: epoch {len(rows)} از {total} | بهترین mAP50-95 تا الان {float(best['metrics/mAP50-95(B)']):.3f} "
                 f"(epoch {best['epoch'].strip()}) | mAP50 {float(best['metrics/mAP50(B)']):.3f}")
    else:
        state = "هنوز شروع نشده"
    print(f"[{stage}] {state}")
    if os.path.exists(f"{d}/progress.png"):
        display(Image(f"{d}/progress.png", width=1100))
'''),
    md("## 6) آموزش\nمرحله‌ی `any`: همه‌ی المان‌ها را بدون توجه به نوع پیدا می‌کند. مرحله‌ی `fine`: نوع را مشخص می‌کند (از وزن‌های `any` شروع می‌شود).\nهر epoch روی Drive ذخیره می‌شود. قطع شد؟ همین بلوک را دوباره اجرا کن.\n\n"
       "خروجی زنده است: بعد از هر epoch یک خط خلاصه می‌آید (loss، دقت روی val، بهترین epoch، زمان هر epoch و زمان باقی‌مانده). "
       "نمودار زنده هم بالای خروجی می‌آید و بعد از هر epoch به‌روز می‌شود. "
       "همین‌ها روی Drive هم هست تا از گوشی ببینی: `<RUNS>/<stage>_<model>/progress.txt` و `progress.png`.\n\n"
       "**ایمنی:** قبل از ذخیره‌ی هر epoch، نسخه‌ی قبلی `last.pt` در `last_prev.pt` نگه داشته می‌شود؛ اگر قطع شدن وسط ذخیره فایل را خراب کند، خودش از نسخه‌ی قبلی ادامه می‌دهد."),
    code('''
# --- ایمنی: ادامه از آموزش قبلی، نه شروع از صفر ---
# (مسیر داخل checkpoint را خود اسکریپت آموزش با این پوشه یکی می‌کند، پس آموزش حساب/پوشه‌ی دیگر هم درست ادامه پیدا می‌کند)
import glob as _g, torch as _t
_ck = _g.glob(f"{RUNS}/*/weights/last.pt")
print("RUNS =", RUNS, flush=True)
print("محتوای پوشه:", sorted(os.listdir(RUNS)), flush=True)
for stage in STAGES.split():
    _d = f"{RUNS}/{stage}_{os.path.splitext(MODEL)[0]}"
    _w = sorted(os.listdir(f"{_d}/weights")) if os.path.isdir(f"{_d}/weights") else []
    if os.path.exists(f"{_d}/DONE"):
        _s = "تمام شده (DONE) -> رد می‌شود"
    elif "last.pt" in _w:
        try:
            _e = _t.load(f"{_d}/weights/last.pt", map_location="cpu", weights_only=False).get("epoch", -1)
            _s = f"ادامه از epoch {_e + 2}" if _e >= 0 else "آموزش تمام شده، فقط تست می‌شود"
        except Exception as ex:
            _s = f"last.pt خراب است ({ex}) -> از نسخه‌ی پشتیبان ادامه می‌دهد"
    else:
        _s = "last.pt ندارد -> از اول شروع می‌شود"
    print(f"[{stage}] {_d} | weights: {_w} | => {_s}", flush=True)
if not _ck and not os.path.exists(f"{RUNS}/final/fine_best.pt"):
    assert ALLOW_NEW_RUN, ("❌ هیچ آموزش قبلی (last.pt) در پوشه‌ی پروژه نیست؛ آموزش از صفر شروع می‌شد. "
                           "اگر آموزش قبلی جای دیگری است مسیرش را در RUNS_DIR بنویس. اگر واقعا آموزش جدید می‌خواهی ALLOW_NEW_RUN = True")
cmd = [sys.executable, f"{WORK}/train_yolo26.py", "--data-root", DATA, "--project", RUNS,
       "--stages", *STAGES.split(), "--model", MODEL, "--imgsz", str(IMGSZ), "--epochs", str(EPOCHS), *EXTRA.split()]
if SESSION_HOURS:
    cmd += ["--hours", str(SESSION_HOURS)]
if BATCH:
    cmd += ["--batch", str(BATCH)]
if not GPU:
    print("⚠️ بدون GPU: آموزش روی CPU (هر epoch چند ساعت). وزن‌ها فقط آخر هر epoch ذخیره می‌شوند.")
    cmd += ["--device", "cpu", "--allow-cpu", "--workers", "2"]
run(cmd, watch=f"{RUNS}/*_{os.path.splitext(MODEL)[0]}/progress.png")
'''),
    md("## 7) نتیجه"),
    code('''
import json, glob
for stage in STAGES.split():
    p = f"{RUNS}/{stage}_{os.path.splitext(MODEL)[0]}/test_metrics.json"
    if os.path.exists(p):
        m = json.load(open(p, encoding="utf8"))
        print(f"[{stage}] TEST  P {m['precision']:.3f}  R {m['recall']:.3f}  mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50-95']:.3f}")
        pc = m.get("per_class", {})
        for n, v in sorted(pc.items(), key=lambda kv: -kv[1]["mAP50"]) if len(pc) > 1 else []:
            print(f"     {n:<24} P {v['precision']:.3f}  R {v['recall']:.3f}  mAP50 {v['mAP50']:.3f}")
print("بهترین وزن‌ها:", glob.glob(f"{RUNS}/final/*.pt"))

out = f"{RUNS}/test_preview"
if not glob.glob(out + "/*_elements.jpg") and os.path.exists(f"{RUNS}/final/any_best.pt"):
    mode = "both" if os.path.exists(f"{RUNS}/final/fine_best.pt") else "any"
    args = ["--any", f"{RUNS}/final/any_best.pt", "--mode", mode, "--source", @PREVIEW@, "--out", out,
            "--tile", "@TILE@", "--imgsz", str(IMGSZ)]
    if mode == "both":
        args += ["--fine", f"{RUNS}/final/fine_best.pt"]
    run([sys.executable, f"{WORK}/predict.py", *args])
try:
    from IPython.display import Image, display
    for p in sorted(glob.glob(out + "/*_elements.jpg"))[:3]:
        print("1) جای المان‌ها:", os.path.basename(p)); display(Image(p, width=1400))
        t = p.replace("_elements.jpg", "_types.jpg")
        if os.path.exists(t):
            print("2) نوع المان‌ها:"); display(Image(t, width=1400))
except ImportError:
    pass
'''),
    md("## 8) تست روی عکس خودت\n`IMAGE` را خالی بگذار تا دکمه‌ی آپلود بیاید، یا مسیر عکس/پوشه را بنویس. `MODE`: `any` (هر المان)، `fine` (فقط نام‌دار)، `both` (پیدا کن و نام ببر؛ ناشناخته‌ها `?`)."),
    code('''
IMAGE = ""        # مثلا "/content/drive/MyDrive/pcb/my_board.jpg" یا خالی برای آپلود
MODE  = "both"
CONF  = 0.20      # کمتر = المان بیشتر (و اشتباه بیشتر)
TILE  = @TILE@      # 1024 = عکس به تکه‌های 1024 پیکسلی (مثل آموزش) بریده می‌شود؛ 0 = کل عکس یکجا
SCALE = 1.0       # اگر قطعه‌ها در عکس خیلی بزرگ‌اند (عکس خیلی نزدیک) 0.5 بگذار

any_w, fine_w = f"{RUNS}/final/any_best.pt", f"{RUNS}/final/fine_best.pt"
if not IMAGE:
    try:
        from google.colab import files
        up = files.upload(); IMAGE = os.path.join(WORK, list(up)[0]); open(IMAGE, "wb").write(list(up.values())[0])
    except ImportError:
        print("آدرس عکس را در IMAGE بنویس")
if IMAGE and os.path.exists(any_w):
    out = f"{WORK}/my_test"
    args = ["--source", IMAGE, "--mode", MODE, "--conf", str(CONF), "--tile", str(TILE), "--scale", str(SCALE),
            "--imgsz", str(IMGSZ), "--out", out, "--any", any_w]
    if MODE in ("fine", "both"):
        args += ["--fine", fine_w]
    run([sys.executable, f"{WORK}/predict.py", *args])
    try:
        from IPython.display import Image, display
        for p in sorted(glob.glob(out + "/*_elements.jpg")):
            print("1) جای المان‌ها (یک رنگ):", os.path.basename(p)); display(Image(p, width=1400))
            t = p.replace("_elements.jpg", "_types.jpg")
            if os.path.exists(t):
                print("2) نوع هر المان:"); display(Image(t, width=1400))
    except ImportError:
        pass
'''),
]

for c in cells:
    c["source"] = [ln.replace("@ZIP@", V["ZIP"]).replace("@DATA@", V["DATA"]).replace("@RUNS@", V["RUNS"])
                   .replace("@BATCH@", V["BATCH"]).replace("@EXTRA@", V["EXTRA"]).replace("@PREVIEW@", V["PREVIEW"])
                   .replace("@TILE@", V["TILE"]).replace("@EPOCHS@", V["EPOCHS"]).replace("@FILEID@", V["FILEID"])
                   .replace("@RUNS_DIR@", V["RUNS_DIR"]).replace("@RUNS_ID@", V["RUNS_ID"]).replace("@ULTRALYTICS@", ULTRALYTICS) for ln in c["source"]]
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"name": OUT.name, "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.parent.mkdir(exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("notebook ->", OUT)
