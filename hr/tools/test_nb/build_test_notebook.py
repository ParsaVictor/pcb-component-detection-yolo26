"""Build upload_to_drive/PCB_Test/PCB_Test_Colab.ipynb (test only, no training) from the two code files here.

  py -I tools/test_nb/build_test_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "upload_to_drive" / "PCB_Test" / "PCB_Test_Colab.ipynb"


RTL = '<div style="direction:rtl;text-align:right;font-family:B Lotus, B Nazanin, Tahoma">'


def md(s):
    s = RTL + "\n\n" + s.strip("\n") + "\n\n</div>"
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s, hide=False):
    meta = {"cellView": "form"} if hide else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


INTRO = r"""
# تست مدل‌های تشخیص المان روی برد (بدون آموزش)

سه مدل، پشت سر هم:
1. **`any`**: فقط *جای* هر المان (یک رنگ)
2. **`fine`**: نوع هر المان (مقاومت، خازن، IC ...)؛ با `any` ترکیب می‌شود
3. **درخت تصمیم (جنگل تصادفی)**: در بلوک ۲ نوع همه‌ی المان‌ها را می‌گوید؛ در بلوک ۳ برای المان‌هایی که مدل دوم نام نداد یا اطمینانش کمتر از ۳۰٪ بود

| بلوک | کار | سرعت |
|---|---|---|
| ⚙️ تنظیمات | آستانه‌ها و روش ادغام (اسلایدر) | فوری |
| 1 آماده‌سازی | Drive، کتابخانه، بارگذاری ۳ مدل، بردهای تست | یک بار |
| 2 انتخاب عکس | شماره‌ی برد تست، تصادفی، یا عکس خودت | فوری |
| 3 | بلوک ۱: فقط مدل اول (جای المان‌ها) + نقشه‌ی خطا | فوری (از کش) |
| 4 | بلوک ۲: مدل اول + درخت تصمیم برای همه + نقشه‌ی خطا | بار اول کندتر |
| 5 ✅ | بلوک ۳ (اصلی): همه‌ی المان‌ها، مدل دوم نام می‌دهد، درخت برای بی‌نام‌ها و نام‌های ضعیف | بار اول کمی کندتر |

**اجرای اول:** `Runtime ▸ Run all`. بعد از آن فقط بلوک‌هایی را که لازم داری اجرا کن.

**تنظیم سریع:** هر بلوک نمایش (۳ تا ۵) اسلایدر «حداقل اطمینان» و «حالت» خودش را دارد: عوض کن و همان بلوک را دوباره اجرا کن. اگر عکس و حالت عوض نشده باشد، مدل‌ها دوباره اجرا نمی‌شوند (نتیجه‌ی خام نگه داشته شده) و جواب فوری است. بقیه‌ی تنظیم‌ها در بلوک ⚙️ است.

**GPU:** `Runtime ▸ Change runtime type ▸ T4 GPU`. اگر GPU نبود خودکار روی CPU اجرا می‌شود (کندتر؛ زمان‌ها همان را نشان می‌دهند).

**درصد دقت:** روی بردهای تست (برچسب دارند) دقت واقعی نشان داده می‌شود: precision (چند درصد جعبه‌ها درست‌اند)، recall (چند درصد المان‌ها پیدا شدند)، F1، درصد نوع درست، و تعداد تکراری‌ها. روی عکس خودت بدون برچسب فقط میانگین اطمینان مدل نشان داده می‌شود.

فایل‌های لازم، همه در پوشه‌ی `runs_hr` روی Drive:
`final/any_best.pt`، `final/fine_best.pt`، پوشه‌ی `tree/` (مدل درخت)، `pcb_data_hr.zip` (فقط برای بردهای تست)
"""

SETTINGS = r'''
#@title ⚙️ تنظیمات — مقدارها را عوض کن، این بلوک را اجرا کن، بعد یکی از بلوک‌های نمایش (۳ تا ۵) را
#@markdown ### ۱) آستانه‌ها و ادغام جعبه‌ها (فوری؛ مدل دوباره اجرا نمی‌شود)
#@markdown حداقل اطمینان برای قبول یک المان (کمتر = المان بیشتر، و خطای بیشتر)
حداقل_اطمینان_پیدا_کردن = 0.15  #@param {type:"slider", min:0.05, max:0.95, step:0.05}
حداقل_اطمینان_نوع = 0.15  #@param {type:"slider", min:0.05, max:0.95, step:0.05}
#@markdown اگر همپوشانی دو جعبه‌ی هم‌نوع بیشتر از این باشد، یکی حذف می‌شود
همپوشانی_حذف_تکراری = 0.5  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
#@markdown «هوشمند» = رفع چند بار دیده شدن المان‌های بزرگ (پیشنهاد) | «ساده» = روش قبلی
روش_ادغام = "هوشمند"  #@param ["هوشمند", "ساده"]
#@markdown تکه‌ای از المان بزرگ که لبه‌ی برش عکس آن را بریده، اگر این مقدار از آن داخل یک المان کامل باشد حذف می‌شود
سهم_داخل_المان_بزرگ = 0.6  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
#@markdown تکه‌های یک المان که در دو برش کنار هم افتاده‌اند یکی شوند
یکی_کردن_تکه_ها = True  #@param {type:"boolean"}
#@markdown جعبه‌ی تکراری هم‌اندازه: اگر این مقدار داخل جعبه‌ی دیگر باشد **و** اندازه‌اش نزدیک آن باشد حذف می‌شود (المان ریز زیر المان بزرگ حذف نمی‌شود)
سهم_داخل_تکراری = 0.85  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
حداقل_نسبت_اندازه_تکراری = 0.35  #@param {type:"slider", min:0.05, max:1.0, step:0.05}
#@markdown ### ۲) ترکیب سه مدل
#@markdown همپوشانی لازم برای اینکه نوعِ مدل دوم به جعبه‌ی مدل اول داده شود
همپوشانی_تطبیق_دو_مدل = 0.4  #@param {type:"slider", min:0.1, max:0.9, step:0.05}
#@markdown المانی که فقط مدل دوم پیدا کرده، با این اطمینان یا بیشتر اضافه می‌شود
اطمینان_افزودن_از_مدل_دوم = 0.5  #@param {type:"slider", min:0.05, max:1.0, step:0.05}
#@markdown درخت تصمیم فقط وقتی نوع را تعیین می‌کند که احتمالش حداقل این باشد
حداقل_اطمینان_درخت = 0.5  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
#@markdown اگر بیشتر از صفر باشد، درخت برای نوع‌هایی که مدل دوم با اطمینان کمتر از این داده هم نظر می‌دهد
درخت_برای_نوع_ضعیف = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
#@markdown المان‌های کوچک‌تر از این (پیکسل) به درخت داده نمی‌شوند
کمترین_اندازه_برای_درخت = 12  #@param {type:"slider", min:4, max:64, step:2}
#@markdown ### ۳) نمایش
نوشته_روی_جعبه = "خودکار"  #@param ["خودکار", "نام و درصد", "فقط درصد", "هیچ"]
نمایش_نقشه_خطا = True  #@param {type:"boolean"}
#@markdown ### ۴) تشخیص (تغییر این‌ها مدل‌ها را دوباره اجرا می‌کند؛ بعد از تغییر «پردازنده» یا «مدل درخت» بلوک ۱ را هم دوباره اجرا کن)
پردازنده = "خودکار"  #@param ["خودکار", "GPU", "CPU"]
اندازه_برش = 1024  #@param {type:"integer"}
همپوشانی_برش = 0.25  #@param {type:"slider", min:0.0, max:0.5, step:0.05}
اندازه_ورودی_مدل = 1024  #@param {type:"integer"}
#@markdown عکس‌هایی که المان‌هایشان بزرگ است مثل داده‌ی آموزش کوچک می‌شوند (خاموش = «مقیاس_دستی»)
مقیاس_خودکار = True  #@param {type:"boolean"}
مقیاس_دستی = 1.0  #@param {type:"number"}
#@markdown یک بار هم کل برد یکجا دیده شود (برای المان‌های بزرگ)
نگاه_به_کل_برد = True  #@param {type:"boolean"}
مدل_درخت = "بردهای واقعی"  #@param ["بردهای واقعی", "مدل اصلی"]
#@markdown ### ۵) پوشه در Google Drive
شناسه_پوشه = "1yq4gdsJe_sAKt0c10tzWhBHi40pEiiop"  #@param {type:"string"}
نام_پوشه = "runs_hr"  #@param {type:"string"}
مسیر_دستی_پوشه = ""  #@param {type:"string"}

# ---- تبدیل به نام‌هایی که بقیه‌ی نوت‌بوک استفاده می‌کند (لازم نیست تغییر کند) ----
CONF_ANY, CONF_FINE, IOU = حداقل_اطمینان_پیدا_کردن, حداقل_اطمینان_نوع, همپوشانی_حذف_تکراری
MERGE_MODE = {"هوشمند": "smart", "ساده": "simple"}[روش_ادغام]
LABEL_TEXT = {"خودکار": "auto", "نام و درصد": "names", "فقط درصد": "conf", "هیچ": "none"}[نوشته_روی_جعبه]
SHOW_ERRORS = نمایش_نقشه_خطا
TILE, OVERLAP, IMGSZ = int(اندازه_برش), float(همپوشانی_برش), int(اندازه_ورودی_مدل)
TREE_MODEL = {"بردهای واقعی": "boards", "مدل اصلی": "original"}[مدل_درخت]
RUNS_FOLDER_ID, RUNS_NAME, RUNS_DIR = شناسه_پوشه, نام_پوشه, مسیر_دستی_پوشه

import os, warnings
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"
import torch
os.makedirs("/content/work", exist_ok=True)
_GPU = torch.cuda.is_available()
if پردازنده == "GPU" and not _GPU:
    print("⚠️ GPU در دسترس نیست، روی CPU اجرا می‌شود (Runtime ▸ Change runtime type ▸ T4 GPU)")
DEV = 0 if (_GPU and پردازنده != "CPU") else "cpu"
CFG = dict(tile=TILE, overlap=OVERLAP, imgsz=IMGSZ, global_pass=bool(نگاه_به_کل_برد), edge_px=4,
           half=DEV != "cpu", max_det=1000, batch=8 if DEV != "cpu" else 2, device=DEV,
           auto_scale=bool(مقیاس_خودکار), auto_scale_tol=1.2, scale=float(مقیاس_دستی))
P = dict(mode=MERGE_MODE, conf_any=CONF_ANY, conf_fine=CONF_FINE, iou=IOU, cross_iou=0.7, contain=سهم_داخل_المان_بزرگ,
         join=یکی_کردن_تکه_ها, join_align=0.6, dup_ios=سهم_داخل_تکراری, dup_area=حداقل_نسبت_اندازه_تکراری,
         global_min_px=0.25 * TILE, match_iou=همپوشانی_تطبیق_دو_مدل, add_fine_conf=اطمینان_افزودن_از_مدل_دوم,
         tree_min_prob=حداقل_اطمینان_درخت, tree_if_fine_below=درخت_برای_نوع_ضعیف, tree_min_px=کمترین_اندازه_برای_درخت,
         tree_weight=0.4, tree_skip_above=0.85, fine_min=0.3)
print("اجرا روی:", "GPU" if DEV != "cpu" else "CPU", "| روش ادغام:", روش_ادغام,
      "| حداقل اطمینان:", CONF_ANY, "/", CONF_FINE, "| مدل درخت:", مدل_درخت)
'''

SETUP = r'''
#@title 1) آماده‌سازی (یک بار): Drive، کتابخانه، بارگذاری مدل‌ها، بردهای تست
import os, sys, glob, random, zipfile, subprocess, importlib
WORK = "/content/work"
os.makedirs(WORK, exist_ok=True)
if WORK not in sys.path:
    sys.path.insert(0, WORK)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Colab نیست - RUNS_DIR را دستی بنویس")

def drive_id(path):
    try:
        return os.getxattr(path, "user.drive.id").decode()
    except Exception:
        return None

def find_runs():
    if RUNS_DIR:
        return RUNS_DIR
    pats = [f"/content/drive/.shortcut-targets-by-id/{RUNS_FOLDER_ID}/{RUNS_NAME}",
            f"/content/drive/MyDrive/{RUNS_NAME}", f"/content/drive/MyDrive/*/{RUNS_NAME}", f"/content/drive/MyDrive/*/*/{RUNS_NAME}",
            f"/content/drive/.shortcut-targets-by-id/*/{RUNS_NAME}", f"/content/drive/.shortcut-targets-by-id/*/*/{RUNS_NAME}"]
    cands = [c for pat in pats for c in glob.glob(pat) if os.path.isdir(c)]
    for c in cands:
        if RUNS_FOLDER_ID in c or drive_id(c) == RUNS_FOLDER_ID:
            return c
    with_w = [c for c in cands if os.path.exists(f"{c}/final/any_best.pt")]
    if with_w:
        return with_w[0]
    raise FileNotFoundError(f"پوشه‌ی {RUNS_NAME} دیده نمی‌شود: https://drive.google.com/drive/folders/{RUNS_FOLDER_ID} "
                            "▸ راست‌کلیک ▸ Organize ▸ Add shortcut ▸ My Drive، بعد دوباره اجرا کن")

RUNS = find_runs()
print("پوشه:", RUNS)

# same Ultralytics version as the training (the weights were saved with it)
try:
    from importlib.metadata import version
    _v = version("ultralytics")
except Exception:
    _v = None
if _v != "8.4.174":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"], check=True)
import warnings
warnings.filterwarnings("ignore")
import ultralytics, torch, cv2, numpy as np
np.seterr(all="ignore")
import pcb_infer as pi, tree_features as tf
importlib.reload(tf); importlib.reload(pi)
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ندارد (CPU)")

def find_weights(stage):
    """final/<stage>_best.pt (training finished) or else the best.pt of the run that is still going"""
    for w in (f"{RUNS}/final/{stage}_best.pt", *sorted(glob.glob(f"{RUNS}/{stage}_*/weights/best.pt"))):
        if os.path.exists(w):
            return w
    raise FileNotFoundError(f"وزن مدل {stage} در {RUNS} پیدا نشد (final/{stage}_best.pt یا {stage}_yolo26s/weights/best.pt)")
ANY_W, FINE_W = find_weights("any"), find_weights("fine")
M = dict(tags=dict(any=pi.model_tag(ANY_W) + str(DEV), fine=pi.model_tag(FINE_W) + str(DEV)))
M["any"] = pi.load_yolo(ANY_W, DEV, IMGSZ, half=DEV != "cpu")
M["fine"] = pi.load_yolo(FINE_W, DEV, IMGSZ, half=DEV != "cpu")
M["fine_names"] = list(M["fine"].names.values())
M["to_name"] = lambda c: tf.TO_YOLO.get(c, c)
print(f"any  : {ANY_W}  (warm-up {M['any'].warmup_ms:.0f} ms)")
print(f"fine : {FINE_W}  (warm-up {M['fine'].warmup_ms:.0f} ms)  classes {M['fine_names']}")

_tree = {"boards": ("tree_rf_boards.joblib", "tree_features_boards.csv"), "original": ("tree_rf.joblib", "tree_features.csv")}[TREE_MODEL]
_tj, _tc = f"{RUNS}/tree/{_tree[0]}", f"{RUNS}/tree/{_tree[1]}"
if os.path.exists(_tj) or os.path.exists(_tc):
    M["rf"], _how = tf.load_forest(_tj, _tc)
    print(f"tree : {_tj}  ({_how})  classes {list(M['rf'].classes_)}")
else:
    M["rf"] = None
    print(f"⚠️ مدل درخت پیدا نشد ({_tj}) - مرحله‌ی ۳ بدون درخت است. پوشه‌ی tree را در {RUNS} بگذار.")

# labelled test boards (46 boards never used in training) - only these files are read from the zip
BT = f"{WORK}/final_dataset_hr/boards_test"
if not glob.glob(f"{BT}/images/*.jpg"):
    zp = f"{RUNS}/pcb_data_hr.zip"
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.startswith("final_dataset_hr/boards_test/"):
                    z.extract(n, WORK)
    else:
        print("⚠️ pcb_data_hr.zip پیدا نشد: فقط عکس خودت را می‌توانی تست کنی")
BOARDS = sorted(glob.glob(f"{BT}/images/*.jpg"))
GT_NAMES = M["fine_names"]

def read_gt(path, W, H):
    t = os.path.join(os.path.dirname(os.path.dirname(path)), "labels_fine", os.path.splitext(os.path.basename(path))[0] + ".txt")
    return pi.read_labels(t, W, H, GT_NAMES)

print(f"\nبردهای تست: {len(BOARDS)}")
for i, b in enumerate(BOARDS, 1):
    print(f"{i:3d} {os.path.basename(b)[7:-8][:30]:30s}", end="\n" if i % 4 == 0 else "")
print()
'''

PICK = r'''
#@title 2) انتخاب یک عکس (فوری؛ فقط همین عکس بررسی می‌شود)
#@markdown شماره‌ی برد تست از لیست بلوک ۱؛ **0 = تصادفی**
SAMPLE = 0  #@param {type:"integer"}
#@markdown یا عکس خودت: مسیر روی Drive، یا کلمه‌ی `upload` برای دکمه‌ی آپلود (خالی = برد تست)
MY_IMAGE = ""  #@param {type:"string"}
#@markdown اختیاری: فایل برچسب YOLO همان عکس (برای درصد دقت واقعی)
MY_LABELS = ""  #@param {type:"string"}

if MY_IMAGE.strip() == "upload":
    from google.colab import files
    _up = files.upload()
    IMG_PATH = os.path.join(WORK, list(_up)[0]); open(IMG_PATH, "wb").write(list(_up.values())[0])
elif MY_IMAGE.strip():
    IMG_PATH = MY_IMAGE.strip()
else:
    assert BOARDS, "برد تستی نیست - عکس خودت را بده"
    k = SAMPLE if 1 <= SAMPLE <= len(BOARDS) else random.randint(1, len(BOARDS))
    IMG_PATH = BOARDS[k - 1]
    print(f"برد شماره {k}")
IMG = cv2.imread(IMG_PATH)
assert IMG is not None, f"عکس خوانده نشد: {IMG_PATH}"
_H, _W = IMG.shape[:2]
if MY_IMAGE.strip():
    GT = pi.read_labels(MY_LABELS.strip(), _W, _H, GT_NAMES) if MY_LABELS.strip() else None
else:
    GT = read_gt(IMG_PATH, _W, _H)
NAME = os.path.basename(IMG_PATH)
print(f"{NAME}  {_W}x{_H}  | برچسب: {len(GT[0]) if GT else 'ندارد'}")
pi.show(IMG, 700)
print("حالا بلوک ۳، ۴ یا ۵ را اجرا کن (فقط همین یک عکس بررسی می‌شود).")
'''

def mode_params(conf):
    return '''#@markdown **حالت:** «کامل» = هیچ المانی جا نماند (همپوشانی بیشتر برش‌ها + یک نگاه بزرگ‌شده روی بردهایی که المان ریز دارند؛ کمی کندتر) | «سریع» = فقط یک نگاه
حالت = "کامل (همه‌ی المان‌ها)"  #@param ["کامل (همه‌ی المان‌ها)", "سریع"]
#@markdown **حداقل اطمینان پیدا کردن المان:** کمتر = المان بیشتر (و چند جعبه‌ی اشتباه بیشتر)
حداقل_اطمینان = %s  #@param {type:"slider", min:0.05, max:0.9, step:0.05}
_c, _p = pi.settings_for(CFG, P, "complete" if حالت.startswith("کامل") else "fast", حداقل_اطمینان)
''' % conf


S1 = r'''
#@title 3) بلوک ۱: فقط مدل اول (جای المان‌ها)
#@markdown **چه می‌بینی:** عکس ۱ = همه‌ی المان‌ها با رنگ سبز. عکس ۲ (روی بردهای تست) = نقشه‌ی خطا در مقایسه با برچسب‌ها. زیر آن زمان و درصد دقت (P = چند درصد جعبه‌ها درست‌اند، R = چند درصد المان‌ها پیدا شدند).
''' + mode_params("0.15") + r'''
OUT1 = pi.view(M, IMG, GT, _c, _p, 1, LABEL_TEXT, SHOW_ERRORS, name=NAME)
'''
S2 = r'''
#@title 4) بلوک ۲: مدل اول + درخت تصمیم (درخت نوع همه‌ی المان‌ها را می‌گوید)
#@markdown **چه می‌بینی:** مدل اول المان‌ها را پیدا می‌کند و درخت تصمیم نوع **همه**‌ی آن‌ها را می‌گوید. عکس ۱ = نوع‌ها، عکس ۲ = نقشه‌ی خطا. اگر درخت مطمئن نباشد «?». اولین اجرا کندتر است (هر المان حدود ۵۰ میلی‌ثانیه)، بعد از کش.
''' + mode_params("0.1") + r'''
#@markdown درخت فقط وقتی نوع می‌دهد که احتمالش حداقل این باشد
حداقل_اطمینان_درخت = 0.15  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
_p["tree_min_prob"] = حداقل_اطمینان_درخت
assert M["rf"] is not None, "مدل درخت بارگذاری نشده (خروجی بلوک ۱ آماده‌سازی را ببین)"
OUT2 = pi.view(M, IMG, GT, _c, _p, 2, LABEL_TEXT, SHOW_ERRORS, name=NAME)
'''
S4 = r'''
#@title 5) بلوک ۳ (اصلی): همه‌ی المان‌ها — مدل اول پیدا می‌کند، مدل دوم نام می‌دهد، درخت برای بقیه
#@markdown **مراحل:** ۱) مدل اول (و مدل دوم) همه‌ی المان‌ها را پیدا می‌کنند ۲) مدل دوم نوع را می‌گوید ۳) هر المانی که مدل دوم نامش را نداد **یا** اطمینانش کمتر از «مدل_دوم_کمتر_از» بود به درخت داده می‌شود و اگر درخت حداقل «حداقل_اطمینان_درخت» مطمئن بود نوعش را درخت تعیین می‌کند.
#@markdown **چه می‌بینی:** عکس ۱ = همه‌ی المان‌ها با نوعشان (نام‌های درخت گوشه‌ی سفید و «(tree)» دارند). عکس ۲ (با فاصله) = نقشه‌ی خطا. زیرش زمان هر مدل و درصد دقت.
''' + mode_params("0.1") + r'''
#@markdown حداقل اطمینان مدل دوم برای اینکه نوعی بدهد
حداقل_اطمینان_مدل_دوم = 0.1  #@param {type:"slider", min:0.05, max:0.9, step:0.05}
#@markdown اگر اطمینان مدل دوم کمتر از این بود، از درخت پرسیده می‌شود
مدل_دوم_کمتر_از = 0.3  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
#@markdown درخت فقط وقتی نوع را عوض می‌کند که احتمالش حداقل این باشد
حداقل_اطمینان_درخت = 0.15  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
ذخیره_در_درایو = True  #@param {type:"boolean"}
_p.update(conf_fine=حداقل_اطمینان_مدل_دوم, fine_min=مدل_دوم_کمتر_از, tree_min_prob=حداقل_اطمینان_درخت)
assert M["rf"] is not None, "مدل درخت بارگذاری نشده (خروجی بلوک ۱ آماده‌سازی را ببین)"
OUT4 = pi.view(M, IMG, GT, _c, _p, 4, LABEL_TEXT, SHOW_ERRORS, name=NAME)
if ذخیره_در_درایو:
    import csv
    _d = f"{RUNS}/test_results"; os.makedirs(_d, exist_ok=True)
    with open(f"{_d}/{os.path.splitext(NAME)[0]}_parts.csv", "w", newline="", encoding="utf8") as fh:
        _w = csv.writer(fh); _w.writerow(["id", "type", "conf", "named_by", "x0", "y0", "x1", "y1"])
        _r = OUT4["R"]["vote"]
        for i, b in enumerate(_r["boxes"]):
            _w.writerow([i + 1, _r["labels"][i], f"{_r['label_conf'][i]:.3f}", _r["src"][i], *[int(v) for v in b]])
    print("فهرست المان‌ها ذخیره شد:", f"{_d}/{os.path.splitext(NAME)[0]}_parts.csv")
'''

def main():
    tree_src = (HERE / "tree_features.py").read_text(encoding="utf8")
    infer_src = (HERE / "pcb_infer.py").read_text(encoding="utf8")
    cells = [
        md(INTRO),
        code(SETTINGS, hide=True),
        md("## کد (دو فایل؛ لازم نیست بازشان کنی - فقط اجرا شوند)"),
        code("%%writefile /content/work/pcb_infer.py\n" + infer_src),
        code("%%writefile /content/work/tree_features.py\n" + tree_src),
        md("## اجرا"),
        code(SETUP, hide=True),
        code(PICK, hide=True),
        code(S1, hide=True),
        code(S2, hide=True),
        code(S4, hide=True),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf8")
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
