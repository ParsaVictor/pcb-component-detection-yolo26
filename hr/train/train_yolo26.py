"""Train YOLO26 in two stages with checkpoints that survive a Colab disconnect.

  stage "any"  : one class "component"  -> find every element on the board, whatever it is
  stage "fine" : resistor / capacitor / IC ... -> optionally starts from the "any" weights

Everything (last.pt every epoch, best.pt, snapshots, metrics, progress) is written to --project,
which should be a Google Drive folder.  Run the same command again after a disconnect
and every stage resumes by itself; finished stages are skipped.

Per stage, in <project>/<stage>_<model>/ :
  progress.txt   one line per epoch (losses, val metrics, best epoch, speed, ETA)
  progress.png   live training curves, redrawn every epoch
  results.csv    Ultralytics' own per-epoch table
  weights/       last.pt (every epoch), last_prev.pt (one epoch older, used if last.pt is damaged), best.pt
  test_metrics.json  final test-set metrics, per class too

  python train_yolo26.py --data-root /content/final_dataset --project /content/drive/MyDrive/pcb_runs
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

T0 = time.time()

# Augmentation: colour is kept (no grayscale), geometry is mild so boxes stay tight.
AUG = dict(
    hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,        # lighting / camera differences; colour is kept (there is no grayscale augmentation)
    degrees=5.0, translate=0.10, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.5, fliplr=0.5,                  # a board can be seen from any side
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
)


class SessionTimeUp(Exception):
    """Raised after a checkpoint is saved when --hours is used up; the run resumes next time."""


def hms(s):
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def safe_copy(src, dst):
    tmp = Path(str(dst) + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def loads(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"   ! {path.name} cannot be read ({type(e).__name__}: {e})", flush=True)
        return None


def finished(ckpt):
    """Ultralytics marks a checkpoint as finished (epoch = -1) when training ends; such a file cannot be resumed."""
    return ckpt.get("epoch", -1) == -1


def plot_progress(run_dir, stage):
    """Redraw progress.png from results.csv (cheap, a few hundred ms)."""
    f = run_dir / "results.csv"
    if not f.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(f, encoding="utf8"))]
    rows = sorted({int(float(r["epoch"])): r for r in rows}.values(), key=lambda r: float(r["epoch"]))  # repeated epochs after a fallback: keep the latest
    if not rows:
        return
    col = lambda k: [float(r[k]) for r in rows if r.get(k, "").strip()]
    ep = col("epoch")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    losses = [k[6:] for k in rows[0] if k.startswith("train/") and k.endswith("_loss")]  # box/cls/dfl or box/cls/l1 ...
    for i, k in enumerate(losses):
        ax[0].plot(ep, col(f"train/{k}"), f"C{i}", label=f"train {k[:-5]}")
        if len(col(f"val/{k}")) == len(ep):
            ax[0].plot(ep, col(f"val/{k}"), f"C{i}--", label=f"val {k[:-5]}")
    ax[0].set_title("loss (lower is better)")
    for k, lab in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")):
        ax[1].plot(ep, col(k), marker=".", label=lab)
    m = col("metrics/mAP50-95(B)")
    if m:
        b = max(range(len(m)), key=m.__getitem__)
        ax[1].axvline(ep[b], color="grey", ls=":", label=f"best epoch {int(ep[b])}")
    ax[1].set_title("val accuracy (higher is better)")
    ax[2].plot(ep, col("metrics/precision(B)"), label="precision")
    ax[2].plot(ep, col("metrics/recall(B)"), label="recall")
    ax[2].set_title("precision / recall")
    for x in ax:
        x.grid(alpha=.3); x.legend(fontsize=7); x.set_xlabel("epoch")
    fig.suptitle(f"stage '{stage}' - {len(ep)} epochs - updated {time.strftime('%H:%M')}")
    fig.tight_layout()
    tmp = run_dir / "progress.tmp.png"
    fig.savefig(tmp, dpi=80); plt.close(fig)
    os.replace(tmp, run_dir / "progress.png")


def add_callbacks(model, stage, run_dir, a):
    """One clear line per epoch (also in progress.txt), live curves, a backup of the previous checkpoint,
    and a clean pause when the session budget (--hours) is used up."""
    log = run_dir / "progress.txt"
    last, prev = run_dir / "weights" / "last.pt", run_dir / "weights" / "last_prev.pt"
    seen = set()

    def say(msg):
        print(msg, flush=True)
        with open(log, "a", encoding="utf8") as fh:
            fh.write(msg + "\n")

    def on_train_start(tr):
        n_train = len(tr.train_loader.dataset) if getattr(tr, "train_loader", None) else "?"
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu = f"{p.name} {p.total_memory / 2**30:.0f} GB"
        else:
            gpu = "CPU"
        say(f"\n{'=' * 78}\n[{stage}] TRAINING START  {time.strftime('%Y-%m-%d %H:%M')}  | {gpu}\n"
            f"[{stage}] epochs {tr.start_epoch + 1}->{tr.epochs} | train images {n_train} | batch {tr.batch_size} "
            f"| imgsz {tr.args.imgsz} | classes {tr.data['nc']} | patience {tr.args.patience}\n{'=' * 78}")

    def on_train_epoch_end(tr):
        # runs right before Ultralytics overwrites last.pt: keep the previous one as a fallback
        if last.exists():
            try:
                safe_copy(last, prev)
            except OSError as e:
                print(f"   ! backup of last.pt failed: {e}", flush=True)

    def on_fit_epoch_end(tr):
        # the progress line is only a report: a problem in it (e.g. a newer Ultralytics) must never stop training
        try:
            report(tr)
        except SessionTimeUp:
            raise
        except Exception as e:
            print(f"   ! progress line failed ({type(e).__name__}: {e}) - training continues", flush=True)

    def loss_text(tr):
        t = tr.tloss  # tensor in older Ultralytics, {name: value} dict in newer ones
        if t is None:
            return ""
        items = t.items() if isinstance(t, dict) else zip(tr.loss_names, t)
        return "  ".join(f"{str(n).replace('_loss', '')} {float(v):.3f}" for n, v in items)

    def report(tr):
        m = tr.metrics or {}
        g = lambda k: float(m.get(k, 0.0))
        val = (f"P {g('metrics/precision(B)'):.3f}  R {g('metrics/recall(B)'):.3f}  "
               f"mAP50 {g('metrics/mAP50(B)'):.3f}  mAP50-95 {g('metrics/mAP50-95(B)'):.3f}")
        if tr.epoch in seen or tr.epoch >= tr.epochs:  # Ultralytics calls this once more after the final validation of best.pt
            say(f"[{stage}] FINAL (best.pt on val) | {val}")
            return
        seen.add(tr.epoch)
        ep = tr.epoch + 1
        loss = loss_text(tr)
        st, dt = tr.stopper, tr.epoch_time or 0
        mem = f" | GPU {torch.cuda.max_memory_reserved() / 2**30:.1f}G" if torch.cuda.is_available() else ""
        say(f"[{stage}] epoch {ep:>3}/{tr.epochs} | loss {loss} | val {val} | best epoch {st.best_epoch} "
            f"(no gain {ep - st.best_epoch}/{tr.args.patience}) | {hms(dt)}/epoch, ETA {hms(dt * (tr.epochs - ep))}"
            f"{mem} | session {hms(time.time() - T0)} | saved")
        try:
            plot_progress(run_dir, stage)
        except Exception as e:  # a plot must never stop training
            print(f"   ! progress plot failed: {e}", flush=True)
        if a.hours and ep < tr.epochs and time.time() - T0 + 1.2 * dt > a.hours * 3600:
            raise SessionTimeUp

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def pick_checkpoint(run_dir):
    """Return ('resume' | 'finished' | 'new', ckpt path). If last.pt is damaged, falls back to last_prev.pt
    (one epoch older), then to the newest epochN.pt snapshot."""
    w = run_dir / "weights"
    last, prev = w / "last.pt", w / "last_prev.pt"
    snaps = sorted(w.glob("epoch*.pt"), key=lambda p: -int(p.stem[5:]) if p.stem[5:].isdigit() else 0)
    tried = False
    for p in (last, prev, *snaps):
        if not p.exists():
            continue
        tried = True
        ck = loads(p)
        if ck is None:
            continue
        if p != last:
            print(f"   using the backup {p.name} instead of the damaged last.pt", flush=True)
            if last.exists():
                os.replace(last, w / "last_damaged.pt")
            safe_copy(p, last)
        return ("finished" if finished(ck) else "resume"), last
    if tried:
        print("   ! no readable checkpoint found - this stage starts again from the beginning", flush=True)
    return "new", None


def retarget(ckpt_path, run_dir, data):
    """Ultralytics resumes into the absolute folder stored inside last.pt. When the run was started on another
    Google account / Drive folder that path is wrong here, so training would write somewhere else (or look like
    it starts over). Point the checkpoint at the folder it actually lives in now."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck.get("train_args") or {}
    want = {"project": str(run_dir.parent), "name": run_dir.name, "save_dir": str(run_dir), "data": data, "exist_ok": True}
    old = {k: ta.get(k) for k in want}
    if any(str(old[k]) != str(v) for k, v in want.items()):
        ta.update(want)
        ck["train_args"] = ta
        tmp = Path(str(ckpt_path) + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_path)
        print(f"   checkpoint moved: {old['save_dir'] or old['project']}  ->  {run_dir}", flush=True)


def train_kwargs(a):
    """Settings the user gave on the command line that may be changed when resuming (e.g. a smaller batch on another GPU)."""
    out = {k: getattr(a, k) for k in ("batch", "workers", "cache", "device", "patience") if getattr(a, k) is not None}
    return out


def run_stage(stage, a):
    run_dir = Path(a.project) / f"{stage}_{Path(a.model).stem}"
    best, done = run_dir / "weights" / "best.pt", run_dir / "DONE"
    # data.yaml was written on another computer: point its "path" at the real folder
    ds = Path(a.data_root).resolve() / stage
    cfg = yaml.safe_load(open(ds / "data.yaml", encoding="utf8"))
    cfg["path"] = str(ds)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = str(run_dir / "data_local.yaml")
    yaml.safe_dump(cfg, open(data, "w", encoding="utf8"), allow_unicode=True, sort_keys=False)

    if done.exists():
        print(f"[{stage}] already finished -> {best}", flush=True)
        return best
    state, last = pick_checkpoint(run_dir)
    if state == "finished" and best.exists():
        # training ended but the test step below did not (e.g. disconnect): just redo the test step
        print(f"[{stage}] training already complete, running the test evaluation only", flush=True)
    elif state == "resume":
        print(f"[{stage}] resuming from {last}", flush=True)
        retarget(last, run_dir, data)
        model = YOLO(str(last))
        add_callbacks(model, stage, run_dir, a)
        model.train(resume=True, **train_kwargs(a))
    else:
        start = a.model
        init = Path(a.project) / f"any_{Path(a.model).stem}" / "weights" / "best.pt"
        if stage == "fine" and not a.scratch and init.exists():
            start = str(init)
            print(f"[fine] initialising from the 'any' detector: {init}", flush=True)
        model = YOLO(start)
        add_callbacks(model, stage, run_dir, a)
        # batch 8: the automatic batch size picked 1 on Colab's T4 (slow, noisy); boards have hundreds of parts
        # per image, so max_det is raised from 300 for the validation during training
        kw = dict(batch=8, workers=4, cache="ram", patience=30) | train_kwargs(a)
        model.train(
            data=data, epochs=a.epochs, imgsz=a.imgsz, project=a.project, name=run_dir.name, exist_ok=True,
            save=True, save_period=a.save_period, seed=0, plots=True, fraction=a.fraction, cos_lr=a.cos_lr,
            max_det=a.max_det, **kw, **AUG,
        )
    plot_progress(run_dir, stage)

    # always keep a copy of the best weights and the test metrics next to the run
    print(f"[{stage}] evaluating best.pt on the TEST boards ...", flush=True)
    best_model = YOLO(str(best))
    m = best_model.val(data=data, split="test", imgsz=a.imgsz, plots=True, device=a.device, max_det=a.max_det,
                       project=str(run_dir), name="test", exist_ok=True)
    metrics = {"precision": float(m.box.mp), "recall": float(m.box.mr),
               "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": {}}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        metrics["per_class"][best_model.names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                                         "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf8")
    final = Path(a.project) / "final"
    final.mkdir(exist_ok=True)
    shutil.copy2(best, final / f"{stage}_best.pt")
    if a.export:
        f = best_model.export(format=a.export, imgsz=a.imgsz)
        shutil.copy2(f, final / f"{stage}_best{Path(f).suffix}")
    done.write_text("ok")

    print(f"\n[{stage}] TEST  P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
          f"mAP50 {metrics['mAP50']:.3f}  mAP50-95 {metrics['mAP50-95']:.3f}", flush=True)
    if len(metrics["per_class"]) > 1:
        print(f"   {'class':<24}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
        for n, v in sorted(metrics["per_class"].items(), key=lambda kv: -kv[1]["mAP50"]):
            print(f"   {n:<24}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['mAP50']:>8.3f}{v['mAP50-95']:>10.3f}")
    print(f"[{stage}] best weights -> {final / (stage + '_best.pt')}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder that contains any/ and fine/ (final_dataset)")
    ap.add_argument("--project", required=True, help="output folder, put it on Google Drive")
    ap.add_argument("--stages", nargs="+", default=["any", "fine"], choices=["any", "fine"])
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=float, default=None, help="8 by default, -1 = auto; also allowed when resuming, e.g. on a smaller GPU")
    ap.add_argument("--max-det", type=int, default=1000, help="max detections per image in validation/test (boards have many parts)")
    ap.add_argument("--patience", type=int, default=None, help="stop after this many epochs without improvement (default 30)")
    ap.add_argument("--save-period", type=int, default=10, help="extra snapshot every N epochs (last.pt is every epoch)")
    ap.add_argument("--cache", default=None, help="ram (default) | disk | False")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers (default 4)")
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this share of the training set (dry runs)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--cos-lr", action="store_true", help="cosine learning-rate schedule (new runs only)")
    ap.add_argument("--scratch", action="store_true", help="do not start 'fine' from the 'any' weights")
    ap.add_argument("--hours", type=float, default=0, help="pause cleanly (after a saved epoch) before this many hours; 0 = off")
    ap.add_argument("--export", default="", help="also export the best model, e.g. onnx")
    ap.add_argument("--allow-cpu", action="store_true", help="train even without a GPU (very slow)")
    args = ap.parse_args()
    if args.batch is not None:
        args.batch = int(args.batch) if args.batch == int(args.batch) else args.batch
    if args.cache == "False":
        args.cache = False
    if not torch.cuda.is_available() and str(args.device) != "cpu" and not args.allow_cpu:
        sys.exit("No GPU found. In Colab: Runtime > Change runtime type > GPU (or pass --allow-cpu).")
    try:
        for s in args.stages:
            run_stage(s, args)
    except SessionTimeUp:
        print(f"\nSession budget of {args.hours} h reached - stopped after a saved epoch. "
              f"Run the same block again (in a new session) to continue.", flush=True)
