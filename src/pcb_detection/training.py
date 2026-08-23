"""Training and evaluation wrappers around Ultralytics YOLOv8."""

from pathlib import Path

DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "workers": 8,
    "project": "runs/train",
    "name": "pcb_yolov8",
    "device": 0,
}


def train(data_yaml, **overrides):
    """Fine-tune YOLOv8 on the component dataset. Returns the Ultralytics results."""
    from ultralytics import YOLO

    config = {**DEFAULT_CONFIG, **overrides}
    weights = config.pop("model")

    Path(config["project"]).mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    results = model.train(data=str(data_yaml), **config)
    return model, results


def evaluate(model_or_weights, data_yaml, device=0, split="val"):
    """Validate a model and return the headline detection metrics."""
    from ultralytics import YOLO

    model = (
        model_or_weights
        if hasattr(model_or_weights, "val")
        else YOLO(str(model_or_weights))
    )
    metrics = model.val(data=str(data_yaml), device=device, split=split)

    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "raw": metrics,
    }


def best_weights(project="runs/train", name="pcb_yolov8"):
    """Path to ``best.pt`` for a run, or ``None`` if the run has not produced one."""
    path = Path(project) / name / "weights" / "best.pt"
    return path if path.exists() else None
