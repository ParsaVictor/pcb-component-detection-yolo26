# Notebooks

## `pcb_detection_yolov8.ipynb`

The original training notebook — dataset download, 50 epochs of YOLOv8n on a
Colab GPU, validation, and inference on the test split. Outputs are preserved,
so the full training log and per-class metrics are readable without re-running
anything.

Translated to English, with the Roboflow API key replaced by a
`ROBOFLOW_API_KEY` environment variable.

For running the pipeline, prefer the scripts:

```bash
python scripts/download_dataset.py
python scripts/train.py --data PCB-1/data.yaml --epochs 50 --device 0
```
