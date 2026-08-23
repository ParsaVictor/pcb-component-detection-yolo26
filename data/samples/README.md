# Sample images

Six boards from the test split, included so that inference can be run without
downloading the full 724 MB dataset:

```bash
python scripts/predict.py --weights best.pt --source data/samples --save
```

The complete dataset lives on
[Roboflow Universe](https://universe.roboflow.com/parsa-karkooti/pcb-wik5v-zonih)
and is fetched with `python scripts/download_dataset.py`.
