# Baseline metrics

YOLOv8n · 50 epochs · 640 px · batch 16 · lr0 0.01 · Colab GPU
Trained on the **un-merged** 50-class taxonomy, so that the effect of merging
the duplicate classes can be measured against this.

## Overall (validation split: 128 images, 23,327 instances)

| Metric | Value |
|---|---|
| Precision | 0.343 |
| Recall | 0.209 |
| mAP@50 | 0.231 |
| mAP@50-95 | 0.151 |

## Per class

| Class | Images | Instances | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| Button | 1 | 1 | 0.333 | 1.000 | 0.995 | 0.697 |
| Capacitor | 47 | 4,640 | 0.249 | 0.071 | 0.083 | 0.039 |
| Capacitor Jumper | 61 | 4,988 | 0.405 | 0.142 | 0.174 | 0.073 |
| Capacitor Network | 3 | 67 | 0.000 | 0.000 | 0.124 | 0.071 |
| Clock | 4 | 4 | 0.237 | 0.320 | 0.370 | 0.333 |
| Connector | 78 | 616 | 0.378 | 0.715 | 0.541 | 0.324 |
| Diode | 39 | 419 | 0.745 | 0.007 | 0.053 | 0.032 |
| EM | 8 | 20 | 0.830 | 0.245 | 0.606 | 0.375 |
| Fiducial | 4 | 14 | 0.230 | 0.143 | 0.175 | 0.140 |
| Flex Cable | 1 | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| Fuse | 27 | 66 | 0.000 | 0.000 | 0.000 | 0.000 |
| IC | 116 | 1,394 | 0.433 | 0.760 | 0.638 | 0.356 |
| Inductor | 32 | 322 | 0.133 | 0.012 | 0.051 | 0.025 |
| Jumper | 8 | 43 | 1.000 | 0.000 | 0.000 | 0.000 |
| Led | 5 | 13 | 0.837 | 0.769 | 0.865 | 0.631 |
| Pads | 11 | 54 | 0.169 | 0.019 | 0.049 | 0.016 |
| Pins | 4 | 4 | 0.842 | 1.000 | 0.995 | 0.860 |
| Resistor | 52 | 4,479 | 0.000 | 0.000 | 0.021 | 0.007 |
| Resistor Jumper | 64 | 4,486 | 0.265 | 0.040 | 0.065 | 0.024 |
| Resistor Network | 16 | 391 | 0.304 | 0.128 | 0.116 | 0.063 |
| Ressistor | 2 | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| Switch | 6 | 8 | 0.248 | 0.375 | 0.393 | 0.332 |
| Test Point | 6 | 23 | 0.000 | 0.000 | 0.000 | 0.000 |
| Transducer | 2 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| Transformer | 4 | 38 | 1.000 | 0.336 | 0.854 | 0.353 |
| Transistor | 69 | 701 | 0.195 | 0.063 | 0.103 | 0.039 |
| Unknown Unlabeled | 1 | 3 | 0.000 | 0.000 | 0.004 | 0.002 |
| Zener Diode | 1 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| capacitor | 13 | 314 | 0.146 | 0.592 | 0.189 | 0.094 |
| iC | 7 | 17 | 1.000 | 0.000 | 0.000 | 0.000 |

## Reading the table

- `Resistor` at **0.021 mAP@50 on 4,479 instances** is the headline failure,
  and it is a data problem: the label competes with `Resistance`, `Resistor1`,
  `Resistort` and `Ressistor` for the same appearance.
- `capacitor` (lowercase, 0.189) scoring higher than `Capacitor` (0.083) is the
  same effect seen from the other side.
- Single-instance classes (`Button` at 0.995, `Pins` at 0.995) are noise, not
  performance.
- `IC` at 0.638 is the fair measure of what this setup achieves on a
  well-represented, unambiguous class.

See [`../docs/dataset-analysis.md`](../docs/dataset-analysis.md) for the full diagnosis.
