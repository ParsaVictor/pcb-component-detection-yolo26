# Dataset analysis — why the baseline mAP is what it is

The baseline reaches **mAP@50 = 0.231**. That number is low, and the reason is
almost entirely in the dataset rather than in the training setup. This document
sets out the diagnosis, because knowing *why* a baseline underperforms is what
makes the next iteration worth running.

## The dataset

| | |
|---|---|
| Source | [Roboflow Universe](https://universe.roboflow.com/parsa-karkooti/pcb-wik5v-zonih) |
| Boards | 3,170 annotated images |
| Annotations | 675,598 |
| Classes | 50 |
| Size | ~724 MB |
| Licence | CC BY 4.0 |
| Label integrity | **0 malformed, 0 out-of-range, 0 out-of-bounds** |

The annotations themselves are clean — `validate_labels` passes on all 675,598
of them. The problem is the class taxonomy and the distribution, not the
geometry.

## Problem 1 — the taxonomy contains duplicates

Seven of the 50 classes are the same physical component under a different
spelling or letter case:

| Duplicate | Canonical | Index → |
|---|---|---|
| `Resistance` | `Resistor` | 32 → 33 |
| `Resistor1` | `Resistor` | 36 → 33 |
| `Resistort` | `Resistor` | 37 → 33 |
| `Ressistor` | `Resistor` | 38 → 33 |
| `capacitor` | `Capacitor` | 47 → 4 |
| `Electrolytic capacitor` | `Electrolytic Capacitor` | 14 → 13 |
| `iC` | `IC` | 49 → 21 |

This is not cosmetic. The detector is being trained to *distinguish* between
`Resistor` and `Ressistor`, which are the same object. Every such pair
guarantees confusion that no amount of training can resolve, and it splits the
training signal for the affected classes across several labels.

The effect is visible in the per-class results. `Resistor` scores
**0.000 precision and 0.000 recall** despite 4,479 validation instances —
the model has no coherent target to learn, because those instances are
competing with four other labels for the same appearance.

`apply_class_merges()` in `pcb_detection.dataset` performs the collapse, and
`scripts/train.py --merge-duplicates` wires it into training. **This is the
single highest-value change available**, and it has not yet been run to
completion — the baseline below is deliberately the *un-merged* result, so
that the improvement can be measured against it.

## Problem 2 — extreme class imbalance

The validation split runs from 4,988 instances (`Capacitor Jumper`) down to
1 or 2 (`Button`, `Pins`, `Transducer`, `Zener Diode`, `Flex Cable`).

Classes with a handful of instances produce meaningless metrics in both
directions: `Button` shows mAP@50 = 0.995 on a single instance, and `Fuse`
shows 0.000 on 66. Neither number describes the model.

## Problem 3 — component size relative to the frame

A populated board downscaled to 640 px leaves small passives a few pixels
across. With 23,327 instances across 128 validation images — an average of
**182 components per board** — most objects are far below the resolution
YOLOv8 can work with at this input size.

Two remedies, both implemented but not yet benchmarked:

- **Tiled inference** (`pcb_detection.tiling`) — run on overlapping crops at
  native resolution, then merge with class-aware NMS.
- **Larger `imgsz`** — 1280 rather than 640, at roughly 4× the training cost.

## Baseline results (50 epochs, YOLOv8n, 640 px)

| Metric | Value |
|---|---|
| Precision | 0.343 |
| Recall | 0.209 |
| mAP@50 | 0.231 |
| mAP@50-95 | 0.151 |

### Classes that work

Where a class is visually distinct and adequately represented, the detector
performs respectably:

| Class | Instances | P | R | mAP@50 |
|---|---|---|---|---|
| Led | 13 | 0.837 | 0.769 | 0.865 |
| IC | 1,394 | 0.433 | 0.760 | 0.638 |
| EM | 20 | 0.830 | 0.245 | 0.606 |
| Connector | 616 | 0.378 | 0.715 | 0.541 |

`IC` is the informative one: 1,394 instances, visually distinct, no duplicate
label competing with it — and it reaches 0.638. That is what the architecture
can do when the data cooperates.

### Classes that fail

| Class | Instances | mAP@50 | Why |
|---|---|---|---|
| Resistor | 4,479 | 0.021 | split across 5 duplicate labels |
| Capacitor | 4,640 | 0.083 | duplicate `capacitor`, tiny objects |
| Fuse | 66 | 0.000 | too few, visually ambiguous |
| Test Point | 23 | 0.000 | a few pixels across at 640 px |

## The ordered fix list

1. **Merge duplicate classes** (50 → 43). Highest value, lowest cost; the code
   is written and tested.
2. **Retrain at `imgsz=1280`** so small passives are resolvable at all.
3. **Group the rare classes** or drop those below a minimum instance count,
   so the metrics describe something real.
4. **Move up to YOLOv8s/m** — the Nano backbone is the right choice for a
   baseline, not for the final model.
5. **Benchmark tiled inference** against the whole-image baseline.

Only after (1) and (2) is a comparison against published PCB-detection results
meaningful. The current number is a starting line, not a result.
