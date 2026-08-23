#!/usr/bin/env python
"""Render the dataset-analysis figure used in the README.

    python scripts/make_analysis_figure.py --out results/dataset-analysis.png

Three panels making the case that the low baseline mAP is a data problem:
the duplicate labels, the class imbalance, and the relationship between how
well-represented a class is and how well it scores.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Per-class validation results from the 50-epoch baseline
# (see results/baseline-metrics.md).
PER_CLASS = [
    ("Button", 1, 0.995), ("Capacitor", 4640, 0.083), ("Capacitor Jumper", 4988, 0.174),
    ("Capacitor Network", 67, 0.124), ("Clock", 4, 0.370), ("Connector", 616, 0.541),
    ("Diode", 419, 0.053), ("EM", 20, 0.606), ("Fiducial", 14, 0.175),
    ("Flex Cable", 4, 0.0), ("Fuse", 66, 0.0), ("IC", 1394, 0.638),
    ("Inductor", 322, 0.051), ("Jumper", 43, 0.0), ("Led", 13, 0.865),
    ("Pads", 54, 0.049), ("Pins", 4, 0.995), ("Resistor", 4479, 0.021),
    ("Resistor Jumper", 4486, 0.065), ("Resistor Network", 391, 0.116),
    ("Ressistor", 6, 0.0), ("Switch", 8, 0.393), ("Test Point", 23, 0.0),
    ("Transducer", 2, 0.0), ("Transformer", 38, 0.854), ("Transistor", 701, 0.103),
    ("Unknown Unlabeled", 3, 0.004), ("Zener Diode", 2, 0.0),
    ("capacitor", 314, 0.189), ("iC", 17, 0.0),
]

DUPLICATE_GROUPS = [
    ("Resistor", ["Resistance", "Resistor1", "Resistort", "Ressistor"]),
    ("Capacitor", ["capacitor"]),
    ("Electrolytic Capacitor", ["Electrolytic capacitor"]),
    ("IC", ["iC"]),
]

BAD = "#c53030"
GOOD = "#2f855a"
ACCENT = "#2b6cb0"
MUTED = "#718096"
INK = "#1a202c"


def panel_duplicates(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Seven of the 50 classes are the same component",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14)

    y = 0.86
    for canonical, duplicates in DUPLICATE_GROUPS:
        labels = [canonical] + duplicates
        height = 0.052 * len(labels)

        for i, label in enumerate(labels):
            top = y - i * 0.052
            is_canonical = i == 0
            ax.add_patch(
                FancyBboxPatch(
                    (0.02, top - 0.040), 0.40, 0.040,
                    boxstyle="round,pad=0,rounding_size=0.008",
                    facecolor="#f0fff4" if is_canonical else "#fff5f5",
                    edgecolor=GOOD if is_canonical else BAD,
                    linewidth=1.3,
                )
            )
            ax.text(0.22, top - 0.020, label, ha="center", va="center",
                    fontsize=8.5, family="monospace",
                    color="#22543d" if is_canonical else "#742a2a",
                    fontweight="bold" if is_canonical else "normal")

        mid = y - (height - 0.052) / 2 - 0.020
        ax.add_patch(
            FancyArrowPatch((0.45, mid), (0.58, mid), arrowstyle="-|>",
                            mutation_scale=12, color=MUTED, linewidth=1.5)
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.61, mid - 0.020), 0.36, 0.040,
                boxstyle="round,pad=0,rounding_size=0.008",
                facecolor="#f0fff4", edgecolor=GOOD, linewidth=1.6,
            )
        )
        ax.text(0.79, mid, canonical, ha="center", va="center", fontsize=8.5,
                family="monospace", fontweight="bold", color="#22543d")

        y -= height + 0.030

    ax.text(0.5, 0.10,
            "The detector is trained to tell `Resistor` from `Ressistor`.\n"
            "It cannot — and the signal for resistors is split five ways.",
            ha="center", va="center", fontsize=9, color="#742a2a",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff5f5",
                      edgecolor="#feb2b2"))


def panel_imbalance(ax):
    ordered = sorted(PER_CLASS, key=lambda r: -r[1])
    names = [r[0] for r in ordered]
    counts = [r[1] for r in ordered]
    colours = [BAD if n in ("Resistor", "Capacitor") else ACCENT for n in names]

    ax.barh(range(len(names)), counts, color=colours, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Validation instances (log scale)", fontsize=9.5)
    ax.set_title("Class frequency spans four orders of magnitude",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    ax.text(0.97, 0.04,
            "red = the two most common components,\nboth wrecked by duplicate labels",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#742a2a",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff5f5",
                      edgecolor="#feb2b2"))


def panel_scatter(ax):
    counts = np.array([r[1] for r in PER_CLASS], dtype=float)
    scores = np.array([r[2] for r in PER_CLASS])
    names = [r[0] for r in PER_CLASS]

    highlight = {"Resistor", "Capacitor", "Resistor Jumper", "capacitor"}
    colours = [BAD if n in highlight else (GOOD if s > 0.5 else ACCENT)
               for n, s in zip(names, scores)]

    ax.scatter(counts, scores, c=colours, s=58, alpha=0.85,
               edgecolor="white", linewidth=1.1, zorder=3)

    annotations = {
        "Resistor": (-62, 26), "Capacitor": (-6, -30), "IC": (-42, 20),
        "Led": (-30, -26), "Connector": (-70, 16),
    }
    for name, count, score in zip(names, counts, scores):
        if name in annotations:
            dx, dy = annotations[name]
            ax.annotate(name, (count, score), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8,
                        color=BAD if name in highlight else "#2d3748",
                        arrowprops=dict(arrowstyle="-", color="#cbd5e0", linewidth=0.9))

    ax.set_xscale("log")
    ax.set_xlabel("Validation instances (log scale)", fontsize=9.5)
    ax.set_ylabel("mAP@50", fontsize=9.5)
    ax.set_title("More data does not help when the label is ambiguous",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=8)

    ax.text(0.5, 0.30,
            "Resistor: 4,479 instances, 0.021 mAP\nIC: 1,394 instances, 0.638 mAP",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="#2d3748",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7fafc",
                      edgecolor="#cbd5e0"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/dataset-analysis.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(17, 6.2))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.05], wspace=0.30,
                            left=0.035, right=0.985, top=0.83, bottom=0.11)

    fig.suptitle(
        "Why the baseline sits at mAP@50 = 0.231 — a data problem, not a training problem",
        fontsize=15, fontweight="bold", color=INK, y=0.955,
    )

    panel_duplicates(fig.add_subplot(grid[0, 0]))
    panel_imbalance(fig.add_subplot(grid[0, 1]))
    panel_scatter(fig.add_subplot(grid[0, 2]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=125, facecolor="white", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
