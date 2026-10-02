"""Render Phase 8 figures from frozen result summaries; no model/data access.

Only the five named PNG outputs are written. Scientific CSV/JSON inputs and
the existing qualitative image are never modified.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["s2_weighted", "s1_weighted", "early_fusion", "modality_dropout",
         "occlusion_trained", "terramind_frozen"]
NAMES = ["S2 weighted", "S1 weighted", "Naive early fusion", "Modality dropout",
         "Dropout + occlusion", "Frozen TerraMind"]
COLORS = ["#718096", "#526778", "#9c718f", "#3274a1", "#d07828", "#428577"]


def save(fig, name):
    fig.savefig(ROOT / "figures" / name, dpi=180, facecolor="white")
    plt.close(fig)


def render():
    with (ROOT / "results/final_test/clean_test_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = {r["model"]: r for r in csv.DictReader(handle)}
    summary = json.loads((ROOT / "results/final_test/final_summary.json").read_text())
    robust = {(r["model"], r["condition"]): r for r in summary["robustness_summary"]}
    assert set(rows) == set(ORDER), "Unexpected model set; inspect frozen sources."
    assert [int(rows[k]["n"]) for k in ORDER] == [1, 1, 1, 3, 3, 3]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, key, title, limit in zip(
        axes, ["mean_iou", "macro_dice", "binary_algae_dice"],
        ["Macro mIoU", "Macro Dice", "Binary algae Dice"], [0.30, 0.45, 0.83]
    ):
        for y, model in enumerate(ORDER):
            row = rows[model]
            mean = float(row[key + "_mean"])
            sd = float(row[key + "_std"]) if int(row["n"]) == 3 else 0.0
            ax.barh(y, mean, color=COLORS[y], height=0.6,
                    xerr=sd if sd else None, capsize=3)
            ax.text(mean + sd + 0.008, y, f"{mean:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, limit)
        ax.set_title(title, weight="bold", pad=14)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(6), [n + (" (single)" if i < 3 else " (3 seeds)")
                                for i, n in enumerate(NAMES)])
    axes[0].invert_yaxis()
    fig.suptitle("Frozen final comparison · clean September TEST", fontsize=15, weight="bold")
    fig.text(0.02, 0.065, "Error bars: sample SD across training seeds, not confidence intervals. Single runs have no SD estimate.", fontsize=9)
    fig.text(0.02, 0.025, "114 tiles · S2: 3,474,284 valid pixels; S1/fusion/TerraMind: 3,169,093 · No significance claim.", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    save(fig, "final_clean_model_comparison.png")

    keys = ["iou_background", "iou_low", "iou_mid", "iou_high"]
    values = [[float(rows[m][k + "_mean"]) for k in keys] for m in ORDER]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=0.55, aspect="auto")
    ax.set_xticks(range(4), ["0 · Background", "1 · Low algae", "2 · Mid algae", "3 · High algae"])
    ax.set_yticks(range(6), [n + (" (single)" if i < 3 else " (3 seeds)")
                           for i, n in enumerate(NAMES)])
    for y, model in enumerate(ORDER):
        for x, key in enumerate(keys):
            label = f"{values[y][x]:.4f}"
            if int(rows[model]["n"]) == 3:
                label += f"\n± {float(rows[model][key + '_std']):.4f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=11,
                    color="white" if values[y][x] > 0.31 else "#17212b")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="IoU")
    ax.set_title("Four-class segmentation · clean September TEST", fontsize=15, weight="bold", pad=18)
    fig.text(0.02, 0.065, "± sample SD across 3 training seeds where shown. Class 255 is ignored.", fontsize=9)
    fig.text(0.02, 0.025, "S2 has more valid scoring pixels than SAR-based methods. No statistical-significance claim.", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "final_per_class_iou.png")

    fig, (ax, missing) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for model, name, color, dx in [
        ("modality_dropout", "Modality dropout", COLORS[3], -0.03),
        ("occlusion_trained", "Dropout + occlusion", COLORS[4], 0.03),
    ]:
        xs = [0, 10, 30, 50, 70]
        conds = ["clean", "occlusion_10", "occlusion_30", "occlusion_50", "occlusion_70"]
        series = [robust[(model, c)] for c in conds]
        assert [r["n"] for r in series] == [3, 9, 9, 9, 9]
        ax.errorbar(xs, [r["mean_iou_mean"] for r in series],
                    yerr=[r["mean_iou_std"] for r in series], marker="o", capsize=4,
                    label=name, color=color, linewidth=2)
        miss = [robust[(model, c)] for c in ["clean", "missing_s1", "missing_s2"]]
        assert all(r["n"] == 3 for r in miss)
        missing.errorbar([i + dx for i in range(3)], [r["mean_iou_mean"] for r in miss],
                         yerr=[r["mean_iou_std"] for r in miss], marker="o", capsize=4,
                         linestyle="none", color=color)
    ax.set_xticks([0, 10, 30, 50, 70])
    ax.set_xlabel("Simulated optical occlusion (%)")
    ax.set_title("Optical degradation")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    missing.set_xticks([0, 1, 2], ["Clean", "S1 missing", "S2 missing"])
    missing.set_title("Complete missing modalities")
    missing.set_xlim(-0.3, 2.3)
    for a in [ax, missing]:
        a.set_ylim(0, 0.28)
        a.set_ylabel("Macro mIoU")
        a.grid(alpha=0.2)
    fig.suptitle("Frozen robust fusion models · September TEST", fontsize=15, weight="bold")
    fig.text(0.04, 0.10, "Error bars: clean/missing = SD over 3 training seeds; occlusion = pooled SD over 3 × 3 runs.", fontsize=9)
    fig.text(0.04, 0.055, "Corruption seeds: 101, 202, 303. These SD populations differ; neither is a confidence interval.", fontsize=9)
    fig.text(0.04, 0.015, "Synthetic perturbations, not measured cloud cover. Non-monotonic scores do not imply added information.", fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    save(fig, "final_robustness_curves.png")

    # Completed TerraMind values are transcribed checkpoint validation metadata,
    # not the historical incomplete exports under results/terramind_reproducibility.
    evidence = json.loads((ROOT / "docs/TERRAMIND_VALIDATION_EVIDENCE.json").read_text())
    tm = evidence["aggregate"]
    tmkeys = ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid",
              "iou_high", "binary_algae_iou", "binary_algae_dice"]
    labels = ["Macro mIoU", "Macro Dice", "IoU 0: background", "IoU 1: low", "IoU 2: mid",
              "IoU 3: high", "Binary algae IoU", "Binary algae Dice"]
    assert all(tm[k]["n"] == 3 for k in tmkeys)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for y, (k, label) in enumerate(zip(tmkeys, labels)):
        mean, sd = tm[k]["mean"], tm[k]["std"]
        ax.barh(y, mean, xerr=sd, capsize=3, color=COLORS[5], height=0.58)
        ax.text(mean + sd + 0.015, y, f"{mean:.4f} ± {sd:.4f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.86)
    ax.set_xlabel("Score")
    ax.set_title("Frozen TerraMind · completed clean VALIDATION", weight="bold", pad=15)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.06, "Mean ± sample SD over seeds 42, 7, 123 (n=3 for every metric).", fontsize=9)
    fig.text(0.02, 0.02, "Source: saved checkpoint validation metadata/confusion counts; no Phase 8 inference.", fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "terramind_reproducibility.png")

    with (ROOT / "results/occlusion_training/aggregate_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        val = {r["method"]: r for r in csv.DictReader(handle) if r["primary_condition"] == "clean"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    models = ["modality_dropout", "occlusion_training", "terramind_frozen"]
    for ax, key, label in zip(axes, ["mean_iou", "macro_dice"], ["Macro mIoU", "Macro Dice"]):
        for y, m in enumerate(models):
            mean = tm[key]["mean"] if m == "terramind_frozen" else float(val[m][key + "_mean"])
            sd = tm[key]["std"] if m == "terramind_frozen" else float(val[m][key + "_std"])
            ax.barh(y, mean, xerr=sd, capsize=3, height=0.55, color=COLORS[y + 3])
            ax.text(mean + sd + 0.008, y, f"{mean:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, 0.38 if key == "mean_iou" else 0.54)
        ax.set_title(label, weight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(3), ["Modality dropout", "Dropout + occlusion", "Frozen TerraMind"])
    axes[0].invert_yaxis()
    fig.suptitle("Three-seed clean VALIDATION comparison", fontsize=15, weight="bold")
    fig.text(0.02, 0.06, "Error bars: sample SD over 3 training seeds for every method; no significance claim.", fontsize=9)
    fig.text(0.02, 0.02, "TerraMind: completed checkpoint metadata. DeepLab: frozen Phase 4B2 summaries. Not TEST results.", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    save(fig, "terramind_vs_robust_deeplab.png")


if __name__ == "__main__":
    render()
