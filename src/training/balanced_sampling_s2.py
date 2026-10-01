from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.diagnostics import binary_metrics_from_multiclass
from src.models.deeplab import create_s2_deeplab
from src.training.class_weighted_s2 import load_baseline_validation
from src.training.train_s2_deeplab import make_loader, run_epoch, seed_everything

RESULT_DIR = PROJECT_ROOT / "results" / "s2_deeplab_balanced_sampling"
WEIGHTED_DIR = PROJECT_ROOT / "results" / "s2_deeplab_weighted"
FIGURE_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MIN_ALGAE_FRACTION = 0.01
MIN_ALGAE_PIXELS = 128
MAX_SAMPLER_WEIGHT_RATIO = 4.0

STRATA = ["background/no algae", "low", "mid", "high"]
EXPERIMENT_ORDER = ["baseline", "weighted_loss", "balanced_sampling"]


def assign_tile_strata(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in train.itertuples(index=False):
        valid = int(row.valid_pixel_count)
        stratum = "background/no algae"
        stratum_class = 0
        for klass, label in ((3, "high"), (2, "mid"), (1, "low")):
            count = int(getattr(row, f"class_{klass}_count"))
            fraction = count / valid if valid else 0.0
            if count >= MIN_ALGAE_PIXELS and fraction >= MIN_ALGAE_FRACTION:
                stratum = label
                stratum_class = klass
                break
        rows.append({"stratum": stratum, "stratum_class": stratum_class})
    return pd.concat([train.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def sampler_weights(stratified_train: pd.DataFrame) -> tuple[torch.DoubleTensor, pd.DataFrame]:
    counts = stratified_train["stratum"].value_counts().reindex(STRATA, fill_value=0)
    present_counts = counts[counts > 0]
    raw_by_stratum = {label: 0.0 for label in STRATA}
    for label, count in present_counts.items():
        raw_by_stratum[label] = float(1.0 / np.sqrt(count))
    present_raw = np.array([raw_by_stratum[label] for label in present_counts.index], dtype=np.float64)
    if present_raw.size:
        present_raw = present_raw / present_raw.mean()
    normalized = {label: 0.0 for label in STRATA}
    for label, value in zip(present_counts.index, present_raw):
        normalized[label] = float(value)
    min_present = min(value for value in normalized.values() if value > 0)
    cap_value = min_present * MAX_SAMPLER_WEIGHT_RATIO
    capped = {label: (min(value, cap_value) if value > 0 else 0.0) for label, value in normalized.items()}
    weights = torch.DoubleTensor([capped[label] for label in stratified_train["stratum"]])
    summary = pd.DataFrame(
        [
            {
                "stratum": label,
                "tile_count": int(counts[label]),
                "raw_inverse_sqrt_weight": raw_by_stratum[label],
                "normalized_weight": normalized[label],
                "sampler_weight_capped": capped[label],
                "expected_fraction_before_sampling": float(counts[label] / counts.sum()) if counts.sum() else 0.0,
                "expected_fraction_after_sampling": float((counts[label] * capped[label]) / (counts * pd.Series(capped)).sum()) if (counts * pd.Series(capped)).sum() else 0.0,
            }
            for label in STRATA
        ]
    )
    return weights, summary


def build_train_loader() -> tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    stratified = assign_tile_strata(train)
    weights, summary = sampler_weights(stratified)
    dataset = S2RGBSegmentationDataset(
        PROJECT_ROOT / "results" / "tile_manifest.csv",
        project_root=PROJECT_ROOT,
        split="train",
        augment=True,
    )
    generator = torch.Generator().manual_seed(42)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=0, pin_memory=True)
    return loader, stratified, summary


def prediction_distribution(model: nn.Module, device: torch.device) -> dict[str, object]:
    loader = make_loader(PROJECT_ROOT, "validation", batch_size=8)
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    total = 0
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(images)["out"].argmax(dim=1)
            valid = targets != 255
            total += int(valid.sum().item())
            counts += torch.bincount(preds[valid], minlength=4)
    counts_cpu = counts.cpu().numpy().astype(np.int64)
    return {
        **{f"predicted_class_{klass}_count": int(counts_cpu[klass]) for klass in range(4)},
        **{f"predicted_class_{klass}_fraction": float(counts_cpu[klass] / total) if total else 0.0 for klass in range(4)},
    }


def train_balanced_sampling() -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    device = torch.device("cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader, stratified, strata_summary = build_train_loader()
    validation_loader = make_loader(PROJECT_ROOT, "validation", batch_size=8)
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "s2_deeplab_balanced_sampling_best.pt"
    history = []
    best_miou = -float("inf")
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, 11):
        train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_metrics, validation_conf = run_epoch(model, validation_loader, criterion, device)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(
            f"balanced epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} val_miou={validation_metrics['mean_iou']:.4f}"
        )
        if validation_metrics["mean_iou"] > best_miou:
            best_miou = validation_metrics["mean_iou"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "validation_confusion_matrix": validation_conf.numpy().tolist(),
                    "strata_thresholds": {
                        "min_algae_fraction": MIN_ALGAE_FRACTION,
                        "min_algae_pixels": MIN_ALGAE_PIXELS,
                        "max_sampler_weight_ratio": MAX_SAMPLER_WEIGHT_RATIO,
                    },
                    "strata_summary": strata_summary.to_dict(orient="records"),
                },
                checkpoint_path,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_conf = run_epoch(model, validation_loader, criterion, device)
    binary = binary_metrics_from_multiclass(model, device)
    pred_dist = prediction_distribution(model, device)
    peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
    pd.DataFrame(history).to_csv(RESULT_DIR / "history.csv", index=False)
    strata_summary.to_csv(RESULT_DIR / "strata_summary.csv", index=False)
    stratified.to_csv(RESULT_DIR / "train_tile_strata.csv", index=False)
    metrics_payload = {
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_validation_miou_during_training": best_miou,
        "validation_metrics": validation_metrics,
        "validation_binary_metrics": binary,
        "validation_prediction_distribution": pred_dist,
        "validation_confusion_matrix": validation_conf.numpy().tolist(),
        "peak_gpu_memory_mb": peak_memory_mb,
        "strata_thresholds": {
            "min_algae_fraction": MIN_ALGAE_FRACTION,
            "min_algae_pixels": MIN_ALGAE_PIXELS,
            "max_sampler_weight_ratio": MAX_SAMPLER_WEIGHT_RATIO,
        },
        "strata_summary": strata_summary.to_dict(orient="records"),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return {
        "model": model,
        "metrics": validation_metrics,
        "binary": binary,
        "prediction_distribution": pred_dist,
        "strata_summary": strata_summary,
        "training_summary": metrics_payload,
    }


def load_weighted_validation_row() -> dict[str, object]:
    device = torch.device("cuda")
    validation_loader = make_loader(PROJECT_ROOT, "validation", batch_size=8)
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    checkpoint = torch.load(CHECKPOINT_DIR / "s2_deeplab_weighted_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metrics, _ = run_epoch(model, validation_loader, criterion, device)
    binary = binary_metrics_from_multiclass(model, device)
    pred = prediction_distribution(model, device)
    return {"experiment": "weighted_loss", **metrics, **binary, **pred}


def comparison_rows(balanced: dict[str, object]) -> pd.DataFrame:
    baseline = load_baseline_validation()
    baseline_binary = binary_metrics_from_multiclass(baseline["model"], torch.device("cuda"))
    baseline_pred = prediction_distribution(baseline["model"], torch.device("cuda"))
    rows = [
        {"experiment": "baseline", **baseline["metrics"], **baseline_binary, **baseline_pred},
        load_weighted_validation_row(),
        {"experiment": "balanced_sampling", **balanced["metrics"], **balanced["binary"], **balanced["prediction_distribution"]},
    ]
    frame = pd.DataFrame(rows)
    for col in [f"predicted_class_{klass}_fraction" for klass in range(4)]:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def save_comparison_figure(rows: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    colors = {
        "baseline": (80, 130, 210),
        "weighted_loss": (215, 105, 60),
        "balanced_sampling": (92, 160, 95),
    }
    panels = [
        ("Macro metrics", [("mean_iou", "mIoU"), ("macro_dice", "Dice"), ("binary_algae_iou", "Bin IoU"), ("binary_algae_dice", "Bin Dice")]),
        ("Per-class IoU", [("iou_background", "C0"), ("iou_low", "C1"), ("iou_mid", "C2"), ("iou_high", "C3")]),
        ("Prediction class fraction", [("predicted_class_0_fraction", "C0"), ("predicted_class_1_fraction", "C1"), ("predicted_class_2_fraction", "C2"), ("predicted_class_3_fraction", "C3")]),
    ]
    canvas = Image.new("RGB", (1180, 760), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "S2 imbalance method comparison on validation", fill=(0, 0, 0))
    legend_x = 650
    for idx, experiment in enumerate(EXPERIMENT_ORDER):
        y = 18 + idx * 24
        draw.rectangle((legend_x, y, legend_x + 18, y + 14), fill=colors[experiment])
        draw.text((legend_x + 26, y - 2), experiment, fill=(0, 0, 0))
    panel_y = 70
    for title, metrics in panels:
        draw.text((24, panel_y), title, fill=(0, 0, 0))
        y = panel_y + 36
        max_value = max(float(rows[col].max(skipna=True)) for col, _ in metrics if col in rows.columns)
        max_value = max(max_value, 1e-6)
        for col, label in metrics:
            draw.text((24, y + 17), label, fill=(0, 0, 0))
            for exp_idx, experiment in enumerate(EXPERIMENT_ORDER):
                value = float(rows.loc[rows["experiment"] == experiment, col].iloc[0]) if col in rows.columns else float("nan")
                bar_y = y + exp_idx * 22
                if np.isfinite(value):
                    width = int(800 * value / max_value)
                    draw.rectangle((150, bar_y, 150 + width, bar_y + 15), fill=colors[experiment])
                    draw.text((158 + width, bar_y - 2), f"{value:.3f}", fill=(0, 0, 0))
                else:
                    draw.text((150, bar_y - 2), "n/a", fill=(120, 120, 120))
            y += 84
        panel_y = y + 28
    canvas.save(FIGURE_DIR / "s2_imbalance_methods_comparison.png")


def recommend_configuration(rows: pd.DataFrame) -> str:
    ranked = rows.sort_values(["mean_iou", "macro_dice"], ascending=False).reset_index(drop=True)
    top = str(ranked.loc[0, "experiment"])
    if top == "balanced_sampling":
        return "balanced_sampling"
    if top == "weighted_loss":
        return "weighted_loss"
    return "baseline"


def write_selection_doc(rows: pd.DataFrame, balanced: dict[str, object]) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    strata = balanced["strata_summary"]
    rec = recommend_configuration(rows)
    row = rows[rows["experiment"] == rec].iloc[0]
    table_lines = [
        "| Experiment | Macro mIoU | Macro Dice | IoU 0 | IoU 1 | IoU 2 | IoU 3 | Binary algae IoU | Binary algae Dice | Pred C0 | Pred C1 | Pred C2 | Pred C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in EXPERIMENT_ORDER:
        r = rows[rows["experiment"] == exp].iloc[0]
        table_lines.append(
            f"| {exp} | {r['mean_iou']:.4f} | {r['macro_dice']:.4f} | {r['iou_background']:.4f} | "
            f"{r['iou_low']:.4f} | {r['iou_mid']:.4f} | {r['iou_high']:.4f} | "
            f"{r['binary_algae_iou']:.4f} | {r['binary_algae_dice']:.4f} | "
            f"{r.get('predicted_class_0_fraction', np.nan):.4f} | {r.get('predicted_class_1_fraction', np.nan):.4f} | "
            f"{r.get('predicted_class_2_fraction', np.nan):.4f} | {r.get('predicted_class_3_fraction', np.nan):.4f} |"
        )
    strata_lines = [
        "| Stratum | Tiles | Capped sampler weight | Expected sample fraction |",
        "|---|---:|---:|---:|",
    ]
    for s in strata.itertuples(index=False):
        strata_lines.append(
            f"| {s.stratum} | {int(s.tile_count)} | {float(s.sampler_weight_capped):.4f} | {float(s.expected_fraction_after_sampling):.4f} |"
        )
    doc = f"""# S2 Baseline Selection

Phase 2G compares three Sentinel-2-only configurations on the validation split only. The Phase 2D test result remains frozen and was not re-evaluated.

## Balanced-Sampling Rule

Each training tile is assigned to the highest algae severity stratum that is meaningfully present. A class is meaningful when it has at least `{MIN_ALGAE_PIXELS}` valid pixels and at least `{MIN_ALGAE_FRACTION:.1%}` of the tile's valid pixels. The scan order is high, mid, low; tiles with no algae class passing both thresholds are assigned to background/no algae.

Sampler weights use inverse-square-root stratum frequency and are capped so the largest present stratum weight is at most `{MAX_SAMPLER_WEIGHT_RATIO:.1f}` times the smallest present stratum weight.

{chr(10).join(strata_lines)}

## Validation Comparison

{chr(10).join(table_lines)}

## Recommendation

Carry forward `{rec}` as the conventional optical baseline. It has validation macro mIoU `{row['mean_iou']:.4f}`, macro Dice `{row['macro_dice']:.4f}`, and binary algae IoU `{row['binary_algae_iou']:.4f}`. The recommendation is based on validation performance and class behaviour only.

## Notes

The balanced-sampling run kept the loss unweighted. No SAR, fusion, focal loss, architecture change, or test-set evaluation was used.
"""
    (DOCS_DIR / "S2_BASELINE_SELECTION.md").write_text(doc, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    balanced = train_balanced_sampling()
    rows = comparison_rows(balanced)
    rows.to_csv(RESULT_DIR / "validation_comparison.csv", index=False)
    save_comparison_figure(rows)
    write_selection_doc(rows, balanced)
    payload = {
        "comparison": rows.to_dict(orient="records"),
        "balanced_sampling": balanced["training_summary"],
        "recommendation": recommend_configuration(rows),
    }
    (RESULT_DIR / "s2_baseline_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

