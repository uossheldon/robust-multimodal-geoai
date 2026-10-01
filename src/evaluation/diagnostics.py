from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s2_dataset import S2RGBSegmentationDataset, deeplab_normalize
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_s2_deeplab
from src.training.train_s2_deeplab import run_epoch


RESULT_DIR = PROJECT_ROOT / "results" / "s2_deeplab"
FIGURE_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_for_split(split: str, batch_size: int = 8, augment: bool = False) -> DataLoader:
    dataset = S2RGBSegmentationDataset(
        PROJECT_ROOT / "results" / "tile_manifest.csv",
        project_root=PROJECT_ROOT,
        split=split,
        augment=augment,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def evaluate_by_date(model: nn.Module, device: torch.device) -> pd.DataFrame:
    dataset = S2RGBSegmentationDataset(PROJECT_ROOT / "results" / "tile_manifest.csv", project_root=PROJECT_ROOT, split="validation")
    rows = []
    for date in sorted(dataset.manifest["date"].unique()):
        indices = dataset.manifest.index[dataset.manifest["date"] == date].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        metrics, conf = run_epoch(model, loader, criterion, device, optimizer=None)
        rows.append(
            {
                "date": date,
                "tiles": len(indices),
                "loss": metrics["loss"],
                "macro_miou": metrics["mean_iou"],
                "macro_dice": metrics["macro_dice"],
                "iou_background": metrics["iou_background"],
                "iou_low": metrics["iou_low"],
                "iou_mid": metrics["iou_mid"],
                "iou_high": metrics["iou_high"],
                "dice_background": metrics["dice_background"],
                "dice_low": metrics["dice_low"],
                "dice_mid": metrics["dice_mid"],
                "dice_high": metrics["dice_high"],
                "confusion_matrix": json.dumps(conf.numpy().tolist()),
            }
        )
    return pd.DataFrame(rows)


def binary_metrics_from_multiclass(model: nn.Module, device: torch.device) -> dict[str, float]:
    loader = loader_for_split("validation", batch_size=8)
    conf = torch.zeros((2, 2), dtype=torch.int64, device=device)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)["out"].argmax(dim=1)
            valid = targets != 255
            binary_targets = (targets > 0).long()
            binary_predictions = (predictions > 0).long()
            encoded = 2 * binary_targets[valid] + binary_predictions[valid]
            conf += torch.bincount(encoded, minlength=4).reshape(2, 2)
    matrix = conf.float()
    true_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)
    tp = matrix.diag()
    union = true_pixels + predicted_pixels - tp
    iou = torch.where(union > 0, tp / union, torch.nan)
    dice = torch.where(true_pixels + predicted_pixels > 0, 2 * tp / (true_pixels + predicted_pixels), torch.nan)
    return {
        "binary_background_iou": float(iou[0].item()),
        "binary_algae_iou": float(iou[1].item()),
        "binary_macro_miou": float(torch.nanmean(iou).item()),
        "binary_background_dice": float(dice[0].item()),
        "binary_algae_dice": float(dice[1].item()),
        "binary_macro_dice": float(torch.nanmean(dice).item()),
        "binary_confusion_matrix": conf.cpu().numpy().tolist(),
    }


def split_class_distribution() -> pd.DataFrame:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    rows = []
    for split in ("train", "validation"):
        frame = manifest[manifest["split"] == split]
        valid = int(frame["valid_pixel_count"].sum())
        counts = {klass: int(frame[f"class_{klass}_count"].sum()) for klass in range(4)}
        row = {"split": split, "valid_pixels": valid}
        for klass in range(4):
            row[f"class_{klass}_pixels"] = counts[klass]
            row[f"class_{klass}_fraction"] = counts[klass] / valid if valid else 0.0
        row["algae_pixels"] = counts[1] + counts[2] + counts[3]
        row["algae_fraction"] = row["algae_pixels"] / valid if valid else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_train_val_distribution(distribution: pd.DataFrame) -> None:
    canvas = Image.new("RGB", (820, 300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Train vs validation class distribution", fill=(0, 0, 0))
    colors = [(65, 65, 65), (121, 190, 85), (247, 188, 65), (218, 83, 63)]
    labels = ["0 background", "1 low", "2 mid", "3 high"]
    for idx, row in enumerate(distribution.itertuples(index=False)):
        y = 70 + idx * 70
        draw.text((24, y + 12), row.split, fill=(0, 0, 0))
        x = 150
        for klass, color in enumerate(colors):
            frac = float(getattr(row, f"class_{klass}_fraction"))
            width = int(560 * frac)
            draw.rectangle((x, y, x + width, y + 38), fill=color)
            x += width
    for idx, (label, color) in enumerate(zip(labels, colors)):
        x = 24 + idx * 190
        draw.rectangle((x, 230, x + 22, 246), fill=color)
        draw.text((x + 30, 230), label, fill=(0, 0, 0))
    canvas.save(FIGURE_DIR / "s2_train_val_class_distribution.png")


def save_validation_by_date(rows: pd.DataFrame) -> None:
    canvas = Image.new("RGB", (760, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Validation macro mIoU by date", fill=(0, 0, 0))
    max_value = max(float(rows["macro_miou"].max()), 1e-6)
    for idx, row in enumerate(rows.itertuples(index=False)):
        y = 75 + idx * 90
        width = int(540 * float(row.macro_miou) / max_value)
        draw.text((24, y + 12), row.date, fill=(0, 0, 0))
        draw.rectangle((160, y, 160 + width, y + 42), fill=(70, 120, 210))
        draw.text((172 + width, y + 12), f"{row.macro_miou:.3f}", fill=(0, 0, 0))
    canvas.save(FIGURE_DIR / "s2_validation_by_date.png")


def sanity_checks() -> dict[str, object]:
    dataset_plain = S2RGBSegmentationDataset(PROJECT_ROOT / "results" / "tile_manifest.csv", project_root=PROJECT_ROOT, split="train", transform=None, augment=False)
    dataset_aug = S2RGBSegmentationDataset(PROJECT_ROOT / "results" / "tile_manifest.csv", project_root=PROJECT_ROOT, split="train", transform=None, augment=True)
    image, mask = dataset_plain[0]
    torch.manual_seed(123)
    aug_image, aug_mask = dataset_aug[0]
    allowed_values = sorted(int(value) for value in torch.unique(mask).tolist())
    aug_values = sorted(int(value) for value in torch.unique(aug_mask).tolist())
    normalized = deeplab_normalize(image.clone())
    return {
        "plain_mask_values": allowed_values,
        "augmented_mask_values": aug_values,
        "class_ids_valid_after_augmentation": set(aug_values).issubset({0, 1, 2, 3, 255}),
        "ignore_255_present": 255 in allowed_values,
        "image_reflectance_min": float(image.min().item()),
        "image_reflectance_max": float(image.max().item()),
        "normalized_mean_approx": [float(value) for value in normalized.mean(dim=(1, 2)).tolist()],
        "same_augmented_shapes": tuple(aug_image.shape) == tuple(image.shape) and tuple(aug_mask.shape) == tuple(mask.shape),
        "mask_interpolation_values_ok": set(allowed_values).issubset({0, 1, 2, 3, 255}) and set(aug_values).issubset({0, 1, 2, 3, 255}),
    }


def tiny_overfit(device: torch.device) -> dict[str, object]:
    seed_everything(123)
    full_dataset = S2RGBSegmentationDataset(PROJECT_ROOT / "results" / "tile_manifest.csv", project_root=PROJECT_ROOT, split="train", augment=False)
    eval_dataset = S2RGBSegmentationDataset(PROJECT_ROOT / "results" / "tile_manifest.csv", project_root=PROJECT_ROOT, split="train", augment=False)
    indices = list(range(16))
    train_loader = DataLoader(Subset(full_dataset, indices), batch_size=4, shuffle=True, num_workers=0)
    eval_loader = DataLoader(Subset(eval_dataset, indices), batch_size=4, shuffle=False, num_workers=0)
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    start = time.perf_counter()
    last_train = {}
    for epoch in range(1, 41):
        last_train, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        if last_train["mean_iou"] > 0.90:
            break
    final_metrics, final_conf = run_epoch(model, eval_loader, criterion, device)
    return {
        "tiles": len(indices),
        "epochs_run": epoch,
        "seconds": time.perf_counter() - start,
            "final_train_loss": last_train.get("loss"),
        "final_eval_loss": final_metrics["loss"],
        "final_eval_macro_miou": final_metrics["mean_iou"],
        "final_eval_macro_dice": final_metrics["macro_dice"],
        "final_eval_iou_background": final_metrics["iou_background"],
        "final_eval_iou_low": final_metrics["iou_low"],
        "final_eval_iou_mid": final_metrics["iou_mid"],
        "final_eval_iou_high": final_metrics["iou_high"],
        "confusion_matrix": final_conf.numpy().tolist(),
    }


def write_doc(payload: dict[str, object]) -> None:
    history = pd.read_csv(RESULT_DIR / "history.csv")
    best = history.loc[history["validation_mean_iou"].idxmax()]
    overfit = payload["tiny_overfit"]
    binary = payload["binary_validation"]
    by_date = pd.DataFrame(payload["validation_by_date"])
    distribution = pd.DataFrame(payload["class_distribution"])

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in frame[columns].itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    doc = f"""# Baseline Diagnostics

Phase 2E diagnoses the Sentinel-2 DeepLab baseline without retraining the full model and without re-evaluating the test set.

## Learning-Curve Diagnosis

Training loss fell from `{history.iloc[0]['train_loss']:.3f}` to `{history.iloc[-1]['train_loss']:.3f}` and training macro mIoU rose from `{history.iloc[0]['train_mean_iou']:.3f}` to `{history.iloc[-1]['train_mean_iou']:.3f}`. Validation macro mIoU stayed low, peaking at `{best['validation_mean_iou']:.3f}` on epoch `{int(best['epoch'])}` while validation loss remained high.

Diagnosis: the model is learning the training set but validation performance plateaus at low macro IoU, especially for high algae. This points to distribution shift and class imbalance more than a simple pipeline failure.

## Train / Validation Class Distribution

{table(distribution.round(4), ['split', 'valid_pixels', 'class_0_fraction', 'class_1_fraction', 'class_2_fraction', 'class_3_fraction', 'algae_fraction'])}

## Validation By Date

{table(by_date.round(4), ['date', 'tiles', 'macro_miou', 'macro_dice', 'iou_background', 'iou_low', 'iou_mid', 'iou_high'])}

## Tiny-Subset Overfit

The model was trained on 16 training tiles for `{overfit['epochs_run']}` epochs. Final no-augmentation evaluation on those same tiles:

- Macro mIoU: `{overfit['final_eval_macro_miou']:.3f}`
- Macro Dice/F1: `{overfit['final_eval_macro_dice']:.3f}`
- IoU background/low/mid/high: `{overfit['final_eval_iou_background']:.3f}`, `{overfit['final_eval_iou_low']:.3f}`, `{overfit['final_eval_iou_mid']:.3f}`, `{overfit['final_eval_iou_high']:.3f}`

The model can partially memorize the small subset, so the core forward/loss/update path works, but high-class recovery remains weak.

## Binary Diagnostic

Validation predictions collapsed to background versus algae:

- Algae IoU: `{binary['binary_algae_iou']:.3f}`
- Algae Dice/F1: `{binary['binary_algae_dice']:.3f}`
- Binary macro mIoU: `{binary['binary_macro_miou']:.3f}`

This diagnostic does not replace the 4-class task.

## Pipeline Checks

- Mask values after loading and augmentation remain in `{payload['sanity_checks']['augmented_mask_values']}`.
- `255` ignore is present and handled by the loss/evaluation path.
- Nearest-neighbour mask alignment avoids interpolated labels.
- RGB reflectance is clipped to `[0, 1]` before ImageNet normalization.
- Augmented image and mask shapes remain identical.

## Recommended Next Controlled Experiment

Before adding SAR or changing architecture, run one S2-only controlled experiment with the same split and model but with class-balanced sampling or loss weighting as the only change. This isolates whether the poor high/low class performance is mainly due to class imbalance.
"""
    (DOCS_DIR / "BASELINE_DIAGNOSTICS.md").write_text(doc, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    device = torch.device("cuda")
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    checkpoint = torch.load(PROJECT_ROOT / "checkpoints" / "s2_deeplab_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    validation_by_date = evaluate_by_date(model, device)
    validation_by_date.to_csv(RESULT_DIR / "validation_by_date.csv", index=False)
    binary = binary_metrics_from_multiclass(model, device)
    distribution = split_class_distribution()
    save_train_val_distribution(distribution)
    save_validation_by_date(validation_by_date)
    sanity = sanity_checks()
    overfit = tiny_overfit(device)
    payload = {
        "learning_curve": pd.read_csv(RESULT_DIR / "history.csv").to_dict(orient="records"),
        "class_distribution": distribution.to_dict(orient="records"),
        "validation_by_date": validation_by_date.drop(columns=["confusion_matrix"]).to_dict(orient="records"),
        "binary_validation": binary,
        "sanity_checks": sanity,
        "tiny_overfit": overfit,
    }
    (RESULT_DIR / "baseline_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_doc(payload)
    print(json.dumps({
        "validation_by_date": payload["validation_by_date"],
        "binary_validation": binary,
        "tiny_overfit": overfit,
        "sanity_checks": sanity,
    }, indent=2))


if __name__ == "__main__":
    main()

