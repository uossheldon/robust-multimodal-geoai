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
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s2_dataset import S2RGBSegmentationDataset, deeplab_normalize
from src.models.deeplab import create_s2_deeplab
from src.training.train_s2_deeplab import run_epoch


RESULT_DIR = PROJECT_ROOT / "results" / "s2_deeplab"


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
    print(json.dumps({
        "validation_by_date": payload["validation_by_date"],
        "binary_validation": binary,
        "tiny_overfit": overfit,
        "sanity_checks": sanity,
    }, indent=2))


if __name__ == "__main__":
    main()
