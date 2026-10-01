from __future__ import annotations

import torch


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    valid = targets != ignore_index
    encoded = num_classes * targets[valid] + predictions[valid]
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def segmentation_metrics(confusion: torch.Tensor) -> dict[str, float]:
    matrix = confusion.float()
    true_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)
    true_positive = matrix.diag()
    union = true_pixels + predicted_pixels - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    dice_denominator = true_pixels + predicted_pixels
    dice = torch.where(dice_denominator > 0, 2 * true_positive / dice_denominator, torch.nan)
    total = matrix.sum()
    accuracy = true_positive.sum() / total if total > 0 else torch.tensor(float("nan"), device=matrix.device)
    return {
        "pixel_accuracy": float(accuracy.item()),
        "mean_iou": float(torch.nanmean(iou).item()),
        "macro_dice": float(torch.nanmean(dice).item()),
        "iou_background": float(iou[0].item()),
        "iou_low": float(iou[1].item()),
        "iou_mid": float(iou[2].item()),
        "iou_high": float(iou[3].item()),
        "dice_background": float(dice[0].item()),
        "dice_low": float(dice[1].item()),
        "dice_mid": float(dice[2].item()),
        "dice_high": float(dice[3].item()),
    }

