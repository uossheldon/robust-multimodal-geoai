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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s1_dataset import S1SARSegmentationDataset, fit_sar_train_statistics
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_sar_deeplab
from src.training.train_s2_deeplab import class_metrics_frame, run_epoch, seed_everything

RESULT_DIR = PROJECT_ROOT / "results" / "s1_deeplab_weighted"
FIGURE_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
MANIFEST_PATH = PROJECT_ROOT / "results" / "tile_manifest.csv"

CLASS_NAMES = ["background", "low_algae", "mid_algae", "high_algae"]
CLASS_COLORS = {
    0: (40, 40, 40),
    1: (121, 190, 85),
    2: (247, 188, 65),
    3: (218, 83, 63),
    255: (210, 210, 210),
}


def train_class_weights() -> tuple[dict[str, int], dict[str, float]]:
    manifest = pd.read_csv(MANIFEST_PATH)
    train = manifest[manifest["split"] == "train"]
    counts = np.array([train[f"class_{klass}_count"].sum() for klass in range(4)], dtype=np.float64)
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    return (
        {f"class_{klass}": int(counts[klass]) for klass in range(4)},
        {f"class_{klass}": float(weights[klass]) for klass in range(4)},
    )


def make_s1_loader(split: str, stats: dict[str, object], *, batch_size: int = 8, augment: bool = False, shuffle: bool = False) -> DataLoader:
    dataset = S1SARSegmentationDataset(
        MANIFEST_PATH,
        project_root=PROJECT_ROOT,
        split=split,
        sar_mean=stats["mean"],
        sar_std=stats["std"],
        augment=augment,
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator, pin_memory=True)


def binary_metrics_from_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | list[list[int]]]:
    conf = torch.zeros((2, 2), dtype=torch.int64, device=device)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(images)["out"].argmax(dim=1)
            valid = targets != 255
            binary_targets = (targets > 0).long()
            binary_preds = (preds > 0).long()
            encoded = 2 * binary_targets[valid] + binary_preds[valid]
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


def prediction_distribution(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    total = 0
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(images)["out"].argmax(dim=1)
            valid = targets != 255
            total += int(valid.sum().item())
            counts += torch.bincount(preds[valid], minlength=4)
    out = {f"predicted_class_{klass}_count": int(counts[klass].item()) for klass in range(4)}
    out.update({f"predicted_class_{klass}_fraction": float(counts[klass].item() / total) if total else 0.0 for klass in range(4)})
    return out


def smoke_test(stats: dict[str, object], weights: dict[str, float], device: torch.device) -> dict[str, object]:
    seed_everything(42)
    loader = make_s1_loader("train", stats, batch_size=8, augment=True, shuffle=True)
    images, targets = next(iter(loader))
    model = create_sar_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        weight=torch.tensor([weights[f"class_{klass}"] for klass in range(4)], dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    model.train()
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(images)["out"]
        loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return {
        "input_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "cuda_used": bool(device.type == "cuda" and images.is_cuda and logits.is_cuda),
    }


def save_curves(history: pd.DataFrame) -> None:
    canvas = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "S1 DeepLab weighted training curves", fill=(0, 0, 0))
    panels = [("Loss", "train_loss", "validation_loss", 50, 70), ("Validation macro mIoU", None, "validation_mean_iou", 500, 70)]
    for title, train_col, val_col, x0, y0 in panels:
        draw.text((x0, y0 - 28), title, fill=(0, 0, 0))
        draw.rectangle((x0, y0, x0 + 330, y0 + 220), outline=(0, 0, 0))
        values = []
        if train_col:
            values.extend(history[train_col].tolist())
        values.extend(history[val_col].tolist())
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1.0
        def points(column: str) -> list[tuple[int, int]]:
            series = history[column].tolist()
            return [(x0 + int(idx * 330 / max(1, len(series) - 1)), y0 + 220 - int((value - lo) * 220 / (hi - lo))) for idx, value in enumerate(series)]
        if train_col:
            draw.line(points(train_col), fill=(70, 120, 210), width=3)
            draw.text((x0, y0 + 232), "train", fill=(70, 120, 210))
        draw.line(points(val_col), fill=(210, 80, 80), width=3)
        draw.text((x0 + 90, y0 + 232), "validation", fill=(210, 80, 80))
    canvas.save(FIGURE_DIR / "s1_training_curves.png")


def save_per_class_iou(metrics: dict[str, float]) -> None:
    frame = class_metrics_frame(metrics)
    canvas = Image.new("RGB", (760, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "S1 validation per-class IoU", fill=(0, 0, 0))
    max_iou = max(float(frame["iou"].max()), 1e-6)
    for idx, row in enumerate(frame.itertuples(index=False)):
        y = 70 + idx * 64
        width = int(500 * float(row.iou) / max_iou)
        draw.text((24, y + 12), str(row.class_name), fill=(0, 0, 0))
        draw.rectangle((170, y, 170 + width, y + 42), fill=CLASS_COLORS[int(row.class_id)])
        draw.text((182 + width, y + 12), f"{row.iou:.3f}", fill=(0, 0, 0))
    canvas.save(FIGURE_DIR / "s1_per_class_iou.png")


def color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in CLASS_COLORS.items():
        out[mask == value] = color
    return out


def stretch_channel(channel: torch.Tensor) -> np.ndarray:
    array = channel.detach().cpu().numpy().astype(np.float32)
    finite = np.isfinite(array)
    if finite.any():
        lo, hi = np.percentile(array[finite], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        array = np.clip((array - lo) / (hi - lo), 0, 1)
    else:
        array = np.zeros_like(array)
    return (array * 255).astype(np.uint8)


def save_qualitative(model: nn.Module, loader: DataLoader, stats: dict[str, object], device: torch.device, max_examples: int = 4) -> None:
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(2, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(2, 1, 1)
    examples = []
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, targets in loader:
            logits = model(images.to(device))["out"]
            preds = logits.argmax(dim=1).cpu()
            raw = images.cpu() * std + mean
            for image, target, pred in zip(raw, targets, preds):
                valid = target != 255
                error = torch.full_like(target, 255)
                error[valid & (pred == target)] = 0
                error[valid & (pred != target)] = 3
                vv_rgb = np.repeat(stretch_channel(image[0])[:, :, None], 3, axis=2)
                vh_rgb = np.repeat(stretch_channel(image[1])[:, :, None], 3, axis=2)
                examples.append((vv_rgb, vh_rgb, color_mask(target.numpy()), color_mask(pred.numpy()), color_mask(error.numpy())))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break
    tile = 224
    label_h = 28
    canvas = Image.new("RGB", (tile * 5, (tile + label_h) * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["VV", "VH", "Ground Truth", "SAR Prediction", "Error Map"]
    for row, example in enumerate(examples):
        y = row * (tile + label_h)
        for col, array in enumerate(example):
            x = col * tile
            draw.text((x + 8, y + 7), labels[col], fill=(0, 0, 0))
            canvas.paste(Image.fromarray(array), (x, y + label_h))
    canvas.save(FIGURE_DIR / "s1_qualitative_predictions.png")


def save_s1_vs_s2(s1: dict[str, float], s2: dict[str, float]) -> None:
    rows = pd.DataFrame([
        {"experiment": "S2 weighted", **s2},
        {"experiment": "S1 weighted", **s1},
    ])
    canvas = Image.new("RGB", (980, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Validation: selected S2 weighted vs S1 SAR weighted", fill=(0, 0, 0))
    metrics = [("mean_iou", "macro mIoU"), ("macro_dice", "macro Dice"), ("iou_background", "IoU 0"), ("iou_low", "IoU 1"), ("iou_mid", "IoU 2"), ("iou_high", "IoU 3"), ("binary_algae_iou", "Binary algae IoU")]
    max_value = max(float(rows[column].max()) for column, _ in metrics) or 1.0
    y = 65
    for column, label in metrics:
        draw.text((24, y + 10), label, fill=(0, 0, 0))
        for idx, experiment in enumerate(("S2 weighted", "S1 weighted")):
            value = float(rows.loc[rows["experiment"] == experiment, column].iloc[0])
            width = int(620 * value / max_value)
            x = 190
            yy = y + idx * 22
            color = (80, 130, 210) if idx == 0 else (92, 160, 95)
            draw.rectangle((x, yy, x + width, yy + 16), fill=color)
            draw.text((x + width + 8, yy), f"{experiment} {value:.3f}", fill=(0, 0, 0))
        y += 60
    canvas.save(FIGURE_DIR / "s1_vs_s2_validation.png")


def load_s2_weighted_metrics() -> dict[str, float]:
    rows = pd.read_csv(PROJECT_ROOT / "results" / "s2_deeplab_balanced_sampling" / "validation_comparison.csv")
    row = rows[rows["experiment"] == "weighted_loss"].iloc[0]
    return {key: float(row[key]) for key in ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid", "iou_high", "binary_algae_iou", "binary_algae_dice"]}


def write_config(stats: dict[str, object], weights: dict[str, float]) -> None:
    text = f"""experiment: s1_deeplab_weighted
model: DeepLabV3-MobileNetV3-Large
input_channels: [VV, VH]
pretrained_weights: torchvision DeepLabV3 MobileNetV3 Large default
input_stem_initialization: mean RGB stem weights repeated to 2 channels and scaled by 1.5
split: configs/split_v1.yaml
tile_size: 224
batch_size: 8
epochs: 10
optimizer:
  name: AdamW
  backbone_lr: 1.0e-5
  classifier_lr: 1.0e-4
  weight_decay: 1.0e-3
loss:
  name: CrossEntropyLoss
  ignore_index: 255
  class_weights:
    class_0: {weights['class_0']:.12f}
    class_1: {weights['class_1']:.12f}
    class_2: {weights['class_2']:.12f}
    class_3: {weights['class_3']:.12f}
sar_preprocessing:
  units: GAMMA0_TERRAIN dB
  nodata: -9999
  align_to: Sentinel-2 B04 grid
  vv_vh_resampling: bilinear
  mask_resampling: nearest
  train_mean: [{stats['mean'][0]:.12f}, {stats['mean'][1]:.12f}]
  train_std: [{stats['std'][0]:.12f}, {stats['std'][1]:.12f}]
  invalid_policy: fill SAR invalid pixels with train mean, set target to 255
model_selection: validation macro mIoU
use_test_set: false
"""
    (CONFIG_DIR / "s1_deeplab_weighted.yaml").write_text(text, encoding="utf-8")


def write_sar_doc(stats: dict[str, object]) -> None:
    text = f"""# SAR Preprocessing Recovered from Day 2

Phase 3A inspected only the Day 2 lab notebook sections needed for SAR input handling.

Recovered preprocessing:

- Sentinel-1 channels are `VV` and `VH` in that order.
- SAR values are terrain-corrected Gamma0 backscatter in dB: `GAMMA0_TERRAIN` with `output_scale = db`.
- SAR nodata is `-9999`.
- VV/VH are aligned to the Sentinel-2 B04 grid with bilinear resampling.
- Masks are aligned to the B04 grid with nearest-neighbour resampling.
- The SAR-only DeepLab adapter receives a finite `[2, H, W]` tensor in `VV,VH` order.
- RGB reflectance scaling must not be applied to SAR.
- SAR CNN inputs are standardized with mean/std fit from valid training pixels only.
- Day 2 adapts the RGB DeepLab input stem to two channels by repeating the mean RGB stem weights and scaling by `3/2`.

Phase 3A fitted train-only SAR statistics on the fixed split:

- VV mean: `{stats['mean'][0]:.6f}`, std: `{stats['std'][0]:.6f}`
- VH mean: `{stats['mean'][1]:.6f}`, std: `{stats['std'][1]:.6f}`
- valid train pixels used per channel: `{stats['valid_pixel_counts']}`

This project keeps the four-class target rather than Day 2's teaching binary target.
"""
    (DOCS_DIR / "SAR_PREPROCESSING.md").write_text(text, encoding="utf-8")


def train_full() -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    stats = fit_sar_train_statistics(MANIFEST_PATH, project_root=PROJECT_ROOT)
    train_counts, weights = train_class_weights()
    write_config(stats, weights)
    write_sar_doc(stats)
    smoke = smoke_test(stats, weights, device)
    if not smoke["loss_finite"] or not smoke["cuda_used"]:
        raise RuntimeError(f"Smoke test failed: {smoke}")
    train_loader = make_s1_loader("train", stats, batch_size=8, augment=True, shuffle=True)
    validation_loader = make_s1_loader("validation", stats, batch_size=8)
    model = create_sar_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=255,
        weight=torch.tensor([weights[f"class_{klass}"] for klass in range(4)], dtype=torch.float32, device=device),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    checkpoint_path = CHECKPOINT_DIR / "s1_deeplab_weighted_best.pt"
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
        print(f"s1 epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} val_miou={validation_metrics['mean_iou']:.4f}")
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
                    "sar_statistics": stats,
                    "class_weights": weights,
                },
                checkpoint_path,
            )
    torch.cuda.synchronize()
    training_time = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_conf = run_epoch(model, validation_loader, criterion, device)
    binary = binary_metrics_from_loader(model, validation_loader, device)
    pred_dist = prediction_distribution(model, validation_loader, device)
    peak_gpu = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(RESULT_DIR / "history.csv", index=False)
    per_class = class_metrics_frame(validation_metrics)
    per_class.to_csv(RESULT_DIR / "per_class_metrics.csv", index=False)
    s2_metrics = load_s2_weighted_metrics()
    s1_compare = {**validation_metrics, **binary}
    comparison = pd.DataFrame([
        {"experiment": "s2_weighted", **s2_metrics},
        {"experiment": "s1_weighted", **{key: s1_compare[key] for key in s2_metrics}},
    ])
    comparison.to_csv(RESULT_DIR / "validation_vs_s2.csv", index=False)
    metrics_payload = {
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_validation_miou_during_training": best_miou,
        "peak_gpu_memory_mb": peak_gpu,
        "sar_statistics": stats,
        "train_class_counts": train_counts,
        "class_weights": weights,
        "smoke_test": smoke,
        "validation_metrics": validation_metrics,
        "validation_binary_metrics": binary,
        "validation_prediction_distribution": pred_dist,
        "validation_confusion_matrix": validation_conf.numpy().tolist(),
        "s2_weighted_validation_metrics": s2_metrics,
        "validation_comparison": comparison.to_dict(orient="records"),
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (RESULT_DIR / "train_summary.json").write_text(
        json.dumps(
            {
                "model": "DeepLabV3-MobileNetV3-Large",
                "input": "Sentinel-1 VV/VH",
                "epochs": 10,
                "batch_size": 8,
                "optimizer": "AdamW backbone lr 1e-5, classifier lr 1e-4, weight_decay 1e-3",
                "loss": "inverse-square-root class-weighted CrossEntropyLoss(ignore_index=255)",
                "model_selection": "validation macro mIoU",
                "test_evaluated": False,
                "sar_preprocessing_doc": "docs/SAR_PREPROCESSING.md",
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_curves(history_frame)
    save_per_class_iou(validation_metrics)
    save_qualitative(model, validation_loader, stats, device)
    save_s1_vs_s2(s1_compare, s2_metrics)
    print(json.dumps(metrics_payload, indent=2))
    return metrics_payload


if __name__ == "__main__":
    train_full()
