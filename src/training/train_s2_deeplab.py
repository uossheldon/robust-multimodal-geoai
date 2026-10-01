from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_s2_deeplab


CLASS_NAMES = ["background", "low_algae", "mid_algae", "high_algae"]
CLASS_COLORS = {
    0: (40, 40, 40),
    1: (121, 190, 85),
    2: (247, 188, 65),
    3: (218, 83, 63),
    255: (210, 210, 210),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(project_root: Path, split: str, batch_size: int, *, augment: bool = False, shuffle: bool = False) -> DataLoader:
    dataset = S2RGBSegmentationDataset(
        project_root / "results" / "tile_manifest.csv",
        project_root=project_root,
        split=split,
        augment=augment,
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator, pin_memory=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    mixed_precision: bool = True,
) -> tuple[dict[str, float], torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=training and mixed_precision)
    total_loss = 0.0
    total_valid = 0
    conf = torch.zeros((4, 4), dtype=torch.int64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=mixed_precision):
                logits = model(images)["out"]
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        valid = int((targets != 255).sum().item())
        total_loss += float(loss.detach().item()) * valid
        total_valid += valid
        predictions = logits.detach().argmax(dim=1)
        conf += confusion_matrix(predictions, targets, num_classes=4, ignore_index=255).to(device)
    metrics = segmentation_metrics(conf)
    metrics["loss"] = total_loss / total_valid if total_valid else float("nan")
    return metrics, conf.detach().cpu()


def class_metrics_frame(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"class_id": 0, "class_name": "background", "iou": metrics["iou_background"], "dice": metrics["dice_background"]},
            {"class_id": 1, "class_name": "low_algae", "iou": metrics["iou_low"], "dice": metrics["dice_low"]},
            {"class_id": 2, "class_name": "mid_algae", "iou": metrics["iou_mid"], "dice": metrics["dice_mid"]},
            {"class_id": 3, "class_name": "high_algae", "iou": metrics["iou_high"], "dice": metrics["dice_high"]},
        ]
    )


def save_curves(history: pd.DataFrame, path: Path) -> None:
    width, height = 900, 360
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "S2 DeepLab training curves", fill=(0, 0, 0))
    panels = [
        ("Loss", "train_loss", "validation_loss", 50, 70),
        ("Validation macro mIoU", None, "validation_mean_iou", 500, 70),
    ]
    for title, train_col, val_col, x0, y0 in panels:
        draw.text((x0, y0 - 28), title, fill=(0, 0, 0))
        draw.rectangle((x0, y0, x0 + 330, y0 + 220), outline=(0, 0, 0))
        values = []
        if train_col:
            values.extend(history[train_col].tolist())
        values.extend(history[val_col].tolist())
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1
        def pts(column: str) -> list[tuple[int, int]]:
            series = history[column].tolist()
            out = []
            for idx, value in enumerate(series):
                x = x0 + int(idx * 330 / max(1, len(series) - 1))
                y = y0 + 220 - int((value - lo) * 220 / (hi - lo))
                out.append((x, y))
            return out
        if train_col:
            draw.line(pts(train_col), fill=(70, 120, 210), width=3)
            draw.text((x0, y0 + 232), "train", fill=(70, 120, 210))
        draw.line(pts(val_col), fill=(210, 80, 80), width=3)
        draw.text((x0 + 90, y0 + 232), "validation", fill=(210, 80, 80))
    canvas.save(path)


def save_confusion_matrix(conf: torch.Tensor, path: Path) -> None:
    matrix = conf.numpy().astype(np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    cell = 92
    canvas = Image.new("RGB", (560, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 18), "Test confusion matrix, row-normalized", fill=(0, 0, 0))
    x0, y0 = 130, 80
    for i, name in enumerate(CLASS_NAMES):
        draw.text((25, y0 + i * cell + 35), name, fill=(0, 0, 0))
        draw.text((x0 + i * cell + 16, 55), name.replace("_", "\n"), fill=(0, 0, 0))
        for j in range(4):
            shade = int(255 - normalized[i, j] * 220)
            draw.rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell, y0 + (i + 1) * cell), fill=(shade, shade, 255), outline=(120, 120, 120))
            draw.text((x0 + j * cell + 18, y0 + i * cell + 32), f"{normalized[i, j]:.2f}", fill=(0, 0, 0))
    canvas.save(path)


def save_per_class_iou(class_metrics: pd.DataFrame, path: Path) -> None:
    canvas = Image.new("RGB", (760, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Test per-class IoU", fill=(0, 0, 0))
    max_iou = max(float(class_metrics["iou"].max()), 1e-6)
    for idx, row in enumerate(class_metrics.itertuples(index=False)):
        y = 70 + idx * 64
        width = int(500 * float(row.iou) / max_iou)
        draw.text((24, y + 12), str(row.class_name), fill=(0, 0, 0))
        draw.rectangle((170, y, 170 + width, y + 42), fill=CLASS_COLORS[int(row.class_id)])
        draw.text((182 + width, y + 12), f"{row.iou:.3f}", fill=(0, 0, 0))
    canvas.save(path)


def color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in CLASS_COLORS.items():
        out[mask == value] = color
    return out


def stretch_rgb(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (image.cpu() * std + mean).clamp(0, 1).numpy()
    return (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)


def save_qualitative(model: nn.Module, loader: DataLoader, device: torch.device, path: Path, max_examples: int = 4) -> None:
    model.eval()
    examples = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        for images, targets in loader:
            logits = model(images.to(device))["out"]
            predictions = logits.argmax(dim=1).cpu()
            for image, target, prediction in zip(images, targets, predictions):
                valid = target != 255
                error = torch.full_like(target, 255)
                error[valid & (prediction == target)] = 0
                error[valid & (prediction != target)] = 3
                examples.append((stretch_rgb(image), color_mask(target.numpy()), color_mask(prediction.numpy()), color_mask(error.numpy())))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break
    tile = 224
    label_h = 28
    canvas = Image.new("RGB", (tile * 4, (tile + label_h) * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["RGB", "Ground Truth", "Prediction", "Error Map"]
    for row, example in enumerate(examples):
        y = row * (tile + label_h)
        for col, array in enumerate(example):
            x = col * tile
            draw.text((x + 8, y + 7), labels[col], fill=(0, 0, 0))
            canvas.paste(Image.fromarray(array), (x, y + label_h))
    canvas.save(path)


def train_full(project_root: Path) -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(project_root / ".torch"))
    seed_everything(42)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = 8
    train_loader = make_loader(project_root, "train", batch_size, augment=True, shuffle=True)
    validation_loader = make_loader(project_root, "validation", batch_size)
    test_loader = make_loader(project_root, "test", batch_size)

    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "s2_deeplab_best.pt"
    result_dir = project_root / "results" / "s2_deeplab"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = project_root / "figures"
    figure_dir.mkdir(exist_ok=True)

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
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} val_miou={validation_metrics['mean_iou']:.4f}")
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
                },
                checkpoint_path,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_conf = run_epoch(model, test_loader, criterion, device)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(result_dir / "history.csv", index=False)
    class_frame = class_metrics_frame(test_metrics)
    class_frame.to_csv(result_dir / "per_class_metrics.csv", index=False)
    metrics = {
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "peak_gpu_memory": torch.cuda.max_memory_allocated(device),
        "best_validation_miou": best_miou,
        "batch_size": batch_size,
        "epochs": 10,
        "optimizer": "AdamW",
        "backbone_lr": 1e-5,
        "classifier_lr": 1e-4,
        "weight_decay": 1e-3,
        "scheduler": None,
        "augmentation": "train-only random horizontal flip, vertical flip, and 0/90/180/270 rotation",
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / "test_metrics.json").write_text(json.dumps({**test_metrics, "confusion_matrix": test_conf.numpy().tolist()}, indent=2), encoding="utf-8")
    save_curves(history_frame, figure_dir / "s2_training_curves.png")
    save_confusion_matrix(test_conf, figure_dir / "s2_confusion_matrix.png")
    save_per_class_iou(class_frame, figure_dir / "s2_per_class_iou.png")
    save_qualitative(model, test_loader, device, figure_dir / "s2_qualitative_predictions.png")
    return {**metrics, "test": test_metrics}

