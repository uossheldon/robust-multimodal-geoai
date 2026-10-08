from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_s2_deeplab


CLASS_NAMES = ["background", "low_algae", "mid_algae", "high_algae"]


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
    return {**metrics, "test": test_metrics}
