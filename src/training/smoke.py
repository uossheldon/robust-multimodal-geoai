from __future__ import annotations

import random
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_s2_deeplab


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(project_root: Path, batch_size: int, train_subset: int, validation_subset: int) -> tuple[DataLoader, DataLoader]:
    manifest = project_root / "results" / "tile_manifest.csv"
    train_dataset = S2RGBSegmentationDataset(manifest, project_root=project_root, split="train")
    validation_dataset = S2RGBSegmentationDataset(manifest, project_root=project_root, split="validation")
    train_indices = list(range(min(train_subset, len(train_dataset))))
    validation_indices = list(range(min(validation_subset, len(validation_dataset))))
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(Subset(validation_dataset, validation_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, validation_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    mixed_precision: bool = True,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    scaler_enabled = mixed_precision and training and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    total_loss = 0.0
    total_valid = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                logits = model(images)["out"]
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        valid = int((targets != ignore_index).sum().item())
        total_loss += float(loss.detach().item()) * valid
        total_valid += valid
        predictions = logits.detach().argmax(dim=1)
        conf += confusion_matrix(predictions, targets, num_classes=num_classes, ignore_index=ignore_index).to(device)
    metrics = segmentation_metrics(conf)
    metrics["loss"] = total_loss / total_valid if total_valid else float("nan")
    return metrics


def run_smoke(project_root: Path) -> dict[str, object]:
    seed_everything(42)
    os.environ.setdefault("TORCH_HOME", str(project_root / ".torch"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 2C smoke tests.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    train_loader, validation_loader = make_loaders(project_root, batch_size=2, train_subset=8, validation_subset=4)
    images, targets = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert targets.shape[1:] == (224, 224)
    assert (targets == 255).any(), "Smoke batch should include ignored pixels."
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        logits = model(images.to(device))["out"]
    assert logits.shape == (images.shape[0], 4, 224, 224)
    history = []
    for epoch in range(1, 3):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device, optimizer=None)
        assert np.isfinite(train_metrics["loss"])
        assert np.isfinite(validation_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
        "peak_gpu_memory": torch.cuda.max_memory_allocated(device),
        "train_tiles": len(train_loader.dataset),
        "validation_tiles": len(validation_loader.dataset),
        "history": history,
    }

