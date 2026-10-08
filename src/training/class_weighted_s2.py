from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.diagnostics import binary_metrics_from_multiclass
from src.models.deeplab import create_s2_deeplab
from src.training.train_s2_deeplab import make_loader, run_epoch, seed_everything


RESULT_DIR = PROJECT_ROOT / "results" / "s2_deeplab_weighted"
BASELINE_DIR = PROJECT_ROOT / "results" / "s2_deeplab"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def class_counts_for_indices(indices: list[int]) -> dict[str, int]:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    subset = train.iloc[indices]
    return {f"class_{klass}": int(subset[f"class_{klass}_count"].sum()) for klass in range(4)}


def representative_indices() -> list[int]:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    selected: list[int] = []
    for column in ("class_3_count", "class_2_count", "class_1_count", "class_0_count"):
        for idx in train.sort_values(column, ascending=False).index:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= 16:
                return selected
    return selected[:16]


def train_class_counts_and_weights() -> tuple[dict[str, int], dict[str, float]]:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    train = manifest[manifest["split"] == "train"]
    counts = np.array([train[f"class_{klass}_count"].sum() for klass in range(4)], dtype=np.float64)
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    return (
        {f"class_{klass}": int(counts[klass]) for klass in range(4)},
        {f"class_{klass}": float(weights[klass]) for klass in range(4)},
    )


def train_weighted(weights: dict[str, float]) -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    device = torch.device("cuda")
    train_loader = make_loader(PROJECT_ROOT, "train", batch_size=8, augment=True, shuffle=True)
    validation_loader = make_loader(PROJECT_ROOT, "validation", batch_size=8)
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    weight_tensor = torch.tensor([weights[f"class_{klass}"] for klass in range(4)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-3,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "s2_deeplab_weighted_best.pt"
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
        print(f"weighted epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} val_miou={validation_metrics['mean_iou']:.4f}")
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
                    "class_weights": weights,
                },
                checkpoint_path,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    unweighted_criterion = nn.CrossEntropyLoss(ignore_index=255)
    validation_metrics, validation_conf = run_epoch(model, validation_loader, unweighted_criterion, device)
    pd.DataFrame(history).to_csv(RESULT_DIR / "history.csv", index=False)
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "training_time_seconds": training_time,
                "best_epoch": best_epoch,
                "best_validation_miou": best_miou,
                "class_weights": weights,
                "validation_metrics_unweighted": validation_metrics,
                "validation_confusion_matrix": validation_conf.numpy().tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "metrics": validation_metrics,
        "confusion": validation_conf,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
    }


def load_baseline_validation() -> dict[str, object]:
    device = torch.device("cuda")
    validation_loader = make_loader(PROJECT_ROOT, "validation", batch_size=8)
    model = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    checkpoint = torch.load(CHECKPOINT_DIR / "s2_deeplab_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    metrics, conf = run_epoch(model, validation_loader, criterion, device)
    return {"model": model, "metrics": metrics, "confusion": conf}


def binary_metrics(model: torch.nn.Module) -> dict[str, float]:
    return binary_metrics_from_multiclass(model, torch.device("cuda"))


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    original_counts = class_counts_for_indices(list(range(16)))
    rep_indices = representative_indices()
    representative_counts = class_counts_for_indices(rep_indices)
    train_counts, weights = train_class_counts_and_weights()
    weighted = train_weighted(weights)
    baseline = load_baseline_validation()
    baseline_binary = binary_metrics(baseline["model"])
    weighted_binary = binary_metrics(weighted["model"])
    comparison = pd.DataFrame(
        [
            {"experiment": "baseline", **baseline["metrics"], **baseline_binary},
            {"experiment": "weighted", **weighted["metrics"], **weighted_binary},
        ]
    )
    comparison.to_csv(RESULT_DIR / "validation_comparison.csv", index=False)
    payload = {
        "train_class_counts": train_counts,
        "class_weights": weights,
        "tiny_original_counts": original_counts,
        "tiny_representative_indices": rep_indices,
        "tiny_representative_counts": representative_counts,
        "weighted_best_epoch": weighted["best_epoch"],
        "weighted_training_time_seconds": weighted["training_time_seconds"],
        "comparison": comparison.to_dict(orient="records"),
    }
    (RESULT_DIR / "class_imbalance_experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
