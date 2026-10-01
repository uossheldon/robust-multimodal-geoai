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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.terramind_dataset import TerraMindMultimodalDataset, terramind_collate
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.terramind_segmentation import create_terramind_frozen_segmenter
from src.training.train_s1_deeplab import color_mask, stretch_channel
from src.training.train_s2_deeplab import class_metrics_frame

MANIFEST_PATH = PROJECT_ROOT / "results" / "tile_manifest.csv"
RESULT_DIR = PROJECT_ROOT / "results" / "terramind_frozen"
FIGURE_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CLASS_COLORS = {0: (40, 40, 40), 1: (121, 190, 85), 2: (247, 188, 65), 3: (218, 83, 63), 255: (210, 210, 210)}
EXPECTED_SPLIT_COUNTS = {"train": 621, "validation": 111}
TEST_DATES = {"2025-09-08", "2025-09-21"}
SELECTED_DEEPLAB_CLASS_WEIGHTS = {
    "class_0": 0.243332998497,
    "class_1": 0.709198873576,
    "class_2": 0.651737877057,
    "class_3": 2.395730250869,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_class_weights() -> tuple[dict[str, int], dict[str, float]]:
    """Return train counts plus the locked DeepLab selected class weights.

    The Phase 6C protocol requires using the same inverse-square-root weights
    selected for the weighted DeepLab experiments, not recomputing a new scheme.
    Counts are recorded only for audit metadata.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    train = manifest[manifest["split"] == "train"]
    counts = {f"class_{klass}": int(train[f"class_{klass}_count"].sum()) for klass in range(4)}
    return counts, SELECTED_DEEPLAB_CLASS_WEIGHTS.copy()


def validate_manifest_protocol() -> dict[str, object]:
    manifest = pd.read_csv(MANIFEST_PATH)
    split_counts = manifest["split"].value_counts().to_dict()
    issues = []
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        observed = int(split_counts.get(split, 0))
        if observed != expected:
            issues.append(f"{split} tile count expected {expected}, observed {observed}")
    loaded = manifest[manifest["split"].isin(["train", "validation"])]
    leaked_test_dates = sorted(set(loaded["date"].astype(str)) & TEST_DATES)
    if leaked_test_dates:
        issues.append(f"test dates present in train/validation loader scope: {leaked_test_dates}")
    tile_sizes = set(zip(loaded["window_height"].astype(int), loaded["window_width"].astype(int)))
    if tile_sizes != {(224, 224)}:
        issues.append(f"non-224 tile sizes observed: {sorted(tile_sizes)}")
    observed_mask_values = set()
    for value in loaded["mask_values_observed"].dropna().astype(str):
        observed_mask_values.update(int(part) for part in value.split())
    if not observed_mask_values.issubset({0, 1, 2, 3, 255}):
        issues.append(f"unexpected mask values in manifest: {sorted(observed_mask_values)}")
    if issues:
        raise RuntimeError("Manifest protocol validation failed: " + "; ".join(issues))
    return {
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "train_tiles": int(split_counts.get("train", 0)),
        "validation_tiles": int(split_counts.get("validation", 0)),
        "test_dates_excluded_from_train_validation": True,
        "tile_size": 224,
        "mask_values_observed_train_validation": sorted(observed_mask_values),
    }

def make_loader(split: str, *, batch_size: int, augment: bool = False, shuffle: bool = False) -> DataLoader:
    dataset = TerraMindMultimodalDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split=split, augment=augment)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=True,
        collate_fn=terramind_collate,
    )


def to_device(inputs: dict[str, torch.Tensor], targets: torch.Tensor, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}, targets.to(device, non_blocking=True)


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
    model.backbone.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=training and mixed_precision and device.type == "cuda")
    total_loss = 0.0
    total_valid = 0
    conf = torch.zeros((4, 4), dtype=torch.int64, device=device)
    for inputs, targets in loader:
        inputs, targets = to_device(inputs, targets, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                logits = model(inputs)["out"]
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


def smoke_test(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    inputs, targets = next(iter(loader))
    inputs, targets = to_device(inputs, targets, device)
    model.train()
    model.backbone.eval()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_backbone = all(not parameter.requires_grad for parameter in model.backbone.parameters())
    only_decoder_trainable = bool(trainable_names) and all(name.startswith("decoder.") for name in trainable_names)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(inputs)["out"]
        loss = criterion(logits, targets)
    loss.backward()
    backbone_grad_tensors = [parameter.grad for parameter in model.backbone.parameters() if parameter.grad is not None]
    decoder_grad_tensors = [parameter.grad for parameter in model.decoder.parameters() if parameter.grad is not None]
    backbone_has_grad = any(torch.isfinite(grad).any().item() for grad in backbone_grad_tensors)
    decoder_has_grad = any(torch.isfinite(grad).any().item() for grad in decoder_grad_tensors)
    model.zero_grad(set_to_none=True)
    return {
        "rgb_shape": list(inputs["RGB"].shape),
        "s1rtc_shape": list(inputs["S1RTC"].shape),
        "target_shape": list(targets.shape),
        "logit_shape": list(logits.shape),
        "expected_logit_shape": [inputs["RGB"].shape[0], 4, 224, 224],
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "cuda_used": bool(logits.is_cuda),
        "backbone_frozen": bool(frozen_backbone),
        "only_decoder_trainable": bool(only_decoder_trainable),
        "trainable_parameter_prefixes": sorted({name.split(".")[0] for name in trainable_names}),
        "backbone_received_gradients": bool(backbone_has_grad),
        "decoder_received_gradients": bool(decoder_has_grad),
    }

def save_curves(history: pd.DataFrame) -> None:
    canvas = Image.new("RGB", (900, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "Frozen TerraMind training curves", fill=(0, 0, 0))
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
            hi = lo + 1
        def pts(column: str) -> list[tuple[int, int]]:
            series = history[column].tolist()
            return [(x0 + int(idx * 330 / max(1, len(series) - 1)), y0 + 220 - int((value - lo) * 220 / (hi - lo))) for idx, value in enumerate(series)]
        if train_col:
            draw.line(pts(train_col), fill=(70, 120, 210), width=3)
            draw.text((x0, y0 + 232), "train", fill=(70, 120, 210))
        draw.line(pts(val_col), fill=(210, 80, 80), width=3)
        draw.text((x0 + 90, y0 + 232), "validation", fill=(210, 80, 80))
    canvas.save(FIGURE_DIR / "terramind_training_curves.png")


def save_per_class_iou(metrics: dict[str, float]) -> None:
    frame = class_metrics_frame(metrics)
    canvas = Image.new("RGB", (760, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "TerraMind validation per-class IoU", fill=(0, 0, 0))
    max_iou = max(float(frame["iou"].max()), 1e-6)
    for idx, row in enumerate(frame.itertuples(index=False)):
        y = 70 + idx * 64
        width = int(500 * float(row.iou) / max_iou)
        draw.text((24, y + 12), str(row.class_name), fill=(0, 0, 0))
        draw.rectangle((170, y, 170 + width, y + 42), fill=CLASS_COLORS[int(row.class_id)])
        draw.text((182 + width, y + 12), f"{row.iou:.3f}", fill=(0, 0, 0))
    canvas.save(FIGURE_DIR / "terramind_per_class_iou.png")


def rgb_for_display(rgb_bgr_255: torch.Tensor) -> np.ndarray:
    rgb = (rgb_bgr_255[[2, 1, 0]].cpu().float() / 255.0).clamp(0, 1).numpy()
    return (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)


def sar_for_display(s1rtc: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mean = torch.tensor([-10.930, -17.329]).view(2, 1, 1)
    std = torch.tensor([4.391, 4.459]).view(2, 1, 1)
    raw = s1rtc.cpu().float() * std + mean
    vv = np.repeat(stretch_channel(raw[0])[:, :, None], 3, axis=2)
    vh = np.repeat(stretch_channel(raw[1])[:, :, None], 3, axis=2)
    return vv, vh


def save_qualitative(model: nn.Module, loader: DataLoader, device: torch.device, max_examples: int = 4) -> None:
    model.eval()
    examples = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for inputs, targets in loader:
            device_inputs, _ = to_device(inputs, targets, device)
            predictions = model(device_inputs)["out"].argmax(dim=1).cpu()
            for idx, (target, prediction) in enumerate(zip(targets, predictions)):
                valid = target != 255
                error = torch.full_like(target, 255)
                error[valid & (prediction == target)] = 0
                error[valid & (prediction != target)] = 3
                vv, vh = sar_for_display(inputs["S1RTC"][idx])
                examples.append((rgb_for_display(inputs["RGB"][idx]), vv, vh, color_mask(target.numpy()), color_mask(prediction.numpy()), color_mask(error.numpy())))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break
    tile = 224
    label_h = 28
    canvas = Image.new("RGB", (tile * 6, (tile + label_h) * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["RGB", "VV", "VH", "Ground Truth", "TerraMind Prediction", "Error Map"]
    for row, example in enumerate(examples):
        y = row * (tile + label_h)
        for col, array in enumerate(example):
            x = col * tile
            draw.text((x + 8, y + 7), labels[col], fill=(0, 0, 0))
            canvas.paste(Image.fromarray(array), (x, y + label_h))
    canvas.save(FIGURE_DIR / "terramind_qualitative_predictions.png")


def load_existing_comparisons() -> pd.DataFrame:
    rows = []
    candidates = [
        ("s2_weighted", PROJECT_ROOT / "results" / "s2_deeplab_balanced_sampling" / "validation_comparison.csv", "weighted_loss"),
        ("s1_weighted", PROJECT_ROOT / "results" / "s1_deeplab_weighted" / "validation_vs_s2.csv", "s1_weighted"),
        ("s1_s2_early_fusion", PROJECT_ROOT / "results" / "s1_s2_early_fusion" / "validation_modality_comparison.csv", "s1_s2_early_fusion"),
        ("modality_dropout_3seed_mean", PROJECT_ROOT / "results" / "occlusion_training" / "aggregate_results.csv", "modality_dropout|clean"),
        ("occlusion_trained_3seed_mean", PROJECT_ROOT / "results" / "occlusion_training" / "aggregate_results.csv", "occlusion_training|clean"),
    ]
    keys = ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid", "iou_high", "binary_algae_iou", "binary_algae_dice"]
    for name, path, selector in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        row = None
        if "|" in selector and {"method", "primary_condition"}.issubset(frame.columns):
            method, condition = selector.split("|", 1)
            selected = frame[(frame["method"].astype(str) == method) & (frame["primary_condition"].astype(str) == condition)]
            if len(selected):
                row = selected.iloc[0]
        elif "experiment" in frame.columns and selector in set(frame["experiment"].astype(str)):
            row = frame[frame["experiment"].astype(str) == selector].iloc[0]
        elif "condition" in frame.columns and selector in set(frame["condition"].astype(str)):
            row = frame[frame["condition"].astype(str) == selector].iloc[0]
        elif "primary_condition" in frame.columns and selector in set(frame["primary_condition"].astype(str)):
            row = frame[frame["primary_condition"].astype(str) == selector].iloc[0]
        elif "method" in frame.columns and selector in set(frame["method"].astype(str)):
            row = frame[frame["method"].astype(str) == selector].iloc[0]
        elif len(frame):
            row = frame.iloc[0]
        if row is None:
            continue
        metrics = {}
        for key in keys:
            if key in row.index:
                metrics[key] = float(row[key])
            elif f"{key}_mean" in row.index:
                metrics[key] = float(row[f"{key}_mean"])
        rows.append({"experiment": name, "comparison_type": "3seed_mean" if name.endswith("3seed_mean") else "single_run", **metrics})
    return pd.DataFrame(rows)


def save_comparison_figure(comparison: pd.DataFrame) -> None:
    if comparison.empty or "mean_iou" not in comparison.columns:
        return
    canvas = Image.new("RGB", (1160, 560), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "TerraMind vs DeepLab validation comparison", fill=(0, 0, 0))
    metrics = [("mean_iou", "macro mIoU"), ("macro_dice", "macro Dice"), ("iou_background", "IoU 0"), ("iou_low", "IoU 1"), ("iou_mid", "IoU 2"), ("iou_high", "IoU 3")]
    maxv = max(float(comparison[col].max()) for col, _ in metrics if col in comparison.columns)
    y = 60
    palette = [(80, 130, 210), (92, 160, 95), (210, 120, 50), (155, 95, 180), (210, 80, 80), (80, 80, 80)]
    for col, label in metrics:
        if col not in comparison.columns:
            continue
        draw.text((24, y + 18), label, fill=(0, 0, 0))
        for idx, row in enumerate(comparison.itertuples(index=False)):
            value = float(getattr(row, col))
            width = int(650 * value / max(maxv, 1e-6))
            yy = y + idx * 18
            draw.rectangle((165, yy, 165 + width, yy + 12), fill=palette[idx % len(palette)])
            draw.text((175 + width, yy - 3), f"{row.experiment} {value:.3f}", fill=(0, 0, 0))
        y += 105
    canvas.save(FIGURE_DIR / "terramind_vs_deeplab_validation.png")


def evaluate_for_binary_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    conf = torch.zeros((2, 2), dtype=torch.int64, device=device)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for inputs, targets in loader:
            inputs, targets = to_device(inputs, targets, device)
            predictions = model(inputs)["out"].argmax(dim=1)
            valid = targets != 255
            target_binary = (targets > 0).long()
            pred_binary = (predictions > 0).long()
            encoded = 2 * target_binary[valid] + pred_binary[valid]
            conf += torch.bincount(encoded, minlength=4).reshape(2, 2)
    matrix = conf.float()
    tp = matrix.diag()
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - tp
    iou = torch.where(union > 0, tp / union, torch.nan)
    dice_den = matrix.sum(dim=1) + matrix.sum(dim=0)
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.nan)
    return {"binary_algae_iou": float(iou[1].item()), "binary_algae_dice": float(dice[1].item())}


def train_full(batch_size: int = 4, epochs: int = 10, seed: int = 42, use_class_weights: bool = True) -> dict[str, object]:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(seed)
    for directory in (RESULT_DIR, FIGURE_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TerraMind benchmark run.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    protocol = validate_manifest_protocol()
    train_counts, weights = train_class_weights()
    class_weight_tensor = torch.tensor([weights[f"class_{klass}"] for klass in range(4)], dtype=torch.float32, device=device) if use_class_weights else None
    train_loader = make_loader("train", batch_size=batch_size, augment=True, shuffle=True)
    validation_loader = make_loader("validation", batch_size=batch_size, augment=False, shuffle=False)

    model = create_terramind_frozen_segmenter(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    smoke = smoke_test(model, train_loader, criterion, device)
    expected_shape = [batch_size, 4, 224, 224]
    if smoke["rgb_shape"][0] != batch_size:
        expected_shape[0] = smoke["rgb_shape"][0]
    if (
        not smoke["loss_finite"]
        or not smoke["cuda_used"]
        or smoke["logit_shape"] != expected_shape
        or not smoke["backbone_frozen"]
        or not smoke["only_decoder_trainable"]
        or smoke["backbone_received_gradients"]
        or not smoke["decoder_received_gradients"]
    ):
        raise RuntimeError(f"TerraMind smoke test failed: {smoke}")

    checkpoint_path = CHECKPOINT_DIR / "terramind_frozen_best.pt"
    history = []
    best_miou = -float("inf")
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_metrics, validation_conf = run_epoch(model, validation_loader, criterion, device)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(f"terramind epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} val_miou={validation_metrics['mean_iou']:.4f}")
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
                    "class_weights": weights if use_class_weights else None,
                    "config": {"batch_size": batch_size, "epochs": epochs, "seed": seed, "use_class_weights": use_class_weights},
                },
                checkpoint_path,
            )
    torch.cuda.synchronize()
    training_time = time.perf_counter() - start

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, validation_conf = run_epoch(model, validation_loader, criterion, device)
    binary_metrics = evaluate_for_binary_metrics(model, validation_loader, device)
    validation_metrics.update(binary_metrics)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(RESULT_DIR / "history.csv", index=False)
    class_metrics_frame(validation_metrics).to_csv(RESULT_DIR / "per_class_metrics.csv", index=False)

    existing = load_existing_comparisons()
    comparison_row = {"experiment": "terramind_frozen", **{key: validation_metrics[key] for key in ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid", "iou_high", "binary_algae_iou", "binary_algae_dice"]}}
    comparison = pd.concat([existing, pd.DataFrame([comparison_row])], ignore_index=True)
    comparison.to_csv(RESULT_DIR / "validation_comparison.csv", index=False)

    save_curves(history_frame)
    save_per_class_iou(validation_metrics)
    save_qualitative(model, validation_loader, device)
    save_comparison_figure(comparison)

    payload = {
        "training_time_seconds": training_time,
        "best_epoch": best_epoch,
        "best_validation_miou_during_training": best_miou,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "use_class_weights": use_class_weights,
        "manifest_protocol": protocol,
        "train_class_counts": train_counts,
        "class_weights": weights if use_class_weights else None,
        "smoke_test": smoke,
        "validation_metrics": validation_metrics,
        "validation_confusion_matrix": validation_conf.numpy().tolist(),
        "validation_comparison": comparison.to_dict(orient="records"),
        "test_evaluated": False,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULT_DIR / "train_summary.json").write_text(json.dumps({
        "model": "terramind_v1_tiny frozen RGB+S1RTC backbone",
        "decoder": "Conv/GN/GELU upsampling head, 192x14x14 to 4x224x224",
        "optimizer": "AdamW decoder only, lr=1e-3, weight_decay=1e-3",
        "loss": "CrossEntropyLoss(ignore_index=255) with locked selected DeepLab inverse-square-root class weights" if use_class_weights else "CrossEntropyLoss(ignore_index=255)",
        "model_selection": "validation macro mIoU",
        "test_evaluated": False,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    train_full()


