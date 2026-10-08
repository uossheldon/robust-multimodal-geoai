from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import S1SARSegmentationDataset
from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_fusion_deeplab, create_s2_deeplab, create_sar_deeplab


RESULT_DIR = PROJECT_ROOT / "results" / "robustness"
MANIFEST_PATH = PROJECT_ROOT / "results" / "tile_manifest.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OCCLUSION_FRACTIONS = [0.0, 0.1, 0.3, 0.5, 0.7]
SEEDS = [101, 202, 303]
METRIC_KEYS = ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid", "iou_high", "binary_algae_iou", "binary_algae_dice"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_metrics(conf4: torch.Tensor) -> dict[str, float | list[list[int]]]:
    conf = torch.zeros((2, 2), dtype=torch.int64)
    conf[0, 0] = conf4[0, 0]
    conf[0, 1] = conf4[0, 1:].sum()
    conf[1, 0] = conf4[1:, 0].sum()
    conf[1, 1] = conf4[1:, 1:].sum()
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
        "binary_confusion_matrix": conf.numpy().tolist(),
    }


def metrics_from_conf(conf: torch.Tensor) -> dict[str, float | list[list[int]]]:
    out = segmentation_metrics(conf)
    out.update(binary_metrics(conf.cpu()))
    return out


def stable_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def irregular_occlusion_mask(batch: int, height: int, width: int, fraction: float, seed: int, tile_ids: list[str], device: torch.device) -> torch.Tensor:
    if fraction <= 0:
        return torch.zeros((batch, height, width), dtype=torch.bool, device=device)
    masks = []
    low_h = max(4, height // 28)
    low_w = max(4, width // 28)
    for tile_id in tile_ids:
        generator = torch.Generator(device="cpu").manual_seed(stable_int(seed, tile_id, fraction))
        field = torch.rand((1, 1, low_h, low_w), generator=generator)
        field = torch.nn.functional.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)[0, 0]
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        threshold = torch.quantile(field.flatten(), 1.0 - fraction)
        mask = field >= threshold
        masks.append(mask)
    return torch.stack(masks, dim=0).to(device)


def apply_optical_occlusion(inputs: torch.Tensor, fraction: float, seed: int, tile_ids: list[str], *, rgb_channels: slice) -> torch.Tensor:
    if fraction <= 0:
        return inputs
    out = inputs.clone()
    mask = irregular_occlusion_mask(inputs.shape[0], inputs.shape[-2], inputs.shape[-1], fraction, seed, tile_ids, inputs.device)
    out[:, rgb_channels][mask.unsqueeze(1).expand(-1, rgb_channels.stop - rgb_channels.start, -1, -1)] = 0.0
    return out


def dataset_dates(dataset) -> list[str]:
    return dataset.manifest["date"].astype(str).tolist()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, *, model_name: str, condition: str, occlusion_fraction: float | None = None, seed: int | None = None, date_filter: str | None = None, input_mode: str = "clean") -> dict[str, object]:
    model.eval()
    conf = torch.zeros((4, 4), dtype=torch.int64, device=device)
    total_valid = 0
    offset = 0
    manifest = loader.dataset.manifest
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for images, targets in loader:
            batch_rows = manifest.iloc[offset:offset + len(images)]
            offset += len(images)
            if date_filter is not None:
                keep = torch.tensor((batch_rows["date"].astype(str).values == date_filter), dtype=torch.bool)
                if not keep.any():
                    continue
                images = images[keep]
                targets = targets[keep]
                batch_rows = batch_rows[keep.numpy()]
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tile_ids = batch_rows["tile_id"].astype(str).tolist()
            if input_mode == "s2_occlusion":
                images = apply_optical_occlusion(images, float(occlusion_fraction), int(seed), tile_ids, rgb_channels=slice(0, 3))
            elif input_mode == "fusion_occlusion":
                images = apply_optical_occlusion(images, float(occlusion_fraction), int(seed), tile_ids, rgb_channels=slice(0, 3))
            elif input_mode == "fusion_missing_s2":
                images = images.clone(); images[:, 0:3] = 0.0
            elif input_mode == "fusion_missing_s1":
                images = images.clone(); images[:, 3:5] = 0.0
            logits = model(images)["out"]
            preds = logits.argmax(dim=1)
            conf += confusion_matrix(preds, targets, num_classes=4, ignore_index=255).to(device)
            total_valid += int((targets != 255).sum().item())
    metrics = metrics_from_conf(conf.detach().cpu())
    return {
        "model": model_name,
        "condition": condition,
        "occlusion_fraction": occlusion_fraction,
        "seed": seed,
        "date": date_filter or "all_validation",
        "valid_pixels": total_valid,
        **{k: metrics[k] for k in METRIC_KEYS},
    }


def relative_degradation(value: float, clean: float) -> float:
    return float((clean - value) / clean) if clean else float("nan")


def add_degradation(rows: pd.DataFrame) -> pd.DataFrame:
    clean_lookup = {}
    for model in rows["model"].unique():
        clean = rows[(rows["model"] == model) & (rows["condition"] == "clean") & (rows["date"] == "all_validation")]
        if not clean.empty:
            clean_lookup[model] = clean.iloc[0].to_dict()
    for key in METRIC_KEYS:
        rows[f"relative_degradation_{key}"] = [relative_degradation(float(row[key]), float(clean_lookup.get(row["model"], {}).get(key, np.nan))) for _, row in rows.iterrows()]
    return rows


def summarize_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows[rows["date"] == "all_validation"].groupby(["model", "condition", "occlusion_fraction"], dropna=False)
    out = grouped[METRIC_KEYS + [f"relative_degradation_{k}" for k in METRIC_KEYS]].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join(str(part) for part in col if part != "") for col in out.columns]
    return out


def load_models(device: torch.device):
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    s2 = create_s2_deeplab(num_classes=4, pretrained=True).to(device)
    s2.load_state_dict(torch.load(CHECKPOINT_DIR / "s2_deeplab_weighted_best.pt", map_location=device, weights_only=True)["model_state_dict"])
    s1 = create_sar_deeplab(num_classes=4, pretrained=True).to(device)
    s1.load_state_dict(torch.load(CHECKPOINT_DIR / "s1_deeplab_weighted_best.pt", map_location=device, weights_only=True)["model_state_dict"])
    fusion = create_fusion_deeplab(num_classes=4, pretrained=True).to(device)
    fusion.load_state_dict(torch.load(CHECKPOINT_DIR / "s1_s2_early_fusion_best.pt", map_location=device, weights_only=True)["model_state_dict"])
    return s2, s1, fusion


def make_loaders():
    s1_metrics = load_json(PROJECT_ROOT / "results" / "s1_deeplab_weighted" / "metrics.json")
    stats = s1_metrics["sar_statistics"]
    s2_ds = S2RGBSegmentationDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split="validation", augment=False)
    s1_ds = S1SARSegmentationDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split="validation", sar_mean=stats["mean"], sar_std=stats["std"], augment=False)
    fusion_ds = S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split="validation", sar_mean=stats["mean"], sar_std=stats["std"], augment=False)
    return (
        DataLoader(s2_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(s1_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(fusion_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
    )


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    s2_model, s1_model, fusion_model = load_models(device)
    s2_loader, s1_loader, fusion_loader = make_loaders()
    dates = sorted(s2_loader.dataset.manifest["date"].astype(str).unique().tolist())
    occlusion_rows = []
    for frac in OCCLUSION_FRACTIONS:
        for seed in SEEDS:
            for date in [None, *dates]:
                occlusion_rows.append(evaluate_model(s2_model, s2_loader, device, model_name="s2_weighted", condition="simulated_optical_occlusion" if frac else "clean", occlusion_fraction=frac, seed=seed, date_filter=date, input_mode="s2_occlusion"))
                occlusion_rows.append(evaluate_model(s1_model, s1_loader, device, model_name="s1_weighted_control", condition="simulated_optical_occlusion" if frac else "clean", occlusion_fraction=frac, seed=seed, date_filter=date, input_mode="clean"))
                occlusion_rows.append(evaluate_model(fusion_model, fusion_loader, device, model_name="s1_s2_early_fusion", condition="simulated_optical_occlusion" if frac else "clean", occlusion_fraction=frac, seed=seed, date_filter=date, input_mode="fusion_occlusion"))
    occlusion = add_degradation(pd.DataFrame(occlusion_rows))
    occlusion.to_csv(RESULT_DIR / "optical_occlusion.csv", index=False)
    missing_rows = []
    for condition, mode in [("clean", "clean"), ("missing_s2_zero", "fusion_missing_s2"), ("missing_s1_zero", "fusion_missing_s1")]:
        for date in [None, *dates]:
            missing_rows.append(evaluate_model(fusion_model, fusion_loader, device, model_name="s1_s2_early_fusion", condition=condition, occlusion_fraction=np.nan, seed=np.nan, date_filter=date, input_mode=mode))
    missing = add_degradation(pd.DataFrame(missing_rows))
    missing.to_csv(RESULT_DIR / "missing_modality.csv", index=False)
    summary = summarize_seeds(occlusion)
    summary.to_csv(RESULT_DIR / "optical_occlusion_summary.csv", index=False)
    print(json.dumps({"summary_rows": len(summary), "occlusion_rows": len(occlusion), "missing_rows": len(missing), "validation_dates": dates}, indent=2))


if __name__ == "__main__":
    main()
