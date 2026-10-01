from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import S1SARSegmentationDataset
from src.data.s2_dataset import S2RGBSegmentationDataset
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.models.deeplab import create_fusion_deeplab, create_s2_deeplab, create_sar_deeplab
from src.training.train_fusion_deeplab import denormalize_rgb
from src.training.train_s1_deeplab import color_mask, stretch_channel

RESULT_DIR = PROJECT_ROOT / "results" / "robustness"
FIGURE_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
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


def save_line_figure(summary: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    canvas = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, fill=(0, 0, 0))
    models = ["s2_weighted", "s1_weighted_control", "s1_s2_early_fusion"]
    colors = {"s2_weighted": (80, 130, 210), "s1_weighted_control": (92, 160, 95), "s1_s2_early_fusion": (210, 120, 50)}
    x0, y0, w, h = 90, 70, 720, 360
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(0, 0, 0))
    values = []
    for model in models:
        frame = summary[summary["model"] == model]
        values += frame[f"{metric}_mean"].dropna().tolist()
    lo, hi = 0.0, max(values) if values else 1.0
    hi = max(hi, 1e-6)
    for tick in range(6):
        frac = tick / 5
        y = y0 + h - int(frac * h)
        draw.line((x0 - 5, y, x0, y), fill=(0, 0, 0))
        draw.text((24, y - 8), f"{lo + frac * (hi - lo):.2f}", fill=(0, 0, 0))
    for tick, frac in enumerate(OCCLUSION_FRACTIONS):
        x = x0 + int(tick * w / (len(OCCLUSION_FRACTIONS) - 1))
        draw.line((x, y0 + h, x, y0 + h + 5), fill=(0, 0, 0))
        draw.text((x - 12, y0 + h + 12), f"{int(frac*100)}%", fill=(0, 0, 0))
    for model in models:
        frame = summary[(summary["model"] == model) & (summary["condition"] == "simulated_optical_occlusion")].sort_values("occlusion_fraction")
        points = []
        for row in frame.itertuples(index=False):
            x = x0 + int(OCCLUSION_FRACTIONS.index(float(row.occlusion_fraction)) * w / (len(OCCLUSION_FRACTIONS)-1))
            value = float(getattr(row, f"{metric}_mean"))
            y = y0 + h - int((value - lo) * h / (hi - lo))
            points.append((x, y))
            std = getattr(row, f"{metric}_std")
            if pd.notna(std):
                y_hi = y0 + h - int((value + float(std) - lo) * h / (hi - lo))
                y_lo = y0 + h - int((value - float(std) - lo) * h / (hi - lo))
                draw.line((x, y_hi, x, y_lo), fill=colors[model])
        if len(points) > 1:
            draw.line(points, fill=colors[model], width=3)
        for p in points:
            draw.ellipse((p[0]-4, p[1]-4, p[0]+4, p[1]+4), fill=colors[model])
    lx = 560
    for idx, model in enumerate(models):
        y = 22 + idx * 24
        draw.rectangle((lx, y, lx + 18, y + 14), fill=colors[model])
        draw.text((lx + 26, y - 2), model, fill=(0, 0, 0))
    canvas.save(path)


def save_per_class_figure(summary: pd.DataFrame) -> None:
    metrics = [("iou_background", "0 bg"), ("iou_low", "1 low"), ("iou_mid", "2 mid"), ("iou_high", "3 high")]
    canvas = Image.new("RGB", (980, 520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Per-class IoU under 70% simulated optical occlusion", fill=(0, 0, 0))
    frame = summary[(summary["condition"] == "simulated_optical_occlusion") & (summary["occlusion_fraction"] == 0.7)]
    models = ["s2_weighted", "s1_weighted_control", "s1_s2_early_fusion"]
    colors = [(80,130,210),(92,160,95),(210,120,50)]
    y = 70
    maxv = max(float(frame[f"{m}_mean"].max()) for m,_ in metrics)
    for metric, label in metrics:
        draw.text((24, y + 20), label, fill=(0,0,0))
        for idx, model in enumerate(models):
            val = float(frame[frame["model"] == model][f"{metric}_mean"].iloc[0])
            width = int(650 * val / max(maxv, 1e-6))
            yy = y + idx * 22
            draw.rectangle((160, yy, 160 + width, yy + 16), fill=colors[idx])
            draw.text((168 + width, yy - 1), f"{model} {val:.3f}", fill=(0,0,0))
        y += 95
    canvas.save(FIGURE_DIR / "robustness_per_class.png")


def save_missing_modality_figure(rows: pd.DataFrame) -> None:
    frame = rows[rows["date"] == "all_validation"].copy()
    canvas = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Fusion missing-modality validation", fill=(0,0,0))
    metrics = [("mean_iou", "macro mIoU"), ("binary_algae_dice", "binary algae Dice"), ("iou_mid", "IoU mid"), ("iou_high", "IoU high")]
    conditions = ["clean", "missing_s2_zero", "missing_s1_zero"]
    colors = [(210,120,50), (90,90,90), (160,80,160)]
    maxv = max(float(frame[m].max()) for m,_ in metrics)
    y = 70
    for metric, label in metrics:
        draw.text((24, y + 20), label, fill=(0,0,0))
        for idx, cond in enumerate(conditions):
            val = float(frame[frame["condition"] == cond][metric].iloc[0])
            width = int(600 * val / max(maxv, 1e-6))
            yy = y + idx * 22
            draw.rectangle((170, yy, 170 + width, yy + 16), fill=colors[idx])
            draw.text((178 + width, yy - 1), f"{cond} {val:.3f}", fill=(0,0,0))
        y += 82
    canvas.save(FIGURE_DIR / "missing_modality_comparison.png")


def save_qualitative(s2_model, fusion_model, s2_loader, fusion_loader, device):
    s2_ds = s2_loader.dataset
    fusion_ds = fusion_loader.dataset
    idx = 0
    s2_img, target = s2_ds[idx]
    fusion_img, _ = fusion_ds[idx]
    tile_id = str(s2_ds.manifest.iloc[idx].tile_id)
    seed = SEEDS[0]
    panels = []
    labels = []
    panels.append(denormalize_rgb(s2_img)); labels.append("Clean RGB")
    for frac in [0.1, 0.3, 0.5, 0.7]:
        x = s2_img.unsqueeze(0).to(device)
        occ = apply_optical_occlusion(x, frac, seed, [tile_id], rgb_channels=slice(0,3))[0].cpu()
        panels.append(denormalize_rgb(occ)); labels.append(f"{int(frac*100)}% occluded RGB")
    panels.append(color_mask(target.numpy())); labels.append("Ground Truth")
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        s2_pred = s2_model(s2_img.unsqueeze(0).to(device))["out"].argmax(dim=1)[0].cpu()
        fusion_occ = apply_optical_occlusion(fusion_img.unsqueeze(0).to(device), 0.7, seed, [tile_id], rgb_channels=slice(0,3))
        fusion_pred = fusion_model(fusion_occ)["out"].argmax(dim=1)[0].cpu()
    panels.append(color_mask(s2_pred.numpy())); labels.append("S2 prediction")
    panels.append(color_mask(fusion_pred.numpy())); labels.append("Fusion prediction")
    tile = 224
    label_h = 36
    canvas = Image.new("RGB", (tile * 4, (tile + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        row = i // 4
        col = i % 4
        x = col * tile
        y = row * (tile + label_h)
        draw.text((x + 6, y + 8), label, fill=(0,0,0))
        canvas.paste(Image.fromarray(panel), (x, y + label_h))
    canvas.save(FIGURE_DIR / "robustness_qualitative_examples.png")


def write_doc(summary: pd.DataFrame, missing: pd.DataFrame) -> None:
    def fmt(model: str, frac: float, metric: str) -> str:
        condition = "clean" if frac == 0.0 else "simulated_optical_occlusion"
        row = summary[(summary["model"] == model) & (summary["condition"] == condition) & (summary["occlusion_fraction"] == frac)].iloc[0]
        return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"

    missing_lines = [
        "| Condition | Macro mIoU | Macro Dice | Binary algae IoU | Binary algae Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in missing[missing["date"] == "all_validation"].itertuples(index=False):
        missing_lines.append(
            f"| {row.condition} | {row.mean_iou:.4f} | {row.macro_dice:.4f} | {row.binary_algae_iou:.4f} | {row.binary_algae_dice:.4f} |"
        )

    doc = f"""# Robustness Baseline

Phase 4A evaluates frozen validation models only. No model was retrained and no held-out test date was evaluated.

## Corruption Definition

The optical corruption is **simulated optical occlusion**, not measured cloud cover. For each validation tile, fixed seed and occlusion fraction, a low-resolution random field is bicubically upsampled and thresholded to produce a spatially coherent irregular mask. The same masks are applied to the S2-only and fusion models for fair comparison. Ground-truth masks and SAR channels are unchanged.

Occluded optical values are set to zero after optical normalization. Zero therefore means the neutral normalized value, not a raw reflectance value. Complete missing-modality fusion conditions also use zero tensors for the missing normalized modality.

## Main Validation Curve

| Model | 0% macro mIoU | 30% macro mIoU | 70% macro mIoU | 70% binary Dice |
|---|---:|---:|---:|---:|
| S2 weighted | {fmt('s2_weighted', 0.0, 'mean_iou')} | {fmt('s2_weighted', 0.3, 'mean_iou')} | {fmt('s2_weighted', 0.7, 'mean_iou')} | {fmt('s2_weighted', 0.7, 'binary_algae_dice')} |
| S1 weighted control | {fmt('s1_weighted_control', 0.0, 'mean_iou')} | {fmt('s1_weighted_control', 0.3, 'mean_iou')} | {fmt('s1_weighted_control', 0.7, 'mean_iou')} | {fmt('s1_weighted_control', 0.7, 'binary_algae_dice')} |
| S1+S2 fusion | {fmt('s1_s2_early_fusion', 0.0, 'mean_iou')} | {fmt('s1_s2_early_fusion', 0.3, 'mean_iou')} | {fmt('s1_s2_early_fusion', 0.7, 'mean_iou')} | {fmt('s1_s2_early_fusion', 0.7, 'binary_algae_dice')} |

## Missing Modality

{chr(10).join(missing_lines)}

## Interpretation

The benchmark measures controlled degradation under simulated missing optical information. It does not estimate real cloud performance because cloud physics, shadows, haze and cloud-mask errors are not simulated.
"""
    (DOCS_DIR / "ROBUSTNESS_BASELINE.md").write_text(doc, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
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
    save_line_figure(summary, "mean_iou", FIGURE_DIR / "robustness_miou_vs_occlusion.png", "Macro mIoU vs simulated optical occlusion")
    save_line_figure(summary, "binary_algae_dice", FIGURE_DIR / "robustness_binary_dice_vs_occlusion.png", "Binary algae Dice vs simulated optical occlusion")
    save_per_class_figure(summary)
    save_missing_modality_figure(missing)
    save_qualitative(s2_model, fusion_model, s2_loader, fusion_loader, device)
    payload = {
        "occlusion_fractions": OCCLUSION_FRACTIONS,
        "seeds": SEEDS,
        "validation_dates": dates,
        "simulated_optical_occlusion": "low-resolution random field, bicubic upsample, thresholded to requested fraction; applied after normalization as zeros",
        "missing_modality_convention": "missing normalized modality represented by zero tensors",
        "summary": summary.to_dict(orient="records"),
        "missing_modality": missing.to_dict(orient="records"),
    }
    (RESULT_DIR / "robustness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_doc(summary, missing)
    print(json.dumps({"summary_rows": len(summary), "occlusion_rows": len(occlusion), "missing_rows": len(missing), "validation_dates": dates}, indent=2))


if __name__ == "__main__":
    main()
