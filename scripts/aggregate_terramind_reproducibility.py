from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "terramind_reproducibility"
FIGURE_DIR = PROJECT_ROOT / "figures"
ROBUST_AGGREGATE = PROJECT_ROOT / "results" / "occlusion_training" / "aggregate_results.csv"
SEEDS = [42, 7, 123]
METRICS = [
    "mean_iou",
    "macro_dice",
    "iou_background",
    "iou_low",
    "iou_mid",
    "iou_high",
    "binary_algae_iou",
    "binary_algae_dice",
]

SEED_42_KNOWN = {
    "seed": 42,
    "best_epoch": 7,
    "mean_iou": 0.283511,
    "macro_dice": 0.434891,
    "source": "user_reported_completed_colab_run",
}


def _load_seed_metrics(seed: int) -> dict[str, Any]:
    candidates = [
        RESULT_DIR / f"seed_{seed}" / "metrics.json",
        RESULT_DIR / f"seed_{seed}_metrics.json",
        PROJECT_ROOT / "results" / f"terramind_frozen_seed{seed}" / "metrics.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            validation = payload.get("validation_metrics", {})
            row = {
                "seed": seed,
                "best_epoch": payload.get("best_epoch"),
                "training_time_seconds": payload.get("training_time_seconds"),
                "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb"),
                "source": str(path.relative_to(PROJECT_ROOT)),
            }
            for metric in METRICS:
                row[metric] = validation.get(metric)
            return row
    if seed == 42:
        return {**SEED_42_KNOWN, **{metric: SEED_42_KNOWN.get(metric) for metric in METRICS if metric not in SEED_42_KNOWN}}
    return {"seed": seed, "source": "pending_colab_run", **{metric: None for metric in METRICS}}


def _aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        values = pd.to_numeric(per_seed[metric], errors="coerce").dropna()
        rows.append(
            {
                "experiment": "terramind_frozen",
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "n": int(len(values)),
                "required_n": 3,
            }
        )
    return pd.DataFrame(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value

def _robust_summary() -> list[dict[str, Any]]:
    if not ROBUST_AGGREGATE.exists():
        return []
    frame = pd.read_csv(ROBUST_AGGREGATE)
    selected = frame[
        frame["method"].isin(["modality_dropout", "occlusion_training"])
        & (frame["primary_condition"] == "clean")
    ]
    return selected.to_dict(orient="records")


def _bar_figure(per_seed: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "TerraMind frozen reproducibility", fill=(0, 0, 0))
    metrics = [("mean_iou", "macro mIoU"), ("macro_dice", "macro Dice")]
    x0 = 170
    y = 80
    for metric, label in metrics:
        draw.text((24, y + 8), label, fill=(0, 0, 0))
        for i, row in enumerate(per_seed.itertuples(index=False)):
            value = getattr(row, metric)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                text = f"seed {row.seed}: pending"
                width = 0
            else:
                width = int(float(value) * 520)
                text = f"seed {row.seed}: {float(value):.3f}"
            yy = y + i * 26
            draw.rectangle((x0, yy, x0 + width, yy + 16), fill=(80, 130, 210))
            draw.text((x0 + 530, yy - 2), text, fill=(0, 0, 0))
        y += 130
    canvas.save(FIGURE_DIR / "terramind_reproducibility.png")


def _comparison_figure(aggregate: pd.DataFrame, robust: list[dict[str, Any]]) -> None:
    canvas = Image.new("RGB", (960, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "TerraMind vs robust DeepLab validation summary", fill=(0, 0, 0))
    rows = []
    miou = aggregate[(aggregate["metric"] == "mean_iou") & (aggregate["n"] == 3)]
    if len(miou):
        rows.append(("TerraMind frozen", float(miou.iloc[0]["mean"]), float(miou.iloc[0]["std"])))
    for item in robust:
        label = "Modality dropout" if item["method"] == "modality_dropout" else "Dropout + occlusion training"
        rows.append((label, float(item["mean_iou_mean"]), float(item["mean_iou_std"])))
    if not rows:
        draw.text((24, 90), "Pending complete TerraMind per-seed metrics.", fill=(0, 0, 0))
    else:
        max_value = max(value for _, value, _ in rows)
        for idx, (label, value, std) in enumerate(rows):
            y = 85 + idx * 72
            draw.text((24, y + 10), label, fill=(0, 0, 0))
            width = int(560 * value / max(max_value, 1e-6))
            draw.rectangle((260, y, 260 + width, y + 32), fill=(92, 160, 95))
            draw.text((830, y + 8), f"{value:.3f} ± {std:.3f}", fill=(0, 0, 0))
    canvas.save(FIGURE_DIR / "terramind_vs_robust_deeplab.png")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame([_load_seed_metrics(seed) for seed in SEEDS])
    aggregate = _aggregate(per_seed)
    robust = _robust_summary()

    per_seed.to_csv(RESULT_DIR / "per_seed_results.csv", index=False)
    aggregate.to_csv(RESULT_DIR / "aggregate_results.csv", index=False)
    summary = {
        "phase": "6C-R",
        "status": "complete" if bool((aggregate["n"] == 3).all()) else "pending_colab_seed_metrics",
        "seeds": SEEDS,
        "fixed_settings": {
            "model": "frozen terramind_v1_tiny RGB+S1RTC",
            "decoder": "same lightweight 4-class decoder as Phase 6C",
            "batch_size": 4,
            "epochs": 10,
            "split": "fixed date-level split, validation only",
            "test_evaluated": False,
            "class_weights": "locked selected DeepLab inverse-square-root weights",
        },
        "per_seed": per_seed.to_dict(orient="records"),
        "aggregate": aggregate.to_dict(orient="records"),
        "robust_deeplab_clean_3seed_means": robust,
        "notes": [
            "Seed 42 is reused from the completed Colab run. Full per-class aggregation requires its metrics.json alongside seed 7 and 123 metrics.json files.",
            "No test dates are evaluated by this workflow.",
        ],
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _bar_figure(per_seed, aggregate)
    _comparison_figure(aggregate, robust)
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()

