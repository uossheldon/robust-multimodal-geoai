from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "SummerSchool_Subset"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

TILE_SIZE = 224
MIN_VALID_RATIO = 0.20
DATES = (
    "2025-01-01",
    "2025-01-31",
    "2025-03-12",
    "2025-04-08",
    "2025-04-09",
    "2025-05-16",
    "2025-05-18",
    "2025-05-21",
    "2025-06-20",
    "2025-08-12",
    "2025-09-08",
    "2025-09-21",
)
MASK_IGNORE = 255
MASK_CLASSES = (0, 1, 2, 3, 255)


def layer_path(date: str, layer: str) -> Path:
    return DATASET_ROOT / "images" / date / "layers" / f"{date}_{layer}.tif"


def mask_path(date: str) -> Path:
    return DATASET_ROOT / "masks" / f"{date.replace('-', '_')}.tiff"


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def same_grid(src: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> bool:
    return (
        src.crs == reference.crs
        and src.width == reference.width
        and src.height == reference.height
        and src.transform.almost_equals(reference.transform)
    )


def read_on_reference(
    path: Path,
    reference: rasterio.io.DatasetReader,
    window: Window,
    *,
    resampling: Resampling,
    nodata: float | int,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if same_grid(src, reference):
            return src.read(1, window=window, masked=False)
        with WarpedVRT(
            src,
            crs=reference.crs,
            transform=reference.transform,
            width=reference.width,
            height=reference.height,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=nodata,
        ) as vrt:
            return vrt.read(1, window=window, masked=False)


def finite_valid(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        valid &= values != nodata
    return valid


def tile_record(date: str, row: int, col: int) -> dict[str, object] | None:
    window = Window(col_off=col, row_off=row, width=TILE_SIZE, height=TILE_SIZE)
    with rasterio.open(layer_path(date, "B04")) as reference:
        red = read_on_reference(layer_path(date, "B04"), reference, window, resampling=Resampling.bilinear, nodata=0)
        green = read_on_reference(layer_path(date, "B03"), reference, window, resampling=Resampling.bilinear, nodata=0)
        blue = read_on_reference(layer_path(date, "B02"), reference, window, resampling=Resampling.bilinear, nodata=0)
        vv = read_on_reference(layer_path(date, "VV"), reference, window, resampling=Resampling.bilinear, nodata=np.nan).astype(np.float32)
        vh = read_on_reference(layer_path(date, "VH"), reference, window, resampling=Resampling.bilinear, nodata=np.nan).astype(np.float32)
        mask = read_on_reference(mask_path(date), reference, window, resampling=Resampling.nearest, nodata=MASK_IGNORE).astype(np.uint8)

        labels = sorted(int(value) for value in np.unique(mask))
        unexpected = sorted(set(labels) - set(MASK_CLASSES))
        if unexpected:
            raise ValueError(f"Unexpected mask values for {date} row {row} col {col}: {unexpected}")

        s2_valid = finite_valid(red, 0) & finite_valid(green, 0) & finite_valid(blue, 0)
        s1_valid = finite_valid(vv, np.nan) & finite_valid(vh, np.nan)
        mask_valid = mask != MASK_IGNORE
        common_valid = s2_valid & s1_valid & mask_valid
        valid_count = int(common_valid.sum())
        valid_ratio = valid_count / (TILE_SIZE * TILE_SIZE)
        if valid_ratio < MIN_VALID_RATIO:
            return None

        counts = {value: int(((mask == value) & common_valid).sum()) for value in (0, 1, 2, 3)}
        ignore_count = TILE_SIZE * TILE_SIZE - valid_count
        left, bottom, right, top = window_bounds(window, reference.transform)
        record: dict[str, object] = {
            "date": date,
            "tile_id": f"{date}_r{row:04d}_c{col:04d}",
            "tile_row": row // TILE_SIZE,
            "tile_col": col // TILE_SIZE,
            "row_offset": row,
            "col_offset": col,
            "window_height": TILE_SIZE,
            "window_width": TILE_SIZE,
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            "valid_pixel_count": valid_count,
            "valid_pixel_ratio": valid_ratio,
            "ignore_pixel_count": ignore_count,
            "s1_valid": bool(s1_valid.any()),
            "s2_valid": bool(s2_valid.any()),
            "common_valid": bool(valid_count > 0),
            "source_B04": rel(layer_path(date, "B04")),
            "source_B03": rel(layer_path(date, "B03")),
            "source_B02": rel(layer_path(date, "B02")),
            "source_VV": rel(layer_path(date, "VV")),
            "source_VH": rel(layer_path(date, "VH")),
            "source_mask": rel(mask_path(date)),
            "mask_values_observed": " ".join(str(value) for value in labels),
        }
        for value, name in [(0, "class_0"), (1, "class_1"), (2, "class_2"), (3, "class_3")]:
            count = counts[value]
            record[f"{name}_count"] = count
            record[f"{name}_fraction"] = count / valid_count if valid_count else 0.0
        record["algae_count"] = counts[1] + counts[2] + counts[3]
        record["algae_fraction"] = record["algae_count"] / valid_count if valid_count else 0.0
        return record


def build_full_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in DATES:
        with rasterio.open(layer_path(date, "B04")) as reference:
            for row in range(0, reference.height - TILE_SIZE + 1, TILE_SIZE):
                for col in range(0, reference.width - TILE_SIZE + 1, TILE_SIZE):
                    record = tile_record(date, row, col)
                    if record is not None:
                        rows.append(record)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No usable tiles found.")
    return manifest


def per_date_statistics(manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = manifest.groupby("date", sort=True)
    rows: list[dict[str, object]] = []
    for date, group in grouped:
        valid = int(group["valid_pixel_count"].sum())
        counts = {klass: int(group[f"class_{klass}_count"].sum()) for klass in range(4)}
        rows.append(
            {
                "date": date,
                "valid_tiles": int(len(group)),
                "valid_pixels": valid,
                "class_0_pixels": counts[0],
                "class_1_pixels": counts[1],
                "class_2_pixels": counts[2],
                "class_3_pixels": counts[3],
                "class_0_fraction": counts[0] / valid if valid else 0.0,
                "class_1_fraction": counts[1] / valid if valid else 0.0,
                "class_2_fraction": counts[2] / valid if valid else 0.0,
                "class_3_fraction": counts[3] / valid if valid else 0.0,
                "algae_pixels": counts[1] + counts[2] + counts[3],
                "algae_fraction": (counts[1] + counts[2] + counts[3]) / valid if valid else 0.0,
                "tiles_with_low": int((group["class_1_count"] > 0).sum()),
                "tiles_with_mid": int((group["class_2_count"] > 0).sum()),
                "tiles_with_high": int((group["class_3_count"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def subset_stats(stats: pd.DataFrame, dates: tuple[str, ...]) -> dict[str, object]:
    frame = stats[stats["date"].isin(dates)]
    valid = int(frame["valid_pixels"].sum())
    counts = {klass: int(frame[f"class_{klass}_pixels"].sum()) for klass in range(4)}
    return {
        "dates": ", ".join(dates),
        "n_dates": len(dates),
        "valid_tiles": int(frame["valid_tiles"].sum()),
        "valid_pixels": valid,
        "class_0": counts[0],
        "class_1": counts[1],
        "class_2": counts[2],
        "class_3": counts[3],
        "algae_fraction": (counts[1] + counts[2] + counts[3]) / valid if valid else 0.0,
        "has_low": counts[1] > 0,
        "has_mid": counts[2] > 0,
        "has_high": counts[3] > 0,
    }


def candidate_splits(stats: pd.DataFrame) -> list[dict[str, tuple[str, ...]]]:
    dates = tuple(stats["date"])
    viable_pairs = []
    for pair in itertools.combinations(dates, 2):
        row = subset_stats(stats, pair)
        if row["has_low"] and row["has_mid"] and row["has_high"]:
            viable_pairs.append(pair)

    candidates: list[dict[str, tuple[str, ...]]] = []
    for validation, test in itertools.permutations(viable_pairs, 2):
        if set(validation) & set(test):
            continue
        train = tuple(date for date in dates if date not in set(validation) | set(test))
        if len(train) == 8:
            candidates.append({"train": train, "validation": validation, "test": test})

    def score(candidate: dict[str, tuple[str, ...]]) -> float:
        parts = {split: subset_stats(stats, dates_) for split, dates_ in candidate.items()}
        total = sum(float(part["valid_pixels"]) for part in parts.values())
        target = {"train": 0.70, "validation": 0.15, "test": 0.15}
        size_penalty = sum(abs(float(parts[split]["valid_pixels"]) / total - target[split]) for split in target)
        algae = [float(parts[split]["algae_fraction"]) for split in ("train", "validation", "test")]
        algae_penalty = max(algae) - min(algae)
        return size_penalty + algae_penalty

    candidates.sort(key=score)
    return candidates[:3]


def split_summary(stats: pd.DataFrame, candidates: list[dict[str, tuple[str, ...]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        for split, dates in candidate.items():
            row = subset_stats(stats, dates)
            row["candidate"] = f"candidate_{index}"
            row["split"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def save_class_distribution_by_date(stats: pd.DataFrame) -> None:
    width, height = 1100, 620
    left_margin, top_margin = 130, 60
    plot_width, row_height = 850, 36
    colors = {
        "class_0_pixels": (65, 65, 65),
        "class_1_pixels": (121, 190, 85),
        "class_2_pixels": (247, 188, 65),
        "class_3_pixels": (218, 83, 63),
    }
    labels = {
        "class_0_pixels": "0 background",
        "class_1_pixels": "1 low",
        "class_2_pixels": "2 mid",
        "class_3_pixels": "3 high",
    }
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "Class distribution by date, valid pixels only", fill=(0, 0, 0))
    for row_index, row in enumerate(stats.itertuples(index=False)):
        y = top_margin + row_index * row_height
        draw.text((20, y + 8), row.date, fill=(0, 0, 0))
        x = left_margin
        total = max(int(row.valid_pixels), 1)
        for column in colors:
            value = int(getattr(row, column))
            segment = int(round(plot_width * value / total))
            draw.rectangle((x, y, x + segment, y + 22), fill=colors[column])
            x += segment
    legend_x, legend_y = left_margin, height - 70
    for index, column in enumerate(colors):
        x = legend_x + index * 200
        draw.rectangle((x, legend_y, x + 24, legend_y + 16), fill=colors[column])
        draw.text((x + 32, legend_y), labels[column], fill=(0, 0, 0))
    canvas.save(FIGURES_DIR / "class_distribution_by_date.png")


def write_split_analysis(stats: pd.DataFrame, split_rows: pd.DataFrame, total_tiles: int) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in frame[columns].itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    per_date = stats.copy()
    for column in ("class_0_fraction", "class_1_fraction", "class_2_fraction", "class_3_fraction", "algae_fraction"):
        per_date[column] = per_date[column].map(lambda value: f"{value:.3f}")
    split_display = split_rows.copy()
    split_display["algae_fraction"] = split_display["algae_fraction"].map(lambda value: f"{value:.3f}")

    doc = f"""# Split Analysis

Phase 2B builds the full manifest and proposes date-level splits only. No model training and no duplicated raster tile cache were created.

## Manifest Summary

- Tile size: `224 x 224`
- Tiling: non-overlapping windows on the B04 reference grid
- Minimum valid-pixel ratio: `{MIN_VALID_RATIO}`
- Total usable tiles: `{total_tiles}`
- Source rasters are read on demand from file paths recorded in `results/tile_manifest.csv`.

## Per-Date Statistics

{table(per_date, [
    "date",
    "valid_tiles",
    "valid_pixels",
    "class_0_pixels",
    "class_1_pixels",
    "class_2_pixels",
    "class_3_pixels",
    "algae_pixels",
    "algae_fraction",
    "tiles_with_low",
    "tiles_with_mid",
    "tiles_with_high",
])}

## Candidate Date-Level Splits

Each candidate uses 8 train dates, 2 validation dates and 2 test dates. No date appears in more than one split. Validation and test sets each contain classes `1`, `2` and `3`.

{table(split_display, [
    "candidate",
    "split",
    "n_dates",
    "dates",
    "valid_tiles",
    "valid_pixels",
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "algae_fraction",
    "has_low",
    "has_mid",
    "has_high",
])}

## Trade-Offs

The candidates differ mainly in algae fraction and valid-pixel volume. Because bloom severity is date-dependent, the validation and test pairs cannot be perfectly balanced while preserving date-level leakage safety. Choose the final split only after deciding whether the priority is temporal order, class balance, or stress-testing on high-algae dates.
"""
    (DOCS_DIR / "SPLIT_ANALYSIS.md").write_text(doc, encoding="utf-8")


def update_preprocessing_note(total_tiles: int) -> None:
    path = DOCS_DIR / "PREPROCESSING.md"
    text = path.read_text(encoding="utf-8")
    note = f"""

## Phase 2B Manifest Note

Phase 2B extended the rasterio pipeline to all 12 dates and wrote `results/tile_manifest.csv` with `{total_tiles}` usable tiles. It still avoids saving duplicated raster tile arrays; future datasets should read the source rasters on demand from manifest file paths and windows.
"""
    if "## Phase 2B Manifest Note" not in text:
        path.write_text(text.rstrip() + note, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_full_manifest()
    stats = per_date_statistics(manifest)
    candidates = candidate_splits(stats)
    splits = split_summary(stats, candidates)

    manifest.to_csv(RESULTS_DIR / "tile_manifest.csv", index=False)
    stats.to_csv(RESULTS_DIR / "per_date_statistics.csv", index=False)
    save_class_distribution_by_date(stats)
    write_split_analysis(stats, splits, total_tiles=len(manifest))
    update_preprocessing_note(total_tiles=len(manifest))

    print(f"usable_tiles={len(manifest)}")
    print(stats[["date", "valid_tiles", "class_0_pixels", "class_1_pixels", "class_2_pixels", "class_3_pixels", "algae_fraction"]].to_string(index=False))
    print(splits[["candidate", "split", "dates", "valid_tiles", "class_1", "class_2", "class_3", "algae_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
