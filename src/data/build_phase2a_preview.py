from __future__ import annotations

import csv
import json
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

TILE_SIZE = 224
MIN_VALID_RATIO = 0.20
PREVIEW_DATES = ("2025-01-01", "2025-06-20")
EXPECTED_DATES = (
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
LAYERS = ("B02", "B03", "B04", "B08", "NDVI", "NDWI", "VV", "VH")
MASK_IGNORE = 255
MASK_CLASSES = (0, 1, 2, 3, 255)


def layer_path(date: str, layer: str) -> Path:
    return DATASET_ROOT / "images" / date / "layers" / f"{date}_{layer}.tif"


def mask_path(date: str) -> Path:
    return DATASET_ROOT / "masks" / f"{date.replace('-', '_')}.tiff"


def metadata_path(date: str, name: str) -> Path:
    return DATASET_ROOT / "images" / date / "layers" / name


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def raster_summary(date: str, kind: str, path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        return {
            "date": date,
            "kind": kind,
            "path": rel(path),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else "",
            "transform": tuple(round(v, 9) for v in src.transform),
            "res_x": src.res[0],
            "res_y": src.res[1],
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            "nodata": src.nodata if src.nodata is not None else "",
            "selected_item_id": "",
            "selected_acquisition_datetime": "",
        }


def summarize_inputs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in EXPECTED_DATES:
        for layer in LAYERS:
            rows.append(raster_summary(date, layer, layer_path(date, layer)))
        rows.append(raster_summary(date, "mask", mask_path(date)))

        with metadata_path(date, "metadata.json").open("r", encoding="utf-8") as handle:
            optical = json.load(handle)
        with metadata_path(date, "metadata_s1.json").open("r", encoding="utf-8") as handle:
            sar = json.load(handle)
        rows.append(
            {
                "date": date,
                "kind": "metadata",
                "path": rel(metadata_path(date, "metadata.json")),
                "width": "",
                "height": "",
                "dtype": "",
                "crs": f"EPSG:{optical.get('output_epsg')}",
                "transform": "",
                "res_x": optical.get("resolution_m"),
                "res_y": optical.get("resolution_m"),
                "left": "",
                "bottom": "",
                "right": "",
                "top": "",
                "nodata": "",
                "selected_item_id": optical.get("selected_item_id", ""),
                "selected_acquisition_datetime": optical.get("selected_acquisition_datetime", ""),
            }
        )
        rows.append(
            {
                "date": date,
                "kind": "metadata_s1",
                "path": rel(metadata_path(date, "metadata_s1.json")),
                "width": "",
                "height": "",
                "dtype": "",
                "crs": f"EPSG:{sar.get('output_epsg')}",
                "transform": "",
                "res_x": sar.get("resolution_m"),
                "res_y": sar.get("resolution_m"),
                "left": "",
                "bottom": "",
                "right": "",
                "top": "",
                "nodata": sar.get("nodata", ""),
                "selected_item_id": sar.get("selected_item_id", ""),
                "selected_acquisition_datetime": sar.get("selected_acquisition_datetime", ""),
            }
        )
    return pd.DataFrame(rows)


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


def alignment_report(date: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with rasterio.open(layer_path(date, "B04")) as reference:
        for name, path, method in [
            ("B03", layer_path(date, "B03"), "direct-or-warp"),
            ("B02", layer_path(date, "B02"), "direct-or-warp"),
            ("VV", layer_path(date, "VV"), "bilinear-to-B04"),
            ("VH", layer_path(date, "VH"), "bilinear-to-B04"),
            ("mask", mask_path(date), "nearest-to-B04"),
        ]:
            with rasterio.open(path) as src:
                rows.append(
                    {
                        "date": date,
                        "source": name,
                        "method": method,
                        "source_crs": src.crs.to_string() if src.crs else "",
                        "reference_crs": reference.crs.to_string() if reference.crs else "",
                        "same_crs": src.crs == reference.crs,
                        "same_transform": src.transform.almost_equals(reference.transform),
                        "same_width": src.width == reference.width,
                        "same_height": src.height == reference.height,
                        "source_bounds": tuple(round(v, 6) for v in src.bounds),
                        "reference_bounds": tuple(round(v, 6) for v in reference.bounds),
                    }
                )
    return rows


def tile_record(date: str, row: int, col: int) -> dict[str, object] | None:
    window = Window(col_off=col, row_off=row, width=TILE_SIZE, height=TILE_SIZE)
    with rasterio.open(layer_path(date, "B04")) as reference:
        red = read_on_reference(layer_path(date, "B04"), reference, window, resampling=Resampling.bilinear, nodata=0)
        green = read_on_reference(layer_path(date, "B03"), reference, window, resampling=Resampling.bilinear, nodata=0)
        blue = read_on_reference(layer_path(date, "B02"), reference, window, resampling=Resampling.bilinear, nodata=0)
        vv = read_on_reference(layer_path(date, "VV"), reference, window, resampling=Resampling.bilinear, nodata=np.nan).astype(np.float32)
        vh = read_on_reference(layer_path(date, "VH"), reference, window, resampling=Resampling.bilinear, nodata=np.nan).astype(np.float32)
        mask = read_on_reference(mask_path(date), reference, window, resampling=Resampling.nearest, nodata=MASK_IGNORE).astype(np.uint8)

        optical_valid = finite_valid(red, 0) & finite_valid(green, 0) & finite_valid(blue, 0)
        sar_valid = finite_valid(vv, np.nan) & finite_valid(vh, np.nan)
        mask_valid = mask != MASK_IGNORE
        common_valid = optical_valid & sar_valid & mask_valid
        valid_ratio = float(common_valid.mean())
        if valid_ratio < MIN_VALID_RATIO:
            return None

        found_labels = sorted(int(value) for value in np.unique(mask))
        unexpected = sorted(set(found_labels) - set(MASK_CLASSES))
        if unexpected:
            raise ValueError(f"Unexpected mask labels for {date} at row {row}, col {col}: {unexpected}")

        counts = {value: int(((mask == value) & common_valid).sum()) for value in (0, 1, 2, 3)}
        left, bottom, right, top = window_bounds(window, reference.transform)
        return {
            "date": date,
            "row_offset": row,
            "col_offset": col,
            "tile_size": TILE_SIZE,
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top,
            "valid_pixel_ratio": valid_ratio,
            "valid_pixels": int(common_valid.sum()),
            "background_pixels": counts[0],
            "low_pixels": counts[1],
            "mid_pixels": counts[2],
            "high_pixels": counts[3],
            "ignore_or_invalid_pixels": TILE_SIZE * TILE_SIZE - int(common_valid.sum()),
            "algae_pixels": counts[1] + counts[2] + counts[3],
            "mask_labels_found": " ".join(str(v) for v in found_labels),
            "_rgb": np.stack([red, green, blue]),
            "_vv": vv,
            "_vh": vh,
            "_mask": mask,
            "_common_valid": common_valid,
        }


def build_preview_manifest() -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample: dict[str, object] | None = None
    for date in PREVIEW_DATES:
        with rasterio.open(layer_path(date, "B04")) as reference:
            date_count = 0
            for row in range(0, reference.height - TILE_SIZE + 1, TILE_SIZE):
                for col in range(0, reference.width - TILE_SIZE + 1, TILE_SIZE):
                    record = tile_record(date, row, col)
                    if record is None:
                        continue
                    if sample is None:
                        sample = dict(record)
                    rows.append({key: value for key, value in record.items() if not key.startswith("_")})
                    date_count += 1
                    if date_count >= 12:
                        break
                if date_count >= 12:
                    break
    if sample is None:
        raise RuntimeError("No valid preview tile found.")
    return pd.DataFrame(rows), sample


def stretch_to_uint8(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    data = values.astype(np.float32)
    sample = data[valid] if valid is not None else data[np.isfinite(data)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(sample, [2, 98])
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip((data - lo) * 255 / (hi - lo), 0, 255)
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def save_alignment_figure(sample: dict[str, object]) -> None:
    rgb = sample["_rgb"].astype(np.float32)  # type: ignore[index]
    common_valid = sample["_common_valid"]  # type: ignore[index]
    rgb_image = np.stack([stretch_to_uint8(rgb[index], common_valid) for index in range(3)], axis=2)
    vv_image = stretch_to_uint8(sample["_vv"], common_valid)  # type: ignore[index]
    vh_image = stretch_to_uint8(sample["_vh"], common_valid)  # type: ignore[index]
    mask = sample["_mask"]  # type: ignore[index]
    colors = {0: (35, 35, 35), 1: (121, 190, 85), 2: (247, 188, 65), 3: (218, 83, 63), 255: (210, 210, 210)}
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, color in colors.items():
        mask_rgb[mask == value] = color
    panels = [
        ("S2 RGB", Image.fromarray(rgb_image)),
        ("S1 VV", Image.fromarray(vv_image).convert("RGB")),
        ("S1 VH", Image.fromarray(vh_image).convert("RGB")),
        ("Mask", Image.fromarray(mask_rgb)),
    ]
    label_height = 28
    canvas = Image.new("RGB", (TILE_SIZE * len(panels), TILE_SIZE + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * TILE_SIZE
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 7), label, fill=(0, 0, 0))
    canvas.save(FIGURES_DIR / "data_alignment_check.png")


def save_class_distribution(manifest: pd.DataFrame) -> None:
    classes = [
        ("background_pixels", "Background", (70, 70, 70)),
        ("low_pixels", "Low", (121, 190, 85)),
        ("mid_pixels", "Mid", (247, 188, 65)),
        ("high_pixels", "High", (218, 83, 63)),
    ]
    totals = [(label, int(manifest[column].sum()), color) for column, label, color in classes]
    max_value = max(value for _, value, _ in totals) or 1
    canvas = Image.new("RGB", (760, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "Class distribution in Phase 2A.1 rasterio preview tiles", fill=(0, 0, 0))
    for index, (label, value, color) in enumerate(totals):
        y = 70 + index * 64
        width = int(500 * value / max_value)
        draw.text((24, y + 12), label, fill=(0, 0, 0))
        draw.rectangle((170, y, 170 + width, y + 42), fill=color)
        draw.text((182 + width, y + 12), f"{value:,}", fill=(0, 0, 0))
    canvas.save(FIGURES_DIR / "class_distribution.png")


def mask_label_values_for_preview() -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    for date in PREVIEW_DATES:
        with rasterio.open(mask_path(date)) as src:
            values[date] = sorted(int(item) for item in np.unique(src.read(1)))
    return values


def write_preprocessing_doc(summary: pd.DataFrame, manifest: pd.DataFrame, alignment: pd.DataFrame) -> None:
    s2 = summary[(summary.date == "2025-01-01") & (summary.kind == "B04")].iloc[0]
    s1 = summary[(summary.date == "2025-01-01") & (summary.kind == "VV")].iloc[0]
    mask = summary[(summary.date == "2025-01-01") & (summary.kind == "mask")].iloc[0]
    totals = {
        "background": int(manifest.background_pixels.sum()),
        "low": int(manifest.low_pixels.sum()),
        "mid": int(manifest.mid_pixels.sum()),
        "high": int(manifest.high_pixels.sum()),
        "valid": int(manifest.valid_pixels.sum()),
    }
    labels = mask_label_values_for_preview()
    alignment_rows = alignment[["date", "source", "method", "same_crs", "same_transform", "same_width", "same_height"]]
    alignment_lines = [
        "| Date | Source | Method | Same CRS | Same transform | Same width | Same height |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in alignment_rows.itertuples(index=False):
        alignment_lines.append(
            f"| {row.date} | {row.source} | {row.method} | {row.same_crs} | "
            f"{row.same_transform} | {row.same_width} | {row.same_height} |"
        )
    alignment_text = "\n".join(alignment_lines)
    doc = f"""# Preprocessing

Phase 2A.1 replaces the temporary preview sampler with a rasterio-based geospatial preprocessing path. No model training, final split, or full 12-date cache was created.

## Environment

Local venv: `.venv`

Installed preprocessing packages: `rasterio`, `numpy`, `pandas`, `pillow`.

## Alignment Findings

All inspected rasters are in EPSG:32629. Sentinel-2 optical bands share the B04 grid. Sentinel-1 VV/VH and masks use different dimensions/transforms and must be aligned to B04 before tiling.

Representative `2025-01-01` grids:

| Source | Size | Resolution | Nodata |
|---|---:|---|---|
| Sentinel-2 B04 | {s2.width} x {s2.height} | {float(s2.res_x):.6f} m x {float(s2.res_y):.6f} m | `{s2.nodata}` |
| Sentinel-1 VV/VH | {s1.width} x {s1.height} | {float(s1.res_x):.6f} m x {float(s1.res_y):.6f} m | `{s1.nodata}` |
| Mask | {mask.width} x {mask.height} | {float(mask.res_x):.6f} m x {float(mask.res_y):.6f} m | `{mask.nodata}` |

Numerical alignment check for preview dates:

{alignment_text}

## Resampling Rules

- B04 is the reference raster grid.
- B03 and B02 are read directly when they already match B04; otherwise they are warped to B04.
- VV and VH are warped to B04 with bilinear resampling.
- Masks are warped to B04 with nearest-neighbour resampling only.
- Mask labels are preserved as integer classes: `0` background, `1` low algae, `2` mid algae, `3` high algae, `255` ignore.

Exact mask values found in preview source masks:

| Date | Values |
|---|---|
| 2025-01-01 | `{labels['2025-01-01']}` |
| 2025-06-20 | `{labels['2025-06-20']}` |

## Tiling Protocol

The baseline protocol is now the original notebook's `224 x 224` non-overlapping B04-grid tiles. This Phase 2A.1 run generated a small preview only: first valid tiles from two representative dates, with valid-pixel ratio at least 20%.

For every candidate tile, optical nodata, SAR nodata/non-finite pixels, and mask value `255` are excluded. The manifest records projected bounds, row/column offsets, valid-pixel ratio, and class statistics.

## Preview Class Balance

Preview tiles: {len(manifest)} across {manifest.date.nunique()} dates.

| Class | Pixels |
|---|---:|
| Background `0` | {totals['background']} |
| Low algae `1` | {totals['low']} |
| Mid algae `2` | {totals['mid']} |
| High algae `3` | {totals['high']} |
| Algae total | {totals['low'] + totals['mid'] + totals['high']} |
| Valid total | {totals['valid']} |

## Outputs

- `results/data_summary.csv`
- `results/tile_manifest_preview.csv`
- `figures/data_alignment_check.png`
- `figures/class_distribution.png`

## Remaining Before Full Cache

- Run the rasterio pipeline over all 12 dates only after Phase 2B is explicitly requested.
- Decide whether final experiments use all valid 224-pixel tiles or the notebook's small selected subsets.
- Record checksums for raw data and generated manifests.
"""
    (PROJECT_ROOT / "docs" / "PREPROCESSING.md").write_text(doc, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_inputs()
    alignment = pd.DataFrame([row for date in PREVIEW_DATES for row in alignment_report(date)])
    summary = pd.concat([summary, alignment.assign(kind="alignment_check")], ignore_index=True)
    manifest, sample = build_preview_manifest()
    summary.to_csv(RESULTS_DIR / "data_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest.to_csv(RESULTS_DIR / "tile_manifest_preview.csv", index=False)
    save_alignment_figure(sample)
    save_class_distribution(manifest)
    write_preprocessing_doc(summary, manifest, alignment)
    print(f"summary_rows={len(summary)}")
    print(f"preview_tiles={len(manifest)}")
    print("mask_values=", mask_label_values_for_preview())
    print("class_totals=", {
        "background": int(manifest.background_pixels.sum()),
        "low": int(manifest.low_pixels.sum()),
        "mid": int(manifest.mid_pixels.sum()),
        "high": int(manifest.high_pixels.sum()),
        "valid": int(manifest.valid_pixels.sum()),
    })


if __name__ == "__main__":
    main()
