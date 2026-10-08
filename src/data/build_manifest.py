from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "SummerSchool_Subset"
RESULTS_DIR = PROJECT_ROOT / "results"

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


def main() -> None:
    from src.data.apply_split import split_for_date
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_full_manifest()
    manifest["split"] = manifest["date"].map(split_for_date)
    stats = per_date_statistics(manifest)
    manifest.to_csv(RESULTS_DIR / "tile_manifest.csv", index=False)
    stats.to_csv(RESULTS_DIR / "per_date_statistics.csv", index=False)
    print(f"usable_tiles={len(manifest)}")


if __name__ == "__main__":
    main()
