from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GEOTIFF_MODEL_PIXEL_SCALE = 33550
GEOTIFF_MODEL_TIEPOINT = 33922
GEOTIFF_GEO_KEYS = 34735
GEOTIFF_NODATA = 42113


@dataclass(frozen=True)
class RasterGrid:
    path: Path
    width: int
    height: int
    dtype: str
    mode: str
    nodata: float | int | None
    scale_x: float
    scale_y: float
    origin_x: float
    origin_y: float
    epsg: int | None

    @property
    def left(self) -> float:
        return self.origin_x

    @property
    def right(self) -> float:
        return self.origin_x + self.width * self.scale_x

    @property
    def top(self) -> float:
        return self.origin_y

    @property
    def bottom(self) -> float:
        return self.origin_y - self.height * self.scale_y

    @property
    def resolution(self) -> tuple[float, float]:
        return self.scale_x, self.scale_y

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.bottom, self.right, self.top


def _epsg_from_geokeys(raw: tuple[int, ...] | None) -> int | None:
    if not raw or len(raw) < 4:
        return None
    count = raw[3]
    for index in range(count):
        base = 4 + index * 4
        if base + 3 >= len(raw):
            break
        key_id, tag_location, _value_count, value_offset = raw[base : base + 4]
        if key_id == 3072 and tag_location == 0:
            return int(value_offset)
    return None


def read_grid(path: Path) -> RasterGrid:
    with Image.open(path) as image:
        tags = image.tag_v2
        pixel_scale = tags.get(GEOTIFF_MODEL_PIXEL_SCALE)
        tiepoint = tags.get(GEOTIFF_MODEL_TIEPOINT)
        if not pixel_scale or not tiepoint:
            raise ValueError(f"Missing GeoTIFF transform tags: {path}")
        nodata = tags.get(GEOTIFF_NODATA)
        if isinstance(nodata, str):
            try:
                nodata = float(nodata) if "." in nodata or "e" in nodata.lower() else int(nodata)
            except ValueError:
                pass
        return RasterGrid(
            path=path,
            width=image.width,
            height=image.height,
            dtype=str(np.asarray(image).dtype),
            mode=image.mode,
            nodata=nodata,
            scale_x=float(pixel_scale[0]),
            scale_y=float(pixel_scale[1]),
            origin_x=float(tiepoint[3]),
            origin_y=float(tiepoint[4]),
            epsg=_epsg_from_geokeys(tags.get(GEOTIFF_GEO_KEYS)),
        )


def read_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def world_coordinates_for_window(
    reference: RasterGrid, row_offset: int, col_offset: int, tile_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = row_offset + np.arange(tile_size, dtype=np.float64) + 0.5
    cols = col_offset + np.arange(tile_size, dtype=np.float64) + 0.5
    x = reference.origin_x + cols * reference.scale_x
    y = reference.origin_y - rows * reference.scale_y
    return np.meshgrid(x, y)


def source_pixel_coordinates(source: RasterGrid, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = (x - source.origin_x) / source.scale_x - 0.5
    rows = (source.origin_y - y) / source.scale_y - 0.5
    return rows, cols


def sample_nearest(
    source_array: np.ndarray,
    source: RasterGrid,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float | int,
) -> np.ndarray:
    rows, cols = source_pixel_coordinates(source, x, y)
    nearest_rows = np.rint(rows).astype(np.int64)
    nearest_cols = np.rint(cols).astype(np.int64)
    valid = (
        (nearest_rows >= 0)
        & (nearest_rows < source.height)
        & (nearest_cols >= 0)
        & (nearest_cols < source.width)
    )
    output = np.full(rows.shape, fill_value, dtype=source_array.dtype)
    output[valid] = source_array[nearest_rows[valid], nearest_cols[valid]]
    return output


def sample_bilinear(
    source_array: np.ndarray,
    source: RasterGrid,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    rows, cols = source_pixel_coordinates(source, x, y)
    row0 = np.floor(rows).astype(np.int64)
    col0 = np.floor(cols).astype(np.int64)
    row1 = row0 + 1
    col1 = col0 + 1
    valid = (row0 >= 0) & (row1 < source.height) & (col0 >= 0) & (col1 < source.width)
    output = np.full(rows.shape, fill_value, dtype=np.float32)
    if not valid.any():
        return output
    r0 = row0[valid]
    r1 = row1[valid]
    c0 = col0[valid]
    c1 = col1[valid]
    dr = (rows[valid] - r0).astype(np.float32)
    dc = (cols[valid] - c0).astype(np.float32)
    top = source_array[r0, c0] * (1 - dc) + source_array[r0, c1] * dc
    bottom = source_array[r1, c0] * (1 - dc) + source_array[r1, c1] * dc
    output[valid] = top * (1 - dr) + bottom * dr
    return output


def window_bounds(reference: RasterGrid, row_offset: int, col_offset: int, tile_size: int) -> tuple[float, float, float, float]:
    left = reference.origin_x + col_offset * reference.scale_x
    right = reference.origin_x + (col_offset + tile_size) * reference.scale_x
    top = reference.origin_y - row_offset * reference.scale_y
    bottom = reference.origin_y - (row_offset + tile_size) * reference.scale_y
    return left, bottom, right, top


def valid_data_mask(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, dtype=bool)
    if nodata is not None:
        valid &= values != nodata
    return valid

