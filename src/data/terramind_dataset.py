from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from torch.utils.data import Dataset

TERRAMIND_S1RTC_MEAN = torch.tensor([-10.930, -17.329], dtype=torch.float32).view(2, 1, 1)
TERRAMIND_S1RTC_STD = torch.tensor([4.391, 4.459], dtype=torch.float32).view(2, 1, 1)
SAR_NODATA = -9999.0
VALID_CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)


def default_terramind_data_root(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "raw" / "SummerSchool_Subset"


from src.data.paths import resolve_terramind_paths


def _same_grid(src: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> bool:
    return (
        src.crs == reference.crs
        and src.width == reference.width
        and src.height == reference.height
        and src.transform.almost_equals(reference.transform)
    )


def _read_on_reference(
    path: Path,
    reference: rasterio.io.DatasetReader,
    window: Window,
    *,
    resampling: Resampling,
    nodata: int | float,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if _same_grid(src, reference):
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


class TerraMindMultimodalDataset(Dataset):
    """Manifest-backed TerraMind RGB + S1RTC dataset.

    The manifest supplies split membership and tile windows. Raster paths are
    reconstructed from ``data_root`` and acquisition date so the same manifest
    works on Windows, Linux, and Colab.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path,
        split: str | None,
        data_root: str | Path | None = None,
        augment: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_root = Path(data_root) if data_root is not None else default_terramind_data_root(self.project_root)
        manifest = pd.read_csv(manifest_path)
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.manifest = manifest.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.manifest.iloc[index]
        date = str(row.date)
        paths = resolve_terramind_paths(self.data_root, date)
        window = Window(
            col_off=int(row.col_offset),
            row_off=int(row.row_offset),
            width=int(row.window_width),
            height=int(row.window_height),
        )
        b04_path = paths["B04"]
        with rasterio.open(b04_path) as reference:
            red = _read_on_reference(b04_path, reference, window, resampling=Resampling.bilinear, nodata=0)
            green = _read_on_reference(paths["B03"], reference, window, resampling=Resampling.bilinear, nodata=0)
            blue = _read_on_reference(paths["B02"], reference, window, resampling=Resampling.bilinear, nodata=0)
            vv = _read_on_reference(paths["VV"], reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            vh = _read_on_reference(paths["VH"], reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            mask = _read_on_reference(paths["mask"], reference, window, resampling=Resampling.nearest, nodata=255)

        rgb = np.stack([red, green, blue]).astype(np.float32) / 10000.0
        rgb = np.clip(rgb, 0.0, 1.0)
        sar = np.stack([vv, vh]).astype(np.float32)
        target = mask.astype(np.int64)
        target[~np.isin(target, VALID_CLASSES)] = 255

        sar_invalid = ~np.isfinite(sar) | np.isclose(sar, SAR_NODATA)
        target[sar_invalid.any(axis=0)] = 255
        mean_np = TERRAMIND_S1RTC_MEAN.numpy().reshape(2)
        for channel in range(2):
            sar[channel][sar_invalid[channel]] = mean_np[channel]

        rgb_tensor = torch.from_numpy(rgb)
        sar_tensor = torch.from_numpy(sar)
        target_tensor = torch.from_numpy(target)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                rgb_tensor = torch.flip(rgb_tensor, dims=[2])
                sar_tensor = torch.flip(sar_tensor, dims=[2])
                target_tensor = torch.flip(target_tensor, dims=[1])
            if torch.rand(1).item() < 0.5:
                rgb_tensor = torch.flip(rgb_tensor, dims=[1])
                sar_tensor = torch.flip(sar_tensor, dims=[1])
                target_tensor = torch.flip(target_tensor, dims=[0])
            rotations = int(torch.randint(0, 4, size=(1,)).item())
            rgb_tensor = torch.rot90(rgb_tensor, k=rotations, dims=[1, 2])
            sar_tensor = torch.rot90(sar_tensor, k=rotations, dims=[1, 2])
            target_tensor = torch.rot90(target_tensor, k=rotations, dims=[0, 1])

        rgb_bgr_255 = rgb_tensor[[2, 1, 0], :, :] * 255.0
        s1rtc = (sar_tensor - TERRAMIND_S1RTC_MEAN) / TERRAMIND_S1RTC_STD
        return {"RGB": rgb_bgr_255.contiguous(), "S1RTC": s1rtc.contiguous()}, target_tensor.contiguous()


def terramind_collate(batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs: dict[str, Any] = {"RGB": [], "S1RTC": []}
    targets = []
    for sample_inputs, target in batch:
        inputs["RGB"].append(sample_inputs["RGB"])
        inputs["S1RTC"].append(sample_inputs["S1RTC"])
        targets.append(target)
    return {key: torch.stack(value, dim=0) for key, value in inputs.items()}, torch.stack(targets, dim=0)
