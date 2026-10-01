from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
SAR_NODATA = -9999.0


def _same_grid(src: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> bool:
    return src.crs == reference.crs and src.width == reference.width and src.height == reference.height and src.transform.almost_equals(reference.transform)


def _read_on_reference(path: Path, reference: rasterio.io.DatasetReader, window: Window, *, resampling: Resampling, nodata: int | float) -> np.ndarray:
    with rasterio.open(path) as src:
        if _same_grid(src, reference):
            return src.read(1, window=window, masked=False)
        with WarpedVRT(src, crs=reference.crs, transform=reference.transform, width=reference.width, height=reference.height, resampling=resampling, src_nodata=src.nodata, nodata=nodata) as vrt:
            return vrt.read(1, window=window, masked=False)


class S1S2EarlyFusionDataset(Dataset):
    """Five-channel early-fusion dataset: normalized RGB + normalized VV/VH."""

    def __init__(self, manifest_path: str | Path, *, project_root: str | Path, split: str | None, sar_mean: Sequence[float], sar_std: Sequence[float], augment: bool = False) -> None:
        self.project_root = Path(project_root)
        manifest = pd.read_csv(manifest_path)
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.manifest = manifest.reset_index(drop=True)
        self.sar_mean = torch.as_tensor(sar_mean, dtype=torch.float32).view(2, 1, 1)
        self.sar_std = torch.as_tensor(sar_std, dtype=torch.float32).view(2, 1, 1)
        if not torch.isfinite(self.sar_mean).all() or not torch.isfinite(self.sar_std).all() or torch.any(self.sar_std <= 0):
            raise ValueError("SAR mean/std must be finite and positive.")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        window = Window(col_off=int(row.col_offset), row_off=int(row.row_offset), width=int(row.window_width), height=int(row.window_height))
        b04_path = self.project_root / str(row.source_B04)
        with rasterio.open(b04_path) as reference:
            red = _read_on_reference(b04_path, reference, window, resampling=Resampling.bilinear, nodata=0)
            green = _read_on_reference(self.project_root / str(row.source_B03), reference, window, resampling=Resampling.bilinear, nodata=0)
            blue = _read_on_reference(self.project_root / str(row.source_B02), reference, window, resampling=Resampling.bilinear, nodata=0)
            vv = _read_on_reference(self.project_root / str(row.source_VV), reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            vh = _read_on_reference(self.project_root / str(row.source_VH), reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            mask = _read_on_reference(self.project_root / str(row.source_mask), reference, window, resampling=Resampling.nearest, nodata=255)
        rgb = np.stack([red, green, blue]).astype(np.float32) / 10000.0
        rgb = np.clip(rgb, 0.0, 1.0)
        sar = np.stack([vv, vh]).astype(np.float32)
        target = mask.astype(np.int64)
        target[~np.isin(target, [0, 1, 2, 3])] = 255
        sar_invalid = ~np.isfinite(sar) | np.isclose(sar, SAR_NODATA)
        target[sar_invalid.any(axis=0)] = 255
        sar_mean_np = self.sar_mean.numpy().reshape(2)
        for channel in range(2):
            sar[channel][sar_invalid[channel]] = sar_mean_np[channel]
        rgb_tensor = torch.from_numpy(rgb)
        sar_tensor = torch.from_numpy(sar)
        target_tensor = torch.from_numpy(target)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                rgb_tensor = torch.flip(rgb_tensor, dims=[2]); sar_tensor = torch.flip(sar_tensor, dims=[2]); target_tensor = torch.flip(target_tensor, dims=[1])
            if torch.rand(1).item() < 0.5:
                rgb_tensor = torch.flip(rgb_tensor, dims=[1]); sar_tensor = torch.flip(sar_tensor, dims=[1]); target_tensor = torch.flip(target_tensor, dims=[0])
            rotations = int(torch.randint(0, 4, size=(1,)).item())
            rgb_tensor = torch.rot90(rgb_tensor, k=rotations, dims=[1, 2]); sar_tensor = torch.rot90(sar_tensor, k=rotations, dims=[1, 2]); target_tensor = torch.rot90(target_tensor, k=rotations, dims=[0, 1])
        rgb_tensor = (rgb_tensor - IMAGENET_MEAN) / IMAGENET_STD
        sar_tensor = (sar_tensor - self.sar_mean) / self.sar_std
        return torch.cat([rgb_tensor, sar_tensor], dim=0), target_tensor
