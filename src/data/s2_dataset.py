from __future__ import annotations

from pathlib import Path
from typing import Callable

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


def deeplab_normalize(image: torch.Tensor) -> torch.Tensor:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


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


class S2RGBSegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path,
        split: str | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = deeplab_normalize,
        reflectance_scale: float = 10000.0,
        augment: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        manifest = pd.read_csv(manifest_path)
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.reflectance_scale = reflectance_scale
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        window = Window(
            col_off=int(row.col_offset),
            row_off=int(row.row_offset),
            width=int(row.window_width),
            height=int(row.window_height),
        )
        b04_path = self.project_root / str(row.source_B04)
        b03_path = self.project_root / str(row.source_B03)
        b02_path = self.project_root / str(row.source_B02)
        mask_path = self.project_root / str(row.source_mask)
        with rasterio.open(b04_path) as reference:
            red = _read_on_reference(b04_path, reference, window, resampling=Resampling.bilinear, nodata=0)
            green = _read_on_reference(b03_path, reference, window, resampling=Resampling.bilinear, nodata=0)
            blue = _read_on_reference(b02_path, reference, window, resampling=Resampling.bilinear, nodata=0)
            mask = _read_on_reference(mask_path, reference, window, resampling=Resampling.nearest, nodata=255)
        image = np.stack([red, green, blue]).astype(np.float32) / self.reflectance_scale
        image = np.clip(image, 0.0, 1.0)
        target = mask.astype(np.int64)
        target[~np.isin(target, [0, 1, 2, 3])] = 255
        image_tensor = torch.from_numpy(image)
        target_tensor = torch.from_numpy(target)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                target_tensor = torch.flip(target_tensor, dims=[1])
            if torch.rand(1).item() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])
                target_tensor = torch.flip(target_tensor, dims=[0])
            rotations = int(torch.randint(0, 4, size=(1,)).item())
            image_tensor = torch.rot90(image_tensor, k=rotations, dims=[1, 2])
            target_tensor = torch.rot90(target_tensor, k=rotations, dims=[0, 1])
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        return image_tensor, target_tensor

