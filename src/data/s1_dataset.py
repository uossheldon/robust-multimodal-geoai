from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from torch.utils.data import Dataset

SAR_NODATA = -9999.0
SAR_CHANNELS = ("VV", "VH")


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


def sar_normalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(2, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(2, 1, 1)
    if torch.any(std_tensor <= 0):
        raise ValueError("SAR standard deviations must be positive.")
    return (image - mean_tensor) / std_tensor


class S1SARSegmentationDataset(Dataset):
    """Manifest-based Sentinel-1 VV/VH segmentation dataset.

    VV and VH are aligned to the B04 reference grid with bilinear resampling.
    Masks are aligned to B04 with nearest-neighbour resampling. SAR nodata
    (-9999) and non-finite pixels are filled with train-derived channel means
    before normalization and excluded from the target with label 255.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path,
        split: str | None = None,
        sar_mean: Sequence[float],
        sar_std: Sequence[float],
        augment: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        manifest = pd.read_csv(manifest_path)
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.manifest = manifest.reset_index(drop=True)
        self.sar_mean = np.asarray(sar_mean, dtype=np.float32).reshape(2)
        self.sar_std = np.asarray(sar_std, dtype=np.float32).reshape(2)
        if not np.isfinite(self.sar_mean).all() or not np.isfinite(self.sar_std).all() or np.any(self.sar_std <= 0):
            raise ValueError("SAR mean/std must be finite and std must be positive.")
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
        vv_path = self.project_root / str(row.source_VV)
        vh_path = self.project_root / str(row.source_VH)
        mask_path = self.project_root / str(row.source_mask)
        with rasterio.open(b04_path) as reference:
            vv = _read_on_reference(vv_path, reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            vh = _read_on_reference(vh_path, reference, window, resampling=Resampling.bilinear, nodata=np.nan)
            mask = _read_on_reference(mask_path, reference, window, resampling=Resampling.nearest, nodata=255)
        image = np.stack([vv, vh]).astype(np.float32)
        target = mask.astype(np.int64)
        sar_invalid = ~np.isfinite(image) | np.isclose(image, SAR_NODATA)
        invalid_any = sar_invalid.any(axis=0)
        target[~np.isin(target, [0, 1, 2, 3])] = 255
        target[invalid_any] = 255
        for channel in range(2):
            image[channel][sar_invalid[channel]] = self.sar_mean[channel]
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
        image_tensor = sar_normalize(image_tensor, self.sar_mean, self.sar_std)
        return image_tensor, target_tensor


def fit_sar_train_statistics(manifest_path: str | Path, *, project_root: str | Path) -> dict[str, object]:
    """Fit VV/VH mean and std from valid train pixels only."""
    project_root = Path(project_root)
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    sums = np.zeros(2, dtype=np.float64)
    sq_sums = np.zeros(2, dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for row in train.itertuples(index=False):
        window = Window(
            col_off=int(row.col_offset),
            row_off=int(row.row_offset),
            width=int(row.window_width),
            height=int(row.window_height),
        )
        b04_path = project_root / str(row.source_B04)
        with rasterio.open(b04_path) as reference:
            values = []
            for source_column in ("source_VV", "source_VH"):
                array = _read_on_reference(
                    project_root / str(getattr(row, source_column)),
                    reference,
                    window,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                ).astype(np.float32)
                values.append(array)
            mask = _read_on_reference(
                project_root / str(row.source_mask),
                reference,
                window,
                resampling=Resampling.nearest,
                nodata=255,
            ).astype(np.int64)
        image = np.stack(values)
        valid = (mask != 255) & np.isin(mask, [0, 1, 2, 3])
        valid &= np.isfinite(image).all(axis=0)
        valid &= ~np.isclose(image, SAR_NODATA).any(axis=0)
        for channel in range(2):
            channel_values = image[channel][valid].astype(np.float64)
            counts[channel] += channel_values.size
            sums[channel] += channel_values.sum()
            sq_sums[channel] += np.square(channel_values).sum()
    means = sums / counts
    variances = np.maximum(sq_sums / counts - np.square(means), 0.0)
    stds = np.sqrt(variances)
    return {
        "channels": list(SAR_CHANNELS),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "valid_pixel_counts": counts.astype(int).tolist(),
        "source": "train split valid pixels only; mask 255, SAR nodata -9999 and non-finite pixels excluded",
    }
