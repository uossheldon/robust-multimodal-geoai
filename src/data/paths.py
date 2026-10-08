"""Portable prepared-dataset paths; no raster or ML dependencies."""
from pathlib import Path, PurePath


def resolve_terramind_paths(data_root: str | Path, date: str) -> dict[str, Path]:
    """Resolve SummerSchool_Subset paths independently of manifest path separators."""
    root = data_root if isinstance(data_root, PurePath) else Path(data_root)
    safe_date = str(date)
    mask_date = safe_date.replace("-", "_")
    layer_dir = root / "images" / safe_date / "layers"
    return {
        "B02": layer_dir / f"{safe_date}_B02.tif",
        "B03": layer_dir / f"{safe_date}_B03.tif",
        "B04": layer_dir / f"{safe_date}_B04.tif",
        "VV": layer_dir / f"{safe_date}_VV.tif",
        "VH": layer_dir / f"{safe_date}_VH.tif",
        "mask": root / "masks" / f"{mask_date}.tiff",
    }
