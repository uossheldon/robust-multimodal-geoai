from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import resolve_terramind_paths


def test_resolve_terramind_paths_from_windows_style_root():
    root = PureWindowsPath(r"C:\project\data\raw\SummerSchool_Subset")
    paths = resolve_terramind_paths(root, "2025-03-12")
    assert paths["B04"].name == "2025-03-12_B04.tif"
    assert paths["VV"].name == "2025-03-12_VV.tif"
    assert paths["mask"].name == "2025_03_12.tiff"
    assert tuple(paths["B04"].parts[-4:]) == (
        "images", "2025-03-12", "layers", "2025-03-12_B04.tif"
    )


def test_resolve_terramind_paths_from_posix_colab_root():
    root = PurePosixPath("/workspace/data/SummerSchool_Subset")
    paths = resolve_terramind_paths(root, "2025-03-12")
    assert paths["B02"] == root / "images/2025-03-12/layers/2025-03-12_B02.tif"
    assert paths["B03"] == root / "images/2025-03-12/layers/2025-03-12_B03.tif"
    assert paths["B04"] == root / "images/2025-03-12/layers/2025-03-12_B04.tif"
    assert paths["VV"] == root / "images/2025-03-12/layers/2025-03-12_VV.tif"
    assert paths["VH"] == root / "images/2025-03-12/layers/2025-03-12_VH.tif"
    assert paths["mask"] == root / "masks/2025_03_12.tiff"
