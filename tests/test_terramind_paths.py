from pathlib import PurePosixPath, PureWindowsPath
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.terramind_dataset import resolve_terramind_paths


def _suffix(path):
    return tuple(path.parts[-5:])


def test_resolve_terramind_paths_from_windows_style_root():
    paths = resolve_terramind_paths(PureWindowsPath(r"D:\CODE\summer School\robust-multimodal-geoai\data\raw\SummerSchool_Subset"), "2025-03-12")
    assert paths["B04"].name == "2025-03-12_B04.tif"
    assert paths["VV"].name == "2025-03-12_VV.tif"
    assert paths["mask"].name == "2025_03_12.tiff"
    assert tuple(paths["B04"].parts[-4:]) == ("images", "2025-03-12", "layers", "2025-03-12_B04.tif")


def test_resolve_terramind_paths_from_posix_colab_root():
    paths = resolve_terramind_paths(PurePosixPath("/content/geoai_data/SummerSchool_Subset"), "2025-03-12")
    assert paths["B02"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/images/2025-03-12/layers/2025-03-12_B02.tif")
    assert paths["B03"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/images/2025-03-12/layers/2025-03-12_B03.tif")
    assert paths["B04"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/images/2025-03-12/layers/2025-03-12_B04.tif")
    assert paths["VV"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/images/2025-03-12/layers/2025-03-12_VV.tif")
    assert paths["VH"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/images/2025-03-12/layers/2025-03-12_VH.tif")
    assert paths["mask"] == PurePosixPath("/content/geoai_data/SummerSchool_Subset/masks/2025_03_12.tiff")

