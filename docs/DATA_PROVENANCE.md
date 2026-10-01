# Data Provenance

Phase 1 provenance notes for the Day 1 and Day 2 data dependencies.

## External References Found

| Source | Reference | Notebook use |
|---|---|---|
| Google Drive file | `https://drive.google.com/file/d/1dj-HIH21LIsCDdofuabgE55V4tp3UD_k/view?usp=sharing` | Day 1 Part 2 `SummerSchool_Subset.zip` download via `gdown`. Also appears as an alternate/commented Day 2 link. |
| Google Drive file | `https://drive.google.com/file/d/1TXzxHG1KwcQMyB-2uOHd4WHeaxeQGbjT/view?usp=sharing` | Day 2 `SummerSchool_Subset.zip` download via `gdown`. |
| NASA SAR explainer | `https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar/image-interpretation` | Day 2 background link only. |
| TerraMind | `https://github.com/IBM/terramind` | Day 2 model background link only. |
| TerraTorch TerraMind implementation | `https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_vit.py` | Day 2 implementation reference only. |

The Drive links were not opened or downloaded during this audit.

## Download Commands In Notebooks

Day 1 Part 2:

```python
SHARE_URL = "https://drive.google.com/file/d/1dj-HIH21LIsCDdofuabgE55V4tp3UD_k/view?usp=sharing"
LOCAL_ZIP = BASE_DIR / "SummerSchool_Subset.zip"
DATA_DIR = BASE_DIR / "session_1_data"
DATASET_ROOT = DATA_DIR / "SummerSchool_Subset"
gdown.download(url=SHARE_URL, output=str(LOCAL_ZIP), quiet=False, fuzzy=True)
```

Day 2:

```python
SHARE_URL = "https://drive.google.com/file/d/1TXzxHG1KwcQMyB-2uOHd4WHeaxeQGbjT/view?usp=sharing"
LOCAL_ZIP = BASE_DIR / "SummerSchool_Subset.zip"
DATA_DIR = BASE_DIR / "session_2_data"
DATASET_ROOT = DATA_DIR / "SummerSchool_Subset"
SESSION1_DATASET_ROOT = BASE_DIR / "session_1_data" / "SummerSchool_Subset"
gdown.download(url=SHARE_URL, output=str(partial_path), quiet=False, fuzzy=True)
```

Day 2 first tries to reuse `session_1_data/SummerSchool_Subset` if it exists.

## Provenance Status

The archive appears to contain a curated Lough Neagh 2025 dataset with Sentinel-2 optical layers, Sentinel-1 radar layers, metadata JSON files, derived indices, and segmentation labels. The notebooks do not provide enough information to prove:

- Which Sentinel product IDs were used.
- Exact crop bounds or projection decisions.
- Whether Sentinel-1 and Sentinel-2 were downloaded from Copernicus, Google Earth Engine, Sentinel Hub, ASF, Microsoft Planetary Computer, or another provider.
- Who created the algae segmentation masks.
- Whether masks are manually labelled, model derived, threshold derived, or otherwise curated.
- License and redistribution terms for the prepared archive and labels.
- Whether the Google Drive links remain live or publicly accessible.

## Licensing Uncertainty

Sentinel-1 and Sentinel-2 source imagery is generally open data, but the prepared archive is more than raw Sentinel data. Its labels, crops, metadata packaging, quicklooks, selected tile manifests, and any preprocessing choices are Summer-School-specific unless separately documented.

Do not redistribute the archive, masks, or derived products from this project until explicit license and attribution terms are confirmed.

## Phase 1 Blockers

- `SummerSchool_Subset.zip` is absent locally.
- Drive link status and permissions are unverified.
- Source product IDs, exact crop bounds, and preprocessing provenance are not documented in the local materials.
- Label origin and license are unknown.
- Public reconstruction can likely recreate comparable imagery, but not the exact labelled dataset without the masks and preparation recipe.

