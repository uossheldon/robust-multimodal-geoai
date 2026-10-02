# Preprocessing

Current pipeline: rasterio alignment, the fixed 846-tile manifest (621 train / 111 validation / 114 TEST), and on-demand source reads. The Phase 2A.1 preview measurements below are historical; model training and the frozen Phase 7 comparison are complete.

## Environment

Local venv: `.venv`

Installed preprocessing packages: `rasterio`, `numpy`, `pandas`, `pillow`.

## Alignment Findings

All inspected rasters are in EPSG:32629. Sentinel-2 optical bands share the B04 grid. Sentinel-1 VV/VH and masks use different dimensions/transforms and must be aligned to B04 before tiling.

Representative `2025-01-01` grids:

| Source | Size | Resolution | Nodata |
|---|---:|---|---|
| Sentinel-2 B04 | 2652 x 3200 | 10.028483 m x 10.024018 m | `0.0` |
| Sentinel-1 VV/VH | 2379 x 2985 | 9.998377 m x 10.000000 m | `-9999.0` |
| Mask | 2559 x 3088 | 10.000000 m x 10.000000 m | `255.0` |

Numerical alignment check for preview dates:

| Date | Source | Method | Same CRS | Same transform | Same width | Same height |
|---|---|---|---:|---:|---:|---:|
| 2025-01-01 | B03 | direct-or-warp | True | True | True | True |
| 2025-01-01 | B02 | direct-or-warp | True | True | True | True |
| 2025-01-01 | VV | bilinear-to-B04 | True | False | False | False |
| 2025-01-01 | VH | bilinear-to-B04 | True | False | False | False |
| 2025-01-01 | mask | nearest-to-B04 | True | False | False | False |
| 2025-06-20 | B03 | direct-or-warp | True | True | True | True |
| 2025-06-20 | B02 | direct-or-warp | True | True | True | True |
| 2025-06-20 | VV | bilinear-to-B04 | True | False | False | False |
| 2025-06-20 | VH | bilinear-to-B04 | True | False | False | False |
| 2025-06-20 | mask | nearest-to-B04 | True | False | False | False |

## Resampling Rules

- B04 is the reference raster grid.
- B03 and B02 are read directly when they already match B04; otherwise they are warped to B04.
- VV and VH are warped to B04 with bilinear resampling.
- Masks are warped to B04 with nearest-neighbour resampling only.
- Mask labels are preserved as integer classes: `0` background, `1` low algae, `2` mid algae, `3` high algae, `255` ignore.

Exact mask values found in preview source masks:

| Date | Values |
|---|---|
| 2025-01-01 | `[0, 1, 2, 3, 255]` |
| 2025-06-20 | `[0, 1, 2, 3, 255]` |

## Tiling Protocol

The baseline protocol is now the original notebook's `224 x 224` non-overlapping B04-grid tiles. This Phase 2A.1 run generated a small preview only: first valid tiles from two representative dates, with valid-pixel ratio at least 20%.

For every candidate tile, optical nodata, SAR nodata/non-finite pixels, and mask value `255` are excluded. The manifest records projected bounds, row/column offsets, valid-pixel ratio, and class statistics.

## Preview Class Balance

Preview tiles: 24 across 2 dates.

| Class | Pixels |
|---|---:|
| Background `0` | 329817 |
| Low algae `1` | 196920 |
| Mid algae `2` | 20451 |
| High algae `3` | 11643 |
| Algae total | 229014 |
| Valid total | 558831 |

## Outputs

- `results/data_summary.csv`
- `results/tile_manifest_preview.csv`
- `figures/data_alignment_check.png`
- `figures/class_distribution.png`

## Final protocol

Phase 2B processed all 12 dates; the fixed manifest retains full 224×224 non-overlapping windows with at least 20% common valid pixels. No duplicated full tile cache is required. Split membership is locked in `configs/split_v1.yaml`; no post-TEST selection changes are allowed.

Manifest validity and runtime scoring support differ: S2 loaders use aligned labels, while SAR/fusion/TerraMind also ignore invalid SAR targets. The resulting unequal TEST pixel counts are disclosed in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md#valid-pixel-comparability); do not treat manifest class counts as each model's evaluation denominator.

## Phase 2B Manifest Note

Phase 2B extended the rasterio pipeline to all 12 dates and wrote `results/tile_manifest.csv` with `846` usable tiles. It still avoids saving duplicated raster tile arrays; the implemented datasets read the source rasters on demand from manifest windows. TerraMind resolves source paths from a configurable data root; legacy DeepLab loaders retain Windows-style manifest strings.
