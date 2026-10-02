# Data Directory

Phase 1.5 downloaded and extracted the Day 1/2 source archive locally. Do not upload, commit, or redistribute it.

The Day 1 Part 2 and Day 2 workflows expect an external archive named `SummerSchool_Subset.zip`, extracted as:

```text
data/raw/
  SummerSchool_Subset.zip
  SummerSchool_Subset/
    images/<YYYY-MM-DD>/layers/
    masks/
```

The project uses the locked date split in `../configs/split_v1.yaml` (621 train / 111 validation / 114 TEST tiles). The earlier notebook split is superseded:

- Train: `2025-01-31`, `2025-03-12`, `2025-04-08`, `2025-04-09`, `2025-05-16`, `2025-05-18`, `2025-05-21`, `2025-08-12`
- Validation: `2025-01-01`, `2025-06-20`
- Test: `2025-09-08`, `2025-09-21`

For each date, the notebooks expect Sentinel-style optical files `B02`, `B03`, `B04`, `B08`, derived `NDVI`/`NDWI`, SAR `VV`/`VH`, metadata JSON files, and a matching mask named with underscores, such as `2025_01_01.tiff`.

Recorded Phase 1.5 local validation:

- Archive size: 1,971,737,334 bytes, about 1.836 GiB.
- Extracted size: 2,144,997,936 bytes, about 1.998 GiB.
- Dates: 12.
- Raster layers: 96.
- Masks: 12.
- Missing expected files: none.
- Representative CRS: EPSG:32629.
- Optical example: uint16 DN, 10 m scale, nodata `0`.
- SAR example: float32 dB, nodata `-9999`.
- Mask example: uint8 labels, nodata/ignore `255`.

Do not place raw archives, extracted rasters, labels, checkpoints, or generated arrays in git. `data/raw/` and `*.zip` are ignored in `.gitignore`. The source archive/label license and permission for publishing derived imagery remain unresolved; see the final provenance qualifications. No source-data license is assumed.

See [data audit](../docs/DATA_AUDIT.md), [provenance](../docs/DATA_PROVENANCE.md) and [reproduction](../docs/REPRODUCTION.md). TerraMind Colab uses `--data-root /content/geoai_data/SummerSchool_Subset`. Phase 7 final TEST results are complete; Phase 8 does not read rasters or rerun inference.

