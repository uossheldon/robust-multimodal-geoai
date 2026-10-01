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

The expected dates are:

- Train: `2025-01-01`, `2025-01-31`, `2025-03-12`, `2025-04-08`, `2025-04-09`, `2025-05-16`, `2025-05-18`, `2025-05-21`
- Validation: `2025-06-20`, `2025-08-12`
- Test: `2025-09-08`, `2025-09-21`

For each date, the notebooks expect Sentinel-style optical files `B02`, `B03`, `B04`, `B08`, derived `NDVI`/`NDWI`, SAR `VV`/`VH`, metadata JSON files, and a matching mask named with underscores, such as `2025_01_01.tiff`.

Current local validation:

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

Do not place raw archives, extracted rasters, labels, checkpoints, or generated arrays in git. `data/raw/` and `*.zip` are ignored in `.gitignore`. Before any Phase 2 work, document final source URLs, checksums, licenses, label origin, crop bounds, CRS/grid, nodata policy, acquisition times, and split rules.

See `../docs/DATA_AUDIT.md` and `../docs/DATA_PROVENANCE.md` for the Phase 1 audit.

