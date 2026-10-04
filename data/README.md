# External data

The original Newcastle University GeoAI Summer School dataset is not distributed. Obtain it separately under its source terms; see [Data provenance](../docs/DATA_PROVENANCE.md).

Default local layout:

```text
data/raw/SummerSchool_Subset/
  images/<YYYY-MM-DD>/layers/
    <YYYY-MM-DD>_B02.tif
    <YYYY-MM-DD>_B03.tif
    <YYYY-MM-DD>_B04.tif
    <YYYY-MM-DD>_VV.tif
    <YYYY-MM-DD>_VH.tif
    metadata.json
    metadata_s1.json
  masks/<YYYY_MM_DD>.tiff
```

The source archive also contains B08, NDVI, NDWI and quicklooks; the implemented RGB/SAR models use the five listed bands. Preserve all 12 dates, the existing [tile manifest](../results/tile_manifest.csv), and [fixed split](../configs/split_v1.yaml).

TerraMind supports `--data-root /content/geoai_data/SummerSchool_Subset` in Colab. Legacy DeepLab loaders expect the original project-local paths. [Methods](../docs/METHODS.md) defines alignment and labels; [Reproduction](../docs/REPRODUCTION.md) explains environment limits.

Raw data, masks, archives, trained checkpoints and model caches remain ignored and must not be committed. Raster-derived imagery is withheld because publication rights remain unresolved.
