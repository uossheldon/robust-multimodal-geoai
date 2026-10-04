# Data and source provenance

This independent research project was developed using data and teaching materials provided through the Newcastle University GeoAI Summer School. The original dataset is not redistributed.

## External materials and independent work

The external Summer School materials supplied the prepared Lough Neagh dataset and Day 1/Day 2 teaching notebooks. The notebook examples established optical/SAR preprocessing and segmentation context. This repository did not create or collect the source Sentinel data or original masks.

Independent project work comprises the alignment and preprocessing implementation, manifest-based data pipelines, four-class segmentation experiments, controlled robustness experiments, TerraMind adapter/benchmark, uncertainty and ensemble diagnostics, result analysis, aggregate plots, schematic and project website.

Pretrained architectures and weights are external dependencies: torchvision's DeepLabV3-MobileNetV3-Large and IBM/ESA TerraMind tiny through TerraTorch. Their respective upstream terms apply separately; they do not establish a license for the prepared dataset or this repository.

## Archive provenance and inventory

The locally validated `SummerSchool_Subset.zip` contained 12 date folders, 96 raster layers (B02/B03/B04/B08/NDVI/NDWI/VV/VH for each date), 12 masks, 12 optical metadata files, 12 SAR metadata files and 12 quicklooks. Archive size was 1,971,737,334 bytes; extracted size was 2,144,997,936 bytes. ZIP integrity passed with 171 members and no missing expected input files.

The Day 1 Part 2 source was [Google Drive file 1dj-HIH21LIsCDdofuabgE55V4tp3UD_k](https://drive.google.com/file/d/1dj-HIH21LIsCDdofuabgE55V4tp3UD_k/view), used successfully for local retrieval on 2026-09-30. The alternate Day 2 reference was [file 1TXzxHG1KwcQMyB-2uOHd4WHeaxeQGbjT](https://drive.google.com/file/d/1TXzxHG1KwcQMyB-2uOHd4WHeaxeQGbjT/view); it was unavailable at retrieval. These are source references, not permission to redistribute or guarantees of continuing access.

Representative metadata recorded:

- Optical product: `S2B_MSIL2A_20250101T113409_N0511_R080_T29UPA_20250101T134107`.
- SAR product: `S1A_IW_GRDH_1SDV_20250101T063857_20250101T063922_057247_070ABF_AB64_COG`.
- Optical process endpoint: `https://sh.dataspace.copernicus.eu`; STAC: `https://stac.dataspace.copernicus.eu/v1/search`.
- Optical units: DN; SAR: GAMMA0_TERRAIN in dB, nodata -9999; output CRS EPSG:32629.
- Optical nodata is 0; masks use 255. The exact aligned project pipeline and fixed split are in [Methods](METHODS.md).

The original teaching split is not the project's final split. Use [split_v1.yaml](../configs/split_v1.yaml) and the fixed manifest, not notebook-era subset selections.

## Availability and unresolved rights

Source TIFFs, original masks/labels, archives, checkpoints and model caches are not redistributed. Nine raster-derived qualitative/alignment images were removed from the current repository and reachable public history before release. The public repository and [live website](https://uossheldon.github.io/robust-multimodal-geoai/) contain independent code, aggregate results, schematics and reviewed numeric plots.

Prepared archive/label licensing, mask authorship and preparation history, and publication rights for withheld raster-derived imagery remain unresolved. No permission to publish those images is inferred. No software license is invented or added. Upstream Sentinel availability does not clear the curated archive or labels.

Comparable optical/SAR imagery may be obtainable independently from public Sentinel sources, but exact reproduction additionally needs product selection, orbit/terrain-correction details, crops/grids and the original labels. Those masks cannot be reconstructed from public Sentinel imagery alone. NDVI/NDWI reproduction also needs the original formulas, scaling and source bands. Acquire the external data under its applicable terms; see [data layout](../data/README.md) and [Reproduction](REPRODUCTION.md).
