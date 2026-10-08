# Data provenance and availability

This independent research project uses data and teaching materials provided through the Newcastle University GeoAI Summer School. The original prepared dataset is not redistributed.

## Dataset used

The supplied subset contains 12 acquisition dates over Lough Neagh. Each date includes Sentinel-2 optical layers (B02/B03/B04/B08, NDVI and NDWI), Sentinel-1 VV/VH layers, metadata and a segmentation mask. The implemented models use Sentinel-2 RGB (B04/B03/B02) and Sentinel-1 VV/VH.

The project keeps a fixed date-level split and an 846-tile manifest. See [Methods](METHODS.md) and the [manifest](../results/tile_manifest.csv) for the exact experimental setup.

Representative processing metadata:

- Sentinel-2: Level-2A optical data, with digital-number scaling handled in preprocessing.
- Sentinel-1: terrain-corrected VV/VH backscatter in dB.
- Output CRS: EPSG:32629.
- Optical nodata: 0.
- SAR nodata: -9999.
- Mask ignore label: 255.

## Project contribution

The repository contains independently implemented alignment and preprocessing code, manifest-based data loading, four-class segmentation experiments, robustness interventions, TerraMind benchmarking, uncertainty analysis, aggregate figures and the project website.

Pretrained architectures and weights remain external dependencies. Their upstream terms apply separately.

## Public availability

The repository does not redistribute source rasters, original masks, the prepared archive, model checkpoints, caches or raster-derived qualitative imagery.

Comparable Sentinel imagery can be obtained independently from public Earth-observation sources, but exact reproduction also requires the original prepared labels and source-specific preprocessing choices. Those labels are not reconstructable from the public Sentinel imagery alone.

See [Reproduction](REPRODUCTION.md) for the supported public workflow.

Public code and aggregate results do not grant redistribution rights to the prepared dataset or labels. Their redistribution terms remain unresolved; no software or data license is added by this repository.
