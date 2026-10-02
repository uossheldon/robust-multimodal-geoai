# Project Scope

## Research question

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

This independent research project was inspired by Newcastle GeoAI Summer School material. It implements four-class algal-bloom segmentation on regional Sentinel-1/Sentinel-2 data; it is not a course-lab reproduction or a publication claim.

## Completed scope

- Rasterio alignment to B04, nearest-neighbour labels, 224×224 non-overlapping tiles and a fixed date split: 621 train / 111 validation / 114 TEST tiles.
- Weighted S2, weighted S1 and naive five-channel DeepLab early fusion.
- Validation diagnosis, modality dropout, simulated optical-occlusion-aware training and three-seed reproducibility.
- Validation-only uncertainty/calibration diagnostics and a three-seed probability ensemble.
- Frozen TerraMind tiny RGB+S1RTC with a lightweight trained decoder, three seeds.
- Frozen Phase 7 comparison on September 8 and 21, without subsequent tuning or checkpoint selection.

Classes: 0 background, 1 low, 2 mid, 3 high algae; 255 ignore. Data/checkpoints are local dependencies, not repository contents.

## Boundaries

No attention/gating, real-cloud benchmark, TerraMind full fine-tuning, TerraMind robustness, calibration fitting or cross-region evaluation is claimed. Date separation prevents same-date tiles crossing sets; repeated geographic coverage remains.

Phase 8 changes documentation and presentation only. Historical Phase 2D test access, differing pixel validity, uncertainty limitations and result provenance are recorded in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md). No statistical-significance test was performed. See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) and [REPRODUCTION.md](REPRODUCTION.md).
