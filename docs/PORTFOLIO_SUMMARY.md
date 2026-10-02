# Portfolio Summary

## Two-sentence description

Built an independent Sentinel-1/Sentinel-2 research pipeline for four-class algal-bloom segmentation, testing optical degradation and missing sensors with controlled DeepLab fusion experiments and a frozen TerraMind benchmark. Three-seed experiments and a frozen final comparison on two September dates quantified robustness gains alongside substantial temporal generalisation failure and limited uncertainty-based error detection.

## Three CV bullets

- Implemented rasterio-aligned, manifest-based Sentinel-1/Sentinel-2 segmentation with a fixed date split and 224×224 tiles, comparing optical, SAR and five-channel fusion baselines.
- Evaluated modality dropout and simulated optical-occlusion training across three seeds; at 70% occlusion on TEST, occlusion-trained fusion reached 0.2122 ± 0.0115 macro mIoU versus 0.1630 ± 0.0055 for dropout alone (SD pooled over nine training-seed × corruption-seed runs).
- Benchmarked frozen TerraMind RGB+S1RTC with a lightweight decoder: TEST binary algae Dice 0.6821 ± 0.0025 across three training seeds, while documenting weaker four-class temporal generalisation and moderate uncertainty error detection.

## Approximately 60-second interview explanation

I asked whether combining radar and optical satellite imagery makes algae segmentation resilient when optical information is degraded or a sensor is unavailable. I aligned Sentinel-1 and Sentinel-2 on the same raster grid, preserved discrete labels, and used a fixed date-level split. Naive fusion was brittle, so I tested modality dropout and then coherent simulated optical occlusion during training, keeping the other DeepLab settings fixed. Across three seeds, occlusion-trained fusion retained numerically higher performance under optical degradation than dropout alone. I also benchmarked a frozen TerraMind backbone with a small decoder. TerraMind had the highest binary algae Dice on the September test dates, but robust DeepLab had numerically higher four-class mIoU. All methods lost four-class performance relative to validation. I documented earlier Phase 2D test access, unequal valid-pixel support and modest uncertainty detection, rather than claiming universal superiority or reliable deployment.

## Key quantitative results

| Finding | Result and population |
|---|---|
| Frozen data split | 621 train / 111 validation / 114 TEST tiles; 12 dates |
| Naive fusion brittleness | Validation mIoU 0.2425 clean → 0.0133 at 70% occlusion; one trained model |
| Dropout clean TEST | mIoU 0.1826 ± 0.0123; 3 training seeds |
| Occlusion-trained clean TEST | mIoU 0.1882 ± 0.0293; 3 training seeds |
| Frozen TerraMind clean TEST | mIoU 0.1718 ± 0.0177; binary algae Dice 0.6821 ± 0.0025; 3 training seeds |
| Temporal shift, robust fusion | Validation mIoU 0.2648 ± 0.0089 → TEST 0.1882 ± 0.0293; 3 training seeds |
| Limited failure detection | Ensemble disagreement AUROC 0.5984 under 70% validation occlusion |

± denotes sample SD, not confidence intervals or significance. TEST dates are September 8 and 21. The final Phase 7 comparison followed frozen model development; earlier Phase 2D access remains part of the record. No post-Phase-7 tuning occurred. S2 and SAR-based methods do not score identical valid-pixel support.

## What I personally implemented

The following are project contributions evidenced by repository source; pretrained architectures, pretrained weights and Summer School source labels are external resources:

- Raster alignment, label-safe windows and manifest datasets: `src/data/`.
- DeepLab input adapters and controlled weighted-loss/dropout/occlusion experiments: `src/models/deeplab.py`, `src/training/`.
- Deterministic simulated occlusion, segmentation, calibration and ensemble diagnostics: `src/evaluation/`.
- TerraMind dataset/preprocessing adapter, frozen-backbone decoder and Colab workflow: `src/data/terramind_dataset.py`, `src/models/terramind_segmentation.py`, `notebooks/TerraMind_Colab.ipynb`.
- Result synthesis, temporal analysis, scientific qualifications and reproducible presentation from saved summaries.

This is an independent research project, not a publication or state-of-the-art claim. Synthetic occlusion is not measured cloud cover; binary algae detection is not four-class severity quality. Source data/checkpoints are excluded. Review label/archive and derived-figure rights before public portfolio release. See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md).
