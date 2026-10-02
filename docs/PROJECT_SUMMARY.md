# Project Summary

## Question and application

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

This independent project, inspired by Newcastle GeoAI Summer School material, studies four-class algal-bloom mapping with Sentinel-2 B04/B03/B02 and Sentinel-1 VV/VH. Classes are background/low/mid/high algae (0/1/2/3), with 255 ignored. Rasterio alignment uses B04 as the reference grid and nearest-neighbour masks. The fixed manifest contains 621 training, 111 validation and 114 TEST tiles of 224×224 pixels.

## Research narrative

Naive fusion was brittle: one selected model fell from 0.2425 clean validation mIoU to 0.0133 at 70% simulated optical occlusion. Modality dropout reduced complete missing-sensor failures, while three-seed optical-occlusion-aware training raised validation mIoU at 70% occlusion from 0.1620 ± 0.0054 to 0.2547 ± 0.0161. On September TEST dates, occlusion-trained fusion had the numerically highest clean macro mIoU, 0.1882 ± 0.0293, and retained numerically higher macro performance under optical degradation than dropout alone. Frozen TerraMind achieved validation mIoU 0.2771 ± 0.0138 and the highest TEST binary algae Dice, 0.6821 ± 0.0025, but TEST four-class mIoU was 0.1718 ± 0.0177. Every model lost four-class performance from validation to TEST, consistent with substantial temporal/domain shift. Ensembling reduced calibration errors, but uncertainty-based failure detection remained modest and inconsistent for missing sensors.

## Headline final TEST results

| Model | Runs | Macro mIoU | Macro Dice | Binary algae Dice |
| --- | --- | --- | --- | --- |
| S2 weighted | Single | 0.1698 | 0.2541 | 0.6252 |
| S1 weighted | Single | 0.1756 | 0.2799 | 0.5101 |
| Naive early fusion | Single | 0.1658 | 0.2700 | 0.6136 |
| Modality dropout | 3 seeds | 0.1826 ± 0.0123 | 0.2987 ± 0.0182 | 0.5731 ± 0.0421 |
| Dropout + occlusion training | 3 seeds | 0.1882 ± 0.0293 | 0.3033 ± 0.0380 | 0.6151 ± 0.0289 |
| Frozen TerraMind | 3 seeds | 0.1718 ± 0.0177 | 0.2843 ± 0.0235 | 0.6821 ± 0.0025 |

Clean ± values are training-seed sample SDs (42, 7, 123); single-run baselines have no estimated training-seed SD. TEST uses September 8 and 21. The final Phase 7 comparison followed frozen development; an earlier unweighted S2 Phase 2D test access is retained and disclosed. No post-Phase-7 tuning or checkpoint selection occurred.

## Robustness and generalisation

| Condition | SD population | Modality dropout mIoU | Occlusion-trained mIoU |
| --- | --- | --- | --- |
| Clean | 3 training seeds | 0.1826 ± 0.0123 | 0.1882 ± 0.0293 |
| 10% occlusion | 9 pooled runs | 0.1840 ± 0.0037 | 0.2173 ± 0.0056 |
| 30% occlusion | 9 pooled runs | 0.1757 ± 0.0040 | 0.2233 ± 0.0131 |
| 50% occlusion | 9 pooled runs | 0.1684 ± 0.0038 | 0.2084 ± 0.0191 |
| 70% occlusion | 9 pooled runs | 0.1630 ± 0.0055 | 0.2122 ± 0.0115 |
| S1 missing | 3 training seeds | 0.1141 ± 0.0166 | 0.1204 ± 0.0264 |
| S2 missing | 3 training seeds | 0.1690 ± 0.0176 | 0.1993 ± 0.0164 |

Optical-occlusion rows pool nine training-seed × corruption-seed runs. Their SDs are not interchangeable with clean/missing-condition SDs over three training seeds. Synthetic occlusion is not real cloud cover, and non-monotonic scores do not imply that removed information improves imagery.

| Model | Runs | Validation macro mIoU | TEST macro mIoU |
| --- | --- | --- | --- |
| S2 weighted | Single | 0.2274 | 0.1698 |
| S1 weighted | Single | 0.2349 | 0.1756 |
| Naive early fusion | Single | 0.2425 | 0.1658 |
| Modality dropout | 3 seeds | 0.2725 ± 0.0157 | 0.1826 ± 0.0123 |
| Dropout + occlusion training | 3 seeds | 0.2648 ± 0.0089 | 0.1882 ± 0.0293 |
| Frozen TerraMind | 3 seeds | 0.2771 ± 0.0138 | 0.1718 ± 0.0177 |

Temporal split safety does not establish geographic generalisation. The held-out dates cover only September in the same region. S2 scores more valid pixels (3,474,284) than SAR-based/fusion/TerraMind evaluation (3,169,093), qualifying comparisons. No statistical-significance claim is made.

## Deliverables and limits

Implemented source includes aligned manifest datasets, controlled DeepLab experiments, fixed corruption masks, three-seed evaluation, uncertainty/risk–coverage diagnostics, a frozen TerraMind decoder and portable Colab data-root handling. [REPRODUCTION.md](REPRODUCTION.md) maps these components. [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) records historical test access, aggregation, support and provenance qualifications.

The dataset is small and severely imbalanced, especially high algae. Binary detection remains distinct from severity quality. TerraMind is frozen, not fully fine-tuned; uncertainty is diagnostic, not a reliable safety mechanism. Raw data, original TIFFs/archives and checkpoints are not distributed. Archive/label provenance and permission for public data-derived figures remain unresolved; public release needs that review.
