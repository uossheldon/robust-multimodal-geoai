# Final test results

The final frozen evaluation compares all selected models on the two September test dates after validation-based development had finished. No tuning or checkpoint selection was performed after this comparison.

## Test set

- Dates: **2025-09-08** and **2025-09-21**
- Tiles: **114** total
- Same geographic region as development data
- Temporal rather than geographic generalisation

An earlier exploratory unweighted Sentinel-2 run had already been evaluated on these dates. The final comparison is therefore not described as a pristine first-touch holdout; see [Scientific audit](SCIENTIFIC_AUDIT.md).

## Clean test performance

| Model | Runs | Macro mIoU | Macro Dice | Binary algae Dice |
|---|---|---:|---:|---:|
| Sentinel-2 weighted | 1 | 0.1698 | 0.2541 | 0.6252 |
| Sentinel-1 weighted | 1 | 0.1756 | 0.2799 | 0.5101 |
| Early fusion | 1 | 0.1658 | 0.2700 | 0.6136 |
| Modality dropout | 3 seeds | 0.1826 ± 0.0123 | 0.2987 ± 0.0182 | 0.5731 ± 0.0421 |
| Occlusion-trained fusion | 3 seeds | **0.1882 ± 0.0293** | **0.3033 ± 0.0380** | 0.6151 ± 0.0289 |
| Frozen TerraMind | 3 seeds | 0.1718 ± 0.0177 | 0.2843 ± 0.0235 | **0.6821 ± 0.0025** |

![Clean test comparison](../figures/final_clean_model_comparison.png)

The robust methods and TerraMind report sample SD across training seeds 42, 7 and 123. Conventional baselines are single selected runs.

## Per-class IoU

| Model | Background | Low algae | Mid algae | High algae |
|---|---:|---:|---:|---:|
| Sentinel-2 weighted | 0.4824 | 0.1054 | 0.0490 | 0.0425 |
| Sentinel-1 weighted | 0.2426 | 0.3251 | 0.0028 | 0.1318 |
| Early fusion | 0.1653 | 0.3287 | 0.0265 | 0.1428 |
| Modality dropout | 0.2764 ± 0.0120 | 0.2402 ± 0.0365 | 0.0475 ± 0.0087 | 0.1662 ± 0.0160 |
| Occlusion-trained fusion | 0.2542 ± 0.0721 | 0.3074 ± 0.0403 | 0.0504 ± 0.0172 | 0.1406 ± 0.0042 |
| Frozen TerraMind | 0.2099 ± 0.0166 | 0.2557 ± 0.0604 | 0.0449 ± 0.0094 | 0.1766 ± 0.0275 |

![Per-class IoU](../figures/final_per_class_iou.png)

Mid-algae segmentation remains weak across all methods. Binary algae detection should therefore not be treated as equivalent to four-class severity mapping.

## Robustness

| Condition | Modality dropout mIoU | Occlusion-trained mIoU |
|---|---:|---:|
| Clean | 0.1826 ± 0.0123 | 0.1882 ± 0.0293 |
| 10% optical occlusion | 0.1840 ± 0.0037 | 0.2173 ± 0.0056 |
| 30% optical occlusion | 0.1757 ± 0.0040 | 0.2233 ± 0.0131 |
| 50% optical occlusion | 0.1684 ± 0.0038 | 0.2084 ± 0.0191 |
| 70% optical occlusion | 0.1630 ± 0.0055 | 0.2122 ± 0.0115 |
| Sentinel-1 missing | 0.1141 ± 0.0166 | 0.1204 ± 0.0264 |
| Sentinel-2 missing | 0.1690 ± 0.0176 | 0.1993 ± 0.0164 |

![Robustness comparison](../figures/final_robustness_curves.png)

Optical-occlusion SD pools 3 training seeds × 3 corruption seeds. Clean and missing-modality SDs are across 3 training seeds. These variability estimates describe different populations and are not confidence intervals.

The corruption experiment uses simulated spatial occlusion of normalized optical channels, not measured cloud cover.

## Temporal shift

| Model | Validation macro mIoU | Test macro mIoU |
|---|---:|---:|
| Sentinel-2 weighted | 0.2274 | 0.1698 |
| Sentinel-1 weighted | 0.2349 | 0.1756 |
| Early fusion | 0.2425 | 0.1658 |
| Modality dropout | 0.2725 ± 0.0157 | 0.1826 ± 0.0123 |
| Occlusion-trained fusion | 0.2648 ± 0.0089 | 0.1882 ± 0.0293 |
| Frozen TerraMind | 0.2771 ± 0.0138 | 0.1718 ± 0.0177 |

All six models lose four-class performance on the September dates. The result is consistent with substantial temporal/domain shift within the study region.

No statistical-significance test is used to rank the models. Valid-pixel support also differs between Sentinel-2-only and SAR-dependent methods; see [Scientific audit](SCIENTIFIC_AUDIT.md).
