# S2 Baseline Selection

Phase 2G compares three Sentinel-2-only configurations on the validation split only. The Phase 2D test result remains frozen and was not re-evaluated.

## Balanced-Sampling Rule

Each training tile is assigned to the highest algae severity stratum that is meaningfully present. A class is meaningful when it has at least `128` valid pixels and at least `1.0%` of the tile's valid pixels. The scan order is high, mid, low; tiles with no algae class passing both thresholds are assigned to background/no algae.

Sampler weights use inverse-square-root stratum frequency and are capped so the largest present stratum weight is at most `4.0` times the smallest present stratum weight.

| Stratum | Tiles | Capped sampler weight | Expected sample fraction |
|---|---:|---:|---:|
| background/no algae | 216 | 0.7894 | 0.3008 |
| low | 66 | 1.4282 | 0.1663 |
| mid | 168 | 0.8951 | 0.2653 |
| high | 171 | 0.8873 | 0.2676 |

## Validation Comparison

| Experiment | Macro mIoU | Macro Dice | IoU 0 | IoU 1 | IoU 2 | IoU 3 | Binary algae IoU | Binary algae Dice | Pred C0 | Pred C1 | Pred C2 | Pred C3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2189 | 0.3263 | 0.4972 | 0.1960 | 0.1740 | 0.0085 | 0.3066 | 0.4693 | 0.6782 | 0.2279 | 0.0872 | 0.0067 |
| weighted_loss | 0.2274 | 0.3497 | 0.4229 | 0.2617 | 0.1846 | 0.0406 | 0.3884 | 0.5595 | 0.4946 | 0.3782 | 0.1085 | 0.0186 |
| balanced_sampling | 0.2169 | 0.3216 | 0.5133 | 0.1845 | 0.1561 | 0.0135 | 0.3099 | 0.4732 | 0.6952 | 0.1928 | 0.1031 | 0.0090 |

## Recommendation

Carry forward `weighted_loss` as the conventional optical baseline. It has validation macro mIoU `0.2274`, macro Dice `0.3497`, and binary algae IoU `0.3884`. The recommendation is based on validation performance and class behaviour only.

## Notes

The balanced-sampling run kept the loss unweighted. No SAR, fusion, focal loss, architecture change, or test-set evaluation was used.

## Final status

The validation-only selection above is frozen. Its conventional weighted S2 baseline is a single run; the historical unweighted Phase 2D TEST result is preserved separately. The later [Phase 7 final comparison](FINAL_TEST_RESULTS.md) occurred after subsequent development was frozen, with no post-Phase-7 tuning or checkpoint selection.
