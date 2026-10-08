# Scientific audit and interpretation limits

This document records the qualifications needed to interpret the reported results.

## Test access and model selection

An earlier exploratory unweighted Sentinel-2 run was evaluated on the test dates before the final comparison. Later model development and checkpoint selection used validation data, and no tuning or checkpoint selection was performed after the final frozen comparison.

The earlier unweighted optical test result (macro mIoU 0.157647; macro Dice 0.225141) remains in [test_metrics.json](../results/s2_deeplab/test_metrics.json).

The September test set is therefore useful as a fixed temporal benchmark, but it should not be described as a pristine first-touch holdout.

## Evaluation support

The fixed split contains 621 training, 111 validation and 114 test tiles. Test dates are 2025-09-08 and 2025-09-21. Geography overlaps across dates, so the experiment measures temporal generalisation within one region rather than geographic generalisation.

Valid-pixel support differs by modality:

| Model support | September 8 | September 21 | Total |
|---|---:|---:|---:|
| Sentinel-2 only | 1,754,102 | 1,720,182 | 3,474,284 |
| Sentinel-1 / fusion / TerraMind | 1,618,660 | 1,550,433 | 3,169,093 |

Sentinel-1, fusion and TerraMind additionally exclude invalid SAR targets. The headline comparison therefore uses the same tiles but not perfectly identical scoring pixels.

## Variability reporting

| Result type | Reported variability |
|---|---|
| Conventional clean test baselines | One selected run; no seed SD |
| Robust models and TerraMind, clean test | Sample SD across 3 training seeds |
| Missing-modality test results | Sample SD across 3 training seeds |
| Optical-occlusion test results | Pooled SD across 3 training seeds × 3 corruption seeds |

These SDs are descriptive and are not confidence intervals. No statistical-significance test supports a ranking claim.

## Robustness interpretation

Optical corruption is simulated by spatially coherent zeroing of normalized Sentinel-2 channels. It is not measured cloud cover, haze or shadow. Non-monotonic scores under corruption do not imply that removing imagery adds information.

Complete loss of Sentinel-1 remains difficult. Robustness to the tested synthetic corruptions should not be interpreted as deployment robustness.

## Uncertainty interpretation

Probability ensembling improved several validation calibration metrics, while failure-detection AUROC remained modest and condition-dependent. The uncertainty analysis is diagnostic only; it does not establish a reliable abstention or safety mechanism.

## Generalisation limits

The study uses one region, two test dates and strongly imbalanced severity classes. Binary algae detection is easier than four-class severity mapping and should not be treated as an equivalent task.

For the frozen numeric record, see [Final test results](FINAL_TEST_RESULTS.md).
