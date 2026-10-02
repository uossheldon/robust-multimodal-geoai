# Ensemble Uncertainty

Historical Phase 5B, **validation only**, using the three frozen occlusion-trained models (42/7/123). Class probabilities are averaged before predicting. No calibration fitting, retraining or TEST inference occurred in this phase; later final TEST results do not extend this uncertainty experiment.

## Calibration comparison

| Condition | Individual mean ECE | Ensemble ECE | Individual mean NLL | Ensemble NLL | Individual mean Brier | Ensemble Brier |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.2019 | 0.1757 | 1.1407 | 1.0680 | 0.6688 | 0.6436 |
| occlusion_30 | 0.1609 | 0.1323 | 1.0261 | 0.9499 | 0.6017 | 0.5732 |
| occlusion_50 | 0.1706 | 0.1363 | 1.0562 | 0.9450 | 0.6226 | 0.5812 |
| occlusion_70 | 0.1395 | 0.1072 | 1.0199 | 0.9321 | 0.6021 | 0.5635 |
| missing_s1_zero | 0.2255 | 0.1730 | 1.3485 | 1.2588 | 0.7634 | 0.7272 |
| missing_s2_zero | 0.1968 | 0.1231 | 1.1834 | 1.0611 | 0.6935 | 0.6331 |

Individual columns are three-seed mean softmax diagnostics from Phase 5A; ensemble columns are metrics of one averaged probability distribution, not means of individual metrics. Clean validation ECE falls from 0.2019 to 0.1757, NLL from 1.1407 to 1.0680, and Brier from 0.6688 to 0.6436. Calibration improvement does not establish reliable failure detection.

## Error detection

| Condition | Seed-42 entropy score | Ensemble entropy | Mutual information | Disagreement |
| --- | --- | --- | --- | --- |
| clean | 0.5418 | 0.5355 | 0.5357 | 0.5472 |
| occlusion_30 | 0.5782 | 0.5797 | 0.5487 | 0.5814 |
| occlusion_50 | 0.5442 | 0.5662 | 0.6080 | 0.6140 |
| occlusion_70 | 0.5607 | 0.5728 | 0.5891 | 0.5984 |
| missing_s1_zero | 0.6142 | 0.5384 | 0.3827 | 0.3897 |
| missing_s2_zero | 0.6016 | 0.5944 | 0.5335 | 0.5402 |

Source: [error_detection.csv](../results/ensemble_uncertainty/error_detection.csv). **All four scores here detect errors of the ensemble prediction.** The stored `single_entropy` column is seed 42's entropy scored against ensemble correctness, not that model's own correctness. Phase 5A contains separate self-error metrics. This limits direct claims that ensembling improves detection relative to a standalone model. The corruption mask seed is 101, shared with Phase 5A.

Disagreement and mutual information are somewhat more informative under severe optical occlusion: at 70%, disagreement AUROC is 0.5984 versus ensemble-entropy AUROC 0.5728. These remain modest. Missing S1 gives disagreement AUROC 0.3897, demonstrating inconsistent failure identification; neither reliable uncertainty nor a calibrated safety score is claimed.

## Risk–coverage and entropy

At 70% occlusion, risk–coverage area is 0.3779 using disagreement versus 0.3961 using ensemble entropy (lower is preferable). With S1 missing, disagreement's area is 0.6802 versus 0.5899 for ensemble entropy. Abstention benefits depend on condition; no threshold was fitted.

Ensemble predictive entropy increases from 0.6978 clean to 0.7844 at 70% occlusion. Mutual information is predictive entropy minus mean individual entropy; disagreement is summed class-probability variance. These quantify model diversity, not proof of accurate uncertainty. Incorrect-pixel entropy exceeds correct-pixel entropy on average, with considerable overlap.

See [metrics.csv](../results/ensemble_uncertainty/metrics.csv), [risk_coverage.csv](../results/ensemble_uncertainty/risk_coverage.csv), and the [scientific audit](SCIENTIFIC_AUDIT.md). The project reports uncertainty diagnostics as secondary evidence, not a solution to reliable failure detection.
