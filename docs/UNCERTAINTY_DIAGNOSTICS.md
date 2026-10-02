# Uncertainty Diagnostics

Historical Phase 5A, **validation only**: frozen original fusion, modality dropout and occlusion-trained fusion, training seeds 42/7/123. No retraining, calibration fitting, MC dropout, ensemble prediction or TEST evaluation occurred in this phase. Later Phase 7 TEST results are separate.

## Calibration and probability metrics

| Method, clean validation | ECE | NLL | Multiclass Brier |
| --- | --- | --- | --- |
| modality_dropout (3-seed mean ± SD) | 0.1491 ± 0.0315 | 1.0248 ± 0.0906 | 0.6127 ± 0.0505 |
| occlusion_training (3-seed mean ± SD) | 0.2019 ± 0.0321 | 1.1407 ± 0.1090 | 0.6688 ± 0.0373 |
| original (3-seed mean ± SD) | 0.1279 ± 0.0093 | 1.0620 ± 0.0137 | 0.6025 ± 0.0120 |

Source: [aggregate_metrics.csv](../results/uncertainty/aggregate_metrics.csv). Values are mean ± sample SD over three training seeds; occlusion here uses fixed corruption seed 101, rather than the three-corruption-seed benchmark average. NLL and multiclass Brier use class probabilities; ECE bins maximum class confidence. Label 255 is excluded everywhere. These are pixel-weighted diagnostics, not independent pixel replications or confidence intervals.

## Occlusion-trained model: entropy and error detection

| Condition | Mean entropy | Correct-pixel entropy | Incorrect-pixel entropy | Entropy error AUROC |
| --- | --- | --- | --- | --- |
| clean | 0.6533 ± 0.0525 | 0.6297 ± 0.0536 | 0.6789 ± 0.0527 | 0.5405 ± 0.0157 |
| occlusion_30 | 0.6542 ± 0.0805 | 0.6205 ± 0.0750 | 0.6975 ± 0.0865 | 0.5737 ± 0.0178 |
| occlusion_50 | 0.6749 ± 0.1101 | 0.6501 ± 0.0995 | 0.7047 ± 0.1225 | 0.5523 ± 0.0385 |
| occlusion_70 | 0.7248 ± 0.0906 | 0.6916 ± 0.0999 | 0.7643 ± 0.0837 | 0.5692 ± 0.0143 |
| missing_s1_zero | 0.9377 ± 0.0532 | 0.9042 ± 0.0554 | 0.9595 ± 0.0525 | 0.5668 ± 0.0454 |
| missing_s2_zero | 0.7872 ± 0.0626 | 0.7333 ± 0.0938 | 0.8277 ± 0.0609 | 0.5887 ± 0.0442 |

Mean entropy rises from clean to severe optical occlusion for the occlusion-trained method, and incorrect pixels have higher mean entropy than correct pixels. The distributions still overlap substantially, as the modest AUROC indicates. The dropout-only model can become more confident under degradation: its mean entropy falls from 0.7520 clean to 0.4648 at 70% occlusion. High entropy alone does not prove calibration, and entropy need not rise whenever performance declines.

## Risk–coverage and limits

Risk–coverage curves remove the highest-entropy pixels and measure the error rate among retained valid pixels. Saved curves/areas are in [risk_coverage.csv](../results/uncertainty/risk_coverage.csv) and [error_detection.csv](../results/uncertainty/error_detection.csv). Lower risk area is preferable, but these exploratory curves do not select an abstention threshold or demonstrate a dependable deployment safeguard. Conditions show different behaviour; failure detection is limited, not solved.

Ensemble analysis is separate in [ENSEMBLE_UNCERTAINTY.md](ENSEMBLE_UNCERTAINTY.md). No uncertainty analysis was extended to TEST in Phase 8. See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) for support and comparison qualifications.
