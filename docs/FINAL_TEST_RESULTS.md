# Final Held-Out Test Results

Phase 7 evaluated the frozen final models after subsequent model development and validation-based selection had finished. No post-Phase-7 tuning or checkpoint selection was performed. Phase 8 reads these saved results only.

## Audit and provenance

- TEST dates: **2025-09-08 (58 tiles), 2025-09-21 (56 tiles)**; total **114**.
- The earlier unweighted S2 Phase 2D TEST evaluation is preserved in `results/s2_deeplab/test_metrics.json`. Phase 7 is the final frozen comparison, not the first historical access to TEST. See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md#test-access-history).
- Checkpoint names and SHA-256 prefixes are recorded in [final_summary.json](../results/final_test/final_summary.json). Validation macro mIoU determined checkpoint selection.
- Clean results come from [clean_test_results.csv](../results/final_test/clean_test_results.csv); robustness from [robustness_test_results.csv](../results/final_test/robustness_test_results.csv) and the saved JSON summary; date results from [per_date_test_results.csv](../results/final_test/per_date_test_results.csv).
- All scientific CSV/JSON records are unchanged by Phase 8. Tables and plots format these saved values.

## Clean held-out TEST

| Model | Runs | Macro mIoU | Macro Dice | Binary algae Dice |
| --- | --- | --- | --- | --- |
| S2 weighted | Single | 0.1698 | 0.2541 | 0.6252 |
| S1 weighted | Single | 0.1756 | 0.2799 | 0.5101 |
| Naive early fusion | Single | 0.1658 | 0.2700 | 0.6136 |
| Modality dropout | 3 seeds | 0.1826 ± 0.0123 | 0.2987 ± 0.0182 | 0.5731 ± 0.0421 |
| Dropout + occlusion training | 3 seeds | 0.1882 ± 0.0293 | 0.3033 ± 0.0380 | 0.6151 ± 0.0289 |
| Frozen TerraMind | 3 seeds | 0.1718 ± 0.0177 | 0.2843 ± 0.0235 | 0.6821 ± 0.0025 |

± denotes sample SD across training seeds 42, 7, 123, not a confidence interval. The three conventional baselines each use one selected run; an absent SD does not mean zero uncertainty.

![Clean TEST comparison](../figures/final_clean_model_comparison.png)

| Model | IoU 0 background | IoU 1 low | IoU 2 mid | IoU 3 high | Binary algae IoU |
| --- | --- | --- | --- | --- | --- |
| S2 weighted | 0.4824 | 0.1054 | 0.0490 | 0.0425 | 0.4547 |
| S1 weighted | 0.2426 | 0.3251 | 0.0028 | 0.1318 | 0.3424 |
| Naive early fusion | 0.1653 | 0.3287 | 0.0265 | 0.1428 | 0.4426 |
| Modality dropout | 0.2764 ± 0.0120 | 0.2402 ± 0.0365 | 0.0475 ± 0.0087 | 0.1662 ± 0.0160 | 0.4024 ± 0.0410 |
| Dropout + occlusion training | 0.2542 ± 0.0721 | 0.3074 ± 0.0403 | 0.0504 ± 0.0172 | 0.1406 ± 0.0042 | 0.4446 ± 0.0299 |
| Frozen TerraMind | 0.2099 ± 0.0166 | 0.2557 ± 0.0604 | 0.0449 ± 0.0094 | 0.1766 ± 0.0275 | 0.5176 ± 0.0029 |

![Per-class IoU](../figures/final_per_class_iou.png)

**Valid-pixel support differs:** S2 evaluates 3,474,284 labelled pixels; S1, fusion and TerraMind evaluate 3,169,093 after SAR-invalid targets are ignored. The same geographic tiles do not imply identical scoring pixels. Mid-algae performance remains weak; binary algae Dice cannot substitute for severity segmentation. No significance test supports ranking claims.

## Frozen robustness protocol on TEST

The evaluator reuses `apply_optical_occlusion` from the Phase 4A implementation, fractions 0/10/30/50/70%, and corruption seeds **101, 202, 303**. Each mask is deterministic from seed, tile ID and fraction. It zeroes normalized S2 channels only; labels and SAR stay fixed. Missing normalized modalities use zeros. TerraMind was evaluated clean only.

| Condition | SD population | Modality dropout mIoU | Occlusion-trained mIoU |
| --- | --- | --- | --- |
| Clean | 3 training seeds | 0.1826 ± 0.0123 | 0.1882 ± 0.0293 |
| 10% occlusion | 9 pooled runs | 0.1840 ± 0.0037 | 0.2173 ± 0.0056 |
| 30% occlusion | 9 pooled runs | 0.1757 ± 0.0040 | 0.2233 ± 0.0131 |
| 50% occlusion | 9 pooled runs | 0.1684 ± 0.0038 | 0.2084 ± 0.0191 |
| 70% occlusion | 9 pooled runs | 0.1630 ± 0.0055 | 0.2122 ± 0.0115 |
| S1 missing | 3 training seeds | 0.1141 ± 0.0166 | 0.1204 ± 0.0264 |
| S2 missing | 3 training seeds | 0.1690 ± 0.0176 | 0.1993 ± 0.0164 |

The occlusion SD is pooled over **3 training seeds × 3 corruption seeds = 9 runs**. Clean and missing-modality SDs describe **3 training seeds**. These are different variability summaries, not nine independent model-training repetitions. Phase 7 CSV `seed` denotes corruption seed on occlusion rows, and training seed on clean/missing rows; training seed is not separately serialized on occlusion rows. Existing pooled summaries are preserved, not relabelled as training-seed-only SDs. The 50% dropout SD rounds to **0.0038** from the final source, superseding an earlier prose rounding of 0.0039.

![TEST robustness](../figures/final_robustness_curves.png)

Occlusion-trained fusion has numerically higher mIoU under the tested optical degradations. The non-monotonic curve does not show that hiding imagery improves underlying information: changed prediction behaviour and class balance under the September distribution are a cautious interpretation, not a demonstrated physical cause. Dropout-only binary algae Dice falls to 0.0327 at 70% occlusion despite macro mIoU of 0.1630, illustrating metric/class-behaviour trade-offs. Complete S1 loss remains difficult for both robust methods.

## Per-date behaviour

| Model | 2025-09-08 mIoU | 2025-09-21 mIoU |
| --- | --- | --- |
| S2 weighted | 0.0914 | 0.2028 |
| S1 weighted | 0.1846 | 0.1639 |
| Naive early fusion | 0.1656 | 0.1540 |
| Modality dropout | 0.1545 ± 0.0264 | 0.1909 ± 0.0032 |
| Dropout + occlusion training | 0.1588 ± 0.0255 | 0.1960 ± 0.0331 |
| Frozen TerraMind | 0.0965 ± 0.0433 | 0.2335 ± 0.0100 |

Multi-seed entries are mean ± sample SD over training seeds. S2 and TerraMind are particularly weak on September 8. Date-wise records cover clean evaluation; per-date corrupted TEST metrics were not saved. Aggregate segmentation metrics pool confusion counts across dates and are not arithmetic averages of the two date scores.

## Validation to TEST

| Model | Runs | Validation macro mIoU | TEST macro mIoU |
| --- | --- | --- | --- |
| S2 weighted | Single | 0.2274 | 0.1698 |
| S1 weighted | Single | 0.2349 | 0.1756 |
| Naive early fusion | Single | 0.2425 | 0.1658 |
| Modality dropout | 3 seeds | 0.2725 ± 0.0157 | 0.1826 ± 0.0123 |
| Dropout + occlusion training | 3 seeds | 0.2648 ± 0.0089 | 0.1882 ± 0.0293 |
| Frozen TerraMind | 3 seeds | 0.2771 ± 0.0138 | 0.1718 ± 0.0177 |

The consistent decline is evidence compatible with substantial temporal/domain shift, not proof of its physical cause. There are only two test dates from the same region. TerraMind validation provenance is documented in [TERRAMIND_REPRODUCIBILITY.md](TERRAMIND_REPRODUCIBILITY.md). No TEST result changed a method, hyperparameter or checkpoint.

## Qualitative examples

![Saved Phase 7 examples](../figures/final_qualitative_test_examples.png)

The existing image is preserved byte-for-byte: first two test tiles (September 8), not examples from both dates. “Robust” is occlusion-trained seed 42; TerraMind is seed 42. S2/S1/fusion are conventional single runs. Colours: dark background, green low, yellow mid, red high, grey ignore. Error-map dark/red/grey means correct/incorrect/ignored under the S2 target mask. That mask differs from SAR-valid scoring support; use the numerical tables for quantitative comparison. No new inference was used to produce Phase 8 presentation updates.
