# Reproducibility Check

Historical Phase 4B1-R validation report. No TEST evaluation occurred in this phase; the later frozen Phase 7 comparison is complete. The earlier Phase 2D access is disclosed in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md).

Training seeds: `[42, 7, 123]`.

| Condition | Original fusion macro mIoU | Modality-dropout macro mIoU |
|---|---:|---:|
| Clean | 0.2592 ± 0.0171 | 0.2725 ± 0.0157 |
| Missing S1 | 0.1539 ± 0.0199 | 0.1886 ± 0.0188 |
| Missing S2 | 0.0275 ± 0.0117 | 0.2104 ± 0.0263 |
| 30% simulated optical occlusion | 0.0947 ± 0.0468 | 0.1870 ± 0.0106 |
| 70% simulated optical occlusion | 0.0334 ± 0.0260 | 0.1620 ± 0.0054 |

Conclusion: Modality dropout improvement is stable across these seeds.

## Aggregation scope

These are three-training-seed validation means ± sample SD (42, 7, 123); occlusion metrics are averaged over the fixed corruption seeds within each training run. Original fusion here is a three-seed mean (clean 0.2592), distinct from the selected single-run baseline (0.2425). In the frozen final TEST comparison, clean/missing conditions use training-seed SD, while optical occlusion uses pooled SD over nine training-seed × corruption-seed runs. See [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md). Numerical differences do not establish statistical significance.
