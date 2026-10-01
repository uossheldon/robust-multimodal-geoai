# Reproducibility Check

Validation only. No held-out test dates were evaluated.

Training seeds: `[42, 7, 123]`.

| Condition | Original fusion macro mIoU | Modality-dropout macro mIoU |
|---|---:|---:|
| Clean | 0.2592 ± 0.0171 | 0.2725 ± 0.0157 |
| Missing S1 | 0.1539 ± 0.0199 | 0.1886 ± 0.0188 |
| Missing S2 | 0.0275 ± 0.0117 | 0.2104 ± 0.0263 |
| 30% simulated optical occlusion | 0.0947 ± 0.0468 | 0.1870 ± 0.0106 |
| 70% simulated optical occlusion | 0.0334 ± 0.0260 | 0.1620 ± 0.0054 |

Conclusion: Modality dropout improvement is stable across these seeds.
