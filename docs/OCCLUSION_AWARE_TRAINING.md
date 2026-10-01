# Occlusion-Aware Training

Phase 4B2 keeps the five-channel early-fusion architecture and modality-dropout probabilities fixed. The only new training change is simulated optical occlusion on half of intact both-modality training samples, with occlusion fraction sampled uniformly from 10% to 70%.

These masks are simulated optical occlusion, not real clouds. No test dates were evaluated.

Training seeds: `[42, 7, 123]`.

| Condition | Original | Modality dropout | Modality dropout + occlusion training |
|---|---:|---:|---:|
| Clean | 0.2592 ± 0.0171 | 0.2725 ± 0.0157 | 0.2648 ± 0.0089 |
| Missing S1 | 0.1539 ± 0.0199 | 0.1886 ± 0.0188 | 0.1746 ± 0.0305 |
| Missing S2 | 0.0275 ± 0.0117 | 0.2104 ± 0.0263 | 0.2130 ± 0.0132 |
| 10% simulated optical occlusion | 0.1671 ± 0.0191 | 0.2170 ± 0.0139 | 0.2621 ± 0.0104 |
| 30% simulated optical occlusion | 0.0947 ± 0.0468 | 0.1870 ± 0.0106 | 0.2729 ± 0.0050 |
| 50% simulated optical occlusion | 0.0539 ± 0.0457 | 0.1702 ± 0.0049 | 0.2632 ± 0.0063 |
| 70% simulated optical occlusion | 0.0334 ± 0.0260 | 0.1620 ± 0.0054 | 0.2547 ± 0.0161 |

Conclusion: Partial-occlusion training improves simulated optical-occlusion macro mIoU beyond modality dropout, with a small clean macro mIoU decrease.
