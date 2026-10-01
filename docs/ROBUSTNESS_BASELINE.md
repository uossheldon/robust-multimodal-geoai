# Robustness Baseline

Phase 4A evaluates frozen validation models only. No model was retrained and no held-out test date was evaluated.

## Corruption Definition

The optical corruption is **simulated optical occlusion**, not measured cloud cover. For each validation tile, fixed seed and occlusion fraction, a low-resolution random field is bicubically upsampled and thresholded to produce a spatially coherent irregular mask. The same masks are applied to the S2-only and fusion models for fair comparison. Ground-truth masks and SAR channels are unchanged.

Occluded optical values are set to zero after optical normalization. Zero therefore means the neutral normalized value, not a raw reflectance value. Complete missing-modality fusion conditions also use zero tensors for the missing normalized modality.

## Main Validation Curve

| Model | 0% macro mIoU | 30% macro mIoU | 70% macro mIoU | 70% binary Dice |
|---|---:|---:|---:|---:|
| S2 weighted | 0.2274 ± 0.0000 | 0.1218 ± 0.0153 | 0.0899 ± 0.0078 | 0.5966 ± 0.0093 |
| S1 weighted control | 0.2349 ± 0.0000 | 0.2349 ± 0.0000 | 0.2349 ± 0.0000 | 0.5076 ± 0.0000 |
| S1+S2 fusion | 0.2425 ± 0.0000 | 0.0516 ± 0.0037 | 0.0133 ± 0.0005 | 0.6193 ± 0.0001 |

## Missing Modality

| Condition | Macro mIoU | Macro Dice | Binary algae IoU | Binary algae Dice |
|---|---:|---:|---:|---:|
| clean | 0.2425 | 0.3667 | 0.4010 | 0.5725 |
| missing_s2_zero | 0.0248 | 0.0482 | 0.4462 | 0.6171 |
| missing_s1_zero | 0.1317 | 0.2058 | 0.2495 | 0.3994 |

## Interpretation

The benchmark measures controlled degradation under simulated missing optical information. It does not estimate real cloud performance because cloud physics, shadows, haze and cloud-mask errors are not simulated.
