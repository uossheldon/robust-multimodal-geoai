# Baseline Diagnostics

Phase 2E diagnoses the Sentinel-2 DeepLab baseline without retraining the full model and without re-evaluating the test set.

## Learning-Curve Diagnosis

Training loss fell from `0.753` to `0.281` and training macro mIoU rose from `0.377` to `0.530`. Validation macro mIoU stayed low, peaking at `0.219` on epoch `8` while validation loss remained high.

Diagnosis: the model is learning the training set but validation performance plateaus at low macro IoU, especially for high algae. This points to distribution shift and class imbalance more than a simple pipeline failure.

## Train / Validation Class Distribution

| split | valid_pixels | class_0_fraction | class_1_fraction | class_2_fraction | class_3_fraction | algae_fraction |
|---|---|---|---|---|---|---|
| train | 21730003 | 0.789 | 0.0929 | 0.11 | 0.0081 | 0.211 |
| validation | 2966478 | 0.5516 | 0.3892 | 0.0452 | 0.014 | 0.4484 |

## Validation By Date

| date | tiles | macro_miou | macro_dice | iou_background | iou_low | iou_mid | iou_high |
|---|---|---|---|---|---|---|---|
| 2025-01-01 | 44 | 0.1523 | 0.2186 | 0.4883 | 0.1135 | 0.0 | 0.0073 |
| 2025-06-20 | 67 | 0.2358 | 0.3483 | 0.5043 | 0.2495 | 0.1809 | 0.0086 |

## Tiny-Subset Overfit

The model was trained on 16 training tiles for `40` epochs. Final no-augmentation evaluation on those same tiles:

- Macro mIoU: `0.184`
- Macro Dice/F1: `0.254`
- IoU background/low/mid/high: `0.165`, `0.566`, `0.000`, `0.005`

The model can partially memorize the small subset, so the core forward/loss/update path works, but high-class recovery remains weak.

## Binary Diagnostic

Validation predictions collapsed to background versus algae:

- Algae IoU: `0.307`
- Algae Dice/F1: `0.469`
- Binary macro mIoU: `0.402`

This diagnostic does not replace the 4-class task.

## Pipeline Checks

- Mask values after loading and augmentation remain in `[0, 1, 2, 3, 255]`.
- `255` ignore is present and handled by the loss/evaluation path.
- Nearest-neighbour mask alignment avoids interpolated labels.
- RGB reflectance is clipped to `[0, 1]` before ImageNet normalization.
- Augmented image and mask shapes remain identical.

## Recommended Next Controlled Experiment

Before adding SAR or changing architecture, run one S2-only controlled experiment with the same split and model but with class-balanced sampling or loss weighting as the only change. This isolates whether the poor high/low class performance is mainly due to class imbalance.
