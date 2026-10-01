# Class Imbalance Experiment

Phase 2F keeps the split, model, RGB preprocessing, augmentations, optimizer, learning rates, batch size and 10-epoch budget fixed. The only full-training change is class-weighted `CrossEntropyLoss`.

## Tiny-Subset Inspection

The original Phase 2E first-16-tile subset had pixel counts:

- class 0: `133224`
- class 1: `361672`
- class 2: `4443`
- class 3: `3150`

Because mid/high were poorly represented, a representative 16-tile subset was selected from training tiles with high class counts:

- class 0: `18982`
- class 1: `16902`
- class 2: `333131`
- class 3: `61465`

This subset was inspected only; it did not change the main experiment.

## Train-Only Class Weights

Inverse-square-root frequency weights, normalized to mean 1:

- class 0: `0.2433`
- class 1: `0.7092`
- class 2: `0.6517`
- class 3: `2.3957`

## Validation Comparison

| Experiment | Macro mIoU | Macro Dice | IoU 0 | IoU 1 | IoU 2 | IoU 3 | Binary algae IoU | Binary algae Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.2189 | 0.3263 | 0.4972 | 0.1960 | 0.1740 | 0.0085 | 0.3066 | 0.4693 |
| Weighted | 0.2274 | 0.3497 | 0.4229 | 0.2617 | 0.1846 | 0.0406 | 0.3884 | 0.5595 |

## Interpretation

Class weighting is a controlled S2-only intervention. It should be judged on validation only. Test results remain frozen from Phase 2D.
