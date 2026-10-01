# Modality Dropout Experiment

Phase 4B1 trains the existing five-channel early-fusion DeepLab architecture with one training change: deterministic sample-level modality dropout after normalization.

Training probabilities:

- 50% keep S1 + S2
- 25% zero S1 channels, S2 only
- 25% zero S2 channels, S1 only
- 0% drop both modalities

No partial optical occlusion was used during training. No test dates were evaluated.

## Result Summary

- Best epoch: `6`
- Training time: `389.0` seconds
- Peak VRAM: `422.3` MB

Clean macro mIoU changed from `0.2425` to `0.2905`.

Missing S1 macro mIoU changed from `0.1317` to `0.1675`.

Missing S2 macro mIoU changed from `0.0248` to `0.2407`.

## Interpretation

This experiment tests robustness to complete missing modalities only. The optical occlusion benchmark is diagnostic because the model was not trained with partial optical occlusion.
