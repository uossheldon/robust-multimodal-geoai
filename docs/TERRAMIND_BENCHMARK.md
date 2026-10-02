# TerraMind Frozen Benchmark

Phase: 6C  
Status: completed across seeds 42, 7, 123. This document records the frozen benchmark design; [TERRAMIND_REPRODUCIBILITY.md](TERRAMIND_REPRODUCIBILITY.md) provides completed validation evidence and [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md) the later frozen clean TEST comparison.

## Benchmark design

The benchmark uses the validated Colab TerraMind setup:

- GPU: Google Colab T4
- TerraTorch: `1.2.13`
- TorchGeo: `0.9.0`
- Backbone: `terramind_v1_tiny`
- Modalities: `RGB` and `S1RTC`
- Input tile size: `224x224`
- TerraMind output used by the decoder: last feature output with shape `[B, 196, 192]`

The backbone is frozen. Only the segmentation decoder is trained.

## Fixed project choices

The experiment keeps the previous project setup fixed:

- Date-level split: `configs/split_v1.yaml`
- Expected manifest counts: 621 train tiles and 111 validation tiles
- Test dates excluded from all Phase 6C loaders: `2025-09-08`, `2025-09-21`
- Manifest: `results/tile_manifest.csv`
- Classes: `0 background`, `1 low algae`, `2 mid algae`, `3 high algae`
- Ignore label: `255`
- Model selection: validation macro mIoU
- TEST was excluded during Phase 6C training/selection; frozen final evaluation occurred in Phase 7.

## Preprocessing

The TerraMind dataset reads the aligned rasters listed in the manifest. It does not reuse DeepLab-normalized tensors.

### RGB

Input bands are B04/B03/B02. They are converted as follows:

1. Divide by `10000` to get reflectance.
2. Clip to `[0, 1]`.
3. Reorder RGB to BGR.
4. Multiply by `255`.

### S1RTC

Input bands are VV/VH in GAMMA0_TERRAIN dB.

- Nodata: `-9999`
- Non-finite and nodata pixels are excluded through the target mask.
- Invalid SAR values are filled with the TerraMind mean before normalization.
- TerraMind normalization:
  - mean `[-10.930, -17.329]`
  - std `[4.391, 4.459]`

## Decoder design

The decoder receives the last TerraMind feature output:

`[B, 196, 192] -> [B, 192, 14, 14]`

It then applies a lightweight convolutional upsampling head:

- Conv 192→128, GroupNorm, GELU, upsample ×2
- Conv 128→96, GroupNorm, GELU, upsample ×2
- Conv 96→64, GroupNorm, GELU, upsample ×2
- Conv 64→32, GroupNorm, GELU, upsample ×2
- Dropout2d `0.10`
- Conv 32→4

Output shape is `[B, 4, 224, 224]`.

## Training setup

The default runner trains for 10 epochs with:

- batch size `4`
- mixed precision
- AdamW on decoder parameters only
- learning rate `1e-3`
- weight decay `1e-3`
- `CrossEntropyLoss(ignore_index=255)`
- locked class weights from the selected weighted DeepLab experiments: `[0.243332998497, 0.709198873576, 0.651737877057, 2.395730250869]`

The completed three-seed experiment used batch size `4`. Earlier feasibility suggestions to increase it are superseded by the locked benchmark settings.

## Outputs

The runner writes the following to its active output workspace; not all original Colab exports are present in this checkout:

- `results/terramind_frozen/history.csv`
- `results/terramind_frozen/metrics.json`
- `results/terramind_frozen/per_class_metrics.csv`
- `results/terramind_frozen/train_summary.json`
- `checkpoints/terramind_frozen_best.pt`
- `figures/terramind_training_curves.png`
- `figures/terramind_per_class_iou.png`
- `figures/terramind_qualitative_predictions.png`
- `figures/terramind_vs_deeplab_validation.png`

## Comparison targets

The comparison table labels conventional baselines as single-run values and robust-method summaries as 3-seed means. It is built from already completed validation results where available:

- S2 weighted baseline
- S1 weighted baseline
- S1+S2 early fusion
- modality-dropout robust fusion, 3-seed clean validation mean
- occlusion-trained robust fusion, 3-seed clean validation mean
- frozen TerraMind RGB+S1RTC

The script does not rerun DeepLab experiments.

## Completed status and provenance

Training completed in Colab. Three-seed clean validation macro mIoU is **0.2771 ± 0.0138** and macro Dice **0.4269 ± 0.0161**. Clean TEST macro mIoU is **0.1718 ± 0.0177**, with binary algae Dice **0.6821 ± 0.0025**. Frozen TerraMind is not universally superior to robust DeepLab for four-class temporal generalisation.

The completed validation summary is corroborated by checkpoint metadata in [TERRAMIND_VALIDATION_EVIDENCE.json](TERRAMIND_VALIDATION_EVIDENCE.json). Historical incomplete exports under `results/terramind_reproducibility/` remain intact and do not describe current training status. Full original per-seed Colab runtime/VRAM exports remain unavailable locally. See [scientific audit](SCIENTIFIC_AUDIT.md).

## Audit updates

- The trainer validates the manifest before training: 621 train tiles, 111 validation tiles, 224x224 windows, allowed labels only, and no test dates in train/validation scope.
- The smoke test verifies real RGB + S1RTC tensors, logits [B,4,224,224], finite loss, CUDA use, frozen backbone, decoder-only trainability, no backbone gradients, and decoder gradients.
- The smoke test does not step the optimizer or alter weights before the timed training run.



## Portability note

The TerraMind dataset reconstructs raster paths from the acquisition date under a configurable data_root. On Colab, run with --data-root /content/geoai_data/SummerSchool_Subset so the manifest keeps its split/window metadata without relying on Windows-style source path strings.

