# Reproduction

Research and final model selection are closed. Independent reproduction must use a separate checkout/output workspace so frozen results are not overwritten. Do not use the held-out test dates for further model selection.

## A. Lightweight presentation reproduction

From the repository root, using Python 3.11 or a compatible installation:

```bash
python scripts/check_lightweight.py
python scripts/check_presentation.py
python -m http.server 8000 --bind 127.0.0.1 --directory .site-build
```

Open `http://127.0.0.1:8000/`. The checks compile source without importing ML frameworks, exercise the pure path resolver, validate displayed metrics and build the 19-file allowlisted static site. No data, checkpoint, GPU or inference is needed.

Optional chart regeneration from existing aggregates requires Matplotlib:

```bash
python scripts/presentation_data.py
python scripts/render_final_presentation.py
python scripts/render_hero.py
python scripts/check_presentation.py
```

These presentation scripts read frozen summaries and the retained TerraMind evidence. They do not execute models. The live project is [GitHub Pages](https://uossheldon.github.io/robust-multimodal-geoai/). The repository's existing manual **Deploy research site to Pages** workflow builds only reviewed assets; an authorized maintainer can run it on `main` with `publish_reviewed_site=true` after reviewing changes.

## B. DeepLab experimental environment

[requirements.txt](../requirements.txt) records the local Windows stack, including torch 2.8.0+cu129. It is not a complete environment lock or a TerraMind recipe; plotting utilities also require Matplotlib. Preserve compatible CUDA-enabled PyTorch/torchvision and geospatial dependencies without changing the system CUDA toolkit.

DeepLab reads `data/raw/SummerSchool_Subset/` using the existing [manifest](../results/tile_manifest.csv). Legacy loader paths contain Windows separators; Linux portability is not established. Keep the [fixed split](../configs/split_v1.yaml) and windows unchanged. [Methods](METHODS.md) records the 10-epoch, batch-size-8 optimizer, normalization, augmentations and locked loss weights.

| Experiment | Implementation | Saved record |
|---|---|---|
| Alignment/manifest | `src/data/geodata.py`, `src/data/build_phase2b_manifest.py` | `results/tile_manifest.csv`, `results/per_date_statistics.csv` |
| Weighted S2 | `src/training/class_weighted_s2.py` | `results/s2_deeplab_weighted/` |
| SAR / early fusion | `src/training/train_s1_deeplab.py`, `src/training/train_fusion_deeplab.py` | `results/s1_deeplab_weighted/`, `results/s1_s2_early_fusion/` |
| Modality dropout / reproducibility | `src/training/train_modality_dropout_fusion.py`, `src/training/reproducibility_check.py` | `results/modality_dropout/`, `results/reproducibility/` |
| Occlusion-aware training | `src/training/occlusion_training.py` | `results/occlusion_training/` |
| Robustness | `src/evaluation/robustness_benchmark.py` | `results/robustness/` |
| Uncertainty / ensemble | `src/evaluation/uncertainty_diagnostics.py`, `src/evaluation/ensemble_uncertainty.py` | `results/uncertainty/`, `results/ensemble_uncertainty/` |
| Final evaluation | `scripts/final_test_evaluation.py` | `results/final_test/` |

Configs describe the experiments; not every runner dynamically reads YAML. Several runners overwrite outputs. `scripts/train.py` is the historical unweighted S2 workflow, including earlier Phase 2D TEST access, not the selected weighted runner. `scripts/evaluate.py` summarizes the manifest. `scripts/final_test_finalize_outputs.py` runs qualitative inference at import/run time: do not use it for presentation updates. Experiment runners may write development-era reports and restricted imagery locally; those generated outputs are not canonical public documentation or cleared for redistribution.

## C. TerraMind Colab environment

Validated environment: Tesla T4 14.56 GB, torch 2.11.0+cu128, numpy 2.2.6, TerraTorch 1.2.13 and TorchGeo 0.9.0. Use [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md) and [TerraMind_Colab.ipynb](../notebooks/TerraMind_Colab.ipynb). The frozen backbone receives independently prepared RGB+S1RTC, not DeepLab-normalized tensors.

```bash
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 42 --data-root /content/geoai_data/SummerSchool_Subset
```

This performs built-in smoke checks followed by decoder training; it is not smoke-only. Seeds 7 and 123 use identical settings. Save each run before starting the next because output paths are shared. Completed validation evidence is retained in [TERRAMIND_VALIDATION_EVIDENCE.json](TERRAMIND_VALIDATION_EVIDENCE.json); older incomplete exports must not override it. See [TerraMind](TERRAMIND.md).

## D. Required external inputs

Obtain the prepared archive, masks and pretrained weights under their source terms. Exact checkpoint-based replication also requires the frozen experiment checkpoints, which are not distributed. [Data layout](../data/README.md) specifies raster and mask paths. TerraMind supports `--data-root`; local Colab storage avoids repeated Drive reads. Retain the exact manifest and split rather than regenerating eligibility.

## E. Intentionally not distributed

Raw imagery, original labels, archives, checkpoints, caches and withheld raster-derived qualitative images are excluded. The repository cannot reconstruct the original labels. [Data provenance](DATA_PROVENANCE.md) explains unresolved rights. Incomplete environment locks, missing full Colab runtime exports, checkpoint hash prefixes and unequal scoring support qualify exact reproduction; see [Scientific audit](SCIENTIFIC_AUDIT.md).
