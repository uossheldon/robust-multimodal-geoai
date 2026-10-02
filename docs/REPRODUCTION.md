# Reproduction Guide

## Frozen record and prerequisites

Phase 8 is documentation/presentation only. Model development and the final Phase 7 comparison are closed. No training, inference, tuning or checkpoint reselection is part of finalisation. Historical commands below describe independent reproduction, which must use a separate output workspace to protect saved results.

Obtain the Summer School archive and labels separately under their source terms. Follow [data setup](../data/README.md); no raw rasters, ZIPs, trained weights or model caches are distributed. The repository is private until publication review. Keep the existing [manifest](../results/tile_manifest.csv) and [split](../configs/split_v1.yaml); do not regenerate them to change selection.

## Data and environments

- Local DeepLab expects `data/raw/SummerSchool_Subset/` under the project root. Its original manifest source strings use Windows separators, and its loaders join these strings directly: Linux portability is not established for those historical loaders.
- TerraMind reconstructs paths from acquisition date and a configurable data root; Colab uses `/content/geoai_data/SummerSchool_Subset`. It never reuses DeepLab-normalized tensors.
- `requirements.txt` records a historical Windows environment (including torch 2.8.0+cu129). It is not a complete environment lock or a Colab installation recipe; utilities also import Matplotlib and other phase-specific dependencies. Do not treat it as a guarantee of reproduction on a new platform.
- Validated TerraMind Colab context: T4 14.56 GB, torch 2.11.0+cu128, numpy 2.2.6, TerraTorch 1.2.13, TorchGeo 0.9.0. [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md) documents the existing dependency recipe. Verify resolved versions there; no environment was rebuilt in Phase 8.
- Data-root portability tests are in `tests/test_terramind_paths.py`. They do not execute the final TEST evaluation.

## Implementation map

| Stage | Existing entry point / source | Frozen outputs |
|---|---|---|
| Alignment and manifest | `src/data/geodata.py`, `build_phase2b_manifest.py`, `apply_split.py` | `results/tile_manifest.csv`, `per_date_statistics.csv` |
| Selected weighted S2 | `src/training/class_weighted_s2.py` | `results/s2_deeplab_weighted/` |
| S1 / naive fusion | `src/training/train_s1_deeplab.py`, `train_fusion_deeplab.py` | `results/s1_deeplab_weighted/`, `s1_s2_early_fusion/` |
| Modality dropout / reproducibility | `src/training/train_modality_dropout_fusion.py`, `reproducibility_check.py` | `results/modality_dropout/`, `reproducibility/` |
| Occlusion-aware training | `src/training/occlusion_training.py` | `results/occlusion_training/` |
| Fixed corruption benchmark | `src/evaluation/robustness_benchmark.py` | `results/robustness/` |
| Uncertainty / ensemble | `src/evaluation/uncertainty_diagnostics.py`, `ensemble_uncertainty.py` | `results/uncertainty/`, `ensemble_uncertainty/` |
| TerraMind | `scripts/run_terramind.py`, `notebooks/TerraMind_Colab.ipynb` | Completed evidence in `docs/TERRAMIND_VALIDATION_EVIDENCE.json` |
| Final TEST record | `scripts/final_test_evaluation.py` | `results/final_test/` |

Some runners have fixed paths and overwrite outputs. `scripts/train.py` runs the historical unweighted S2 workflow, including its Phase 2D TEST evaluation; it is not the selected weighted-S2 runner. `scripts/evaluate.py` summarizes the manifest rather than evaluating a model. `scripts/final_test_finalize_outputs.py` performs qualitative inference and writes documentation at import/run time; it is not a safe presentation-only command. Do not execute these to refresh documentation.

DeepLab uses 10 epochs, batch size 8, AdamW backbone/classifier learning rates 1e-5/1e-4, weight decay 1e-3, and matched spatial augmentations. Selected models use weights `[0.243332998497, 0.709198873576, 0.651737877057, 2.395730250869]` with ignore=255. TerraMind decoder-only AdamW uses LR 1e-3, weight decay 1e-3, batch size 4, 10 epochs and the same weights. The config files describe experiments; not every runner parses YAML dynamically.

## TerraMind Colab

Mount Drive, clone/pull the existing repository, restore the optional Hugging Face cache, copy the archive to `/content`, and extract there. Train/read from local storage and copy results/checkpoints/figures back to Drive. See [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md) for the full historical workflow and seed output preservation.

```bash
# Historical fixed training invocation; not executed in Phase 8.
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 42 --data-root /content/geoai_data/SummerSchool_Subset
```

The smoke test is built into this command, which continues to full decoder training. It is not a smoke-only command. The locked three-seed benchmark always uses batch size 4; do not apply earlier exploratory suggestions to increase it.

## Presentation-only reproduction

```bash
python scripts/render_final_presentation.py
```

Requires Matplotlib in the current Python environment. Reads frozen CSV/JSON summaries and the documented TerraMind metadata transcription; writes summary PNGs only. No dataset, checkpoint, GPU, network or model inference is used. Scientific records and the existing qualitative image remain unchanged.

## Known reproducibility limits

Data/label redistribution rights, incomplete historical environment locks, absent full Colab runtime exports, Windows-only legacy path assumptions and model-specific valid-pixel masks qualify portability and comparison. Phase 7 recorded 16-character SHA-256 prefixes rather than full digests. See [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md) for the complete interpretation record. No new experiment is required or authorized by this guide.
