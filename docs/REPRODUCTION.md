# Reproduction

The public repository supports code inspection, presentation checks and re-running the experiments when the external dataset and required pretrained weights are available.

## Lightweight repository checks

From the repository root:

```bash
python scripts/check_lightweight.py
python scripts/check_presentation.py
python -m http.server 8000 --bind 127.0.0.1 --directory .site-build
```

The first two commands compile the Python sources, verify path handling, validate the published metrics and build the reviewed static site. They do not require the dataset, checkpoints or a GPU.

Optional regeneration of the final presentation figures requires Matplotlib:

```bash
python scripts/presentation_data.py
python scripts/render_final_presentation.py
python scripts/render_hero.py
python scripts/check_presentation.py
```

## DeepLab experiments

The local DeepLab environment is recorded in [requirements.txt](../requirements.txt). It is a working environment record rather than a complete cross-platform lock.

Use the fixed [split](../configs/split_v1.yaml) and [manifest](../results/tile_manifest.csv). [Methods](METHODS.md) defines alignment, preprocessing, class weighting, training settings and evaluation metrics.

Main experiment entry points and saved records:

| Experiment | Implementation | Saved record |
|---|---|---|
| Data alignment / manifest | `src/data/` | `results/tile_manifest.csv`, `results/per_date_statistics.csv` |
| Sentinel-2 / Sentinel-1 / early fusion | `src/training/` | model-specific result folders |
| Modality dropout | `src/training/train_modality_dropout_fusion.py` | `results/modality_dropout/` |
| Occlusion-aware training | `src/training/occlusion_training.py` | `results/occlusion_training/` |
| Robustness evaluation | `src/evaluation/robustness_benchmark.py` | `results/robustness/` |
| Uncertainty analysis | `src/evaluation/` | `results/uncertainty/`, `results/ensemble_uncertainty/` |
| Final frozen evaluation | `scripts/final_test_evaluation.py` | `results/final_test/` |

## TerraMind

The validated Colab stack used a T4 GPU, torch 2.11.0+cu128, numpy 2.2.6, TerraTorch 1.2.13 and TorchGeo 0.9.0.

Use [COLAB_WORKFLOW.md](COLAB_WORKFLOW.md) and:

```bash
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 42 --data-root /content/geoai_data/SummerSchool_Subset
```

Repeat with seeds 7 and 123 using the same settings. The canonical validation record is [terramind_validation.json](../results/terramind_validation.json).

## External inputs

Place the prepared dataset under:

```text
data/raw/SummerSchool_Subset/
```

or provide the TerraMind runner with an explicit `--data-root`.

Raw imagery, masks, checkpoints and caches are intentionally not distributed. See [Data provenance](DATA_PROVENANCE.md) and [data layout](../data/README.md).
