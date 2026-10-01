# Robust Multimodal GeoAI for Algal Bloom Mapping

This repository contains a research prototype for studying robustness in multimodal Earth-observation segmentation models for algal bloom mapping.

## Research question

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

The project compares Sentinel-2 optical, Sentinel-1 SAR, and fused Sentinel-1 + Sentinel-2 semantic segmentation models under clean validation conditions, simulated optical occlusion, missing-modality inputs, and uncertainty diagnostics.

## Data and task setup

The task is 4-class semantic segmentation:

| Class | Meaning |
|---:|---|
| 0 | background |
| 1 | low algae |
| 2 | mid algae |
| 3 | high algae |
| 255 | ignore |

Inputs use aligned 224x224 geographic tiles from the Summer School subset:

- Sentinel-2 RGB: B04, B03, B02
- Sentinel-1 SAR: VV, VH GAMMA0_TERRAIN dB
- Multimodal fusion: B04/B03/B02/VV/VH

The dataset is represented through `results/tile_manifest.csv`; raster tiles are read on demand from source rasters. Raw Summer School data, archives, and model checkpoints are intentionally excluded from Git and are not redistributed.

## Fixed temporal split

The split is date-level to avoid leakage across neighboring tiles from the same acquisition date.

| Split | Dates |
|---|---|
| Train | 2025-01-31, 2025-03-12, 2025-04-08, 2025-04-09, 2025-05-16, 2025-05-18, 2025-05-21, 2025-08-12 |
| Validation | 2025-01-01, 2025-06-20 |
| Test | 2025-09-08, 2025-09-21 |

Current development and robustness results are validation-only unless explicitly stated. Held-out test reporting for the final selected models is pending.

## Implemented baselines

Implemented conventional baselines use DeepLabV3-MobileNetV3-Large with the fixed split and 224x224 tiles.

| Model | Input | Validation macro mIoU | Validation macro Dice | Notes |
|---|---|---:|---:|---|
| S2 weighted | B04/B03/B02 | 0.2274 | 0.3497 | Single selected run with class-weighted CE |
| S1 weighted | VV/VH | 0.2349 | 0.3497 | Single selected run with class-weighted CE |
| S1+S2 early fusion | B04/B03/B02/VV/VH | 0.2425 | 0.3667 | Single selected run |

These values are validation results. They are not held-out test results.

## Robustness experiments

The repository includes a validation-only robustness benchmark with:

- simulated optical occlusion at 0%, 10%, 30%, 50%, and 70%
- missing S1 inputs represented by zeroed normalized SAR channels
- missing S2 inputs represented by zeroed normalized optical channels
- shared deterministic corruption masks for fair comparison

The simulated optical occlusion benchmark is not real cloud-cover measurement.

## Robust training variants

Two controlled robust-fusion training variants were implemented after the early-fusion baseline:

1. **Modality dropout**
   - 50% both modalities
   - 25% S1 missing
   - 25% S2 missing

2. **Modality dropout + optical-occlusion-aware training**
   - same modality-dropout policy
   - when both modalities are present, 50% of intact samples remain clean and 50% receive simulated optical occlusion sampled from 10% to 70%

Three training seeds were used for reproducibility: 42, 7, 123.

### Robust-method validation summary, 3-seed means

| Method | Clean macro mIoU | Missing S1 macro mIoU | Missing S2 macro mIoU | Robustness AUC 0-70% |
|---|---:|---:|---:|---:|
| Original early fusion | 0.2592 ± 0.0171 | 0.1539 ± 0.0199 | 0.0275 ± 0.0117 | 0.1016 |
| Modality dropout | 0.2725 ± 0.0157 | 0.1886 ± 0.0188 | 0.2104 ± 0.0263 | 0.1912 |
| Modality dropout + optical-occlusion training | 0.2648 ± 0.0089 | 0.1746 ± 0.0305 | 0.2130 ± 0.0132 | 0.2646 |

The robust variants substantially reduce missing-modality brittleness. Optical-occlusion-aware training gives the strongest 0-70% robustness AUC in the completed validation benchmark.

## Uncertainty diagnostics

The repository includes validation-only uncertainty analyses for the frozen trained models:

- softmax confidence
- predictive entropy
- negative log likelihood
- multiclass Brier score
- expected calibration error
- entropy on correct versus incorrect pixels
- entropy-based error-detection AUROC
- risk-coverage curves

Ensemble uncertainty was also evaluated for the three-seed occlusion-trained robust model by averaging class probabilities across seeds and computing:

- ensemble predictive entropy
- mean individual-model entropy
- mutual information
- probability variance / model disagreement
- ensemble calibration and error-detection metrics

The ensemble diagnostics are stored under `results/ensemble_uncertainty/`. They are diagnostic only; no calibration fitting has been performed.

## TerraMind benchmark status

TerraMind feasibility and implementation scaffolding are included, but TerraMind Phase 6C training is still pending Colab execution.

Implemented TerraMind components:

- config: `configs/terramind_frozen.yaml`
- dataset: `src/data/terramind_dataset.py`
- model wrapper and decoder: `src/models/terramind_segmentation.py`
- training runner: `src/training/train_terramind.py`
- script entry point: `scripts/run_terramind.py`
- Colab notebook: `notebooks/TerraMind_Colab.ipynb`

Planned TerraMind Phase 6C benchmark:

- frozen `terramind_v1_tiny` backbone
- modalities: RGB + S1RTC
- last TerraMind feature `[B,196,192]` reshaped to `[B,192,14,14]`
- lightweight 4-class upsampling decoder
- validation-only model selection by macro mIoU

No TerraMind validation results are reported yet.

## Repository layout

```text
configs/      experiment configs and fixed split
src/          data pipelines, models, training, evaluation
scripts/      runnable training/evaluation entry points
docs/         phase reports and experiment notes
notebooks/    Colab workflow for TerraMind
results/      small CSV/JSON experiment summaries
figures/      small generated figures for reports
data/         README only; raw data is excluded
checkpoints/  excluded from Git
```

## Reproduction notes

### Local DeepLab workflow

The local environment used a project virtual environment and CUDA-enabled PyTorch. See:

- `requirements.txt`
- `docs/PREPROCESSING.md`
- `docs/S2_BASELINE_SELECTION.md`
- `docs/ROBUSTNESS_BASELINE.md`
- `docs/OCCLUSION_AWARE_TRAINING.md`
- `docs/UNCERTAINTY_DIAGNOSTICS.md`
- `docs/ENSEMBLE_UNCERTAINTY.md`

Raw data must be supplied separately under `data/raw/SummerSchool_Subset/` following `data/README.md` and `docs/DATA_AUDIT.md`.

### TerraMind Colab workflow

Use `notebooks/TerraMind_Colab.ipynb` or `docs/COLAB_WORKFLOW.md`.

Expected Colab setup:

- Tesla T4 GPU
- `torch 2.11.0+cu128`
- `numpy 2.2.6`
- `terratorch 1.2.13`
- `torchgeo 0.9.0`

Expected Google Drive inputs:

- `robust-multimodal-geoai/data/SummerSchool_Subset.zip`
- `robust-multimodal-geoai/model_cache/TerraMind-1.0-tiny-hf-cache.tar.gz` when available

The notebook copies data to local `/content` storage, trains from local storage, and writes checkpoints/results/figures back to Drive.

## Data and model distribution policy

This repository does not distribute:

- original Newcastle GeoAI Summer School datasets
- `SummerSchool_Subset.zip`
- raw Sentinel rasters or masks
- trained model checkpoints
- Hugging Face or TerraMind model caches
- credentials, API keys, or tokens

Small CSV/JSON summaries and figures are included to document completed experiments.
