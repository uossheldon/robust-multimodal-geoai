# Robust Multimodal GeoAI for Algal Bloom Mapping

A completed independent study of optical–radar segmentation, missing-sensor robustness and temporal generalisation over Lough Neagh.

**[Project website](https://uossheldon.github.io/robust-multimodal-geoai/)** · [Methods](docs/METHODS.md) · [Frozen results](docs/FINAL_TEST_RESULTS.md) · [Reproduction](docs/REPRODUCTION.md)

![Study design schematic](figures/hero_overview.svg)

## Research question

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

Sentinel-2 RGB (B04/B03/B02) and Sentinel-1 VV/VH radar were aligned to 224×224 tiles. The fixed date-level split contains 621 training, 111 validation and 114 test tiles; the test dates are 8 and 21 September 2025. Models predict background, low, mid and high algae, ignoring label 255.

## Key findings

1. **Early fusion alone was brittle under optical degradation.** In the initial single-run validation benchmark, macro mIoU fell from 0.2425 clean to 0.0133 at 70% simulated optical occlusion.
2. **Robustness-aware training improved resilience.** Modality dropout reduced missing-sensor failures on validation, and occlusion-aware training improved resistance to simulated optical degradation. Complete Sentinel-1 loss remained difficult on the test dates.
3. **Temporal generalisation remained the dominant unresolved challenge.** Every model lost four-class performance from validation to the September test dates.

## Final held-out test

| Model | Runs | Macro mIoU | Macro Dice | Binary algae Dice |
| --- | --- | --- | --- | --- |
| S2 weighted | Single | 0.1698 | 0.2541 | 0.6252 |
| S1 weighted | Single | 0.1756 | 0.2799 | 0.5101 |
| Naive early fusion | Single | 0.1658 | 0.2700 | 0.6136 |
| Modality dropout | 3 seeds | 0.1826 ± 0.0123 | 0.2987 ± 0.0182 | 0.5731 ± 0.0421 |
| Dropout + occlusion training | 3 seeds | 0.1882 ± 0.0293 | 0.3033 ± 0.0380 | 0.6151 ± 0.0289 |
| Frozen TerraMind | 3 seeds | 0.1718 ± 0.0177 | 0.2843 ± 0.0235 | 0.6821 ± 0.0025 |

Conventional baselines are single runs; robust models and TerraMind report mean ± sample SD across training seeds 42, 7 and 123. The final comparison followed frozen model development, with no post-final-test tuning or checkpoint selection. Earlier unweighted S2 test access and unequal valid-pixel support limit a pristine, perfectly matched holdout claim; see the [scientific audit](docs/SCIENTIFIC_AUDIT.md). Differences are descriptive, not statistically significant rankings.

## Robustness

The benchmark used coherent **simulated optical occlusion**, not real clouds, at 0/10/30/50/70%, plus separate missing-S1 and missing-S2 conditions. Occlusion-trained fusion achieved **0.2122 ± 0.0115** macro mIoU at 70% occlusion on the test dates. Optical-corruption SD pools nine training-seed × corruption-seed runs; clean/missing-sensor SD uses three training seeds. Non-monotonic scores do not imply that removing imagery improves information.

![Frozen robustness comparison](figures/final_robustness_curves.svg)

## Temporal generalisation

All six models declined on September data from the same region. Occlusion-trained fusion had the numerically highest clean four-class test macro mIoU, **0.1882 ± 0.0293**, but severe imbalance and weak mid-algae segmentation remained. Temporal separation does not demonstrate geographic generalisation.

![Validation-to-test performance shift](figures/final_temporal_generalisation.svg)

## TerraMind

A frozen `terramind_v1_tiny` RGB+S1RTC backbone with a lightweight four-class decoder completed three seeds in Colab. Validation macro mIoU was **0.2771 ± 0.0138**; test macro mIoU was **0.1718 ± 0.0177**. Its test binary algae Dice, **0.6821 ± 0.0025**, was the highest among the compared methods, without consistently stronger four-class temporal generalisation. This was frozen-feature benchmarking, not full fine-tuning; TerraMind robustness was not evaluated. [TerraMind details](docs/TERRAMIND.md).

## Repository structure

```text
configs/       Locked experiment settings and date split
src/           Data pipelines, models, training and evaluation
scripts/       Experiment entry points and safe presentation tools
notebooks/     TerraMind Colab reproduction workflow
results/       Frozen metrics, manifests and experiment records
figures/       Aggregate plots and independent schematics
site/          Static project website and results explorer
docs/          Canonical research documentation
```

## Reproduction

To check and preview the presentation without data, checkpoints or a GPU:

```bash
python scripts/check_lightweight.py
python scripts/check_presentation.py
python -m http.server 8000 --bind 127.0.0.1 --directory .site-build
```

Open `http://127.0.0.1:8000/`. Experimental reproduction requires external data/weights and a separate output workspace. The Windows DeepLab stack and TerraMind Colab stack are distinct; `requirements.txt` is not a universal environment lock. See [Reproduction](docs/REPRODUCTION.md) and [Colab workflow](docs/COLAB_WORKFLOW.md).

## Data attribution and availability

This independent research project was developed using data and teaching materials provided through the Newcastle University GeoAI Summer School. The original dataset is not redistributed.

The public repository contains independently implemented code, experiments, analysis, aggregate results and presentation graphics. Raw imagery, original masks, archives, checkpoints and caches are excluded. Nine raster-derived qualitative/alignment images were removed from the current tree and reachable public history before release. Source-data/archive/label licensing and withheld-image publication rights remain unresolved; no redistribution permission or software license is invented. [Data provenance](docs/DATA_PROVENANCE.md).

## Limitations

The study covers one region and two test dates, with severe class imbalance, earlier test exposure and unequal model-specific scoring support. Binary algae detection is not equivalent to severity segmentation. Ensembling improved probability calibration, but uncertainty-based error detection remained modest and condition-dependent. No deployment-safety or statistical-significance claim is made.

## Documentation

- [Methods](docs/METHODS.md)
- [Final test results](docs/FINAL_TEST_RESULTS.md)
- [Reproduction](docs/REPRODUCTION.md)
- [Data provenance](docs/DATA_PROVENANCE.md)
- [Scientific audit](docs/SCIENTIFIC_AUDIT.md)
- [TerraMind benchmark](docs/TERRAMIND.md)
- [Uncertainty and calibration](docs/UNCERTAINTY.md)
- [Colab workflow](docs/COLAB_WORKFLOW.md)
