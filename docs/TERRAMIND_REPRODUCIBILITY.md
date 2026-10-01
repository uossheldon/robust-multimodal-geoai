# TerraMind Reproducibility Check

Phase: 6C-R  
Scope: validation-only reproducibility for the frozen TerraMind benchmark.

## Fixed experiment

This phase keeps the Phase 6C setup unchanged:

- frozen `terramind_v1_tiny` RGB+S1RTC backbone
- same lightweight 4-class decoder
- same 224x224 tiles
- same fixed date-level split
- same 621 train tiles and 111 validation tiles
- same locked class weights
- batch size 4
- 10 epochs
- validation-only model selection by macro mIoU
- no test-date evaluation

## Seeds

Use seeds `42`, `7`, and `123`.

Seed 42 is reused from the completed Colab run:

- best epoch: 7
- validation macro mIoU: 0.283511
- validation macro Dice: 0.434891

The full seed-42 `metrics.json` should be copied into `results/terramind_reproducibility/seed_42/metrics.json` or `results/terramind_reproducibility/seed_42_metrics.json` before final aggregation so per-class IoU and binary algae metrics can be included.

## Colab commands for remaining seeds

Run from the project root after mounting Drive, pulling the latest repository, restoring the TerraMind cache, and extracting data under `/content/geoai_data/SummerSchool_Subset`.

```bash
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 7 --data-root /content/geoai_data/SummerSchool_Subset
mkdir -p results/terramind_reproducibility/seed_7
cp results/terramind_frozen/metrics.json results/terramind_reproducibility/seed_7/metrics.json
cp results/terramind_frozen/history.csv results/terramind_reproducibility/seed_7/history.csv
cp checkpoints/terramind_frozen_best.pt checkpoints/terramind_frozen_seed7_best.pt
```

```bash
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 123 --data-root /content/geoai_data/SummerSchool_Subset
mkdir -p results/terramind_reproducibility/seed_123
cp results/terramind_frozen/metrics.json results/terramind_reproducibility/seed_123/metrics.json
cp results/terramind_frozen/history.csv results/terramind_reproducibility/seed_123/history.csv
cp checkpoints/terramind_frozen_best.pt checkpoints/terramind_frozen_seed123_best.pt
```

After all three seed metrics are present:

```bash
python scripts/aggregate_terramind_reproducibility.py
```

## Outputs

The aggregation script writes:

- `results/terramind_reproducibility/per_seed_results.csv`
- `results/terramind_reproducibility/aggregate_results.csv`
- `results/terramind_reproducibility/summary.json`
- `figures/terramind_reproducibility.png`
- `figures/terramind_vs_robust_deeplab.png`

## Comparison policy

The comparison is against established validation-only 3-seed means:

- modality-dropout fusion
- modality-dropout + optical-occlusion training

No DeepLab models are rerun in this phase. No test dates are evaluated.
