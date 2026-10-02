# TerraMind Reproducibility Check

**Status: completed, three training seeds (42, 7, 123).** The final validation summary is authoritative. Phase 8 verified it by reading the existing checkpoint metadata, without model execution.

## Fixed experiment

Frozen `terramind_v1_tiny` RGB+S1RTC; same lightweight four-class decoder; 224×224 tiles; 621 train / 111 validation tiles; 10 epochs; batch size 4; locked DeepLab inverse-square-root class weights; best checkpoint selected by clean validation macro mIoU. TEST was excluded from these training/validation loaders. The later frozen Phase 7 evaluation is documented in [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md).

## Completed validation results

| Training seed | Best validation epoch | Macro mIoU | Macro Dice |
| --- | --- | --- | --- |
| 42 | 7 | 0.283511 | 0.434891 |
| 7 | 5 | 0.261364 | 0.408358 |
| 123 | 10 | 0.286567 | 0.437345 |

| Validation metric | 3-seed mean ± sample SD | n |
| --- | --- | --- |
| Macro mIoU | 0.277148 ± 0.013754 | 3 |
| Macro Dice | 0.426864 ± 0.016074 | 3 |
| IoU 0 background | 0.269726 ± 0.059600 | 3 |
| IoU 1 low | 0.415583 ± 0.018151 | 3 |
| IoU 2 mid | 0.207178 ± 0.031039 | 3 |
| IoU 3 high | 0.216104 ± 0.008250 | 3 |
| Binary algae IoU | 0.496434 ± 0.012650 | 3 |
| Binary algae Dice | 0.663426 ± 0.011248 | 3 |

These are training-seed statistics, not an ensemble prediction. At headline precision the validation macro mIoU is **0.2771 ± 0.0138** and macro Dice **0.4269 ± 0.0161**. The supplied completed summary reported high-algae SD as 0.0083; saved checkpoint metadata gives 0.008249714, shown above at six decimals to make the precision distinction explicit. No recorded result was overwritten.

## Provenance reconciliation

[TERRAMIND_VALIDATION_EVIDENCE.json](TERRAMIND_VALIDATION_EVIDENCE.json) records each saved seed's validation metrics, binary metrics derived from its saved validation confusion matrix, checkpoint filenames and SHA-256 prefixes matching the Phase 7 inventory. The means agree with the supplied completed summary at the stated precision apart from the minor high-class SD rounding noted above.

The files `results/terramind_reproducibility/per_seed_results.csv`, `aggregate_results.csv`, `summary.json` and `seed_42_metrics.json` are **historical incomplete export snapshots**, preserved intact. Their missing values and pending status do not describe the completed training. They are not the authoritative source for the current three-seed tables or figures. The old aggregation script reads those historical exports; do not rerun it expecting the completed summary until full original Colab metrics files have been restored. Runtime/VRAM for all three runs are not recoverable from the available checkpoint metadata and are not invented.

![Completed TerraMind validation summary](../figures/terramind_reproducibility.png)

![Three-seed validation comparison](../figures/terramind_vs_robust_deeplab.png)

The comparison uses robust DeepLab **three-seed validation means**, not single selected runs. Dropout mIoU is 0.2725 ± 0.0157; occlusion-trained fusion is 0.2648 ± 0.0089. No significance claim is made. TerraMind's TEST mIoU is 0.1718 ± 0.0177 and binary algae Dice is 0.6821 ± 0.0025: strong binary detection does not establish superior severity segmentation or temporal generalisation.

## Historical reproduction commands

The runs are complete. For independent reproduction in a separate output workspace, use the [Colab workflow](COLAB_WORKFLOW.md), with the same data root and fixed settings:

```bash
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 7 --data-root /content/geoai_data/SummerSchool_Subset
python scripts/run_terramind.py --batch-size 4 --epochs 10 --seed 123 --data-root /content/geoai_data/SummerSchool_Subset
```

Each command writes the same default output paths; archive each seed's metrics/history/checkpoint before starting another. Seed 42 was reused, not rerun in Phase 6C-R. Phase 8 runs none of these commands.
