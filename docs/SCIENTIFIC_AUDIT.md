# Scientific audit and interpretation limits

The research and final comparison are complete. Frozen source records take precedence over rounded narrative summaries.

## Evidence hierarchy

Final TEST numbers come from `results/final_test/` (saved Phase 7 CSV/JSON). Validation method summaries come from `results/reproducibility/`, `results/occlusion_training/` and the selected baseline records. Completed TerraMind validation uses the supplied final summary corroborated by saved checkpoint metrics/confusion matrices, transcribed in [TERRAMIND_VALIDATION_EVIDENCE.json](TERRAMIND_VALIDATION_EVIDENCE.json). Historical result outputs are retained, not silently relabelled.

## Test-access history

**An earlier Phase 2D TEST evaluation exists.** `results/s2_deeplab/test_metrics.json` records unweighted S2 macro mIoU 0.157647 and macro Dice 0.225141. The associated training summary and source runner also record TEST access. The later weighted model was selected on validation; the historical unweighted result remains preserved.

Therefore TEST was not literally untouched until Phase 7. The accurate statement is: **the final Phase 7 comparison was performed after subsequent model development and validation-based selection were frozen; no post-Phase-7 tuning or checkpoint selection occurred.** Earlier exposure limits a pristine holdout claim, even though the later selection procedure uses validation. The historical result is preserved; no statistical independence claim is inferred from the split alone.

The fixed date split contains 621 train, 111 validation, 114 TEST tiles; test dates are September 8 and 21. Geography overlaps across dates. The final inventory identifies 12 checkpoint files with SHA-256 prefixes; source results record `post_test_tuning=false`. The final aggregate and per-date/qualitative passes constitute one final comparison, not one literal forward pass over each pixel.

## TerraMind provenance

The three frozen checkpoints contain validation metrics and confusion matrices, along with seed, batch size 4, 10 epochs, enabled fixed class weights and best epochs 7/5/10 for seeds 42/7/123. Their completed mean validation mIoU is 0.2771 ± 0.0138, matching the supplied final summary. Binary metrics were recovered algebraically from saved confusion counts, without predictions or model execution.

The older partial CSV/JSON exports under `results/terramind_reproducibility/` remain unchanged. Their `pending_colab_seed_metrics` status and n=0/1 fields are historical export state, not current training state. Current reports/plots link to the completed evidence. Full original Colab export bundles and per-seed runtime/VRAM remain absent locally; no values are invented. High-algae validation SD is 0.008249714 in the saved checkpoint metadata versus 0.0083 in the supplied rounded summary; the completed evidence uses full available precision.

## Aggregation and uncertainty populations

| Context | Population behind mean ± SD |
|---|---|
| Conventional clean TEST | One selected run; no seed SD estimate |
| Robust/TerraMind clean TEST and robust missing sensors | Three training seeds, sample SD (ddof=1) |
| Robust optical-occlusion TEST | Nine pooled runs: three training seeds × corruption seeds 101/202/303 |
| Initial single-model validation occlusion | Three corruption seeds for one frozen model |
| Reproducibility/occlusion-training validation | Corruption results averaged within a training seed, then sample SD across three training seeds |
| Validation ensemble | One probability ensemble of three models; its metrics are not a mean of individual-model metrics |

Phase 7's occlusion CSV stores the corruption seed in `seed`, without a separate training-seed field. Its saved SD is pooled, not SD over three per-training-seed means. Values are preserved and explicitly labelled. The saved 50% dropout TEST SD rounds to 0.0038, using final-source precision.

Validation robustness AUC integrates macro mIoU with the 0–70% axis normalized to [0,1]. It is a validation selection/development diagnostic, not a TEST headline or a significance test. Corruption masks are synthetic, not measured cloud cover. Non-monotonic TEST responses indicate changed prediction behaviour; they do not establish physical causation or information gain.

## Valid-pixel comparability

| Model support | September 8 | September 21 | Total |
|---|---:|---:|---:|
| S2-only | 1,754,102 | 1,720,182 | 3,474,284 |
| S1 / fusion / TerraMind | 1,618,660 | 1,550,433 | 3,169,093 |

Source: saved per-date TEST records. Manifest tile selection was fixed; scoring masks differ. S2 uses the aligned label validity mask; SAR/fusion/TerraMind additionally exclude invalid SAR. The manifest's common-valid selection statistics are not identical to each model's runtime scoring pixels. Thus same tiles do not provide perfectly like-for-like support; the headline comparison is descriptive, without a common-support reevaluation.

## Uncertainty limitations

Validation ensembling reduced ECE, NLL and Brier compared with mean individual occlusion-trained scores. Entropy/error separation and AUROC were modest; missing-sensor behaviour was inconsistent. The ensemble diagnostic uses corruption seed 101. Its `single_entropy` score is the first model's entropy tested against **ensemble correctness**, not an independent single-model error target; self-error diagnostics are recorded separately in the individual-model analysis. This distinction limits direct single-vs-ensemble failure-detection claims. No calibration fitting, reliable-abstention guarantee or uncertainty solution is claimed.


## Public assets and reproduction limits

Nine raster-derived qualitative/alignment images were removed from the current tree and reachable public history before release. Aggregate metrics and independently generated schematics/plots remain public. This removal does not resolve source-data, label or withheld-image rights; [Data provenance](DATA_PROVENANCE.md) is the canonical rights record.

Historical DeepLab loaders retain Windows-style manifest paths; TerraMind has a configurable data root. The requirements file is not a complete cross-platform environment lock. Full original Colab runtime exports are absent. The final checkpoint inventory records SHA-256 prefixes rather than complete digests. These limits are disclosed in [Reproduction](REPRODUCTION.md), not filled with invented evidence.

No statistical-significance test supports model ranking. Two dates from one region, severe class imbalance, repeated geography and unequal scoring support limit generalisation. Binary algae detection does not establish accurate four-class severity mapping. No post-Phase-7 tuning or checkpoint selection was performed.

Frozen metadata can contain historical report destinations: `results/s1_deeplab_weighted/train_summary.json` names the former SAR report. Its canonical replacement is [Methods](METHODS.md#sentinel-1-preprocessing). That saved JSON is preserved byte-for-byte. Historical report-writing code also retains its output filenames; these are generated destinations, not links to current public documents.
