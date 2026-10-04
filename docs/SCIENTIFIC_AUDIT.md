# Final Scientific and Repository Audit

Phase 8, 2026-10-02. Documentation and result presentation only: no training, inference, hyperparameter changes, new checkpoint selection or alteration of saved scientific results.

## Evidence hierarchy

Final TEST numbers come from `results/final_test/` (saved Phase 7 CSV/JSON). Validation method summaries come from `results/reproducibility/`, `results/occlusion_training/` and the selected baseline records. Completed TerraMind validation uses the supplied final summary corroborated by saved checkpoint metrics/confusion matrices, transcribed in [TERRAMIND_VALIDATION_EVIDENCE.json](TERRAMIND_VALIDATION_EVIDENCE.json). Historical phase outputs are retained, not silently relabelled.

## Test-access history

**An earlier Phase 2D TEST evaluation exists.** `results/s2_deeplab/test_metrics.json` records unweighted S2 macro mIoU 0.157647 and macro Dice 0.225141. The associated training summary and source runner also record TEST access. Phase 2G documentation explicitly retained that result while selecting the weighted model on validation.

Therefore TEST was not literally untouched until Phase 7. The accurate statement is: **the final Phase 7 comparison was performed after subsequent model development and validation-based selection were frozen; no post-Phase-7 tuning or checkpoint selection occurred.** Earlier exposure limits a pristine holdout claim, even though the later selection procedure uses validation. The historical result is preserved; no statistical independence claim is inferred from the split alone.

The fixed date split contains 621 train, 111 validation, 114 TEST tiles; test dates are September 8 and 21. Geography overlaps across dates. The final inventory identifies 12 checkpoint files with SHA-256 prefixes; source results record `post_test_tuning=false`. Phase 8 did not run the evaluator. Phase 7's aggregate and per-date/qualitative passes should be described as one final comparison phase, not one literal forward pass over each test pixel.

## TerraMind provenance

The three frozen checkpoints contain validation metrics and confusion matrices, along with seed, batch size 4, 10 epochs, enabled fixed class weights and best epochs 7/5/10 for seeds 42/7/123. Their completed mean validation mIoU is 0.2771 ± 0.0138, matching the supplied final summary. Binary metrics were recovered algebraically from saved confusion counts, without predictions or model execution.

The older partial CSV/JSON exports under `results/terramind_reproducibility/` remain unchanged. Their `pending_colab_seed_metrics` status and n=0/1 fields are historical export state, not current training state. Current reports/plots link to the completed evidence. Full original Colab export bundles and per-seed runtime/VRAM remain absent locally; no values are invented. High-algae validation SD is 0.008249714 in the saved checkpoint metadata versus 0.0083 in the supplied rounded summary; the completed evidence uses full available precision.

## Aggregation and uncertainty populations

| Context | Population behind mean ± SD |
|---|---|
| Conventional clean TEST | One selected run; no seed SD estimate |
| Robust/TerraMind clean TEST and robust missing sensors | Three training seeds, sample SD (ddof=1) |
| Robust optical-occlusion TEST | Nine pooled runs: three training seeds × corruption seeds 101/202/303 |
| Phase 4A initial validation occlusion | Three corruption seeds for one frozen model |
| Phase 4B reproducibility/occlusion-training validation | Corruption results averaged within a training seed, then sample SD across three training seeds |
| Phase 5 ensemble | One probability ensemble of three models; its metrics are not a mean of individual-model metrics |

Phase 7's occlusion CSV stores the corruption seed in `seed`, without a separate training-seed field. Its saved SD is pooled, not SD over three per-training-seed means. Values are preserved and explicitly labelled. The saved 50% dropout TEST SD rounds to 0.0038, while an earlier prose handoff said 0.0039; final-source precision controls presentation.

Validation robustness AUC integrates macro mIoU with the 0–70% axis normalized to [0,1]. It is a validation selection/development diagnostic, not a TEST headline or a significance test. Corruption masks are synthetic, not measured cloud cover. Non-monotonic TEST responses indicate changed prediction behaviour; they do not establish physical causation or information gain.

## Valid-pixel comparability

| Model support | September 8 | September 21 | Total |
|---|---:|---:|---:|
| S2-only | 1,754,102 | 1,720,182 | 3,474,284 |
| S1 / fusion / TerraMind | 1,618,660 | 1,550,433 | 3,169,093 |

Source: saved per-date TEST records. Manifest tile selection was fixed; scoring masks differ. S2 uses the aligned label validity mask; SAR/fusion/TerraMind additionally exclude invalid SAR. The manifest's common-valid selection statistics are not identical to each model's runtime scoring pixels. Thus same tiles do not provide perfectly like-for-like support; the headline comparison is descriptive, with no common-support reevaluation in Phase 8.

## Uncertainty limitations

Validation ensembling reduced ECE, NLL and Brier compared with mean individual occlusion-trained scores. Entropy/error separation and AUROC were modest; missing-sensor behaviour was inconsistent. Phase 5B uses corruption seed 101. Its `single_entropy` score is the first model's entropy tested against **ensemble correctness**, not an independent single-model error target; self-error diagnostics belong to Phase 5A. This distinction limits direct single-vs-ensemble failure-detection claims. No calibration fitting, reliable-abstention guarantee or uncertainty solution is claimed.

## Figure audit

The original final per-class plot overlapped rows and clipped content. Three final metric charts were redrawn from unchanged saved summaries, with correctly associated model labels, quantitative axes, seed/SD labels and support qualifications. Optical-occlusion x positions reflect 0/10/30/50/70 rather than equal index spacing. Completed TerraMind plots use documented saved validation metadata rather than partial exports.

The historical qualitative image is withheld from the public tree and rewritten history after the public-release cleanup. It contained the first two September 8 tiles, seed-42 robust/TerraMind examples, and an error map under S2 target validity. It is valid as an illustrative prediction panel with this qualification, not a common-support quantitative comparison or a cross-date sample. See [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md).

## Reproduction and publication qualifications

Historical Windows DeepLab loaders still use Windows-style manifest strings; TerraMind alone has the validated configurable data-root fix. The original requirements file is not a complete cross-platform lock. Original Colab metadata exports should accompany a fully packaged reproduction. The final evaluation/finalisation scripts can rerun inference or overwrite reports; use only the new presentation renderer for chart updates.

The repository safety scan found no raw datasets, TIFFs, ZIP archives, checkpoints, model caches, credentials or temporary Colab artifacts tracked, and no tracked/history blob over 20 MB. Pattern-based secret scanning found no token/private-key/credential pattern matches in reachable text history. This is not a guarantee against every possible secret. Ignore probes cover raw data, archives, checkpoints, caches, environments and secrets. Public-facing Markdown has no absolute local Windows paths. Final staged checks passed: 198 tracked files, approximately 6.49 MiB of tracked content, 35 changed files, 21/21 ignore probes, no staged secret-pattern matches and no file over 20 MB. The prior reachable-history scan covered 191 blobs. A read-only GitHub API check confirmed private visibility and main as the default branch; no visibility or license change was made.

## Public portfolio review decision

**Ready for scientific/portfolio review with the disclosed qualifications; not cleared for public release.** The prepared archive/label license, attribution requirements and permission to publish data-derived qualitative figures remain unresolved. Raw data/checkpoints are excluded, visibility is unchanged and no software license is added. Scientific limitations are part of the final narrative, not hidden blockers to candid review.

## Final verification record

The final cross-file check passed 126 assertions: headline TEST tables match frozen CSV/JSON, robustness values and population labels match the saved summaries, all eight TerraMind aggregate metrics have three metadata records, fixed tile counts/dates/windows agree, local Markdown links resolve, notebook experiment code is unchanged except its repository URL, and 107 original scientific/config/result/requirements/ignore/qualitative files match their previous Git content (with normal Git line-ending handling). The two historical TEST report writers changed only their inaccurate first-access sentence. All five redrawn charts were visually inspected; no overlapping/clipped class rows remain.

## Public-release cleanup (2026-10-04)

The nine raster-derived alignment/qualitative figures were removed from the current tree and all reachable public-release history. Their old image blobs are no longer reachable. A verified local-only backup bundle outside the repository retains the private record and must not be published. README and website now explicitly attribute data and teaching materials to the Newcastle University GeoAI Summer School and state that the original dataset is not redistributed. Scientific source code, aggregate results and safe metric figures are unchanged.

This cleanup addresses the retained-imagery blocker identified after the historical Phase 8/9 review above. Dataset/label and withheld-image permissions remain unresolved; removal does not grant redistribution rights. No software license or visibility change is made.
