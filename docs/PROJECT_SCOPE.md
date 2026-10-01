# Project Scope

## Research Question

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

## Motivation

Earth observation segmentation systems often combine optical imagery with radar or foundation-model features. Optical imagery can fail under cloud, haze, acquisition gaps, shadows, or sensor issues. A practical model should make its limits visible when one modality is degraded or missing, instead of producing confident maps that hide sensor failure.

This project is intended to turn the summer school fusion material into a careful robustness study rather than a notebook reproduction.

## Phase 0 Deliverables

Phase 0 includes:

- Repository structure.
- Scope documentation.
- Reference summary from existing analysis files.
- Data and artifact boundaries.
- Placeholder folders for future code, configs, results, figures, and notebooks.

Phase 0 excludes:

- Data download.
- Package installation.
- Notebook execution.
- Model implementation.
- Training or inference.
- New PDF analysis.
- Phase 1 experiment setup.

## Candidate Future Baselines

Future phases may evaluate these families of methods, after data access and licensing are resolved:

- RGB-only segmentation baseline.
- SAR-only segmentation baseline.
- Early-fusion RGB and SAR segmentation baseline.
- Foundation-model RGB features.
- Foundation-model multimodal features.
- Missing-modality and degraded-optical ablations.

These are candidate directions only. They are not implemented in Phase 0.

## Robustness Questions

Future experiments should separate at least four cases:

1. Clean optical input with all modalities available.
2. Degraded optical input with SAR available.
3. Optical-only input when SAR is unavailable.
4. SAR-only or fallback input when optical imagery is unavailable.

Useful degradation settings may include real cloud/invalid-pixel masks, synthetic optical masking, reduced optical quality, and missing-channel tests. Any synthetic corruption should be clearly marked as synthetic.

## Evaluation Principles

Future evaluation should:

- Keep train, validation, and test data split by date, scene, or region.
- Avoid treating nearby pixels as independent samples for uncertainty claims.
- Report per-date or per-scene metrics in addition to aggregate scores.
- Use validation data for thresholds and model selection.
- Evaluate the final test set once after choices are frozen.
- Record missing-modality handling explicitly.
- Distinguish model failure from data-alignment or label-quality failure.

Candidate metrics include mean IoU, class IoU, Dice, precision, recall, false-positive area, calibration error, per-scene variance, and latency or memory if deployment robustness is studied.

## Phase 1 Entry Criteria

Start Phase 1 only after explicit approval and after documenting:

- Which dataset will be used.
- Whether the external summer school archives are accessible.
- Data rights and redistribution limits.
- Compute target and environment plan.
- Minimal baseline to reproduce first.
- Success criteria for a small pilot run.

