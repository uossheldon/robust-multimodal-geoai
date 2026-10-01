# Project Agent Instructions

This project is currently in Phase 0.

## Read-Only Reference Material

Treat these sibling folders as read-only references:

- `../2026 GeoAI Summer School Material/`
- `../analysis/`

Do not modify, rename, move, or delete files in those folders.

## Phase 0 Rules

- Do not download data.
- Do not install packages.
- Do not train models.
- Do not implement model architectures.
- Do not run notebooks.
- Do not re-analyse the source PDFs.
- Do not commit external datasets, pretrained weights, generated rasters, or large binary artifacts.

## Intended Project Direction

The research question is:

> How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

The likely starting point for future work is the summer school SAR and optical fusion material, using the existing analysis files for concise context. Future implementation should preserve strict train, validation, and test separation by date or scene, avoid pixel-level leakage, and report uncertainty at scene/date level where possible.

## Documentation Expectations

Keep documentation clear about what is:

- Present locally.
- Inferred from the summer school notebooks.
- Proposed for future work.
- Dependent on external data or checkpoints.

