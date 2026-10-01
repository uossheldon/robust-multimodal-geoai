# Robust Multimodal GeoAI

Phase 0 project scaffold for a research project on multimodal GeoAI segmentation robustness.

## Research Question

How robust are multimodal GeoAI segmentation models when optical satellite imagery is degraded or one sensing modality is unavailable?

## Phase 0 Status

This repository currently contains only project structure and planning documentation. It does not contain data, trained models, implemented model code, downloaded checkpoints, generated results, or executed experiments.

The reference material used for this Phase 0 scaffold is outside this project folder:

- `../2026 GeoAI Summer School Material/`
- `../analysis/`

Treat both folders as read-only reference material.

## Proposed Study Direction

The project is framed around semantic segmentation for Earth observation, with a focus on optical, SAR, and fused multimodal inputs. The planned robustness questions include:

- Performance under degraded optical conditions.
- Performance when RGB optical input is unavailable.
- Performance when SAR input is unavailable.
- Whether multimodal fusion improves or harms results under matched evaluation conditions.
- Whether foundation-model features are more robust than conventional segmentation baselines.

## Repository Layout

```text
robust-multimodal-geoai/
  README.md
  AGENTS.md
  requirements.txt
  configs/
  data/README.md
  docs/
    PROJECT_SCOPE.md
    SUMMER_SCHOOL_REFERENCE.md
  src/
    data/
    models/
    training/
    evaluation/
    visualization/
  scripts/
  results/
  figures/
  notebooks/
```

## Phase Boundaries

Phase 0 creates the project structure and records the research scope. It intentionally stops before implementation.

Phase 1 may begin later by auditing data access, licenses, environment setup, and a minimal reproducible baseline. Do not start Phase 1 inside this scaffold without an explicit new instruction.

