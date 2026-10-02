# Project Agent Instructions

The research experiments and Phase 7 final test comparison are complete. Phases 8–9 cover documentation, scientific auditing and static presentation only. The Pages workflow is manual; do not deploy or change repository visibility as part of a presentation push.

## Frozen scientific record

- Do not train, tune, rerun inference, select checkpoints using TEST, or alter saved scientific results during finalisation.
- Preserve the split, tile manifest, preprocessing, architecture, loss and experiment settings.
- Distinguish validation/TEST, single runs/three-seed means, and training/corruption seeds. Document contradictions; retain historical evidence.
- Read `docs/SCIENTIFIC_AUDIT.md` before claims about test access, pixel-matched comparisons or reproducibility status.
- Presentation scripts may read saved CSV/JSON without executing models.

## Read-only references

Treat `../2026 GeoAI Summer School Material/` and `../analysis/` as read-only. Do not modify, move, rename or delete them, or re-analyse lecture PDFs for finalisation.

## Distribution and writing

- Do not commit datasets, TIFFs, archives, checkpoints, model caches, credentials or large binaries.
- Do not change visibility or add a software license without explicit authorization.
- Dataset/label provenance and permission to publish data-derived figures need review before public release.
- Use “simulated optical occlusion”; make no real-cloud, universal foundation-model superiority or statistical-significance claims.
- Keep negative findings and temporal generalisation limits visible.
