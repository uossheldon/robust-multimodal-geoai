# Summer School Reference

This summary uses the existing `../analysis/` files as concise context. The source PDFs were not re-analysed for this finalisation. This is reference context; the independent experiments and final status are summarized in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

## Relevant Local Context

The summer school material contains notebooks and lectures covering supervised image learning, EO segmentation, SAR/optical fusion, diffusion-based cloud inpainting, and efficient segmentation. The existing analysis identifies the most connected sequence as:

```text
Day 1 Part 1 -> Day 1 Part 2 -> Day 2 -> Day 4
```

For this project, Day 2 is the most relevant starting point because it compares RGB optical, SAR, and fused segmentation workflows. Day 1 Part 2 provides the RGB algae segmentation baseline context. Day 4 provides later efficiency and deployment context. Day 3 is useful background for optical degradation and cloud-related evidence limits, but its local data contain only one visual scene and one mask, not a cloud-free reference target.

## Most Relevant Analysis Files

- `../analysis/02_lab_analysis.md`: static notebook analysis and limitations.
- `../analysis/03_knowledge_map.md`: conceptual links between lectures and labs.
- `../analysis/04_project_ideas.md`: project direction comparing SAR fusion, cloud inpainting, and deployable segmentation.
- `../analysis/technical_environment.md`: environment notes from the inspected notebooks.

## Working Interpretation

The implemented research direction is a controlled robustness study based on multimodal segmentation:

- RGB-only model.
- SAR-only model.
- RGB+SAR fused model.
- A frozen TerraMind RGB+S1RTC backbone with a lightweight decoder.
- Missing-modality and degraded-optical tests under common splits.

The project should avoid overclaiming geographic generalisation if the data remain limited to one lake or one region. A true cross-region claim requires additional external data with compatible labels.

## Known Constraints From Existing Analysis

- Day 1 Part 2 and Day 2 depend on `SummerSchool_Subset.zip`, retrieved in Phase 1.5 for local experiments and excluded from Git.
- Day 4 depends on an external prepared archive, checkpoint, and helper module that are not present locally.
- Day 2 notebook text and code differ on epoch count; future reproduction should verify code settings directly.
- The available Day 3 TIFF and PNG do not provide a cloud-free reference image.
- Day 3 image generation outputs would not preserve georeferencing unless separately managed.
- Existing Day 4 recorded results are historical notebook outputs and were not reproduced locally.

## Attribution and completed project

The Summer School supplied educational context, notebooks and the external prepared dataset. This project implemented independent aligned data pipelines, controlled robustness training/evaluation, uncertainty diagnostics and foundation-model benchmarking. Source reference folders remain read-only. No publication, novelty or source-data redistribution entitlement is implied.

