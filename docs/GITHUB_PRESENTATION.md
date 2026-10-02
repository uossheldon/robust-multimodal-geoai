# GitHub Presentation and Project Website

Phase 9 changes presentation only. Scientific source, configs, saved results and existing research documentation are preserved. No experiment, inference, training or post-TEST selection was performed. Repository visibility is unchanged; no software license was added.

## Landing page and assets

The README opening now includes the research question, local static technology badges, an original vector schematic and three final TEST findings. The detailed scientific content remains below it, including earlier Phase 2D TEST access, differing valid-pixel support and SD populations.

- `figures/hero_overview.svg`: scalable schematic; no dataset imagery or labels.
- `figures/hero_overview.png`: raster export for GitHub and presentations.
- `figures/social_preview.png`: 1280×640 social preview, derived from the same saved results.
- `figures/badges/`: locally rendered text badges; no external badge service or telemetry.
- `scripts/render_hero.py`: optional asset regeneration using saved metrics and Matplotlib, with no inference. It is not needed for website deployment.

All headline values derive from the frozen Phase 7 summary. Clean robust fusion mIoU is 0.1882 ± 0.0293 (three training seeds), 70% occlusion mIoU is 0.2122 ± 0.0115 (nine pooled training-seed × corruption-seed runs), and TerraMind binary algae Dice is 0.6821 ± 0.0025 (three training seeds). These SD populations differ and are not confidence intervals.

## Static site structure

```text
site/
  index.html     complete semantic research narrative and result table
  style.css      responsive layout, system light/dark themes, print styles
  theme.js       optional theme toggle; content works without JavaScript
  favicon.svg    original schematic mark
scripts/
  build_site.py          explicit deployment allowlist
  check_presentation.py  metric/link/asset/accessibility checks
  check_lightweight.py   static Python compilation and existing path assertions
.site-build/              generated artifact, ignored by Git
```

No Node build chain, external fonts, analytics, user credentials, model packages or GPU are needed. The site uses local assets with relative URLs so it works below the repository's GitHub Pages path. Source-document links go to GitHub and require access while the repository is private. The configured social URLs assume the default Pages hostname; change them only if a custom domain is actually adopted.

## Local preview

Run from the repository root with Python 3.11 or later:

```bash
python scripts/check_lightweight.py
python scripts/check_presentation.py
python -m http.server 8000 --bind 127.0.0.1 --directory .site-build
```

Open `http://127.0.0.1:8000`. The presentation check builds the site. Alternatively, `python scripts/build_site.py` performs just the static copy. Both use standard-library Python only. Stop the server with Ctrl+C.

## What is deployed / withheld

`scripts/build_site.py` copies only four site source files, the hero SVG/PNG, social preview PNG, two existing numerical summary charts (`final_per_class_iou.png`, `final_robustness_curves.png`), and `clean_test_results.csv`, plus an empty `.nojekyll`. It does not copy the repository root, all figures, all results or the data directory. Unexpected artifact files and symlinks cause a failure.

**Withheld from the site:** all source-raster-derived figures, including `final_qualitative_test_examples.png`, `data_alignment_check.png`, the S2/S1/fusion qualitative predictions, corruption qualitative examples and uncertainty/ensemble maps. These may depend on unresolved prepared-dataset/label and derived-figure rights. They remain in the private research record where already present; making the whole repository public would require a separate review of those retained figures. No permission to republish them is inferred.

The deployed images contain only original schematic elements or aggregate numeric summaries. This does not resolve the source archive/label license, imply permission to redistribute source data, or add a software license. Review the complete static artifact before choosing public deployment.

## One-time GitHub Pages settings

GitHub Pages from a private repository requires an eligible paid plan. A Pages website can be publicly accessible even when its repository remains private; repository visibility and website visibility are separate. If private-repository Pages is unavailable, leave the repository private and use the local preview rather than changing visibility. See [GitHub availability and workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [publishing-source/visibility guidance](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

When ready to publish the reviewed static presentation:

1. Open repository **Settings → Pages**. Under **Build and deployment → Source**, select **GitHub Actions**. Do not change repository visibility.
2. In **Settings → Environments → github-pages**, restrict deployment branches to `main`. Add a required reviewer if your plan supports that protection.
3. Open **Actions → Deploy research site to Pages → Run workflow**. Select `main` and enable **publish_reviewed_site** only after checking the static artifact and accepting its intended public accessibility.
4. Run the workflow. It builds/checks the allowlisted artifact, uses GitHub's Pages artifact upload, and deploys through the `github-pages` environment. Approve the deployment if environment protection requests it.
5. Open the URL reported by the deployment. The expected default is `https://uossheldon.github.io/robust-multimodal-geoai/`; it is not asserted live before a successful deployment.

The workflow is **manual only**. A push does not publish or enable Pages. Repeat step 3 for future reviewed updates. No custom PAT, deploy key or repository secret is required; GitHub supplies its short-lived workflow token/OIDC authorization. Pages permissions are confined to the deployment job. Workflow versions follow the [official custom-workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), checked during Phase 9.

## Repository metadata recommendations (not automatically applied)

**Description**

> Robust multimodal GeoAI for Sentinel-1/Sentinel-2 algal bloom segmentation under optical degradation and missing sensors.

**Topics**

`geoai`, `remote-sensing`, `sentinel-1`, `sentinel-2`, `earth-observation`, `multimodal-learning`, `semantic-segmentation`, `robustness`, `terramind`, `pytorch`.

Set these through the repository **About → gear icon**. Set the homepage URL only after a successful Pages deployment.

**Social preview:** repository **Settings → General → Social preview → Edit → Upload an image**; choose `figures/social_preview.png`. Its 1280×640 size follows [GitHub's social-preview guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview). Uploading this schematic requires no dataset imagery.

## Lightweight CI

`.github/workflows/checks.yml` runs on `main` pushes, pull requests and manual dispatch, with read-only repository permissions. It compiles Python source without importing experiment runners, then executes the two existing path-resolution assertions by AST-isolating the actual pure path resolver. This intentionally avoids importing/installing PyTorch, rasterio or TerraTorch. It tests path construction only, not raster loading or model behaviour.

Presentation checks compare all 18 clean TEST table values and three headlines against saved sources, verify relative links/anchors and image alternatives, parse SVG assets, and prove the packaging audit rejects an unexpected TIFF-like file. No data, checkpoint, inference, training or package installation is involved. Existing scientific experiments are never part of CI.

## Review status

Local visual QA covered a 1440×1000 desktop viewport and 390×844 / 320×740 mobile viewports, light/dark themes, stacked method cards, the results table and reproduction section. No page-level horizontal overflow or browser console errors were observed; the wide results table scrolls within its own container. The README landing was also rendered locally from Markdown, with its badges and hero loading correctly. No layout correction was needed. This is browser viewport testing, not testing on physical mobile devices.

The local lightweight checks passed: 38 Python files compile, both existing pure path assertions pass, and all 18 clean TEST table cells plus three headline values match the frozen sources. Relative site links, anchors, image dimensions/alternatives, SVGs and the 11-file deployment allowlist pass. These checks do not execute scientific models.

The project is presentation-ready for review while remaining private. Publication is a separate manual choice; no Pages deployment or visibility change is performed by Phase 9. Source-data/label and retained qualitative-figure licensing issues remain documented in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md).
