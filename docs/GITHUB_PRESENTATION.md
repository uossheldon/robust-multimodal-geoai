# GitHub Presentation and Project Website

Phase 9 and the final presentation pass change presentation only. Scientific source, configs, saved results and existing research documentation are preserved. No experiment, inference, training or post-TEST selection was performed. Repository visibility is unchanged; no software license was added.

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
  explorer.js    three-mode vanilla-JavaScript results explorer
  assets/results_summary.json  exact copy of approved frozen aggregates
  favicon.svg    original schematic mark
scripts/
  presentation_data.py   frozen summary loader / JSON asset generator
  render_final_presentation.py  four scientific PNG/SVG figures
  render_hero.py          original schematic PNG/SVG
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

`scripts/build_site.py` copies exactly 19 reviewed files: five site source files, the aggregate JSON asset, the hero SVG/PNG, social preview PNG, four numeric figure PNG/SVG pairs, the clean TEST CSV and an empty `.nojekyll`. It does not copy the repository root, all figures, all results or the data directory. Unexpected artifact files and symlinks cause a failure.

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

Presentation checks compare all 18 clean TEST table values, three headlines, six validation scores and the complete interactive JSON asset against saved sources, verify relative links/anchors and image alternatives, parse SVG assets, and prove the packaging audit rejects an unexpected TIFF-like file. No data, checkpoint, inference, training or package installation is involved. Existing scientific experiments are never part of CI.

## Review status

The final pass consolidates repeated qualifications in one Evaluation notes disclosure and shortens methods/reproduction/footer prose. Its main structure is hero → study design → three findings → explorer → clean TEST/per-class figures → optical robustness → temporal shift → TerraMind/uncertainty → notes/limitations → reproduction. Source scientific records, model code and prior qualitative figures remain unchanged.

The interactive explorer provides only measured optical-occlusion levels 0/10/30/50/70, six-model clean comparisons for three metrics, and validation–TEST pairs with a signed score difference. It does not interpolate, extrapolate, calculate percentage improvements, perform statistical testing or call models. Clean/missing and pooled optical-occlusion SD populations remain distinct. `site/assets/results_summary.json` is validated against the original CSV/JSON at full stored precision. It uses completed TerraMind evidence, not incomplete historical exports.

The four scientific figures are generated by `scripts/render_final_presentation.py`:

| File stem | Title | Axes |
|---|---|---|
| `final_clean_model_comparison` | Held-out September test performance | (a) x: Score (Macro mIoU / Macro Dice), y: Model; (b) x: Binary algae Dice, shared model rows |
| `final_robustness_curves` | Robustness to optical occlusion on the held-out September test set | (a) x: Optical occlusion rate (%), y: Macro mIoU; (b) x: Missing modality, y: Macro mIoU |
| `final_per_class_iou` | Per-class IoU on the held-out September test set | x: Class; y: Model; colorbar: IoU |
| `final_temporal_generalisation` | Validation-to-test performance shift | x: Macro mIoU; y: Model |

PNG exports are 220 dpi; SVGs retain text with portable sans-serif fallbacks. Error bars show saved sample SDs, with none fabricated for single runs. Robustness is drawn as measured dots only; temporal connectors pair two observed evaluations. Statistical/scoring context sits in the page's Evaluation notes and the detailed reports. The heatmap retains every numeric annotation. Historical TerraMind-specific Phase 8 figures remain unchanged.

Regenerate presentation assets locally (Matplotlib required only for figures):

```bash
python scripts/presentation_data.py
python scripts/render_final_presentation.py
python scripts/render_hero.py
python scripts/check_presentation.py
python scripts/check_lightweight.py
```

CI and Pages use pre-rendered figure assets; their checks/build require only standard-library Python and execute no model or data pipeline. No environment or package installation was added to either workflow. All external document URLs identify existing repository files; access depends on repository permissions.

The project is presentation-ready for review while remaining private. Publication is a separate manual choice; no Pages deployment or visibility change is performed by Phase 9. Source-data/label and retained qualitative-figure licensing issues remain documented in [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md).

## Final QA and deployment check (2026-10-03)

- Desktop 1440×1000, tablet 768×1024 and mobile 390×844 viewport checks passed, including light/dark themes. Wide plots and tables scroll within labelled, keyboard-focusable containers rather than overflowing the page.
- All five discrete optical levels, all three comparison metrics and all six temporal selections displayed their saved values. Native keyboard selection, Enter activation, visible focus and horizontal keyboard scrolling worked. Mobile controls exceed 44 px in height; no hover interaction is required. This is viewport QA, not a claim of physical-device testing.
- The no-script fixture retained the main narrative, static figures, accessible details and numeric TEST table. Interactive controls remain hidden until their local JSON loads. The README was rendered locally from Markdown; referenced images and links resolve.
- The temporal-figure legend was moved clear of data; SVG text uses explicit UTF-8 and portable sans-serif fallbacks. Static charts maintain a readable minimum width on small screens. No scientific value changed.
- Safe checks compile 39 Python files, execute two existing pure path assertions, compare all 18 clean table cells / three headlines / six validation scores / full JSON aggregates, and validate the 19-file artifact. No training, inference, raster processing or weight download is executed.
- GitHub's repository API confirmed private visibility. The Pages endpoint returned HTTP 404 (absent or inaccessible configuration), so an enabled/live deployment cannot be claimed. Check **Settings → Pages → Source: GitHub Actions** while keeping the repository private; the existing manual deployment workflow remains documented above. No visibility or licensing setting was changed.
