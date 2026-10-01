# TerraMind Feasibility Assessment

Phase: 6A  
Scope: feasibility only; no training, no test-set evaluation, no model-weight download.

## Summary recommendation

Use **TerraMind v1 tiny** as a low-cost Earth-observation foundation-model benchmark with a **frozen backbone plus a small segmentation decoder**. The fairest first experiment is the Summer School style **RGB + S1RTC multimodal TerraMind** setup adapted from binary segmentation to this project's 4-class algae task.

This is scientifically useful because it keeps the dataset, date split, tile size, labels, and robustness protocol comparable with the existing DeepLab baselines while testing whether a pretrained EO foundation backbone provides better multimodal representations.

## TerraMind variant used in the Summer School

The Summer School notebooks use:

- Variant: `terramind_v1_tiny`
- TerraTorch registry construction:
  - RGB: `BACKBONE_REGISTRY.build("terramind_v1_tiny", pretrained=True, modalities=["RGB"])`
  - Multimodal: `BACKBONE_REGISTRY.build("terramind_v1_tiny", pretrained=True, modalities=["RGB", "S1RTC"], merge_method="mean")`
- Pretrained weights: TerraTorch downloads the TerraMind tiny pretrained checkpoint, corresponding to IBM/ESA TerraMind 1.0 tiny model weights on Hugging Face.

The official TerraMind model card lists supported raw input modalities including `S2L2A`, `S2L1C`, `S1GRD`, `S1RTC`, `DEM`, and `RGB`. For this project, the Summer School material uses `RGB` and `S1RTC`, not the full Sentinel-2 L2A modality.

Source: official model card, `ibm-esa-geospatial/TerraMind-1.0-tiny`, Apache 2.0 license.

## Relevant supported modalities

### Sentinel-2 / optical

Two TerraMind paths are relevant:

1. **Summer School path: `RGB`**
   - Input bands: B04, B03, B02 as R, G, B.
   - Expected raw project tensor order before adapter: `R, G, B`.
   - Preprocessing recovered from Day 1/Day 2:
     - divide stored Sentinel-2 DN by 10000 where needed to get reflectance in `[0, 1]`
     - clip to `[0, 1]`
     - reorder RGB to BGR
     - multiply by `255.0`
   - This is directly compatible with the current aligned 224x224 B04 reference-grid tiles.

2. **Full Sentinel-2 path: `S2L2A` or `S2L1C`**
   - Official TerraMind supports these modalities and optional band subsets.
   - This would require a different preprocessing route and potentially more Sentinel-2 bands than the current DeepLab RGB baselines use.
   - It is not the lowest-cost fair comparison for the existing project baselines.

### Sentinel-1 / SAR

The Summer School multimodal TerraMind setup uses:

- TerraMind modality key: `S1RTC`
- Bands: VV and VH
- Expected units: terrain-corrected gamma-zero radar backscatter in dB
- Nodata: `-9999`, to be masked or handled before normalization
- Preprocessing recovered from Day 2:
  - VV/VH tensor order: `VV, VH`
  - mean: `[-10.930, -17.329]`
  - std: `[4.391, 4.459]`
  - normalized as `(x - mean) / std`

This differs from the earlier project DeepLab SAR pipeline, which used train-only mean/std statistics. A TerraMind benchmark should use the Summer School TerraMind S1RTC normalization to match the pretrained backbone.

### Multimodal input

The Day 2 notebook builds TerraMind with modalities:

- `RGB`
- `S1RTC`

The paired raw tensor order is:

`R, G, B, VV, VH`

The adapter returns a dictionary:

- `{"RGB": rgb_bgr_255, "S1RTC": normalized_vv_vh}`

The Summer School fusion uses TerraMind's joint encoder behavior and merges corresponding modality output tokens using `merge_method="mean"` before the segmentation decoder.

## Original Summer School segmentation setup

The TerraMind segmentation setup in the Summer School is a frozen-backbone fine-tuning workflow:

- Backbone: `terramind_v1_tiny`, pretrained, frozen
- Decoder: small convolutional segmentation head
- Original target: binary segmentation
  - background
  - algae
  - ignore value `255`
- Loss: `CrossEntropyLoss(ignore_index=255)`
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-3`
- Epochs: `10`
- Checkpoint selection: validation macro mIoU
- Tile size: `224x224`

For this project, the direct adaptation is:

- keep frozen `terramind_v1_tiny`
- change the decoder output from 2 classes to 4 classes
- preserve target classes `0, 1, 2, 3` and ignore `255`
- use the fixed project train/validation date split
- do not use test dates during development

## Required packages and weights

Recovered package requirements from the Summer School notebooks:

- `terratorch==1.2.12`
- `torchgeo==0.9.0`
- `numpy==2.2.6`
- `scipy>=1.15,<1.17`
- `setuptools<81`
- `gdown>=5.2,<6`
- PyTorch and torchvision compatible with the local CUDA runtime

Weights:

- TerraMind tiny pretrained weights from Hugging Face, loaded through TerraTorch with `pretrained=True`.
- The weights were not downloaded in Phase 6A.
- Exact cache size was not measured locally.

Installation risk:

- The current project environment was built for the rasterio and DeepLab workflows.
- Adding TerraTorch and TorchGeo may introduce dependency version constraints, especially around `numpy`, `torch`, `torchvision`, and geospatial packages.
- A separate environment or a carefully pinned install step is safer for the next phase.

## Approximate parameter count and VRAM expectations

Exact parameter count and local VRAM use were not measured in Phase 6A because TerraTorch and the pretrained weights were not installed or loaded.

The TerraMind tiny model card example emits 196 tokens with feature dimension 768 for a 224x224 input. This suggests a ViT-style tiny TerraMind backbone with a substantially larger feature representation than the MobileNetV3 DeepLab baseline.

Practical expectation on an RTX 4060 8GB:

| Strategy | Feasibility on 8GB | Expected notes |
|---|---:|---|
| Inference only | Feasible | Likely batch size 1-4 with mixed precision, depending on modality count and decoder memory. |
| Frozen backbone + segmentation head | Feasible | Best first benchmark. Use `torch.no_grad()` for the backbone, mixed precision, and start with batch size 2-4. Increase only after measuring memory. |
| Partial fine-tuning | Possible but risky | Fine-tuning the final encoder blocks may fit with batch size 1-2, mixed precision, and possibly gradient checkpointing. It is not the lowest-cost first experiment. |
| Full fine-tuning | Not recommended | Likely too memory intensive or unstable on 8GB for practical iteration. It also changes the comparison from foundation-feature benchmark to heavy model tuning. |

The expected safest configuration is frozen backbone plus decoder head. If implemented later, the first run should log parameter count, peak allocated VRAM, and batch-size limits before training.

## Compatibility with the current aligned dataset

The current dataset can support the Summer School TerraMind benchmark with small adapter changes:

- Tile size: compatible at `224x224`.
- Grid: compatible with the B04 reference grid used throughout the project.
- Optical bands: compatible for `RGB` using B04/B03/B02.
- SAR bands: compatible for `S1RTC` using aligned VV/VH, provided nodata and non-finite values are handled before normalization.
- Masks: compatible after changing decoder output to 4 classes and preserving ignore value `255`.

Important preprocessing distinction:

- The existing DeepLab datasets return already normalized tensors for RGB and SAR.
- TerraMind should receive raw aligned values and apply the TerraMind-specific adapter:
  - RGB reflectance to BGR multiplied by 255
  - VV/VH dB normalized by TerraMind S1RTC mean/std

Do not feed the current DeepLab normalized 5-channel tensor directly to TerraMind.

## Fairest benchmark against existing DeepLab models

The fairest low-cost benchmark is:

**Frozen TerraMind v1 tiny RGB + S1RTC segmentation**

Configuration:

- Input: B04/B03/B02 + VV/VH
- TerraMind modalities: `RGB`, `S1RTC`
- Backbone: pretrained `terramind_v1_tiny`, frozen
- Fusion: TerraMind multimodal joint encoder with mean token merge, following Day 2
- Head: same Summer School convolutional decoder, adapted to 4 classes
- Split: existing fixed date-level split
- Tile size: existing 224x224 tiles
- Loss: class-weighted CrossEntropyLoss with `ignore_index=255`, matching the selected DeepLab baseline policy unless the TerraMind experiment is explicitly designed as unweighted
- Training budget: 10 epochs, validation macro mIoU checkpoint selection
- Evaluation during development: validation only

This benchmark answers a clear question: whether a frozen EO foundation backbone provides a stronger or more robust multimodal representation than the trained DeepLab-MobileNetV3 early-fusion baselines under the same data split and robustness protocol.

A secondary, cheaper ablation can be added later if needed:

- Frozen TerraMind RGB-only, 4-class decoder

This would separate TerraMind pretraining effects from multimodal fusion effects, but the primary benchmark should be multimodal because the project's research question concerns robustness when optical information is degraded or one sensing modality is unavailable.

## Blockers before Phase 6B

1. `terratorch` and `torchgeo` are not installed in the current project environment.
2. TerraMind pretrained weights are not downloaded or cached locally.
3. Exact TerraMind parameter count and RTX 4060 peak VRAM must be measured after installing/loading the model.
4. The Summer School TerraMind notebooks are binary segmentation examples; the project needs a 4-class decoder and metric path.
5. TerraMind preprocessing must use raw aligned values, not the current DeepLab-normalized tensors.
6. Dependency conflicts are possible if TerraTorch requires specific PyTorch, TorchGeo, or NumPy versions.

## Phase 6A decision

Proceed, if requested in a later phase, with a **frozen TerraMind v1 tiny RGB + S1RTC benchmark**. This is the lowest-cost scientifically valid TerraMind setup for the current project and should be feasible on an RTX 4060 8GB with conservative batch size and mixed precision.
