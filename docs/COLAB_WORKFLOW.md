# TerraMind Colab reproduction

This workflow reproduces the frozen TerraMind benchmark without relying on any personal cloud-storage layout.

## Environment

Validated setup:

- NVIDIA T4
- torch `2.11.0+cu128`
- numpy `2.2.6`
- TerraTorch `1.2.13`
- TorchGeo `0.9.0`

## 1. Clone the repository

```bash
cd /content
git clone https://github.com/uossheldon/robust-multimodal-geoai.git
cd robust-multimodal-geoai
```

## 2. Install TerraMind dependencies

```bash
pip install -q terratorch==1.2.13 torchgeo==0.9.0 numpy==2.2.6 "setuptools<81"
```

Verify the resolved PyTorch build before training. Do not replace the Colab CUDA toolkit.

## 3. Prepare the dataset

Place or extract the prepared Summer School subset at:

```text
/content/geoai_data/SummerSchool_Subset/
```

Expected structure:

```text
images/<YYYY-MM-DD>/layers/
masks/
```

The dataset itself is not distributed by this repository.

## 4. Train the frozen-backbone decoder

```bash
python scripts/run_terramind.py \
  --batch-size 4 \
  --epochs 10 \
  --seed 42 \
  --data-root /content/geoai_data/SummerSchool_Subset
```

Repeat with seeds `7` and `123` using identical settings.

The runner checks the multimodal batch, output shape, finite loss, frozen backbone, decoder gradients, split counts and exclusion of test dates before full training.

## 5. Evaluation policy

Checkpoint selection uses validation macro mIoU. The September test dates are excluded from training and validation. Do not use test outcomes to select hyperparameters or checkpoints.

The canonical three-seed validation record is [terramind_validation.json](../results/terramind_validation.json); final test results are reported in [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md).
