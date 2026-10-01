# Colab Workflow for TerraMind

Phase: 6C

This workflow uses the validated Colab setup:

- Tesla T4, 14.56 GB
- torch `2.11.0+cu128`
- numpy `2.2.6`
- terratorch `1.2.13`
- torchgeo `0.9.0`

It trains from local `/content` storage and persists outputs back to Google Drive.

## 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Expected Drive files:

- `/content/drive/MyDrive/robust-multimodal-geoai/data/SummerSchool_Subset.zip`
- `/content/drive/MyDrive/robust-multimodal-geoai/model_cache/TerraMind-1.0-tiny-hf-cache.tar.gz`
- `/content/drive/MyDrive/robust-multimodal-geoai/checkpoints/`
- `/content/drive/MyDrive/robust-multimodal-geoai/colab_outputs/`

## 2. Clone or update the repository in local Colab storage

```bash
cd /content
if [ ! -d robust-multimodal-geoai ]; then
  git clone <YOUR_GITHUB_REPO_URL> robust-multimodal-geoai
else
  cd robust-multimodal-geoai
  git pull
fi
```

## 3. Install TerraMind dependencies without changing system CUDA

```bash
pip install -q terratorch==1.2.13 torchgeo==0.9.0 numpy==2.2.6 "setuptools<81"
```

Do not install or modify a system CUDA toolkit.

## 4. Restore TerraMind Hugging Face cache when available

```bash
mkdir -p /root/.cache/huggingface
if [ -f /content/drive/MyDrive/robust-multimodal-geoai/model_cache/TerraMind-1.0-tiny-hf-cache.tar.gz ]; then
  tar -xzf /content/drive/MyDrive/robust-multimodal-geoai/model_cache/TerraMind-1.0-tiny-hf-cache.tar.gz -C /root/.cache/huggingface
fi
```

## 5. Copy and extract the Summer School dataset to `/content`

```bash
cd /content/robust-multimodal-geoai
mkdir -p /content/geoai_data
cp /content/drive/MyDrive/robust-multimodal-geoai/data/SummerSchool_Subset.zip /content/geoai_data/SummerSchool_Subset.zip
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
archive = Path('/content/geoai_data/SummerSchool_Subset.zip')
out = Path('/content/geoai_data/SummerSchool_Subset')
out.mkdir(parents=True, exist_ok=True)
with ZipFile(archive) as zf:
    zf.extractall(out)
print('extracted', out)
PY
```

The runner reads rasters from `/content/geoai_data/SummerSchool_Subset`, not directly from Drive and not from Windows-style manifest paths.

## 6. Smoke test and train

```bash
cd /content/robust-multimodal-geoai
python scripts/run_terramind.py --batch-size 4 --epochs 10 --data-root /content/geoai_data/SummerSchool_Subset
```

The built-in smoke test verifies:

- real RGB + S1RTC batch
- logits `[B,4,224,224]`
- finite loss
- CUDA used
- backbone frozen
- decoder-only trainability
- no backbone gradients
- decoder gradients present
- 621 train tiles and 111 validation tiles
- no test dates loaded

Batch size 8 may be used only after the batch-size 4 Colab run shows it is safe.

## 7. Persist outputs back to Drive

```bash
cd /content/robust-multimodal-geoai
mkdir -p /content/drive/MyDrive/robust-multimodal-geoai/checkpoints
mkdir -p /content/drive/MyDrive/robust-multimodal-geoai/colab_outputs/terramind_frozen
cp checkpoints/terramind_frozen_best.pt /content/drive/MyDrive/robust-multimodal-geoai/checkpoints/
cp -r results/terramind_frozen /content/drive/MyDrive/robust-multimodal-geoai/colab_outputs/
cp figures/terramind_*.png /content/drive/MyDrive/robust-multimodal-geoai/colab_outputs/terramind_frozen/ || true
```

## 8. Test-date policy

Do not evaluate test dates in Phase 6C. The runner only constructs train and validation loaders.

