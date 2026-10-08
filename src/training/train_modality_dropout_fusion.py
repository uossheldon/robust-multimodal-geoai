from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import fit_sar_train_statistics
from src.evaluation.robustness_benchmark import (
    OCCLUSION_FRACTIONS,
    SEEDS,
    add_degradation,
    evaluate_model,
    stable_int,
)
from src.models.deeplab import create_fusion_deeplab

from src.training.train_s1_deeplab import binary_metrics_from_loader, prediction_distribution
from src.training.train_s2_deeplab import run_epoch, seed_everything

RESULT_DIR = PROJECT_ROOT / "results" / "modality_dropout"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs"
MANIFEST_PATH = PROJECT_ROOT / "results" / "tile_manifest.csv"
METRIC_KEYS = ["mean_iou", "macro_dice", "iou_background", "iou_low", "iou_mid", "iou_high", "binary_algae_iou", "binary_algae_dice"]


class ModalityDropoutDataset(S1S2EarlyFusionDataset):
    def __init__(self, *args, dropout_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_seed = dropout_seed

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        tile_id = str(self.manifest.iloc[index].tile_id)
        rng = np.random.default_rng(stable_int(self.dropout_seed, tile_id))
        value = rng.random()
        if value < 0.50:
            mode = "both"
        elif value < 0.75:
            mode = "s2_only"
            image[3:5] = 0.0
        else:
            mode = "s1_only"
            image[0:3] = 0.0
        return image, target


def train_class_weights():
    manifest = pd.read_csv(MANIFEST_PATH)
    train = manifest[manifest["split"] == "train"]
    counts = np.array([train[f"class_{k}_count"].sum() for k in range(4)], dtype=np.float64)
    weights = 1.0 / np.sqrt(counts / counts.sum())
    weights = weights / weights.mean()
    return {f"class_{k}": int(counts[k]) for k in range(4)}, {f"class_{k}": float(weights[k]) for k in range(4)}


def make_train_loader(stats):
    ds = ModalityDropoutDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split="train", sar_mean=stats["mean"], sar_std=stats["std"], augment=True, dropout_seed=42)
    return DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42), pin_memory=True)


def make_val_loader(stats):
    ds = S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split="validation", sar_mean=stats["mean"], sar_std=stats["std"], augment=False)
    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)


def write_config(stats, weights):
    text=f"""experiment: s1_s2_modality_dropout
base_model: DeepLabV3-MobileNetV3-Large early fusion
input_channels: [B04, B03, B02, VV, VH]
split: configs/split_v1.yaml
tile_size: 224
batch_size: 8
epochs: 10
optimizer:
  name: AdamW
  backbone_lr: 1.0e-5
  classifier_lr: 1.0e-4
  weight_decay: 1.0e-3
loss:
  name: CrossEntropyLoss
  ignore_index: 255
  class_weights:
    class_0: {weights['class_0']:.12f}
    class_1: {weights['class_1']:.12f}
    class_2: {weights['class_2']:.12f}
    class_3: {weights['class_3']:.12f}
training_change:
  modality_dropout_after_normalization:
    both_modalities: 0.50
    zero_s1_channels: 0.25
    zero_s2_channels: 0.25
    drop_both: false
    deterministic_seed: 42
preprocessing:
  s2: reflectance divided by 10000, clipped to [0,1], ImageNet normalized
  s1: VV/VH GAMMA0_TERRAIN dB, train-only mean/std normalized
  s1_train_mean: [{stats['mean'][0]:.12f}, {stats['mean'][1]:.12f}]
  s1_train_std: [{stats['std'][0]:.12f}, {stats['std'][1]:.12f}]
model_selection: validation macro mIoU clean input
use_test_set: false
"""
    (CONFIG_DIR / "s1_s2_modality_dropout.yaml").write_text(text, encoding="utf-8")


def load_original_fusion_rows():
    clean = pd.read_csv(PROJECT_ROOT / "results" / "s1_s2_early_fusion" / "validation_modality_comparison.csv")
    orig_clean = clean[clean["experiment"] == "s1_s2_early_fusion"].iloc[0].to_dict()
    missing = pd.read_csv(PROJECT_ROOT / "results" / "robustness" / "missing_modality.csv")
    return orig_clean, missing[(missing["model"] == "s1_s2_early_fusion") & (missing["date"] == "all_validation")]


def train_full():
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    for d in [RESULT_DIR, CHECKPOINT_DIR, CONFIG_DIR]: d.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    stats=fit_sar_train_statistics(MANIFEST_PATH, project_root=PROJECT_ROOT)
    train_counts, weights=train_class_weights(); write_config(stats, weights)
    train_loader=make_train_loader(stats); val_loader=make_val_loader(stats)
    model=create_fusion_deeplab(num_classes=4, pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([weights[f"class_{k}"] for k in range(4)], dtype=torch.float32, device=device))
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":1e-5},{"params":model.classifier.parameters(),"lr":1e-4}], weight_decay=1e-3)
    ckpt=CHECKPOINT_DIR / "s1_s2_modality_dropout_best.pt"
    history=[]; best=-float("inf"); best_epoch=0; start=time.perf_counter()
    for epoch in range(1,11):
        tr,_=run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        va,conf=run_epoch(model, val_loader, criterion, device)
        row={"epoch":epoch}; row.update({f"train_{k}":v for k,v in tr.items()}); row.update({f"validation_{k}":v for k,v in va.items()}); history.append(row)
        print(f"moddrop epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_miou={va['mean_iou']:.4f}")
        if va["mean_iou"] > best:
            best=va["mean_iou"]; best_epoch=epoch
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"validation_metrics":va,"validation_confusion_matrix":conf.numpy().tolist(),"sar_statistics":stats,"class_weights":weights}, ckpt)
    torch.cuda.synchronize(); train_time=time.perf_counter()-start
    checkpoint=torch.load(ckpt, map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    clean, clean_conf=run_epoch(model, val_loader, criterion, device)
    clean_binary=binary_metrics_from_loader(model, val_loader, device); clean_pred=prediction_distribution(model,val_loader,device)
    pd.DataFrame(history).to_csv(RESULT_DIR / "history.csv", index=False)
    clean_payload={"validation_metrics":clean,"validation_binary_metrics":clean_binary,"validation_prediction_distribution":clean_pred,"validation_confusion_matrix":clean_conf.numpy().tolist(),"best_epoch":best_epoch,"training_time_seconds":train_time,"peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated(device)/(1024**2))}
    (RESULT_DIR / "clean_metrics.json").write_text(json.dumps(clean_payload, indent=2), encoding="utf-8")

    # Validation protocol: clean/missing and optical occlusion diagnostic on all validation.
    rows=[]
    for cond,mode in [("clean","clean"),("missing_s1_zero","fusion_missing_s1"),("missing_s2_zero","fusion_missing_s2")]:
        rows.append(evaluate_model(model,val_loader,device,model_name="s1_s2_modality_dropout",condition=cond,occlusion_fraction=np.nan,seed=np.nan,date_filter=None,input_mode=mode))
    missing=add_degradation(pd.DataFrame(rows)); missing.to_csv(RESULT_DIR / "missing_modality.csv", index=False)
    occ=[]
    for frac in OCCLUSION_FRACTIONS:
        for seed in SEEDS:
            occ.append(evaluate_model(model,val_loader,device,model_name="s1_s2_modality_dropout",condition="clean" if frac==0 else "simulated_optical_occlusion",occlusion_fraction=frac,seed=seed,date_filter=None,input_mode="fusion_occlusion"))
    occ=add_degradation(pd.DataFrame(occ)); occ.to_csv(RESULT_DIR / "optical_occlusion_diagnostic.csv", index=False)

    orig_clean, orig_missing=load_original_fusion_rows()
    comparison_rows=[]
    comparison_rows.append({"experiment":"original","condition":"clean", **{k:float(orig_clean[k]) for k in METRIC_KEYS}})
    for cond in ["missing_s1_zero","missing_s2_zero"]:
        r=orig_missing[orig_missing.condition==cond].iloc[0]
        comparison_rows.append({"experiment":"original","condition":cond, **{k:float(r[k]) for k in METRIC_KEYS}})
    for _,r in missing.iterrows():
        comparison_rows.append({"experiment":"modality_dropout","condition":r.condition, **{k:float(r[k]) for k in METRIC_KEYS}})
    comp=pd.DataFrame(comparison_rows)

    def val(exp, cond, metric): return float(comp[(comp.experiment==exp)&(comp.condition==cond)][metric].iloc[0])
    payload={
        "best_epoch":best_epoch,"training_time_seconds":train_time,"peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated(device)/(1024**2)),
        "train_class_counts":train_counts,"class_weights":weights,"clean":clean_payload,"missing_modality":missing.to_dict(orient="records"),"optical_occlusion_diagnostic":occ.to_dict(orient="records"),
        "comparison":{
            "clean":{"original_mean_iou":val("original","clean","mean_iou"),"dropout_mean_iou":val("modality_dropout","clean","mean_iou")},
            "missing_s1":{"original_mean_iou":val("original","missing_s1_zero","mean_iou"),"dropout_mean_iou":val("modality_dropout","missing_s1_zero","mean_iou")},
            "missing_s2":{"original_mean_iou":val("original","missing_s2_zero","mean_iou"),"dropout_mean_iou":val("modality_dropout","missing_s2_zero","mean_iou")},
        },
        "comparison_rows":comp.to_dict(orient="records"),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2))

if __name__ == "__main__":
    train_full()
