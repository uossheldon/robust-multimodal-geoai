from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import fit_sar_train_statistics
from src.models.deeplab import create_fusion_deeplab
from src.training.train_s1_deeplab import binary_metrics_from_loader, prediction_distribution, stretch_channel, color_mask
from src.training.train_s2_deeplab import class_metrics_frame, run_epoch, seed_everything

RESULT_DIR = PROJECT_ROOT / "results" / "s1_s2_early_fusion"
FIGURE_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_DIR = PROJECT_ROOT / "configs"
MANIFEST_PATH = PROJECT_ROOT / "results" / "tile_manifest.csv"
CLASS_COLORS = {0:(40,40,40),1:(121,190,85),2:(247,188,65),3:(218,83,63),255:(210,210,210)}


def train_class_weights():
    manifest = pd.read_csv(MANIFEST_PATH)
    train = manifest[manifest["split"] == "train"]
    counts = np.array([train[f"class_{k}_count"].sum() for k in range(4)], dtype=np.float64)
    freqs = counts / counts.sum()
    weights = 1.0 / np.sqrt(freqs)
    weights = weights / weights.mean()
    return {f"class_{k}": int(counts[k]) for k in range(4)}, {f"class_{k}": float(weights[k]) for k in range(4)}


def make_loader(split, stats, *, batch_size=8, augment=False, shuffle=False):
    dataset = S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split=split, sar_mean=stats["mean"], sar_std=stats["std"], augment=augment)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator, pin_memory=True)


def load_metric_row(path, experiment):
    rows = pd.read_csv(path)
    row = rows[rows["experiment"] == experiment].iloc[0]
    return {key: float(row[key]) for key in ["mean_iou","macro_dice","iou_background","iou_low","iou_mid","iou_high","binary_algae_iou","binary_algae_dice"]}


def smoke_test(stats, weights, device):
    loader = make_loader("train", stats, batch_size=8, augment=True, shuffle=True)
    images, targets = next(iter(loader))
    model = create_fusion_deeplab(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([weights[f"class_{k}"] for k in range(4)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 1e-5}, {"params": model.classifier.parameters(), "lr": 1e-4}], weight_decay=1e-3)
    images = images.to(device); targets = targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(images)["out"]
        loss = criterion(logits, targets)
    loss.backward(); optimizer.step()
    return {"input_shape": list(images.shape), "target_shape": list(targets.shape), "logit_shape": list(logits.shape), "loss": float(loss.detach().item()), "loss_finite": bool(torch.isfinite(loss).item()), "cuda_used": bool(images.is_cuda and logits.is_cuda)}


def save_curves(history):
    canvas = Image.new("RGB", (900,360), "white"); draw=ImageDraw.Draw(canvas)
    draw.text((24,16), "S1+S2 early fusion training curves", fill=(0,0,0))
    for title, train_col, val_col, x0, y0 in [("Loss","train_loss","validation_loss",50,70),("Validation macro mIoU",None,"validation_mean_iou",500,70)]:
        draw.text((x0,y0-28), title, fill=(0,0,0)); draw.rectangle((x0,y0,x0+330,y0+220), outline=(0,0,0))
        vals=[]
        if train_col: vals += history[train_col].tolist()
        vals += history[val_col].tolist(); lo=min(vals); hi=max(vals)
        if hi <= lo: hi=lo+1
        def pts(col):
            s=history[col].tolist(); return [(x0+int(i*330/max(1,len(s)-1)), y0+220-int((v-lo)*220/(hi-lo))) for i,v in enumerate(s)]
        if train_col:
            draw.line(pts(train_col), fill=(70,120,210), width=3); draw.text((x0,y0+232), "train", fill=(70,120,210))
        draw.line(pts(val_col), fill=(210,80,80), width=3); draw.text((x0+90,y0+232), "validation", fill=(210,80,80))
    canvas.save(FIGURE_DIR / "fusion_training_curves.png")


def save_per_class_iou(metrics):
    frame = class_metrics_frame(metrics)
    canvas=Image.new("RGB", (760,360), "white"); draw=ImageDraw.Draw(canvas)
    draw.text((24,18), "Fusion validation per-class IoU", fill=(0,0,0)); max_iou=max(float(frame["iou"].max()), 1e-6)
    for idx,row in enumerate(frame.itertuples(index=False)):
        y=70+idx*64; w=int(500*float(row.iou)/max_iou)
        draw.text((24,y+12), str(row.class_name), fill=(0,0,0)); draw.rectangle((170,y,170+w,y+42), fill=CLASS_COLORS[int(row.class_id)]); draw.text((182+w,y+12), f"{row.iou:.3f}", fill=(0,0,0))
    canvas.save(FIGURE_DIR / "fusion_per_class_iou.png")


def denormalize_rgb(image):
    mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    rgb=(image[:3].cpu()*std+mean).clamp(0,1).numpy()
    return (np.transpose(rgb,(1,2,0))*255).astype(np.uint8)


def save_qualitative(model, loader, stats, device, max_examples=4):
    mean=torch.tensor(stats["mean"], dtype=torch.float32).view(2,1,1); std=torch.tensor(stats["std"], dtype=torch.float32).view(2,1,1)
    examples=[]; model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type=="cuda"):
        for images, targets in loader:
            preds=model(images.to(device))["out"].argmax(dim=1).cpu(); raw_sar=images[:,3:5].cpu()*std+mean
            for image,sar,target,pred in zip(images.cpu(),raw_sar,targets,preds):
                valid=target!=255; err=torch.full_like(target,255); err[valid & (pred==target)] = 0; err[valid & (pred!=target)] = 3
                vv=np.repeat(stretch_channel(sar[0])[:,:,None],3,axis=2); vh=np.repeat(stretch_channel(sar[1])[:,:,None],3,axis=2)
                examples.append((denormalize_rgb(image), vv, vh, color_mask(target.numpy()), color_mask(pred.numpy()), color_mask(err.numpy())))
                if len(examples)>=max_examples: break
            if len(examples)>=max_examples: break
    tile=224; label_h=28; canvas=Image.new("RGB", (tile*6,(tile+label_h)*len(examples)), "white"); draw=ImageDraw.Draw(canvas)
    labels=["RGB","VV","VH","Ground Truth","Fusion Prediction","Error Map"]
    for r,ex in enumerate(examples):
        y=r*(tile+label_h)
        for c,arr in enumerate(ex):
            x=c*tile; draw.text((x+8,y+7), labels[c], fill=(0,0,0)); canvas.paste(Image.fromarray(arr),(x,y+label_h))
    canvas.save(FIGURE_DIR / "fusion_qualitative_predictions.png")


def save_comparison_figure(rows):
    canvas=Image.new("RGB", (1080,600), "white"); draw=ImageDraw.Draw(canvas)
    draw.text((24,18), "Validation modality comparison", fill=(0,0,0))
    colors={"s2_weighted":(80,130,210),"s1_weighted":(92,160,95),"s1_s2_early_fusion":(210,120,50)}
    metrics=[("mean_iou","macro mIoU"),("macro_dice","macro Dice"),("iou_background","IoU 0"),("iou_low","IoU 1"),("iou_mid","IoU 2"),("iou_high","IoU 3"),("binary_algae_iou","Binary algae IoU"),("binary_algae_dice","Binary algae Dice")]
    maxv=max(float(rows[c].max()) for c,_ in metrics)
    y=65
    for col,label in metrics:
        draw.text((24,y+22), label, fill=(0,0,0))
        for i,exp in enumerate(["s2_weighted","s1_weighted","s1_s2_early_fusion"]):
            v=float(rows.loc[rows["experiment"]==exp,col].iloc[0]); w=int(700*v/maxv); yy=y+i*20
            draw.rectangle((190,yy,190+w,yy+14), fill=colors[exp]); draw.text((198+w,yy-2), f"{exp} {v:.3f}", fill=(0,0,0))
        y += 62
    canvas.save(FIGURE_DIR / "modality_comparison_validation.png")


def write_config(stats, weights):
    text=f"""experiment: s1_s2_early_fusion
model: DeepLabV3-MobileNetV3-Large
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
preprocessing:
  s2: reflectance divided by 10000, clipped to [0,1], ImageNet normalized
  s1: VV/VH GAMMA0_TERRAIN dB, train-only mean/std normalized
  s1_train_mean: [{stats['mean'][0]:.12f}, {stats['mean'][1]:.12f}]
  s1_train_std: [{stats['std'][0]:.12f}, {stats['std'][1]:.12f}]
stem_initialization: retain pretrained RGB weights for B04/B03/B02; initialize VV/VH from mean RGB stem weights
model_selection: validation macro mIoU
use_test_set: false
"""
    (CONFIG_DIR / "s1_s2_early_fusion.yaml").write_text(text, encoding="utf-8")


def train_full():
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".torch"))
    seed_everything(42)
    for d in (RESULT_DIR,FIGURE_DIR,CHECKPOINT_DIR,CONFIG_DIR): d.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    stats=fit_sar_train_statistics(MANIFEST_PATH, project_root=PROJECT_ROOT)
    train_counts, weights=train_class_weights(); write_config(stats, weights)
    smoke=smoke_test(stats, weights, device)
    if not smoke["loss_finite"] or not smoke["cuda_used"]: raise RuntimeError(f"Smoke test failed: {smoke}")
    train_loader=make_loader("train", stats, batch_size=8, augment=True, shuffle=True); val_loader=make_loader("validation", stats, batch_size=8)
    model=create_fusion_deeplab(num_classes=4, pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([weights[f"class_{k}"] for k in range(4)], dtype=torch.float32, device=device))
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":1e-5},{"params":model.classifier.parameters(),"lr":1e-4}], weight_decay=1e-3)
    ckpt=CHECKPOINT_DIR / "s1_s2_early_fusion_best.pt"; history=[]; best=-float("inf"); best_epoch=0; start=time.perf_counter()
    for epoch in range(1,11):
        tr,_=run_epoch(model, train_loader, criterion, device, optimizer=optimizer); va,conf=run_epoch(model, val_loader, criterion, device)
        row={"epoch":epoch}; row.update({f"train_{k}":v for k,v in tr.items()}); row.update({f"validation_{k}":v for k,v in va.items()}); history.append(row)
        print(f"fusion epoch={epoch} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_miou={va['mean_iou']:.4f}")
        if va["mean_iou"] > best:
            best=va["mean_iou"]; best_epoch=epoch
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"validation_metrics":va,"validation_confusion_matrix":conf.numpy().tolist(),"sar_statistics":stats,"class_weights":weights}, ckpt)
    torch.cuda.synchronize(); train_time=time.perf_counter()-start
    checkpoint=torch.load(ckpt, map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    va,conf=run_epoch(model, val_loader, criterion, device); binary=binary_metrics_from_loader(model,val_loader,device); pred=prediction_distribution(model,val_loader,device)
    history_frame=pd.DataFrame(history); history_frame.to_csv(RESULT_DIR / "history.csv", index=False)
    class_metrics_frame(va).to_csv(RESULT_DIR / "per_class_metrics.csv", index=False)
    s2=load_metric_row(PROJECT_ROOT/"results"/"s1_s2_early_fusion_dummy.csv", "none") if False else load_metric_row(PROJECT_ROOT/"results"/"s2_deeplab_balanced_sampling"/"validation_comparison.csv", "weighted_loss")
    s1=load_metric_row(PROJECT_ROOT/"results"/"s1_deeplab_weighted"/"validation_vs_s2.csv", "s1_weighted")
    fusion={**va, **binary}
    metric_keys=["mean_iou","macro_dice","iou_background","iou_low","iou_mid","iou_high","binary_algae_iou","binary_algae_dice"]
    comparison=pd.DataFrame([{ "experiment":"s2_weighted", **s2 }, {"experiment":"s1_weighted", **s1}, {"experiment":"s1_s2_early_fusion", **{k:fusion[k] for k in metric_keys}, **pred}])
    comparison.to_csv(RESULT_DIR / "validation_modality_comparison.csv", index=False)
    payload={"training_time_seconds":train_time,"best_epoch":best_epoch,"best_validation_miou_during_training":best,"peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated(device)/(1024**2)),"sar_statistics":stats,"train_class_counts":train_counts,"class_weights":weights,"smoke_test":smoke,"validation_metrics":va,"validation_binary_metrics":binary,"validation_prediction_distribution":pred,"validation_confusion_matrix":conf.numpy().tolist(),"validation_comparison":comparison.to_dict(orient="records")}
    (RESULT_DIR/"metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULT_DIR/"train_summary.json").write_text(json.dumps({"model":"DeepLabV3-MobileNetV3-Large","input":"B04/B03/B02/VV/VH","epochs":10,"batch_size":8,"loss":"inverse-square-root class-weighted CrossEntropyLoss(ignore_index=255)","model_selection":"validation macro mIoU","test_evaluated":False,"checkpoint":str(ckpt.relative_to(PROJECT_ROOT))}, indent=2), encoding="utf-8")
    save_curves(history_frame); save_per_class_iou(va); save_qualitative(model,val_loader,stats,device); save_comparison_figure(comparison)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    train_full()
