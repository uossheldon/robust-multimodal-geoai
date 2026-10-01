from __future__ import annotations

import json, os, sys, time
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
from src.evaluation.robustness_benchmark import OCCLUSION_FRACTIONS, SEEDS as OCC_SEEDS, add_degradation, evaluate_model, stable_int
from src.models.deeplab import create_fusion_deeplab
from src.training.train_s2_deeplab import run_epoch, seed_everything

RESULT_DIR = PROJECT_ROOT / 'results' / 'reproducibility'
FIGURE_DIR = PROJECT_ROOT / 'figures'
DOCS_DIR = PROJECT_ROOT / 'docs'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
MANIFEST_PATH = PROJECT_ROOT / 'results' / 'tile_manifest.csv'
TRAIN_SEEDS = [42, 7, 123]
METRIC_KEYS = ['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high','binary_algae_dice']
PRIMARY = ['clean','missing_s1_zero','missing_s2_zero','occlusion_30','occlusion_70']

class ModalityDropoutDataset(S1S2EarlyFusionDataset):
    def __init__(self, *args, dropout_seed:int=42, **kwargs):
        super().__init__(*args, **kwargs); self.dropout_seed=dropout_seed
    def __getitem__(self, index:int):
        image,target=super().__getitem__(index)
        tile_id=str(self.manifest.iloc[index].tile_id)
        rng=np.random.default_rng(stable_int(self.dropout_seed,tile_id))
        v=rng.random()
        if v < 0.50:
            pass
        elif v < 0.75:
            image[3:5]=0.0
        else:
            image[0:3]=0.0
        return image,target

def train_class_weights():
    m=pd.read_csv(MANIFEST_PATH); tr=m[m.split=='train']
    counts=np.array([tr[f'class_{k}_count'].sum() for k in range(4)], dtype=np.float64)
    w=1/np.sqrt(counts/counts.sum()); w=w/w.mean()
    return {f'class_{k}':float(w[k]) for k in range(4)}

def loaders(stats, method, seed):
    klass = ModalityDropoutDataset if method=='modality_dropout' else S1S2EarlyFusionDataset
    kwargs={'dropout_seed':seed} if method=='modality_dropout' else {}
    train=klass(MANIFEST_PATH, project_root=PROJECT_ROOT, split='train', sar_mean=stats['mean'], sar_std=stats['std'], augment=True, **kwargs)
    val=S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='validation', sar_mean=stats['mean'], sar_std=stats['std'], augment=False)
    return DataLoader(train,batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed),pin_memory=True), DataLoader(val,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)

def checkpoint_for(method, seed):
    if method=='original' and seed==42: return PROJECT_ROOT/'checkpoints/s1_s2_early_fusion_best.pt'
    if method=='modality_dropout' and seed==42: return PROJECT_ROOT/'checkpoints/s1_s2_modality_dropout_best.pt'
    return CHECKPOINT_DIR / f'repro_{method}_seed{seed}_best.pt'

def train_one(method, seed, stats, weights, device):
    ckpt=checkpoint_for(method,seed)
    if ckpt.exists():
        return {'reused': True, 'checkpoint': str(ckpt.relative_to(PROJECT_ROOT)), 'training_time_seconds': None, 'best_epoch': int(torch.load(ckpt,map_location='cpu',weights_only=True)['epoch'])}
    seed_everything(seed)
    tr_loader,val_loader=loaders(stats,method,seed)
    model=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([weights[f'class_{k}'] for k in range(4)],dtype=torch.float32,device=device))
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}], weight_decay=1e-3)
    best=-1; best_epoch=0; start=time.perf_counter()
    for epoch in range(1,11):
        tr,_=run_epoch(model,tr_loader,criterion,device,optimizer=opt)
        va,conf=run_epoch(model,val_loader,criterion,device)
        print(f'{method} seed={seed} epoch={epoch} train_loss={tr["loss"]:.4f} val_miou={va["mean_iou"]:.4f}')
        if va['mean_iou']>best:
            best=va['mean_iou']; best_epoch=epoch
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'validation_metrics':va,'validation_confusion_matrix':conf.numpy().tolist(),'sar_statistics':stats,'method':method,'seed':seed}, ckpt)
    torch.cuda.synchronize(); return {'reused':False,'checkpoint':str(ckpt.relative_to(PROJECT_ROOT)),'training_time_seconds':time.perf_counter()-start,'best_epoch':best_epoch}

def eval_one(method, seed, stats, device):
    _,val_loader=loaders(stats,method,seed)
    model=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
    model.load_state_dict(torch.load(checkpoint_for(method,seed),map_location=device,weights_only=True)['model_state_dict'])
    rows=[]
    for cond,mode in [('clean','clean'),('missing_s1_zero','fusion_missing_s1'),('missing_s2_zero','fusion_missing_s2')]:
        r=evaluate_model(model,val_loader,device,model_name=method,condition=cond,occlusion_fraction=np.nan,seed=np.nan,date_filter=None,input_mode=mode)
        r.update({'method':method,'training_seed':seed,'primary_condition':cond}); rows.append(r)
    for frac in OCCLUSION_FRACTIONS:
        seed_rows=[]
        for occ_seed in OCC_SEEDS:
            r=evaluate_model(model,val_loader,device,model_name=method,condition='clean' if frac==0 else 'simulated_optical_occlusion',occlusion_fraction=frac,seed=occ_seed,date_filter=None,input_mode='fusion_occlusion')
            seed_rows.append(r)
        df=pd.DataFrame(seed_rows)
        r={'method':method,'training_seed':seed,'primary_condition':f'occlusion_{int(frac*100)}','condition':'optical_occlusion_mean','occlusion_fraction':frac,'seed':'mean3','date':'all_validation'}
        for k in METRIC_KEYS: r[k]=float(df[k].mean())
        rows.append(r)
    return rows

def aggregate(df):
    g=df[df.primary_condition.isin(PRIMARY)].groupby(['method','primary_condition'])
    out=g[METRIC_KEYS].agg(['mean','std']).reset_index()
    out.columns=['_'.join([str(x) for x in c if x]) for c in out.columns]
    return out

def save_figures(agg):
    for filename, conditions, title in [
        ('reproducibility_clean_missing.png',['clean','missing_s1_zero','missing_s2_zero'],'Reproducibility: clean and missing modality'),
        ('reproducibility_occlusion.png',['occlusion_0','occlusion_10','occlusion_30','occlusion_50','occlusion_70'],'Reproducibility: optical occlusion diagnostic')]:
        canvas=Image.new('RGB',(980,520),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),title,fill=(0,0,0))
        y=70; maxv=max(float(agg['mean_iou_mean'].max()),1e-6)
        for cond in conditions:
            draw.text((24,y+18),cond,fill=(0,0,0))
            for i,method in enumerate(['original','modality_dropout']):
                row=agg[(agg.method==method)&(agg.primary_condition==cond)]
                if row.empty: continue
                mean=float(row.mean_iou_mean.iloc[0]); std=float(row.mean_iou_std.iloc[0]); w=int(620*mean/maxv); yy=y+i*24
                color=(210,120,50) if method=='original' else (80,130,210)
                draw.rectangle((220,yy,220+w,yy+16),fill=color); draw.text((228+w,yy-2),f'{method} {mean:.3f}±{std:.3f}',fill=(0,0,0))
            y+=78
        canvas.save(FIGURE_DIR/filename)

def write_doc(agg, summary):
    def cell(method,cond):
        r=agg[(agg.method==method)&(agg.primary_condition==cond)].iloc[0]
        return f"{r.mean_iou_mean:.4f} ± {r.mean_iou_std:.4f}"
    doc=f"""# Reproducibility Check

Validation only. No held-out test dates were evaluated.

Training seeds: `{TRAIN_SEEDS}`.

| Condition | Original fusion macro mIoU | Modality-dropout macro mIoU |
|---|---:|---:|
| Clean | {cell('original','clean')} | {cell('modality_dropout','clean')} |
| Missing S1 | {cell('original','missing_s1_zero')} | {cell('modality_dropout','missing_s1_zero')} |
| Missing S2 | {cell('original','missing_s2_zero')} | {cell('modality_dropout','missing_s2_zero')} |
| 30% simulated optical occlusion | {cell('original','occlusion_30')} | {cell('modality_dropout','occlusion_30')} |
| 70% simulated optical occlusion | {cell('original','occlusion_70')} | {cell('modality_dropout','occlusion_70')} |

Conclusion: {summary['conclusion']}
"""
    (DOCS_DIR/'REPRODUCIBILITY_CHECK.md').write_text(doc,encoding='utf-8')

def main():
    os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT/'.torch'))
    for d in [RESULT_DIR,FIGURE_DIR,DOCS_DIR,CHECKPOINT_DIR]: d.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats(device)
    stats=fit_sar_train_statistics(MANIFEST_PATH, project_root=PROJECT_ROOT); weights=train_class_weights()
    train_infos=[]; rows=[]
    for method in ['original','modality_dropout']:
        for seed in TRAIN_SEEDS:
            info=train_one(method,seed,stats,weights,device); info.update({'method':method,'seed':seed}); train_infos.append(info)
            rows.extend(eval_one(method,seed,stats,device))
    per=pd.DataFrame(rows); per.to_csv(RESULT_DIR/'per_seed_results.csv',index=False)
    agg=aggregate(per); agg.to_csv(RESULT_DIR/'aggregate_results.csv',index=False)
    save_figures(agg)
    def get(method,cond): return float(agg[(agg.method==method)&(agg.primary_condition==cond)].mean_iou_mean.iloc[0])
    stable = get('modality_dropout','missing_s2_zero') > get('original','missing_s2_zero') and get('modality_dropout','clean') >= get('original','clean')
    summary={'training_seeds':TRAIN_SEEDS,'occlusion_seeds':OCC_SEEDS,'train_runs':train_infos,'peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated(device)/(1024**2)),'conclusion':'Modality dropout improvement is stable across these seeds.' if stable else 'Modality dropout improvement is not fully stable across these seeds.','aggregate':agg.to_dict(orient='records')}
    (RESULT_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_doc(agg,summary)
    print(json.dumps({'seeds':TRAIN_SEEDS,'conclusion':summary['conclusion'],'peak_gpu_memory_mb':summary['peak_gpu_memory_mb']},indent=2))

if __name__=='__main__': main()
