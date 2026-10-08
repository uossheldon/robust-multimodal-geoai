from __future__ import annotations

import json, os, sys, time
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
from src.evaluation.robustness_benchmark import OCCLUSION_FRACTIONS, SEEDS as OCC_SEEDS, evaluate_model, stable_int, apply_optical_occlusion
from src.models.deeplab import create_fusion_deeplab
from src.training.train_s2_deeplab import run_epoch, seed_everything

RESULT_DIR=PROJECT_ROOT/'results'/'occlusion_training'
CHECKPOINT_DIR=PROJECT_ROOT/'checkpoints'
MANIFEST_PATH=PROJECT_ROOT/'results'/'tile_manifest.csv'
TRAIN_SEEDS=[42,7,123]
PRIMARY=['clean','missing_s1_zero','missing_s2_zero','occlusion_10','occlusion_30','occlusion_50','occlusion_70']
METRIC_KEYS=['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high','binary_algae_dice','predicted_class_0_fraction','predicted_class_1_fraction','predicted_class_2_fraction','predicted_class_3_fraction']

class OcclusionAwareDataset(S1S2EarlyFusionDataset):
    def __init__(self,*args,dropout_seed:int=42,**kwargs):
        super().__init__(*args,**kwargs); self.dropout_seed=dropout_seed
    def __getitem__(self,index:int):
        image,target=super().__getitem__(index)
        tile_id=str(self.manifest.iloc[index].tile_id)
        rng=np.random.default_rng(stable_int(self.dropout_seed,tile_id,'mode'))
        v=rng.random()
        if v < 0.50:
            # both modalities: half clean, half simulated optical occlusion.
            if np.random.default_rng(stable_int(self.dropout_seed,tile_id,'occ_flag')).random() < 0.50:
                frac=float(np.random.default_rng(stable_int(self.dropout_seed,tile_id,'occ_frac')).uniform(0.10,0.70))
                x=image.unsqueeze(0)
                image=apply_optical_occlusion(x, frac, stable_int(self.dropout_seed,tile_id,'train_occ_seed'), [tile_id], rgb_channels=slice(0,3))[0]
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

def loaders(stats, seed, train_variant=True):
    train=OcclusionAwareDataset(MANIFEST_PATH,project_root=PROJECT_ROOT,split='train',sar_mean=stats['mean'],sar_std=stats['std'],augment=True,dropout_seed=seed)
    val=S1S2EarlyFusionDataset(MANIFEST_PATH,project_root=PROJECT_ROOT,split='validation',sar_mean=stats['mean'],sar_std=stats['std'],augment=False)
    return DataLoader(train,batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed),pin_memory=True), DataLoader(val,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)

def checkpoint_for(seed): return CHECKPOINT_DIR/f's1_s2_moddrop_occlusion_seed{seed}_best.pt'

def train_one(seed,stats,weights,device):
    ckpt=checkpoint_for(seed)
    if ckpt.exists():
        cp=torch.load(ckpt,map_location='cpu',weights_only=True)
        return {'seed':seed,'reused':True,'best_epoch':int(cp['epoch']),'training_time_seconds':None,'checkpoint':str(ckpt.relative_to(PROJECT_ROOT))}
    seed_everything(seed)
    tr_loader,val_loader=loaders(stats,seed)
    model=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=255,weight=torch.tensor([weights[f'class_{k}'] for k in range(4)],dtype=torch.float32,device=device))
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=1e-3)
    best=-1; best_epoch=0; start=time.perf_counter()
    for epoch in range(1,11):
        tr,_=run_epoch(model,tr_loader,criterion,device,optimizer=opt)
        va,conf=run_epoch(model,val_loader,criterion,device)
        print(f'occaware seed={seed} epoch={epoch} train_loss={tr["loss"]:.4f} val_miou={va["mean_iou"]:.4f}')
        if va['mean_iou']>best:
            best=va['mean_iou']; best_epoch=epoch
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'validation_metrics':va,'validation_confusion_matrix':conf.numpy().tolist(),'sar_statistics':stats,'seed':seed},ckpt)
    torch.cuda.synchronize(); return {'seed':seed,'reused':False,'best_epoch':best_epoch,'training_time_seconds':time.perf_counter()-start,'checkpoint':str(ckpt.relative_to(PROJECT_ROOT))}


def eval_seed(seed,stats,device):
    _,val_loader=loaders(stats,seed)
    model=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
    model.load_state_dict(torch.load(checkpoint_for(seed),map_location=device,weights_only=True)['model_state_dict'])
    rows=[]
    for cond,mode in [('clean','clean'),('missing_s1_zero','fusion_missing_s1'),('missing_s2_zero','fusion_missing_s2')]:
        r=evaluate_model(model,val_loader,device,model_name='occlusion_training',condition=cond,occlusion_fraction=np.nan,seed=np.nan,date_filter=None,input_mode=mode)
        r.update({'method':'occlusion_training','training_seed':seed,'primary_condition':cond}); rows.append(r)
    for frac in OCCLUSION_FRACTIONS:
        seed_rows=[]
        for occ_seed in OCC_SEEDS:
            r=evaluate_model(model,val_loader,device,model_name='occlusion_training',condition='clean' if frac==0 else 'simulated_optical_occlusion',occlusion_fraction=frac,seed=occ_seed,date_filter=None,input_mode='fusion_occlusion')
            seed_rows.append(r)
        df=pd.DataFrame(seed_rows)
        r={'method':'occlusion_training','training_seed':seed,'primary_condition':f'occlusion_{int(frac*100)}','condition':'optical_occlusion_mean','occlusion_fraction':frac,'seed':'mean3','date':'all_validation'}
        for k in ['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high','binary_algae_iou','binary_algae_dice']:
            r[k]=float(df[k].mean())
        rows.append(r)
    return rows

def load_comparison_methods():
    out=[]
    # original/moddrop aggregate from reproducibility
    repro=pd.read_csv(PROJECT_ROOT/'results/reproducibility/aggregate_results.csv')
    for _,row in repro.iterrows():
        if row.primary_condition in PRIMARY:
            rec={'method':row.method,'primary_condition':row.primary_condition}
            for k in ['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high','binary_algae_dice']:
                rec[f'{k}_mean']=row[f'{k}_mean']; rec[f'{k}_std']=row[f'{k}_std']
            out.append(rec)
    return pd.DataFrame(out)

def aggregate(df):
    g=df.groupby(['method','primary_condition'])
    out=g[['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high','binary_algae_dice']].agg(['mean','std']).reset_index()
    out.columns=['_'.join([str(x) for x in c if x]) for c in out.columns]
    return out

def auc_table(agg):
    xs=np.array([0,10,30,50,70],dtype=float)/70.0
    rows=[]
    for method in agg.method.unique():
        vals=[]
        for cond in ['occlusion_0','occlusion_10','occlusion_30','occlusion_50','occlusion_70']:
            vals.append(float(agg[(agg.method==method)&(agg.primary_condition==cond)].mean_iou_mean.iloc[0]))
        rows.append({'method':method,'macro_miou_auc_0_70':float(np.trapezoid(vals,xs))})
    return pd.DataFrame(rows)


def main():
    os.environ.setdefault('TORCH_HOME',str(PROJECT_ROOT/'.torch'))
    for d in [RESULT_DIR,CHECKPOINT_DIR]: d.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats(device)
    stats=fit_sar_train_statistics(MANIFEST_PATH,project_root=PROJECT_ROOT); weights=train_class_weights()
    infos=[]; rows=[]
    for seed in TRAIN_SEEDS:
        infos.append(train_one(seed,stats,weights,device)); rows.extend(eval_seed(seed,stats,device))
    per=pd.DataFrame(rows); per.to_csv(RESULT_DIR/'per_seed_results.csv',index=False)
    occagg=aggregate(per); baseagg=load_comparison_methods(); allagg=pd.concat([baseagg,occagg],ignore_index=True)
    allagg.to_csv(RESULT_DIR/'aggregate_results.csv',index=False)
    auc=auc_table(allagg); auc.to_csv(RESULT_DIR/'robustness_auc.csv',index=False)
    def val(method,cond): return float(allagg[(allagg.method==method)&(allagg.primary_condition==cond)].mean_iou_mean.iloc[0])
    conclusion='Partial-occlusion training adds value beyond modality dropout for optical occlusion.' if val('occlusion_training','occlusion_70')>val('modality_dropout','occlusion_70') else 'Partial-occlusion training did not improve 70% occlusion macro mIoU over modality dropout.'
    summary={'training_seeds':TRAIN_SEEDS,'occlusion_eval_seeds':OCC_SEEDS,'train_runs':infos,'peak_gpu_memory_mb':float(torch.cuda.max_memory_allocated(device)/(1024**2)),'conclusion':conclusion}
    (RESULT_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'conclusion':conclusion,'peak_gpu_memory_mb':summary['peak_gpu_memory_mb']},indent=2))

if __name__=='__main__': main()
