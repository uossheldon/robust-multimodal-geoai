from __future__ import annotations

import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import fit_sar_train_statistics
from src.evaluation.robustness_benchmark import apply_optical_occlusion
from src.models.deeplab import create_fusion_deeplab


RESULT_DIR=PROJECT_ROOT/'results'/'uncertainty'
MANIFEST_PATH=PROJECT_ROOT/'results'/'tile_manifest.csv'
SEEDS=[42,7,123]
METHODS=['original','modality_dropout','occlusion_training']
CONDITIONS=['clean','occlusion_30','occlusion_50','occlusion_70','missing_s1_zero','missing_s2_zero']
CONDITION_MODES={
 'clean':('clean',None),
 'occlusion_30':('fusion_occlusion',0.30),
 'occlusion_50':('fusion_occlusion',0.50),
 'occlusion_70':('fusion_occlusion',0.70),
 'missing_s1_zero':('fusion_missing_s1',None),
 'missing_s2_zero':('fusion_missing_s2',None),
}
CLASS_NAMES=['background','low','mid','high']

def checkpoint_path(method, seed):
    if method=='original':
        return PROJECT_ROOT/'checkpoints/s1_s2_early_fusion_best.pt' if seed==42 else PROJECT_ROOT/f'checkpoints/repro_original_seed{seed}_best.pt'
    if method=='modality_dropout':
        return PROJECT_ROOT/'checkpoints/s1_s2_modality_dropout_best.pt' if seed==42 else PROJECT_ROOT/f'checkpoints/repro_modality_dropout_seed{seed}_best.pt'
    if method=='occlusion_training':
        return PROJECT_ROOT/f'checkpoints/s1_s2_moddrop_occlusion_seed{seed}_best.pt'
    raise ValueError(method)

def make_loader(stats):
    ds=S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='validation', sar_mean=stats['mean'], sar_std=stats['std'], augment=False)
    return DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,pin_memory=True), ds

def entropy_from_probs(p):
    return -(p.clamp_min(1e-8)*p.clamp_min(1e-8).log()).sum(dim=1)

def confusion(pred,target):
    valid=target!=255
    enc=4*target[valid]+pred[valid]
    return torch.bincount(enc,minlength=16).reshape(4,4)

def seg_metrics(conf):
    m=conf.float(); tp=m.diag(); true=m.sum(1); pred=m.sum(0); union=true+pred-tp
    iou=torch.where(union>0,tp/union,torch.nan); dice=torch.where(true+pred>0,2*tp/(true+pred),torch.nan)
    return {'macro_miou':float(torch.nanmean(iou)),'macro_dice':float(torch.nanmean(dice)),**{f'iou_{i}':float(iou[i]) for i in range(4)}}

def ece(confidences, correct, bins=15):
    conf=np.asarray(confidences); corr=np.asarray(correct).astype(float)
    out=0.0; n=len(conf)
    for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(1/bins,1,bins)):
        mask=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any(): out += mask.mean()*abs(corr[mask].mean()-conf[mask].mean())
    return float(out)

def auroc_error(entropy, incorrect):
    scores=np.asarray(entropy); labels=np.asarray(incorrect).astype(bool)
    n_pos=labels.sum(); n_neg=(~labels).sum()
    if n_pos==0 or n_neg==0: return float('nan')
    order=np.argsort(scores)
    ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    # average ranks for ties
    vals=scores[order]; i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[j]==vals[i]: j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    sum_pos=ranks[labels].sum()
    return float((sum_pos - n_pos*(n_pos+1)/2)/(n_pos*n_neg))

def risk_coverage(entropy, incorrect, steps=np.linspace(0.1,1.0,10)):
    ent=np.asarray(entropy); inc=np.asarray(incorrect).astype(float)
    order=np.argsort(ent) # low uncertainty retained first
    rows=[]
    n=len(ent)
    for cov in steps:
        k=max(1,int(round(cov*n)))
        idx=order[:k]
        rows.append({'coverage':float(cov),'risk':float(inc[idx].mean())})
    auc=float(np.trapezoid([r['risk'] for r in rows],[r['coverage'] for r in rows])/(steps[-1]-steps[0]))
    return rows, auc

def apply_condition(images, condition, tile_ids, device):
    mode, frac=CONDITION_MODES[condition]
    x=images.to(device,non_blocking=True)
    if mode=='fusion_occlusion':
        return apply_optical_occlusion(x, frac, 101, tile_ids, rgb_channels=slice(0,3))
    if mode=='fusion_missing_s1':
        y=x.clone(); y[:,3:5]=0; return y
    if mode=='fusion_missing_s2':
        y=x.clone(); y[:,0:3]=0; return y
    return x

def evaluate(model, loader, ds, device, condition):
    model.eval(); conf=torch.zeros((4,4),dtype=torch.int64)
    nll_sum=0.0; brier_sum=0.0; valid_count=0
    entropies=[]; confidences=[]; corrects=[]
    offset=0
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
        for images,targets in loader:
            rows=ds.manifest.iloc[offset:offset+len(images)]; offset+=len(images)
            tile_ids=rows.tile_id.astype(str).tolist()
            targets=targets.to(device,non_blocking=True)
            x=apply_condition(images,condition,tile_ids,device)
            logits=model(x)['out']; probs=torch.softmax(logits,dim=1)
            pred=probs.argmax(1); maxp=probs.max(1).values; ent=entropy_from_probs(probs)
            valid=targets!=255
            conf += confusion(pred.cpu(),targets.cpu())
            p_valid=probs.permute(0,2,3,1)[valid]
            t_valid=targets[valid]
            nll_sum += float((-torch.log(p_valid[torch.arange(len(t_valid),device=device),t_valid].clamp_min(1e-8))).sum())
            onehot=torch.nn.functional.one_hot(t_valid,num_classes=4).float()
            brier_sum += float(((p_valid-onehot)**2).sum(dim=1).sum())
            valid_count += int(valid.sum())
            c=(pred[valid]==targets[valid])
            entropies.append(ent[valid].float().cpu().numpy()); confidences.append(maxp[valid].float().cpu().numpy()); corrects.append(c.cpu().numpy())
    ent=np.concatenate(entropies); confs=np.concatenate(confidences); corr=np.concatenate(corrects)
    incorrect=~corr
    seg=seg_metrics(conf)
    rc,rc_auc=risk_coverage(ent,incorrect)
    return {
        **seg,
        'nll':nll_sum/valid_count,
        'brier':brier_sum/valid_count,
        'ece':ece(confs,corr),
        'mean_entropy':float(ent.mean()),
        'entropy_correct':float(ent[corr].mean()) if corr.any() else float('nan'),
        'entropy_incorrect':float(ent[incorrect].mean()) if incorrect.any() else float('nan'),
        'error_auroc_entropy':auroc_error(ent,incorrect),
        'risk_coverage_auc':rc_auc,
        'valid_pixels':valid_count,
    }, rc


def main():
    os.environ.setdefault('TORCH_HOME',str(PROJECT_ROOT/'.torch'))
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda')
    stats=fit_sar_train_statistics(MANIFEST_PATH,project_root=PROJECT_ROOT)
    loader,ds=make_loader(stats)
    per=[]; rc_rows=[]
    for method in METHODS:
        for seed in SEEDS:
            ck=checkpoint_path(method,seed)
            model=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
            model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model_state_dict'])
            for cond in CONDITIONS:
                metrics,rc=evaluate(model,loader,ds,device,cond)
                row={'method':method,'seed':seed,'condition':cond,**metrics}; per.append(row)
                for r in rc: rc_rows.append({'method':method,'seed':seed,'condition':cond,**r})
                print(method,seed,cond,metrics['ece'],metrics['mean_entropy'],metrics['error_auroc_entropy'])
    per_df=pd.DataFrame(per); per_df.to_csv(RESULT_DIR/'per_seed_metrics.csv',index=False)
    agg=per_df.groupby(['method','condition']).agg({k:['mean','std'] for k in ['macro_miou','macro_dice','nll','brier','ece','mean_entropy','entropy_correct','entropy_incorrect','error_auroc_entropy','risk_coverage_auc']}).reset_index()
    agg.columns=['_'.join([str(x) for x in c if x]) for c in agg.columns]
    agg.to_csv(RESULT_DIR/'aggregate_metrics.csv',index=False)
    rc_df=pd.DataFrame(rc_rows); rc_df.to_csv(RESULT_DIR/'risk_coverage.csv',index=False)
    print('done')

if __name__=='__main__': main()
