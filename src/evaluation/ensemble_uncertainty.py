from __future__ import annotations

import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT))

from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.s1_dataset import fit_sar_train_statistics
from src.evaluation.robustness_benchmark import apply_optical_occlusion
from src.evaluation.uncertainty_diagnostics import ece, auroc_error, risk_coverage, entropy_from_probs, seg_metrics, confusion
from src.models.deeplab import create_fusion_deeplab
from src.training.train_fusion_deeplab import denormalize_rgb
from src.training.train_s1_deeplab import color_mask

RESULT_DIR=PROJECT_ROOT/'results'/'ensemble_uncertainty'
FIGURE_DIR=PROJECT_ROOT/'figures'
DOCS_DIR=PROJECT_ROOT/'docs'
MANIFEST_PATH=PROJECT_ROOT/'results'/'tile_manifest.csv'
SEEDS=[42,7,123]
CONDITIONS=['clean','occlusion_30','occlusion_50','occlusion_70','missing_s1_zero','missing_s2_zero']
CONDITION_MODES={'clean':('clean',None),'occlusion_30':('fusion_occlusion',0.30),'occlusion_50':('fusion_occlusion',0.50),'occlusion_70':('fusion_occlusion',0.70),'missing_s1_zero':('fusion_missing_s1',None),'missing_s2_zero':('fusion_missing_s2',None)}

def make_loader(stats):
    ds=S1S2EarlyFusionDataset(MANIFEST_PATH,project_root=PROJECT_ROOT,split='validation',sar_mean=stats['mean'],sar_std=stats['std'],augment=False)
    return DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,pin_memory=True), ds

def load_models(device):
    models=[]
    for seed in SEEDS:
        ck=PROJECT_ROOT/f'checkpoints/s1_s2_moddrop_occlusion_seed{seed}_best.pt'
        m=create_fusion_deeplab(num_classes=4,pretrained=True).to(device)
        m.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model_state_dict']); m.eval(); models.append(m)
    return models

def apply_condition(images,condition,tile_ids,device):
    mode,frac=CONDITION_MODES[condition]
    x=images.to(device,non_blocking=True)
    if mode=='fusion_occlusion': return apply_optical_occlusion(x,frac,101,tile_ids,rgb_channels=slice(0,3))
    if mode=='fusion_missing_s1':
        y=x.clone(); y[:,3:5]=0; return y
    if mode=='fusion_missing_s2':
        y=x.clone(); y[:,0:3]=0; return y
    return x

def brier_nll(probs,targets):
    p=probs.permute(0,2,3,1)
    valid=targets!=255
    p_valid=p[valid]; t=targets[valid]
    nll=float((-torch.log(p_valid[torch.arange(len(t),device=t.device),t].clamp_min(1e-8))).mean())
    onehot=torch.nn.functional.one_hot(t,num_classes=4).float()
    brier=float(((p_valid-onehot)**2).sum(1).mean())
    return nll,brier

def disagreement(probs_stack):
    # mean class probability variance across models, summed over classes
    return probs_stack.var(dim=0,unbiased=False).sum(dim=1)

def evaluate(models,loader,ds,device,condition):
    conf=torch.zeros((4,4),dtype=torch.int64)
    nll_sum=brier_sum=0.0; n_valid=0
    ens_entropy=[]; mean_ind_entropy=[]; mi=[]; disag=[]; confs=[]; corrects=[]; single_entropy=[]
    offset=0
    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=True):
        for images,targets in loader:
            rows=ds.manifest.iloc[offset:offset+len(images)]; offset+=len(images); tile_ids=rows.tile_id.astype(str).tolist()
            targets=targets.to(device,non_blocking=True); x=apply_condition(images,condition,tile_ids,device)
            probs_list=[torch.softmax(m(x)['out'],dim=1) for m in models]
            stack=torch.stack(probs_list,dim=0); meanp=stack.mean(dim=0)
            pred=meanp.argmax(1); valid=targets!=255
            conf+=confusion(pred.cpu(),targets.cpu())
            nll,brier=brier_nll(meanp,targets); vc=int(valid.sum()); nll_sum+=nll*vc; brier_sum+=brier*vc; n_valid+=vc
            pe=entropy_from_probs(meanp); indiv=torch.stack([entropy_from_probs(p) for p in probs_list],dim=0).mean(dim=0); mutual=pe-indiv; dis=disagreement(stack)
            ens_entropy.append(pe[valid].float().cpu().numpy()); mean_ind_entropy.append(indiv[valid].float().cpu().numpy()); mi.append(mutual[valid].float().cpu().numpy()); disag.append(dis[valid].float().cpu().numpy())
            single_entropy.append(entropy_from_probs(probs_list[0])[valid].float().cpu().numpy())
            confs.append(meanp.max(1).values[valid].float().cpu().numpy()); corrects.append((pred[valid]==targets[valid]).cpu().numpy())
    ens=np.concatenate(ens_entropy); indiv=np.concatenate(mean_ind_entropy); mi_arr=np.concatenate(mi); dis_arr=np.concatenate(disag); single=np.concatenate(single_entropy); corr=np.concatenate(corrects); confv=np.concatenate(confs); inc=~corr
    seg=seg_metrics(conf)
    metrics={**seg,'nll':nll_sum/n_valid,'brier':brier_sum/n_valid,'ece':ece(confv,corr),'predictive_entropy':float(ens.mean()),'mean_individual_entropy':float(indiv.mean()),'mutual_information':float(mi_arr.mean()),'disagreement':float(dis_arr.mean()),'entropy_correct':float(ens[corr].mean()),'entropy_incorrect':float(ens[inc].mean()),'valid_pixels':n_valid}
    aucs={'single_entropy':auroc_error(single,inc),'ensemble_entropy':auroc_error(ens,inc),'mutual_information':auroc_error(mi_arr,inc),'disagreement':auroc_error(dis_arr,inc)}
    rc_rows=[]
    for name,score in [('single_entropy',single),('ensemble_entropy',ens),('mutual_information',mi_arr),('disagreement',dis_arr)]:
        rc,auc=risk_coverage(score,inc)
        for r in rc: rc_rows.append({'condition':condition,'uncertainty':name,**r})
        aucs[f'{name}_risk_auc']=auc
    return metrics,aucs,rc_rows

def save_figures(metrics,err,rc):
    def bar(path,title,df,value_col):
        canvas=Image.new('RGB',(960,520),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),title,fill=(0,0,0))
        maxv=max(float(df[value_col].max()),1e-6); y=70
        for cond in CONDITIONS:
            r=df[df.condition==cond].iloc[0]; val=float(r[value_col]); w=int(620*val/maxv)
            draw.text((24,y+8),cond,fill=(0,0,0)); draw.rectangle((220,y,220+w,y+20),fill=(90,160,95)); draw.text((228+w,y+2),f'{val:.3f}',fill=(0,0,0)); y+=60
        canvas.save(path)
    bar(FIGURE_DIR/'ensemble_vs_single_calibration.png','Ensemble ECE by condition',metrics,'ece')
    bar(FIGURE_DIR/'ensemble_uncertainty_vs_occlusion.png','Ensemble predictive entropy',metrics,'predictive_entropy')
    # AUROC grouped
    canvas=Image.new('RGB',(1100,560),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),'Error detection AUROC',fill=(0,0,0))
    maxv=1.0; y=70; colors={'single_entropy':(210,120,50),'ensemble_entropy':(80,130,210),'mutual_information':(90,160,95),'disagreement':(160,80,160)}
    for cond in CONDITIONS:
        draw.text((24,y+30),cond,fill=(0,0,0))
        for i,u in enumerate(colors):
            val=float(err[(err.condition==cond)&(err.uncertainty==u)].auroc.iloc[0]); w=int(620*val/maxv); yy=y+i*18
            draw.rectangle((220,yy,220+w,yy+13),fill=colors[u]); draw.text((228+w,yy-3),f'{u} {val:.3f}',fill=(0,0,0))
        y+=92
    canvas.save(FIGURE_DIR/'ensemble_error_detection.png')
    # risk coverage clean
    canvas=Image.new('RGB',(860,520),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),'Ensemble risk coverage, clean',fill=(0,0,0)); x0,y0,w,h=80,70,680,360; draw.rectangle((x0,y0,x0+w,y0+h),outline=(0,0,0))
    for u,c in colors.items():
        d=rc[(rc.condition=='clean')&(rc.uncertainty==u)].sort_values('coverage'); pts=[]
        for r in d.itertuples(index=False): pts.append((x0+int((r.coverage-0.1)/0.9*w), y0+h-int(r.risk*h)))
        if len(pts)>1: draw.line(pts,fill=c,width=3)
    canvas.save(FIGURE_DIR/'ensemble_risk_coverage.png')

def qualitative(models,loader,ds,device):
    images,targets=next(iter(loader)); img=images[0]; target=targets[0]
    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=True):
        x=img.unsqueeze(0).to(device); stack=torch.stack([torch.softmax(m(x)['out'],dim=1) for m in models],dim=0); meanp=stack.mean(0)[0].cpu(); pred=meanp.argmax(0); ent=entropy_from_probs(meanp.unsqueeze(0))[0]
    err=torch.full_like(target,255); valid=target!=255; err[valid&(pred==target)]=0; err[valid&(pred!=target)]=3
    e=ent.numpy(); erg=np.repeat((((e-e.min())/(e.max()-e.min()+1e-8))*255).astype(np.uint8)[:,:,None],3,axis=2)
    panels=[denormalize_rgb(img), color_mask(target.numpy()), color_mask(pred.numpy()), color_mask(err.numpy()), erg]
    labels=['RGB','Ground Truth','Ensemble Prediction','Error Map','Entropy Map']
    tile=224; label_h=28; canvas=Image.new('RGB',(tile*5,tile+label_h),'white'); draw=ImageDraw.Draw(canvas)
    for i,(p,l) in enumerate(zip(panels,labels)):
        x=i*tile; draw.text((x+8,7),l,fill=(0,0,0)); canvas.paste(Image.fromarray(p),(x,label_h))
    canvas.save(FIGURE_DIR/'ensemble_uncertainty_examples.png')

def main():
    os.environ.setdefault('TORCH_HOME',str(PROJECT_ROOT/'.torch'))
    RESULT_DIR.mkdir(parents=True,exist_ok=True); FIGURE_DIR.mkdir(exist_ok=True); DOCS_DIR.mkdir(exist_ok=True)
    device=torch.device('cuda'); stats=fit_sar_train_statistics(MANIFEST_PATH,project_root=PROJECT_ROOT); loader,ds=make_loader(stats); models=load_models(device)
    metric_rows=[]; err_rows=[]; rc_all=[]
    for cond in CONDITIONS:
        met,aucs,rc=evaluate(models,loader,ds,device,cond); metric_rows.append({'condition':cond,**met})
        for u in ['single_entropy','ensemble_entropy','mutual_information','disagreement']:
            err_rows.append({'condition':cond,'uncertainty':u,'auroc':aucs[u],'risk_auc':aucs[f'{u}_risk_auc']})
        rc_all.extend(rc); print(cond,met['macro_miou'],met['ece'],aucs)
    metrics=pd.DataFrame(metric_rows); err=pd.DataFrame(err_rows); rc=pd.DataFrame(rc_all)
    metrics.to_csv(RESULT_DIR/'metrics.csv',index=False); err.to_csv(RESULT_DIR/'error_detection.csv',index=False); rc.to_csv(RESULT_DIR/'risk_coverage.csv',index=False)
    summary={'conditions':CONDITIONS,'metrics':metric_rows,'error_detection':err_rows}
    (RESULT_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    save_figures(metrics,err,rc); qualitative(models,loader,ds,device)
    (DOCS_DIR/'ENSEMBLE_UNCERTAINTY.md').write_text('# Ensemble Uncertainty\n\nValidation only. Used frozen occlusion-trained seeds 42, 7, and 123. No calibration fitting, retraining, or test-date evaluation was performed.\n',encoding='utf-8')
    print('done')

if __name__=='__main__': main()
