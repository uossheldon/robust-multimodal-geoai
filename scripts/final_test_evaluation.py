from __future__ import annotations

import hashlib, json, os, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s2_dataset import S2RGBSegmentationDataset
from src.data.s1_dataset import S1SARSegmentationDataset
from src.data.fusion_dataset import S1S2EarlyFusionDataset
from src.data.terramind_dataset import TerraMindMultimodalDataset, terramind_collate, default_terramind_data_root
from src.evaluation.segmentation import confusion_matrix, segmentation_metrics
from src.evaluation.robustness_benchmark import OCCLUSION_FRACTIONS, SEEDS as CORRUPTION_SEEDS, apply_optical_occlusion, METRIC_KEYS, binary_metrics
from src.models.deeplab import create_s2_deeplab, create_sar_deeplab, create_fusion_deeplab
from src.models.terramind_segmentation import TerraMindFrozenSegmenter
from src.training.train_s1_deeplab import color_mask, stretch_channel

MANIFEST_PATH = PROJECT_ROOT / 'results' / 'tile_manifest.csv'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
RESULT_DIR = PROJECT_ROOT / 'results' / 'final_test'
FIGURE_DIR = PROJECT_ROOT / 'figures'
DOCS_DIR = PROJECT_ROOT / 'docs'
TEST_DATES = ['2025-09-08', '2025-09-21']
BATCH_SIZE = 4

LOCKED_CHECKPOINTS = {
    's2_weighted': ['s2_deeplab_weighted_best.pt'],
    's1_weighted': ['s1_deeplab_weighted_best.pt'],
    'early_fusion': ['s1_s2_early_fusion_best.pt'],
    'modality_dropout': ['s1_s2_modality_dropout_best.pt','repro_modality_dropout_seed7_best.pt','repro_modality_dropout_seed123_best.pt'],
    'occlusion_trained': ['s1_s2_moddrop_occlusion_seed42_best.pt','s1_s2_moddrop_occlusion_seed7_best.pt','s1_s2_moddrop_occlusion_seed123_best.pt'],
    'terramind_frozen': ['terramind_frozen_seed42_best.pt','terramind_frozen_seed7_best.pt','terramind_frozen_seed123_best.pt'],
}
SEED_LABELS = {
    's1_s2_modality_dropout_best.pt': 42,
    'repro_modality_dropout_seed7_best.pt': 7,
    'repro_modality_dropout_seed123_best.pt': 123,
    's1_s2_moddrop_occlusion_seed42_best.pt': 42,
    's1_s2_moddrop_occlusion_seed7_best.pt': 7,
    's1_s2_moddrop_occlusion_seed123_best.pt': 123,
    'terramind_frozen_seed42_best.pt': 42,
    'terramind_frozen_seed7_best.pt': 7,
    'terramind_frozen_seed123_best.pt': 123,
}
CLASS_NAMES = ['background','low','mid','high']


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def audit() -> dict[str, Any]:
    manifest = pd.read_csv(MANIFEST_PATH)
    test = manifest[manifest['split'] == 'test']
    if sorted(test['date'].astype(str).unique().tolist()) != TEST_DATES:
        raise RuntimeError(f'Unexpected test dates: {sorted(test["date"].astype(str).unique().tolist())}')
    if len(test) != 114:
        raise RuntimeError(f'Expected 114 test tiles, found {len(test)}')
    ckpt_rows=[]
    for model, names in LOCKED_CHECKPOINTS.items():
        for name in names:
            path=CHECKPOINT_DIR/name
            if not path.exists():
                raise FileNotFoundError(path)
            ckpt_rows.append({'model':model,'checkpoint':name,'sha256_16':sha16(path),'bytes':path.stat().st_size})
    return {'test_tile_count': int(len(test)), 'test_dates': test['date'].value_counts().to_dict(), 'checkpoints': ckpt_rows, 'corruption_seeds': CORRUPTION_SEEDS, 'occlusion_fractions': OCCLUSION_FRACTIONS, 'post_test_tuning': False}


def load_state(model, ckpt_name: str, device):
    ckpt=torch.load(CHECKPOINT_DIR/ckpt_name, map_location=device, weights_only=False)
    state=ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return ckpt


def make_terramind_model(device):
    from terratorch.registry import BACKBONE_REGISTRY
    backbone = BACKBONE_REGISTRY.build('terramind_v1_tiny', pretrained=False, modalities=['RGB','S1RTC'], merge_method='mean')
    return TerraMindFrozenSegmenter(backbone, feature_dim=192, num_classes=4).to(device)


def to_device(batch_inputs, targets, device):
    if isinstance(batch_inputs, dict):
        return {k:v.to(device, non_blocking=True) for k,v in batch_inputs.items()}, targets.to(device, non_blocking=True)
    return batch_inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def model_forward(model, inputs):
    return model(inputs)['out']


def metrics_from_conf(conf):
    out=segmentation_metrics(conf.cpu())
    out.update(binary_metrics(conf.cpu()))
    return {k: out[k] for k in METRIC_KEYS}


def evaluate(model, loader, device, *, model_name, condition='clean', seed=None, occlusion_fraction=None, mode='clean', date=None):
    conf=torch.zeros((4,4), dtype=torch.int64, device=device)
    total_valid=0
    offset=0
    manifest=loader.dataset.manifest
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
        for inputs, targets in loader:
            rows=manifest.iloc[offset:offset+len(targets)]
            offset += len(targets)
            if date is not None:
                keep_np=(rows['date'].astype(str).values == date)
                if not keep_np.any():
                    continue
                keep=torch.tensor(keep_np, dtype=torch.bool)
                targets=targets[keep]
                if isinstance(inputs, dict):
                    inputs={k:v[keep] for k,v in inputs.items()}
                else:
                    inputs=inputs[keep]
                rows=rows[keep_np]
            inputs, targets=to_device(inputs, targets, device)
            if not isinstance(inputs, dict):
                tile_ids=rows['tile_id'].astype(str).tolist()
                if mode in {'fusion_occlusion','s2_occlusion'}:
                    inputs=apply_optical_occlusion(inputs, float(occlusion_fraction), int(seed), tile_ids, rgb_channels=slice(0,3))
                elif mode == 'fusion_missing_s1':
                    inputs=inputs.clone(); inputs[:,3:5]=0.0
                elif mode == 'fusion_missing_s2':
                    inputs=inputs.clone(); inputs[:,0:3]=0.0
            logits=model_forward(model, inputs)
            preds=logits.argmax(dim=1)
            conf += confusion_matrix(preds, targets, 4, 255).to(device)
            total_valid += int((targets != 255).sum().item())
    return {'model':model_name,'condition':condition,'seed':seed,'occlusion_fraction':occlusion_fraction,'date':date or 'all_test','valid_pixels':total_valid, **metrics_from_conf(conf)}


def make_loaders():
    stats=json.loads((PROJECT_ROOT/'results/s1_deeplab_weighted/metrics.json').read_text())['sar_statistics']
    return {
        's2': DataLoader(S2RGBSegmentationDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='test', augment=False), batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        's1': DataLoader(S1SARSegmentationDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='test', sar_mean=stats['mean'], sar_std=stats['std'], augment=False), batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        'fusion': DataLoader(S1S2EarlyFusionDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='test', sar_mean=stats['mean'], sar_std=stats['std'], augment=False), batch_size=8, shuffle=False, num_workers=0, pin_memory=True),
        'terramind': DataLoader(TerraMindMultimodalDataset(MANIFEST_PATH, project_root=PROJECT_ROOT, split='test', data_root=default_terramind_data_root(PROJECT_ROOT), augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=terramind_collate),
    }


def aggregate(rows):
    df=pd.DataFrame(rows)
    group_cols=['model','condition']
    grouped=df[df['date'].eq('all_test')].groupby(group_cols, dropna=False)
    out=[]
    for keys, grp in grouped:
        rec={'model':keys[0], 'condition':keys[1], 'n': int(len(grp))}
        for m in METRIC_KEYS:
            vals=pd.to_numeric(grp[m], errors='coerce')
            rec[f'{m}_mean']=float(vals.mean())
            rec[f'{m}_std']=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def save_figures(clean_summary, robust_rows, per_date):
    FIGURE_DIR.mkdir(exist_ok=True)
    # clean comparison
    canvas=Image.new('RGB',(1100,560),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((24,18),'Final held-out clean test comparison',fill=(0,0,0))
    y=70
    for row in clean_summary.sort_values('mean_iou_mean', ascending=False).itertuples(index=False):
        v=float(row.mean_iou_mean); w=int(v*700)
        draw.text((24,y+8), f'{row.model} ({row.n} run{"s" if row.n!=1 else ""})', fill=(0,0,0))
        draw.rectangle((330,y,330+w,y+24), fill=(80,130,210))
        std='' if pd.isna(row.mean_iou_std) else f' ± {row.mean_iou_std:.3f}'
        draw.text((1040-170,y+6), f'{v:.3f}{std}', fill=(0,0,0))
        y+=42
    canvas.save(FIGURE_DIR/'final_clean_model_comparison.png')
    # per class
    canvas=Image.new('RGB',(1100,680),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),'Final test per-class IoU',fill=(0,0,0))
    y=70
    for row in clean_summary.itertuples(index=False):
        draw.text((24,y), row.model, fill=(0,0,0)); x=260
        for col,label in [('iou_background_mean','0'),('iou_low_mean','1'),('iou_mid_mean','2'),('iou_high_mean','3')]:
            v=float(getattr(row,col)); h=int(v*160)
            draw.rectangle((x,y+28+160-h,x+35,y+188), fill=(90,160,90)); draw.text((x,y+194),f'{label}:{v:.2f}',fill=(0,0,0)); x+=75
        y+=92
    canvas.save(FIGURE_DIR/'final_per_class_iou.png')
    # robustness curves
    rob=pd.DataFrame(robust_rows)
    canvas=Image.new('RGB',(980,560),'white'); draw=ImageDraw.Draw(canvas); draw.text((24,18),'Final test robustness curves: macro mIoU',fill=(0,0,0))
    x0,y0,w,h=90,80,760,360; draw.rectangle((x0,y0,x0+w,y0+h), outline=(0,0,0))
    colors={'modality_dropout':(80,130,210),'occlusion_trained':(210,100,80)}
    for model,color in colors.items():
        pts=[]
        for i,frac in enumerate(OCCLUSION_FRACTIONS):
            cond='clean' if frac==0 else f'occlusion_{int(frac*100)}'
            sub=rob[(rob.model==model)&(rob.condition==cond)&(rob.date=='all_test')]
            if len(sub):
                val=sub['mean_iou'].mean(); x=x0+int(i*w/(len(OCCLUSION_FRACTIONS)-1)); y=y0+h-int(val*h/0.4); pts.append((x,y))
        if len(pts)>1: draw.line(pts, fill=color, width=3)
        if pts: draw.text((pts[-1][0]+8, pts[-1][1]), model, fill=color)
    for i,frac in enumerate(OCCLUSION_FRACTIONS):
        x=x0+int(i*w/(len(OCCLUSION_FRACTIONS)-1)); draw.text((x-10,y0+h+10),f'{int(frac*100)}%',fill=(0,0,0))
    canvas.save(FIGURE_DIR/'final_robustness_curves.png')
    # qualitative placeholder from first rows not predictions to avoid rerun
    canvas=Image.new('RGB',(900,240),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((24,24),'Qualitative test examples generated from final models are not embedded here.',fill=(0,0,0))
    draw.text((24,60),'See final CSV metrics for held-out test behavior by date and model.',fill=(0,0,0))
    canvas.save(FIGURE_DIR/'final_qualitative_test_examples.png')


def write_doc(clean, robust_summary, per_date, audit_info):
    DOCS_DIR.mkdir(exist_ok=True)
    lines=['# Final Held-Out Test Results','', 'Phase 7: first and final held-out test evaluation. No post-test tuning was performed.','', '## Audit','', f"Test dates: {', '.join(TEST_DATES)}", f"Test tiles: {audit_info['test_tile_count']}", '', '## Clean test summary','', clean.to_markdown(index=False), '', '## Robustness test summary', '', robust_summary.to_markdown(index=False), '', '## Per-date behavior', '', per_date.to_markdown(index=False), '']
    (DOCS_DIR/'FINAL_TEST_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT/'.torch'))
    RESULT_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(exist_ok=True)
    audit_info=audit()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders=make_loaders()
    clean_rows=[]; robust_rows=[]; per_date=[]
    start=time.perf_counter()
    # single baselines
    singles=[('s2_weighted', create_s2_deeplab, 's2', 's2_deeplab_weighted_best.pt'), ('s1_weighted', create_sar_deeplab, 's1', 's1_deeplab_weighted_best.pt'), ('early_fusion', create_fusion_deeplab, 'fusion', 's1_s2_early_fusion_best.pt')]
    for name, factory, loader_key, ckpt in singles:
        model=factory(num_classes=4, pretrained=True).to(device); load_state(model, ckpt, device)
        row=evaluate(model, loaders[loader_key], device, model_name=name); row['run_type']='single_run'; clean_rows.append(row)
        for d in TEST_DATES: per_date.append({**evaluate(model, loaders[loader_key], device, model_name=name, date=d), 'run_type':'single_run'})
        del model; torch.cuda.empty_cache()
    # robust DeepLab seeds
    robust_models=[('modality_dropout', LOCKED_CHECKPOINTS['modality_dropout']), ('occlusion_trained', LOCKED_CHECKPOINTS['occlusion_trained'])]
    for model_name, ckpts in robust_models:
        for ckpt in ckpts:
            seed_label=SEED_LABELS[ckpt]
            model=create_fusion_deeplab(num_classes=4, pretrained=True).to(device); load_state(model, ckpt, device)
            clean=evaluate(model, loaders['fusion'], device, model_name=model_name, seed=seed_label); clean['run_type']='seeded'; clean_rows.append(clean); robust_rows.append(clean)
            for d in TEST_DATES: per_date.append({**evaluate(model, loaders['fusion'], device, model_name=model_name, seed=seed_label, date=d), 'run_type':'seeded'})
            for frac in OCCLUSION_FRACTIONS[1:]:
                for corr_seed in CORRUPTION_SEEDS:
                    robust_rows.append(evaluate(model, loaders['fusion'], device, model_name=model_name, condition=f'occlusion_{int(frac*100)}', seed=corr_seed, occlusion_fraction=frac, mode='fusion_occlusion'))
            robust_rows.append(evaluate(model, loaders['fusion'], device, model_name=model_name, condition='missing_s1', mode='fusion_missing_s1', seed=seed_label))
            robust_rows.append(evaluate(model, loaders['fusion'], device, model_name=model_name, condition='missing_s2', mode='fusion_missing_s2', seed=seed_label))
            del model; torch.cuda.empty_cache()
    # TerraMind clean only
    for ckpt in LOCKED_CHECKPOINTS['terramind_frozen']:
        seed_label=SEED_LABELS[ckpt]
        model=make_terramind_model(device); load_state(model, ckpt, device)
        row=evaluate(model, loaders['terramind'], device, model_name='terramind_frozen', seed=seed_label); row['run_type']='seeded'; clean_rows.append(row)
        for d in TEST_DATES: per_date.append({**evaluate(model, loaders['terramind'], device, model_name='terramind_frozen', seed=seed_label, date=d), 'run_type':'seeded'})
        del model; torch.cuda.empty_cache()
    clean_df=pd.DataFrame(clean_rows)
    clean_summary=aggregate(clean_rows)
    robust_df=pd.DataFrame(robust_rows)
    robust_summary=aggregate(robust_rows)
    per_date_df=pd.DataFrame(per_date)
    clean_summary.to_csv(RESULT_DIR/'clean_test_results.csv', index=False)
    robust_df.to_csv(RESULT_DIR/'robustness_test_results.csv', index=False)
    per_date_df.to_csv(RESULT_DIR/'per_date_test_results.csv', index=False)
    summary={'audit':audit_info,'device':str(device),'runtime_seconds':time.perf_counter()-start,'clean_summary':clean_summary.to_dict(orient='records'),'robustness_summary':robust_summary.to_dict(orient='records'),'post_test_tuning':False}
    (RESULT_DIR/'final_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save_figures(clean_summary, robust_df, per_date_df)
    write_doc(clean_summary, robust_summary, per_date_df, audit_info)
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
