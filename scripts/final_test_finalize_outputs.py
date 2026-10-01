from __future__ import annotations
import json, sys, os
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT=Path.cwd(); os.environ.setdefault('TORCH_HOME', str(PROJECT_ROOT/'.torch')); sys.path.insert(0,str(PROJECT_ROOT))
from scripts.final_test_evaluation import make_loaders, load_state, make_terramind_model, model_forward, TEST_DATES, CHECKPOINT_DIR
from src.models.deeplab import create_s2_deeplab, create_sar_deeplab, create_fusion_deeplab
from src.training.train_s1_deeplab import color_mask, stretch_channel

RESULT_DIR=PROJECT_ROOT/'results/final_test'; FIGURE_DIR=PROJECT_ROOT/'figures'; DOCS_DIR=PROJECT_ROOT/'docs'

def df_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df=df.head(max_rows)
    cols=list(df.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _, row in df.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            if isinstance(v,float): vals.append('' if pd.isna(v) else f'{v:.4f}')
            else: vals.append(str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def denorm_rgb(x):
    mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    arr=((x.cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
    return arr

def terramind_rgb(x):
    return ((x[[2,1,0]].cpu()/255.0).clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')

def make_qualitative():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders=make_loaders()
    models=[]
    specs=[('RGB', create_s2_deeplab(num_classes=4,pretrained=True).to(device),'s2','s2_deeplab_weighted_best.pt'),('SAR', create_sar_deeplab(num_classes=4,pretrained=True).to(device),'s1','s1_deeplab_weighted_best.pt'),('Fusion', create_fusion_deeplab(num_classes=4,pretrained=True).to(device),'fusion','s1_s2_early_fusion_best.pt'),('Robust', create_fusion_deeplab(num_classes=4,pretrained=True).to(device),'fusion','s1_s2_moddrop_occlusion_seed42_best.pt')]
    for label,model,key,ckpt in specs:
        load_state(model, ckpt, device); model.eval(); models.append((label,model,key))
    tm=make_terramind_model(device); load_state(tm,'terramind_frozen_seed42_best.pt',device); tm.eval(); models.append(('TerraMind',tm,'terramind'))
    # first two test samples
    batches={k:next(iter(v)) for k,v in loaders.items()}
    n=2; tile=224; label_h=24; cols=['RGB','GT','S2','S1','Fusion','Robust','TerraMind','Error Robust']
    canvas=Image.new('RGB',(tile*len(cols),(tile+label_h)*n),'white'); draw=ImageDraw.Draw(canvas)
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
        for i in range(n):
            rgb_img=denorm_rgb(batches['s2'][0][i]); gt=batches['s2'][1][i]
            preds={}
            for label,model,key in models:
                inp=batches[key][0]
                if isinstance(inp,dict): inp={kk:vv[i:i+1].to(device) for kk,vv in inp.items()}
                else: inp=inp[i:i+1].to(device)
                preds[label]=model_forward(model, inp).argmax(1)[0].cpu()
            err=torch.full_like(gt,255); valid=gt!=255; err[valid & (preds['Robust']==gt)] = 0; err[valid & (preds['Robust']!=gt)] = 3
            arrays=[rgb_img,color_mask(gt.numpy()),color_mask(preds['RGB'].numpy()),color_mask(preds['SAR'].numpy()),color_mask(preds['Fusion'].numpy()),color_mask(preds['Robust'].numpy()),color_mask(preds['TerraMind'].numpy()),color_mask(err.numpy())]
            y=i*(tile+label_h)
            for c,(name,arr) in enumerate(zip(cols,arrays)):
                x=c*tile; draw.text((x+6,y+5),name,fill=(0,0,0)); canvas.paste(Image.fromarray(arr),(x,y+label_h))
    canvas.save(FIGURE_DIR/'final_qualitative_test_examples.png')

def write_doc():
    clean=pd.read_csv(RESULT_DIR/'clean_test_results.csv')
    robust=pd.read_csv(RESULT_DIR/'robustness_test_results.csv')
    per_date=pd.read_csv(RESULT_DIR/'per_date_test_results.csv')
    summary=json.loads((RESULT_DIR/'final_summary.json').read_text())
    robust_summary=pd.DataFrame(summary['robustness_summary'])
    lines=['# Final Held-Out Test Results','', 'Phase 7: first and final held-out test evaluation. No post-test tuning was performed.','', '## Audit', '', f"Test dates: {', '.join(TEST_DATES)}", f"Test tiles: {summary['audit']['test_tile_count']}", 'TEST results were not used to select checkpoints or alter model settings.', '', '## Clean held-out test results', '', df_md(clean), '', '## Robust-model test curves summary', '', df_md(robust_summary), '', '## Per-date behavior', '', df_md(per_date), '', '## Generalization notes', '', '- All models drop substantially relative to validation macro mIoU, indicating temporal generalization difficulty on September dates.', '- TerraMind has the highest binary algae Dice among clean models, while occlusion-trained fusion has the best clean macro mIoU among robust DeepLab variants.', '- Mid and high algae classes remain the hardest classes across models.', '']
    (DOCS_DIR/'FINAL_TEST_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

make_qualitative()
write_doc()
print('final docs and qualitative figure written')

