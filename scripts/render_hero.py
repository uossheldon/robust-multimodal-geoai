"""Draw original schematic portfolio assets using frozen summary numbers only."""
from pathlib import Path
import csv
import json
import os
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BG='#f5f6f0'; INK='#173c39'; MUTED='#526763'; TEAL='#176c60'; GOLD='#a66020'

def card(ax,x,y,w,h,color='white',edge='#d4ded7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                              facecolor=color,edgecolor=edge,linewidth=1))

def text(ax,x,y,s,size=18,color=INK,weight='normal',**kw):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,va='center',**kw)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,linewidth=1.7,color=TEAL))

def headlines():
    with (ROOT/'results/final_test/clean_test_results.csv').open(newline='',encoding='utf-8') as f:
        rows={r['model']:r for r in csv.DictReader(f)}
    summary=json.loads((ROOT/'results/final_test/final_summary.json').read_text())
    occ=next(r for r in summary['robustness_summary'] if r['model']=='occlusion_trained' and r['condition']=='occlusion_70')
    def value(r,k): return f"{float(r[k+'_mean']):.4f} ± {float(r[k+'_std']):.4f}"
    return [value(rows['occlusion_trained'],'mean_iou'),value(occ,'mean_iou'),value(rows['terramind_frozen'],'binary_algae_dice')]

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'geoai-phase9'})
    vals=headlines()
    fig,ax=plt.subplots(figsize=(14.4,9.6),dpi=150)
    fig.patch.set_facecolor(BG); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    text(ax,.045,.943,'ROBUST MULTIMODAL GEOAI',15,TEAL,'bold')
    text(ax,.045,.884,'When optical information disappears.',30,weight='bold')
    text(ax,.045,.827,'Four-class algal-bloom mapping with Sentinel-1 + Sentinel-2',18,MUTED)
    card(ax,.05,.643,.235,.12)
    text(ax,.068,.720,'Sentinel-2',22,weight='bold'); text(ax,.068,.674,'Optical · B04 / B03 / B02',14,MUTED)
    card(ax,.05,.491,.235,.12)
    text(ax,.068,.568,'Sentinel-1',22,weight='bold'); text(ax,.068,.522,'SAR · VV / VH',14,MUTED)
    card(ax,.38,.542,.245,.175,INK,INK)
    text(ax,.502,.655,'Multimodal',25,'white','bold',ha='center')
    text(ax,.502,.609,'GeoAI',25,'white','bold',ha='center')
    text(ax,.502,.566,'Aligned 224 × 224 tiles',13,'#cfdfd7',ha='center')
    arrow(ax,(.291,.704),(.373,.641)); arrow(ax,(.291,.548),(.373,.603)); arrow(ax,(.637,.63),(.704,.63))
    card(ax,.721,.508,.225,.24)
    text(ax,.738,.708,'Algae severity',22,weight='bold')
    for i,(name,color) in enumerate([('Background','#3d4746'),('Low algae','#6a9950'),('Mid algae','#dfa835'),('High algae','#be5949')]):
        y=.657-i*.04
        ax.add_patch(Rectangle((.741,y-.010),.017,.020,color=color))
        text(ax,.772,y,name,14,MUTED)
    text(ax,.05,.45,'STRESS TESTS',13,TEAL,'bold')
    text(ax,.05,.412,'Simulated optical occlusion',18,weight='bold')
    text(ax,.05,.377,'0 / 10 / 30 / 50 / 70%',14,MUTED)
    text(ax,.38,.412,'Missing modality',18,weight='bold')
    text(ax,.38,.377,'S1 absent or S2 absent',14,MUTED)
    text(ax,.721,.412,'Training interventions',16,weight='bold')
    text(ax,.721,.377,'Modality dropout',14,MUTED)
    text(ax,.721,.349,'+ occlusion-aware training',14,MUTED)
    ax.plot([.05,.945],[.307,.307],color='#c6d3cb',lw=1)
    text(ax,.05,.277,'HELD-OUT TEMPORAL TEST',13,TEAL,'bold')
    text(ax,.95,.277,'114 tiles · 8 & 21 September 2025',14,MUTED,ha='right')
    labels=['Occlusion-trained fusion\nClean TEST','Occlusion-trained fusion\n70% optical occlusion','Frozen TerraMind\nClean TEST']
    foot=['Macro mIoU','Macro mIoU','Binary algae Dice']
    for x,label,value,note in zip([.05,.359,.668],labels,vals,foot):
        card(ax,x,.055,.279,.172)
        text(ax,x+.01,.198,label,13,MUTED)
        text(ax,x+.01,.128,value,23,TEAL,'bold')
        text(ax,x+.01,.083,note,13,MUTED)
    fig.savefig(ROOT/'figures/hero_overview.png',dpi=150)
    fig.savefig(ROOT/'figures/hero_overview.svg',metadata={'Date':None})
    svg=ROOT/'figures/hero_overview.svg'
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif").splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(12.8,6.4),dpi=100); fig.patch.set_facecolor(INK)
    fig.subplots_adjust(0,0,1,1); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    text(ax,.06,.87,'INDEPENDENT EARTH-OBSERVATION RESEARCH',15,'#abd1ba','bold')
    text(ax,.06,.69,'Robust multimodal GeoAI',37,'white','bold')
    text(ax,.06,.56,'Sentinel-1 + Sentinel-2 · Algal-bloom segmentation',19,'#d4e4dc')
    text(ax,.06,.46,'Optical degradation. Missing sensors. Temporal generalisation.',17,'#d4e4dc')
    for x,label,value,note in zip([.06,.365,.67],['CLEAN ROBUST FUSION','70% OPTICAL OCCLUSION','FROZEN TERRAMIND'],vals,['mIoU · 3 training seeds','mIoU · 9 pooled runs¹','Binary algae Dice · 3 seeds']):
        text(ax,x,.30,label,11,'#abd1ba','bold')
        text(ax,x,.23,value,25,'white','bold')
        text(ax,x,.17,note,12,'#d4e4dc')
    text(ax,.06,.075,'Final TEST · ¹ Training × corruption seeds · ± sample SD · No significance claim',12,'#d4e4dc')
    fig.savefig(ROOT/'figures/social_preview.png',dpi=100); plt.close(fig)
    folder=ROOT/'figures/badges'; folder.mkdir(exist_ok=True)
    for name,label in [('python','Python · PyTorch'),('geospatial','Rasterio · S1 + S2'),('seeds','3 training seeds'),('terramind','Frozen TerraMind')]:
        w=len(label)*7+24
        (folder/f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="24" role="img" aria-label="{escape(label)}"><rect width="{w}" height="24" rx="4" fill="{INK}"/><text x="{w/2}" y="16" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="white">{escape(label)}</text></svg>\n',encoding='utf-8')

if __name__=='__main__': main()
