"""Render four presentation figures from saved aggregates only; no inference."""
import os
from presentation_data import ROOT, load_summary
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TEAL='#176c60'; BLUE='#326c9f'; ORANGE='#b05c20'; INK='#203b37'


def save(fig, name):
    for extension in ['png', 'svg']:
        path=ROOT/'figures'/(name+'.'+extension)
        fig.savefig(path,dpi=220,facecolor='white',metadata={'Date':None} if extension=='svg' else None)
        if extension=='svg':
            content=path.read_text(encoding='utf-8').replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif")
            path.write_text('\n'.join(s.rstrip() for s in content.splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def setup(ax,xlabel,ylabel=None):
    ax.set_xlabel(xlabel,labelpad=10)
    if ylabel: ax.set_ylabel(ylabel,labelpad=12)
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e2e8e3',linewidth=.7);ax.tick_params(length=0,pad=8)


def point(ax,value,y,color,marker='o',label=None):
    ax.errorbar(value['mean'],y,xerr=value['std'],fmt=marker,color=color,
                capsize=3,markersize=7,elinewidth=1.3,label=label)


def render():
    data=load_summary(); rows=data['models']; labels=[r['label'] for r in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
                         'axes.labelcolor':INK,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','svg.hashsalt':'geoai-final'})
    fig,axes=plt.subplots(1,2,figsize=(13,6.2),sharey=True,gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
    for y,row in enumerate(rows):
        point(axes[0],row['clean']['mean_iou'],y-.12,TEAL,label='Macro mIoU' if y==0 else None)
        point(axes[0],row['clean']['macro_dice'],y+.12,BLUE,'s','Macro Dice' if y==0 else None)
        point(axes[1],row['clean']['binary_algae_dice'],y,ORANGE)
    axes[0].set_yticks(range(6),labels);axes[0].invert_yaxis()
    axes[0].set_title('(a) Four-class segmentation',pad=15);axes[1].set_title('(b) Binary algae detection',pad=15)
    setup(axes[0],'Score (Macro mIoU / Macro Dice)','Model');setup(axes[1],'Binary algae Dice')
    axes[0].set_xlim(0,.43);axes[1].set_xlim(0,.8);axes[0].legend(loc='lower right',frameon=False)
    fig.suptitle('Held-out September test performance',fontsize=17)
    save(fig,'final_clean_model_comparison')

    robust={(r['model'],r['condition']):r['mean_iou'] for r in data['robustness']}
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    for model,color,marker,offset in [('modality_dropout',BLUE,'o',-.06),('occlusion_trained',ORANGE,'s',.06)]:
        name=next(r['label'] for r in rows if r['id']==model)
        scores=[robust[(model,'clean' if rate==0 else f'occlusion_{rate}')] for rate in data['occlusion_rates']]
        # Measured points only: no connecting line implies unmeasured levels.
        axes[0].errorbar(data['occlusion_rates'],[s['mean'] for s in scores],yerr=[s['std'] for s in scores],
                        fmt=marker,linestyle='none',color=color,capsize=4,markersize=7,label=name)
        missing=[robust[(model,c)] for c in ['missing_s1','missing_s2']]
        axes[1].errorbar([offset,1+offset],[s['mean'] for s in missing],yerr=[s['std'] for s in missing],
                        fmt=marker,linestyle='none',color=color,capsize=4,markersize=7)
    axes[0].set_xticks(data['occlusion_rates']);axes[0].set_xlim(-5,75)
    axes[0].set_title('(a) Simulated optical occlusion',pad=15)
    axes[1].set_title('(b) Missing-modality performance',pad=15)
    axes[1].set_xticks([0,1],['S1 unavailable','S2 unavailable']);axes[1].set_xlim(-.5,1.5)
    for ax,xlabel in zip(axes,['Optical occlusion rate (%)','Missing modality']):
        setup(ax,xlabel,'Macro mIoU');ax.set_ylim(0,.28)
    axes[0].legend(loc='lower left',frameon=False,fontsize=10)
    fig.suptitle('Robustness to optical occlusion on the held-out September test set',fontsize=15)
    save(fig,'final_robustness_curves')

    keys=['iou_background','iou_low','iou_mid','iou_high']
    values=[[r['clean'][key]['mean'] for key in keys] for r in rows]
    fig,ax=plt.subplots(figsize=(11.5,6.4),layout='constrained')
    cmap=LinearSegmentedColormap.from_list('academic_teal',['#f2f5ef','#a0c3b2','#176c60','#103d37'])
    im=ax.imshow(values,cmap=cmap,vmin=0,vmax=.55,aspect='auto')
    ax.set_xticks(range(4),['Background','Low algae','Mid algae','High algae'])
    ax.set_yticks(range(6),labels);ax.set_xlabel('Class',labelpad=12);ax.set_ylabel('Model',labelpad=12)
    ax.tick_params(length=0,pad=10)
    for y,row in enumerate(rows):
        for x,key in enumerate(keys):
            s=row['clean'][key];text=f"{s['mean']:.4f}"
            if s['std'] is not None: text+=f"\n± {s['std']:.4f}"
            ax.text(x,y,text,ha='center',va='center',fontsize=12,color='white' if s['mean']>.29 else INK)
    fig.colorbar(im,ax=ax,label='IoU',fraction=.038,pad=.035)
    ax.set_title('Per-class IoU on the held-out September test set',fontsize=16,pad=18)
    save(fig,'final_per_class_iou')

    fig,ax=plt.subplots(figsize=(10.8,6.2),layout='constrained')
    for y,row in enumerate(rows):
        val=row['validation'];test=row['clean']['mean_iou']
        ax.plot([test['mean'],val['mean']],[y,y],color='#bbc9c1',linewidth=2,zorder=1)
        point(ax,val,y,BLUE,'s','Validation' if y==0 else None)
        point(ax,test,y,TEAL,'o','September TEST' if y==0 else None)
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(.10,.32)
    ax.set_xticks([.10,.15,.20,.25,.30], ['0.10','0.15','0.20','0.25','0.30'])
    setup(ax,'Macro mIoU','Model');ax.legend(loc='upper right',frameon=False)
    ax.set_title('Validation-to-test performance shift',fontsize=17,pad=18)
    save(fig,'final_temporal_generalisation')
    print('Rendered four PNG/SVG figures from frozen summaries only.')


if __name__=='__main__': render()
