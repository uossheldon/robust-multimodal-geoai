"""Render final research charts from canonical saved results."""
import os
from presentation_data import ROOT, load_summary
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.insert(0, str(ROOT))
from src.visualization.style import configure, save as export, grid, TEAL, BLUE, ORANGE, INK


def save(fig, name):
    export(fig, ROOT, name)


def setup(ax,xlabel,ylabel=None):
    ax.set_xlabel(xlabel,labelpad=10)
    if ylabel: ax.set_ylabel(ylabel,labelpad=12)
    grid(ax)


def point(ax,value,y,color,marker='o',label=None):
    ax.errorbar(value['mean'],y,xerr=value['std'],fmt=marker,color=color,
                capsize=3,markersize=7,elinewidth=1.3,label=label)


def render():
    data=load_summary(); rows=data['models']; labels=[r['label'] for r in rows]
    configure()
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
    fig.supxlabel('Single runs or mean ± sample SD across three training seeds.', fontsize=10)
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
    fig.suptitle('September test robustness')
    fig.supxlabel('Sample SD: clean/missing = 3 training seeds; occlusion = 9 pooled training × corruption runs.', fontsize=10)
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
            s=row['clean'][key];text=f"{s['mean']:.3f}"
            if s['std'] is not None: text+=f"\n± {s['std']:.3f}"
            ax.text(x,y,text,ha='center',va='center',fontsize=12,color='white' if s['mean']>.29 else INK)
    fig.colorbar(im,ax=ax,label='IoU',fraction=.038,pad=.035)
    ax.set_title('Per-class IoU on the held-out September test set',fontsize=16,pad=18)
    fig.supxlabel('Single runs or mean ± sample SD across three training seeds.', fontsize=10)
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
    fig.supxlabel('Single runs or mean ± sample SD across three training seeds.', fontsize=10)
    save(fig,'final_temporal_generalisation')
    render_supporting_figures()
    print('Rendered eight SVG charts from canonical saved results.')





def render_supporting_figures():
    from presentation_data import read_csv, read_json
    from src.visualization.style import CLASS_COLORS, MODEL_COLORS
    import numpy as np

    dates = read_csv('results/per_date_statistics.csv')
    fig, ax = plt.subplots(figsize=(12, 5.8), layout='constrained')
    bottom = np.zeros(len(dates))
    for k, (label, color) in enumerate(zip(['Background', 'Low algae', 'Mid algae', 'High algae'], CLASS_COLORS)):
        values = np.array([float(r[f'class_{k}_fraction']) for r in dates])
        ax.bar(range(len(dates)), values, bottom=bottom, color=color, label=label, width=.72)
        bottom += values
    ax.set_xticks(range(len(dates)), [r['date'][5:] for r in dates], rotation=45, ha='right')
    ax.set_ylim(0, 1); ax.set_ylabel('Fraction of common valid pixels')
    ax.set_xlabel('Acquisition date (2025)'); ax.set_title('Class balance across acquisition dates')
    ax.legend(ncols=4, loc='upper center', bbox_to_anchor=(.5, 1.14))
    save(fig, 'class_distribution_by_date')

    rows = read_csv('results/s2_deeplab_balanced_sampling/validation_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained', gridspec_kw={'width_ratios':[1, 1.3]})
    labels = ['Unweighted', 'Weighted loss', 'Balanced sampling']
    colors = ['#647a82', MODEL_COLORS['s2_weighted'], '#a58c57']
    for i, (row, label, color) in enumerate(zip(rows, labels, colors)):
        axes[0].bar(i, float(row['mean_iou']), color=color, width=.6)
        axes[0].text(i, float(row['mean_iou'])+.008, f"{float(row['mean_iou']):.3f}", ha='center')
        axes[1].plot(range(4), [float(row[k]) for k in ['iou_background','iou_low','iou_mid','iou_high']], marker='o', color=color, label=label)
    axes[0].set_xticks(range(3), labels, rotation=20, ha='right'); axes[0].set_ylim(0,.30)
    axes[0].set_ylabel('Macro mIoU'); axes[0].set_title('(a) Validation segmentation')
    axes[1].set_xticks(range(4), ['Background','Low','Mid','High']); axes[1].set_ylabel('IoU')
    axes[1].set_ylim(0,.6); axes[1].set_title('(b) Class behaviour'); axes[1].legend()
    for ax in axes: grid(ax, 'y')
    fig.suptitle('Sentinel-2 class-imbalance strategies · single runs')
    save(fig, 's2_imbalance_methods_comparison')

    terra = read_json('results/terramind_validation.json')
    data = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout='constrained')
    metrics = ['mean_iou','macro_dice','iou_background','iou_low','iou_mid','iou_high']
    for i, key in enumerate(metrics):
        point(axes[0], terra['aggregate'][key], i, TEAL)
    axes[0].set_yticks(range(6), ['Macro mIoU','Macro Dice','Background IoU','Low IoU','Mid IoU','High IoU'])
    axes[0].invert_yaxis(); axes[0].set_xlim(0,.5); axes[0].set_title('(a) Frozen TerraMind')
    robust = [m for m in data['models'] if m['n']==3]
    for i, row in enumerate(robust):
        point(axes[1], row['validation'], i, MODEL_COLORS[row['id']])
    axes[1].set_yticks(range(3), ['Modality\ndropout','Occlusion-trained\nfusion','Frozen\nTerraMind'])
    axes[1].invert_yaxis(); axes[1].set_xlim(.20,.32); axes[1].set_title('(b) Three-seed comparison')
    setup(axes[0], 'Score'); setup(axes[1], 'Macro mIoU')
    fig.suptitle('TerraMind validation performance')
    fig.supxlabel('Mean ± sample SD across three training seeds.', fontsize=10)
    save(fig, 'terramind_validation')

    single = {r['condition']:r for r in read_csv('results/uncertainty/aggregate_metrics.csv') if r['method']=='occlusion_training'}
    ens = read_csv('results/ensemble_uncertainty/metrics.csv')
    errors = read_csv('results/ensemble_uncertainty/error_detection.csv')
    conditions = [r['condition'] for r in ens]
    labels = ['Clean','30%','50%','70%','S1 missing','S2 missing']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout='constrained')
    axes[0].plot(range(6), [float(single[c]['ece_mean']) for c in conditions], 's-', color=BLUE, label='Individual mean (3 seeds)')
    axes[0].plot(range(6), [float(r['ece']) for r in ens], 'o-', color=TEAL, label='Probability ensemble')
    axes[0].set_ylabel('ECE (lower is better)'); axes[0].set_ylim(0,.27)
    axes[0].set_title('(a) Calibration'); axes[0].legend(loc='lower right')
    for key, label, color, marker in [('single_entropy','Seed-42 entropy',BLUE,'s'),('ensemble_entropy','Ensemble entropy',TEAL,'o'),('mutual_information','Mutual information','#76658c','^'),('disagreement','Disagreement',ORANGE,'D')]:
        vals = {r['condition']:float(r['auroc']) for r in errors if r['uncertainty']==key}
        axes[1].plot(range(6), [vals[c] for c in conditions], marker=marker, color=color, label=label)
    axes[1].axhline(.5, color='#8b9693', ls='--', lw=1)
    axes[1].set_ylim(.33,.70); axes[1].set_ylabel('AUROC (ensemble errors)')
    axes[1].set_title('(b) Error detection'); axes[1].legend(ncols=2, loc='upper center', fontsize=9)
    for ax in axes:
        ax.set_xticks(range(6), labels, rotation=30, ha='right')
        ax.set_xlabel('Validation input condition'); grid(ax, 'y')
    fig.suptitle('Ensembling improves calibration, not consistent error detection')
    save(fig, 'ensemble_diagnostics')


if __name__ == '__main__':
    render()
