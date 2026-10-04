/* Progressive enhancement. Only saved aggregates; no interpolation or models. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const names = {mean_iou: 'Macro mIoU', macro_dice: 'Macro Dice', binary_algae_dice: 'Binary algae Dice'};
  const svg = byId('explorer-chart');
  const format = s => s.mean.toFixed(4) + (s.std === null ? '' : ` ± ${s.std.toFixed(4)}`);
  let data;
  let mode = 'optical';
  function node(tag, attributes = {}, text) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    svg.appendChild(el);
    return el;
  }
  const text = (x, y, value, attrs = {}) => node('text', {x, y, ...attrs}, value);
  const line = (x1,y1,x2,y2,cls='plot-grid') => node('line',{x1,y1,x2,y2,class:cls});
  function start(title) {
    svg.replaceChildren(); node('title',{id:'explorer-chart-title'},title);
  }
  function dot(x,y,cls,selected=false) {
    if (selected) node('circle',{cx:x,cy:y,r:11,class:`plot-halo ${cls}`});
    node('circle',{cx:x,cy:y,r:5,class:cls});
  }
  function errorX(s,y,scale,cls) {
    if (s.std!==null) {
      const left=scale(s.mean-s.std),right=scale(s.mean+s.std);
      line(left,y,right,y,cls);line(left,y-5,left,y+5,cls);line(right,y-5,right,y+5,cls);
    }
    dot(scale(s.mean),y,cls);
  }
  function valueCards(items) {
    const target=byId('explorer-values');target.replaceChildren();
    for (const [label,value,note] of items) {
      const card=document.createElement('div');card.className='result-value';
      for (const [tag,content] of [['h4',label],['strong',value],['p',note]]) {
        const el=document.createElement(tag);el.textContent=content;card.appendChild(el);
      }
      target.appendChild(card);
    }
  }
  function horizontalAxis(maximum,label,ticks = [0, maximum/4, maximum/2, maximum*3/4, maximum]) {
    const scale=v=>260+v/maximum*590;
    for(const v of ticks) {
      const x=scale(v);
      line(x,35,x,345);text(x,374,v.toFixed(2),{'text-anchor':'middle'});
    }
    text(555,408,label,{'text-anchor':'middle',class:'axis-label'});
    return scale;
  }
  function renderOptical() {
    const rate=Number(byId('occlusion-rate').value);
    start(`Macro mIoU at measured optical occlusion rates; selected ${rate}%`);
    const x=r=>85+r/70*740,y=v=>335-v/.30*290;
    for (const v of [0,.05,.10,.15,.20,.25,.30]) {
      line(85,y(v),825,y(v));text(70,y(v)+6,v.toFixed(2),{'text-anchor':'end'});
    }
    for(const r of data.occlusion_rates) text(x(r),367,`${r}`,{'text-anchor':'middle'});
    text(455,408,'Optical occlusion rate (%)',{'text-anchor':'middle',class:'axis-label'});
    text(85,23,'Macro mIoU',{class:'axis-label'});
    const cards=[];
    for (const [index,id] of ['modality_dropout','occlusion_trained'].entries()) {
      const model=data.models.find(m=>m.id===id),cls=`series-${index}`;
      const rowFor=r=>data.robustness.find(s=>s.model===id && s.condition===(r===0?'clean':`occlusion_${r}`)).mean_iou;
      for (const r of data.occlusion_rates) {
        const s=rowFor(r),px=x(r),top=y(s.mean+s.std),bottom=y(s.mean-s.std);
        line(px,top,px,bottom,cls);line(px-5,top,px+5,top,cls);line(px-5,bottom,px+5,bottom,cls);
        dot(px,y(s.mean),cls,r===rate);
      }
      const s=rowFor(rate);
      cards.push([model.label,format(s),`${rate}% simulated optical occlusion · Macro mIoU`]);
      node('circle',{cx:390+index*210,cy:21,r:5,class:cls});text(402+index*210,27,model.label,{class:'plot-legend'});
    }
    valueCards(cards);
    byId('explorer-context').textContent=rate===0?'Clean TEST · mean ± sample SD across 3 training seeds.':'TEST · mean ± pooled SD across 9 training-seed × corruption-seed runs.';
    byId('explorer-note').textContent='Non-monotonic scores under simulated corruption do not imply that removing optical information improves imagery.';
  }
  function renderComparison() {
    const metric=byId('comparison-metric').value;
    start(`Clean September test comparison: ${names[metric]}`);
    const scale=horizontalAxis(metric==='mean_iou'?.30:metric==='macro_dice'?.40:.80,names[metric]);
    const cards=[];
    data.models.forEach((m,i)=>{
      const y=60+i*53,s=m.clean[metric];text(240,y+6,m.label,{'text-anchor':'end'});
      errorX(s,y,scale,'series-0');cards.push([m.label,format(s),m.n===1?'Single run':'3 training seeds']);
    });
    valueCards(cards);
    byId('explorer-context').textContent='Clean TEST · single-run baselines and three-seed means; error bars show sample SD where available.';
    byId('explorer-note').textContent=metric==='binary_algae_dice'?'Binary algae Dice measures binary algae detection, not four-class severity segmentation.':'Four-class segmentation: Background, Low algae, Mid algae and High algae.';
  }
  function renderTemporal() {
    const selected=byId('temporal-model').value,model=data.models.find(m=>m.id===selected);
    start(`Validation-to-test Macro mIoU; highlighted ${model.label}`);
    const scale=horizontalAxis(.35,'Macro mIoU',[0,.10,.20,.30]);
    data.models.forEach((m,i)=>{
      const y=60+i*53;
      if(m.id===selected) node('rect',{x:5,y:y-22,width:890,height:45,rx:4,class:'selected-row'});
      text(240,y+6,m.label,{'text-anchor':'end'});
      line(scale(m.clean.mean_iou.mean),y,scale(m.validation.mean),y,'plot-connector');
      errorX(m.validation,y,scale,'series-0');errorX(m.clean.mean_iou,y,scale,'series-1');
    });
    const delta=model.clean.mean_iou.mean-model.validation.mean;
    valueCards([['Validation Macro mIoU',format(model.validation),model.label],
                ['TEST Macro mIoU',format(model.clean.mean_iou),model.label],
                ['Absolute change (TEST − validation)',delta.toFixed(4),'Score difference; not a percentage']]);
    byId('explorer-context').textContent='Blue: validation · Orange: September TEST. Lines pair the two observed evaluations, with sample SD for three-seed methods.';
    byId('explorer-note').textContent='All six models declined on these dates. This is consistent with temporal/domain shift; it does not establish geographic generalisation or physical causation.';
  }
  function render() {
    byId('rate-control').hidden=mode!=='optical';byId('metric-control').hidden=mode!=='comparison';byId('model-control').hidden=mode!=='temporal';
    if(mode==='optical') renderOptical();else if(mode==='comparison') renderComparison();else renderTemporal();
  }
  fetch('assets/results_summary.json').then(response=>{
    if(!response.ok) throw new Error('Aggregate data unavailable');return response.json();
  }).then(summary=>{
    data=summary;
    for(const m of data.models) {
      const option=document.createElement('option');option.value=m.id;option.textContent=m.label;byId('temporal-model').appendChild(option);
    }
    document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{
      mode=button.dataset.mode;
      document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
      render();
    }));
    ['occlusion-rate','comparison-metric','temporal-model'].forEach(id=>byId(id).addEventListener('change',render));
    render();byId('explorer').hidden=false;byId('explorer-fallback').hidden=true;
  }).catch(()=>{
    byId('explorer-fallback').textContent='Interactive data could not load. The static figures and results table below remain available.';
  });
})();
