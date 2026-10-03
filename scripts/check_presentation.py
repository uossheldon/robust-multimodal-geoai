"""Check static links, saved metric values, accessibility basics and packaging."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import csv
import json
import re
import sys
import tempfile
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_site import FILES, build, audit_artifact, OUTPUT
from presentation_data import load_summary


class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.links=[]; self.images=[]; self.metrics={}; self.model=None; self.metric=None; self.text=[]; self.headlines={}; self.headline=None
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='tr': self.model=a.get('data-model')
        if 'data-metric' in a: self.metric=a['data-metric']; self.text=[]
        if 'data-headline' in a: self.headline=a['data-headline']; self.text=[]
        if tag=='img': self.images.append(a)
        for k in ('href','src'):
            if k in a: self.links.append(a[k])
    def handle_data(self, data):
        if self.metric or self.headline: self.text.append(data)
    def handle_endtag(self, tag):
        if tag=='td' and self.metric:
            self.metrics[(self.model,self.metric)]=' '.join(''.join(self.text).split()); self.metric=None
        if tag=='strong' and self.headline:
            self.headlines[self.headline]=' '.join(''.join(self.text).split()); self.headline=None


def check():
    build()
    html=(OUTPUT/'index.html').read_text(encoding='utf-8')
    page=Page(); page.feed(html)
    assert len(set(page.ids))==len(page.ids), 'Duplicate HTML IDs'
    for image in page.images:
        assert image.get('alt') and image.get('width') and image.get('height'), 'Image needs alt text and dimensions'
    for link in page.links:
        parsed=urlsplit(link)
        if parsed.scheme:
            assert parsed.scheme=='https', 'Unexpected external protocol'
            continue
        assert not parsed.path.startswith('/'), 'Project Pages links must be relative'
        if parsed.path: assert (OUTPUT/unquote(parsed.path)).is_file(), f'Missing site asset: {link}'
        if parsed.fragment and not parsed.path: assert parsed.fragment in page.ids, f'Missing anchor: {link}'
    with (ROOT/'results/final_test/clean_test_results.csv').open(newline='',encoding='utf-8') as f:
        results={r['model']:r for r in csv.DictReader(f)}
    def value(row,key):
        return f"{float(row[key+'_mean']):.4f}"+(f" ± {float(row[key+'_std']):.4f}" if int(row['n'])>1 else '')
    assert len(page.metrics)==18
    for (model,metric),text in page.metrics.items(): assert text==value(results[model],metric), f'Metric mismatch: {model}/{metric}'
    summary=json.loads((ROOT/'results/final_test/final_summary.json').read_text())
    occ=next(r for r in summary['robustness_summary'] if r['model']=='occlusion_trained' and r['condition']=='occlusion_70')
    assert page.headlines=={'clean':value(results['occlusion_trained'],'mean_iou'),'occlusion':value(occ,'mean_iou'),'terramind':value(results['terramind_frozen'],'binary_algae_dice')}
    for phrase in ['Phase 2D','post-Phase-7','3,474,284','3,169,093','pool','licensing review','not measured real cloud cover']:
        assert phrase in html, f'Missing scientific qualification: {phrase}'
    assert not re.search(r'\b[A-Za-z]:[\\/]',html), 'Local Windows path in site'
    assert 'secrets.' not in (ROOT/'.github/workflows/pages.yml').read_text(), 'Pages must not need user secrets'
    # Every interactive value must be an exact copy of a frozen aggregate.
    data=json.loads((OUTPUT/'assets/results_summary.json').read_text(encoding='utf-8'))
    assert data==load_summary(), 'Explorer data differs from frozen source summaries'
    assert data['occlusion_rates']==[0,10,30,50,70]
    assert [m['n'] for m in data['models']]==[1,1,1,3,3,3]
    assert len(data['robustness'])==14
    for row in data['robustness']:
        assert row['mean_iou']['n']==(9 if row['condition'].startswith('occlusion_') else 3)
    # Independent rounded cross-check against the approved presentation brief.
    expected_val=[.2274,.2349,.2425,.2725,.2648,.2771]
    assert [round(m['validation']['mean'],4) for m in data['models']]==expected_val
    assert all(m['validation']['mean']>m['clean']['mean_iou']['mean'] for m in data['models'])
    assert all(m['clean']['mean_iou']['std'] is None for m in data['models'][:3])
    assert 'id="evaluation-notes"' in html and '<summary>Evaluation notes</summary>' in html
    assert '<div id="explorer" hidden>' in html, 'Static no-JS fallback required'
    assert re.findall(r'<option value="(\d+)">', html)==['0','10','30','50','70']
    for name in ['final_clean_model_comparison','final_robustness_curves','final_per_class_iou','final_temporal_generalisation']:
        ET.parse(ROOT/'figures'/f'{name}.svg')
    hero=(ROOT/'figures/hero_overview.svg').read_text(encoding='utf-8')
    for value in page.headlines.values(): assert value in hero
    for name in FILES:
        assert not any(s in name.lower() for s in ['qualitative','alignment','checkpoint','.tif','.zip']), 'Unreviewed data-derived asset'
    # Prove the deployment audit rejects an unexpected raw-data-like file.
    with tempfile.TemporaryDirectory(prefix='geoai-presentation-') as temp:
        test=Path(temp)/'artifact'; shutil.copytree(OUTPUT,test)
        (test/'unexpected.tif').write_bytes(b'not data')
        try: audit_artifact(test)
        except ValueError: pass
        else: raise AssertionError('Artifact audit accepted an unlisted file')
    for p in [ROOT/'figures/hero_overview.svg',ROOT/'site/favicon.svg',*(ROOT/'figures/badges').glob('*.svg')]: ET.parse(p)
    # Relative README images must remain viewable on GitHub.
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for ref in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',readme):
        if not ref.startswith('https://'): assert (ROOT/ref).is_file(), f'Broken README image: {ref}'
    for p in OUTPUT.rglob('*'):
        if p.is_file(): assert p.read_bytes()==((ROOT/FILES[p.relative_to(OUTPUT).as_posix()]).read_bytes() if p.name!='.nojekyll' else b'')
    print('PASS: 18 TEST cells, 3 headlines, all explorer aggregates and 6 validation values match frozen sources; links, notes, SVGs and safe artifact pass.')


if __name__=='__main__': check()
