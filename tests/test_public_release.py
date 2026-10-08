"""Public documentation links and the final vector-figure inventory."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    'hero_overview.svg', 'social_preview.png',
    'final_clean_model_comparison.svg', 'final_per_class_iou.svg',
    'final_robustness_curves.svg', 'final_temporal_generalisation.svg',
    'class_distribution_by_date.svg', 's2_imbalance_methods_comparison.svg',
    'terramind_validation.svg', 'ensemble_diagnostics.svg',
}


def anchor_ids(text):
    headings = re.findall(r'^#{1,6}\s+(.+)$', text, flags=re.M)
    return {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}


def test_markdown_links():
    documents = [ROOT/'README.md', ROOT/'data/README.md', ROOT/'results/README.md', *(ROOT/'docs').glob('*.md')]
    for document in documents:
        text = document.read_text(encoding='utf-8')
        for ref in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            target = urlsplit(ref.strip('<>'))
            if target.scheme:
                assert target.scheme == 'https', (document.name, ref)
                continue
            path = (document.parent/unquote(target.path)).resolve() if target.path else document
            assert path.exists(), (document.name, ref)
            if target.fragment and path.suffix=='.md':
                assert target.fragment in anchor_ids(path.read_text(encoding='utf-8')), (document.name, ref)


def test_final_figures_are_referenced_and_public():
    actual = {p.name for p in (ROOT/'figures').iterdir() if p.is_file()}
    assert actual == FIGURES, (actual-FIGURES, FIGURES-actual)
    public = '\n'.join(p.read_text(encoding='utf-8') for p in [ROOT/'README.md', ROOT/'site/index.html', *(ROOT/'docs').glob('*.md')])
    for name in FIGURES:
        assert name in public, f'Orphan figure: {name}'
        if name.endswith('.svg'):
            tree = ET.parse(ROOT/'figures'/name)
            labels = ' '.join(tree.getroot().itertext())
            assert not re.search(r'Phase\s+\d|transcription|reconciliation|checkpoint metadata|historical export|partial export|no inference', labels, re.I), name


def test_no_unresolved_global_symbols():
    import builtins
    import symtable
    issues = []
    for path in [*(ROOT/'src').rglob('*.py'), *(ROOT/'scripts').glob('*.py')]:
        table = symtable.symtable(path.read_text(encoding='utf-8'), str(path), 'exec')
        known = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        known |= set(dir(builtins)) | {'__file__','__name__','__package__','__doc__','__annotations__'}
        def visit(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                    issues.append((str(path.relative_to(ROOT)), scope.get_name(), symbol.get_name()))
            for child in scope.get_children():
                visit(child)
        visit(table)
    assert not issues, issues
