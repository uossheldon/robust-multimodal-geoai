"""Build an explicit, static-only Pages artifact. Standard library only."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '.site-build'
# Destination -> reviewed source. Never copy an entire figures/results directory.
FILES = {
    'index.html': 'site/index.html',
    'style.css': 'site/style.css',
    'theme.js': 'site/theme.js',
    'favicon.svg': 'site/favicon.svg',
    'assets/hero_overview.svg': 'figures/hero_overview.svg',
    'assets/hero_overview.png': 'figures/hero_overview.png',
    'assets/social_preview.png': 'figures/social_preview.png',
    'assets/final_per_class_iou.png': 'figures/final_per_class_iou.png',
    'assets/final_robustness_curves.png': 'figures/final_robustness_curves.png',
    'assets/clean_test_results.csv': 'results/final_test/clean_test_results.csv',
}


def audit_artifact(path):
    path = Path(path)
    entries = list(path.rglob('*'))
    if any(p.is_symlink() for p in [path, *entries]):
        raise ValueError('No symlinks allowed in the deployment artifact.')
    actual = {p.relative_to(path).as_posix() for p in entries if p.is_file()}
    expected = set(FILES) | {'.nojekyll'}
    if actual != expected:
        raise ValueError(f'Unexpected artifact contents: extra={actual-expected}, missing={expected-actual}')
    if any(p.stat().st_size > 5_000_000 for p in entries if p.is_file()):
        raise ValueError('Oversized presentation file (>5 MB).')


def build():
    if OUTPUT.is_symlink():
        raise ValueError('Output must not be a symlink.')
    OUTPUT.mkdir(exist_ok=True)
    # Fail on extra files before copying; do not silently package or delete them.
    for p in OUTPUT.rglob('*'):
        if p.is_symlink() or (p.is_file() and p.relative_to(OUTPUT).as_posix() not in set(FILES) | {'.nojekyll'}):
            raise ValueError(f'Unreviewed output artifact: {p.name}')
    for dest, source in FILES.items():
        src = ROOT / source
        if src.is_symlink() or not src.is_file():
            raise ValueError(f'Invalid presentation source: {source}')
        target = OUTPUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    (OUTPUT / '.nojekyll').write_text('', encoding='utf-8')
    audit_artifact(OUTPUT)
    print(f'Built {len(FILES)+1} reviewed static files in .site-build/; no model execution.')


if __name__ == '__main__':
    build()
