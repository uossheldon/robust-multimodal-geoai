"""Compile sources and run dependency-free behaviour and release tests."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check():
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT / folder).rglob('*.py')]
    for path in paths:
        compile(path.read_text(encoding='utf-8'), str(path.relative_to(ROOT)), 'exec')
    count = 0
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            if name.startswith('test_'):
                getattr(module, name)()
                count += 1
    print(f'PASS: {len(paths)} Python sources compile; {count} lightweight tests pass.')


if __name__ == '__main__':
    check()
