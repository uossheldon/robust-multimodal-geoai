"""Static compilation + existing pure path assertions, without ML imports.

The original path tests import the complete raster/PyTorch dataset module.
Extract only its actual pure resolver function and the two existing assertions
using Python's AST. This avoids installing ML frameworks in presentation CI.
It does not test raster loading, tensors, models or inference.
"""
import ast
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def check():
    paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT / folder).rglob('*.py')]
    for path in paths:
        compile(path.read_text(encoding='utf-8'), str(path.relative_to(ROOT)), 'exec')
    module = ast.parse((ROOT / 'src/data/terramind_dataset.py').read_text(encoding='utf-8'))
    resolver = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_terramind_paths')
    namespace = {'Path': Path, 'PurePath': PurePath, 'PurePosixPath': PurePosixPath, 'PureWindowsPath': PureWindowsPath}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), '<actual path resolver>', 'exec'), namespace)
    original_tests = ast.parse((ROOT / 'tests/test_terramind_paths.py').read_text(encoding='utf-8'))
    names = ['test_resolve_terramind_paths_from_windows_style_root', 'test_resolve_terramind_paths_from_posix_colab_root']
    for name in names:
        fn = next(n for n in original_tests.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<existing path assertion>', 'exec'), namespace)
        namespace[name]()
    print(f'PASS: {len(paths)} Python sources compile; 2 existing pure path tests pass (no ML imports).')


if __name__ == '__main__':
    check()
