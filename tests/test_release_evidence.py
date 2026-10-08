"""Fixed split, tile geometry and uncertainty/replication evidence invariants."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_split_integrity():
    with (ROOT/'results/tile_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert Counter(r['split'] for r in rows) == {'train':621, 'validation':111, 'test':114}
    dates = defaultdict(set)
    for row in rows:
        dates[row['split']].add(row['date'])
        assert int(row['window_height']) == int(row['window_width']) == 224
        assert float(row['valid_pixel_ratio']) >= .20
    assert dates['test'] == {'2025-09-08','2025-09-21'}
    assert not dates['train'] & dates['validation']
    assert not dates['test'] & (dates['train'] | dates['validation'])


def test_terramind_complete_three_seed_evidence():
    data = json.loads((ROOT/'results/terramind_validation.json').read_text(encoding='utf-8'))
    assert {r['seed'] for r in data['per_seed']} == {42,7,123}
    for metric, aggregate in data['aggregate'].items():
        assert aggregate['n'] == 3
        assert all(metric in row for row in data['per_seed'])


def test_ensemble_error_detection_has_all_conditions_and_scores():
    with (ROOT/'results/ensemble_uncertainty/error_detection.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    assert {r['uncertainty'] for r in rows} == {'single_entropy','ensemble_entropy','mutual_information','disagreement'}
    assert all(0 <= float(r['auroc']) <= 1 for r in rows)
