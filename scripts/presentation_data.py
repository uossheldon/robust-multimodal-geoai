"""Read frozen aggregate summaries for presentation. No scientific computation."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'s2_weighted': 'Sentinel-2 weighted', 's1_weighted': 'Sentinel-1 weighted',
          'early_fusion': 'Early fusion', 'modality_dropout': 'Modality dropout',
          'occlusion_trained': 'Occlusion-trained fusion', 'terramind_frozen': 'Frozen TerraMind'}
METRICS = ['mean_iou', 'macro_dice', 'iou_background', 'iou_low', 'iou_mid', 'iou_high',
           'binary_algae_iou', 'binary_algae_dice']
SOURCES = ['results/final_test/clean_test_results.csv', 'results/final_test/final_summary.json',
           'results/s2_deeplab_weighted/metrics.json', 'results/s1_deeplab_weighted/metrics.json',
           'results/s1_s2_early_fusion/metrics.json', 'results/occlusion_training/aggregate_results.csv',
           'docs/TERRAMIND_VALIDATION_EVIDENCE.json']


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(row, metric, n):
    return {'mean': float(row[metric + '_mean']),
            'std': float(row[metric + '_std']) if n > 1 else None, 'n': n}


def load_summary():
    clean = {r['model']: r for r in read_csv(SOURCES[0])}
    validation = {}
    for model, source in zip(list(MODELS)[:3], SOURCES[2:5]):
        row = read_json(source)
        key = 'validation_metrics_unweighted' if model == 's2_weighted' else 'validation_metrics'
        validation[model] = {'mean': row[key]['mean_iou'], 'std': None, 'n': 1}
    for row in read_csv(SOURCES[5]):
        if row['primary_condition'] == 'clean' and row['method'] in ['modality_dropout', 'occlusion_training']:
            model = 'occlusion_trained' if row['method'] == 'occlusion_training' else row['method']
            validation[model] = score(row, 'mean_iou', 3)
    validation['terramind_frozen'] = read_json(SOURCES[6])['aggregate']['mean_iou']
    models = []
    for model, label in MODELS.items():
        n = int(clean[model]['n'])
        models.append({'id': model, 'label': label, 'n': n,
                       'clean': {key: score(clean[model], key, n) for key in METRICS},
                       'validation': validation[model]})
    robustness = []
    for row in read_json(SOURCES[1])['robustness_summary']:
        if row['model'] in ['modality_dropout', 'occlusion_trained']:
            robustness.append({'model': row['model'], 'condition': row['condition'],
                               'mean_iou': score(row, 'mean_iou', int(row['n']))})
    return {'schema_version': 1, 'sources': SOURCES, 'models': models,
            'robustness': robustness, 'occlusion_rates': [0, 10, 30, 50, 70],
            'training_seeds': [42, 7, 123], 'corruption_seeds': [101, 202, 303],
            'test_dates': ['2025-09-08', '2025-09-21'],
            'sd_populations': {'clean_missing': '3 training seeds',
                               'optical_occlusion': '9 pooled training-seed × corruption-seed runs'}}


def write_asset():
    target = ROOT / 'site/assets/results_summary.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(load_summary(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Wrote presentation copy of frozen aggregates; scientific sources unchanged.')


if __name__ == '__main__': write_asset()
