"""Verify saved baseline artifacts and assemble the three-dataset results page."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch

from run_baselines import ROOT, write_json, write_suite_page
from utils import classification_metrics


def verify_dataset(output, expected_epochs=None):
    config = json.loads((output / 'config.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    status = json.loads((output / 'status.json').read_text())
    splits = json.loads((output / 'splits.json').read_text())
    epochs = int(config['epochs']) if expected_epochs is None else expected_epochs
    assert status['stage'] == 'complete' and status['completed_epochs'] == epochs
    assert result['epochs'] == config['epochs'] == epochs
    history = pd.read_csv(output / 'history.csv')
    assert list(history.epoch) == list(range(1, epochs + 1))
    assert np.isfinite(history.to_numpy()).all()
    best_epoch = int(np.argmax(history.val_auc + history.val_f1)) + 1
    checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
    last = torch.load(output / 'last_model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == result['best_epoch'] == best_epoch
    assert last['epoch'] == epochs
    assert all(torch.isfinite(value).all() for value in checkpoint['model_state_dict'].values())
    names = [name for split in splits.values() for name in split]
    assert len(names) == len(set(names))
    features = Path(config['data_dir'])
    if not features.is_file():
        features = ROOT / 'feature' / config['dataset_key'] / 'patch_feats_pretrain_medical_ssl.h5'
    classes = int(config['n_class'])
    with h5py.File(features, 'r') as h5:
        assert set(names) == set(h5)
        for split, saved_key in [('validation', 'Validation'), ('test', 'Test')]:
            frame = pd.read_csv(output / f'{split}_predictions.csv')
            assert list(frame.slide_id) == splits[saved_key]
            labels = frame.label.to_numpy(dtype=np.int64, copy=True)
            assert np.array_equal(labels, [int(h5[name].attrs['label']) for name in frame.slide_id])
            probabilities = frame[[f'p_class_{i}' for i in range(classes)]].to_numpy(
                dtype=np.float32, copy=True
            )
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            assert np.array_equal(probabilities.argmax(axis=1), frame.prediction.to_numpy())
            metrics = classification_metrics(torch.from_numpy(probabilities), torch.from_numpy(labels), classes)
            recorded = json.loads((output / f'{split}_metrics.json').read_text())
            for name in ['acc', 'auc', 'f1']:
                assert np.isclose(metrics[name], recorded[name], atol=1e-5)
            accuracy = 100.0 * (frame.label == frame.prediction).mean()
            assert np.isclose(accuracy, result[f'{split}_accuracy_pct'], atol=1e-5)
            matrix = pd.read_csv(output / f'{split}_confusion_matrix.csv', index_col=0).to_numpy()
            np.testing.assert_array_equal(
                matrix, confusion_matrix(labels, frame.prediction, labels=list(range(classes)))
            )
            per_class = pd.read_csv(output / f'{split}_per_class_metrics.csv')
            np.testing.assert_array_equal(per_class.support, matrix.sum(axis=1))
    panels = output / 'attention_panels'
    manifest = json.loads((panels / 'export.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['checkpoint_epoch'] == best_epoch
    expected = set(splits['Validation'])
    pngs = list((panels / 'validation').glob('*.png'))
    assert {path.stem for path in pngs} == expected
    assert manifest['completed_pngs'] == result['validation_pngs'] == len(expected)
    attention = pd.read_csv(panels / 'summary.csv')
    assert set(attention.slide_id) == expected
    assert len(attention) == len(expected)
    np.testing.assert_allclose(attention.attention_sum, 1, atol=1e-5)
    validation = pd.read_csv(output / 'validation_predictions.csv').set_index('slide_id')
    attention = attention.set_index('slide_id').loc[validation.index]
    np.testing.assert_array_equal(attention.prediction, validation.prediction)
    np.testing.assert_array_equal(attention.label, validation.label)
    np.testing.assert_allclose(attention.confidence, validation.confidence, atol=1e-5)
    for path in pngs:
        with Image.open(path) as image:
            assert image.format == 'PNG' and image.width > 2 * image.height
            image.load()
    return result, {
        'epochs_verified': epochs, 'best_epoch_verified': best_epoch,
        'validation_pngs_decoded': len(pngs), 'prediction_metrics_recomputed': True,
        'slide_coverage_matches_features': True,
    }


def summarize(suite):
    results, checks, configs, source_hashes = [], {}, [], set()
    for key, classes, validation_count in [('C16', 2, 27), ('C17', 4, 30), ('BRACS', 7, 62)]:
        output = suite / key
        result, verification = verify_dataset(output, expected_epochs=30)
        config = json.loads((output / 'config.json').read_text())
        assert config['run_mode'] == result['run_mode'] == 'baseline'
        assert config['n_class'] == classes and result['validation_pngs'] == validation_count
        source = json.loads((output / 'source_snapshot.json').read_text())
        source_hashes.add(source['notebook_definitions_sha256'])
        configs.append(config)
        results.append(result)
        checks[key] = verification
    assert len(source_hashes) == 1, 'Notebook model/training definitions differ between runs.'
    for field in ['seed', 'lr', 'wd', 'B', 'D_feat', 'D_inner', 'D_attn', 'droprate']:
        assert len({config[field] for config in configs}) == 1, f'Inconsistent baseline {field}.'
    write_json(suite / 'verification.json', {
        'status': 'passed', 'datasets': checks,
        'total_validation_pngs': sum(item['validation_pngs_decoded'] for item in checks.values()),
        'same_notebook_definitions_across_datasets': True,
    })
    state = json.loads((suite / 'suite_status.json').read_text())
    devices = {config['device'] for config in configs}
    state.update(status='complete', current_dataset='none',
                 completed_datasets=['C16', 'C17', 'BRACS'], verified=True,
                 device=next(iter(devices)) if len(devices) == 1 else 'mixed',
                 dataset_devices={config['dataset_key']: config['device'] for config in configs})
    write_suite_page(suite, results, state)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, field, heading, limit in zip(
        axes, ['test_accuracy_pct', 'test_macro_auroc', 'test_macro_f1'],
        ['Test slide-level accuracy (%)', 'Test macro AUROC', 'Test macro F1'], [100, 1, 1]
    ):
        values = [result[field] for result in results]
        bars = ax.bar([result['dataset_key'] for result in results], values,
                      color=['#3274a1', '#e1812c', '#3a923a'])
        ax.bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=3)
        ax.set(title=heading, ylim=(0, limit * 1.12))
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('ABMIL baseline | 30 epochs per dataset | validation-selected checkpoint')
    try:
        fig.savefig(suite / 'baseline_test_metrics.png', dpi=150)
    finally:
        plt.close(fig)
    warnings = sorted({warning for result in results for warning in result['warnings']})
    page = suite / 'index.html'
    page.write_text(
        page.read_text()
        + '<h2>Test performance</h2><img style="width:100%;max-width:1300px" '
        'src="baseline_test_metrics.png" alt="ABMIL test accuracy, AUROC and F1">'
        '<p><a href="baseline_report.md">Baseline report</a> | '
        '<a href="verification.json">Artifact verification</a></p><h2>Protocol notes</h2><ul>'
        + ''.join(f'<li>{escape(warning)}</li>' for warning in warnings) + '</ul>'
    )
    lines = [
        '# ABMIL baseline results', '',
        'All three datasets completed 30 epochs using seed 42 and the same notebook '
        'model/training definitions. Checkpoints were selected using validation macro '
        'AUROC + macro F1, never test performance. Earlier epochs win ties.', '',
        '| Dataset | Best epoch / trained | Validation accuracy | Test accuracy | Test macro AUROC | Test macro F1 | Validation PNGs |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for result in results:
        lines.append(
            f'| {result["dataset"]} | {result["best_epoch"]}/30 | '
            f'{result["validation_accuracy_pct"]:.2f}% | {result["test_accuracy_pct"]:.2f}% | '
            f'{result["test_macro_auroc"]:.4f} | {result["test_macro_f1"]:.4f} | '
            f'{result["validation_pngs"]} |'
        )
    lines.extend([
        '', '## How to read these results', '',
        '- Accuracy is whole-slide classification accuracy, not patch accuracy.',
        '- An early selected checkpoint does not mean training stopped early; every history has 30 epochs.',
        '- Attention weights and percentile ranks are not patch tumor probabilities.',
        '- This is one seed, not a multi-seed estimate of uncertainty.',
        '', '## Execution', '',
    ])
    for result, config in zip(results, configs):
        lines.append(f'- {result["dataset"]}: `{config["device"]}`; original run directory `{result["run_directory"]}`.')
    if (suite / 'gpu_preflight.json').exists():
        lines.extend([
            '- CAMELYON16 ran on Antares CPU; CAMELYON17 and BRACS ran on Raccoon GTX 1080 Ti.',
            '- Raccoon used an isolated Python 3.12 / PyTorch 2.5.1+cu121 environment for Pascal GPU support.',
            '- NumPy 2.4.0 was yanked, so the isolated GPU environment uses the 2.4.6 patch release.',
            '- See `environment_raccoon.freeze.txt`, `gpu_preflight.json`, and `migration.json` for provenance.',
        ])
    lines.extend(['', '## Split caveats', '', *[f'- {warning}' for warning in warnings], ''])
    (suite / 'baseline_report.md').write_text('\n'.join(lines))
    migration_path = suite / 'migration.json'
    if migration_path.exists():
        migration = json.loads(migration_path.read_text())
        migration.update(status='complete', results_copied_back=True, artifacts_verified=True)
        write_json(migration_path, migration)
    print(pd.DataFrame(results)[
        ['dataset', 'epochs', 'best_epoch', 'test_accuracy_pct', 'test_macro_auroc',
         'test_macro_f1', 'validation_pngs']
    ].to_string(index=False), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    summarize(parser.parse_args().suite.resolve())
