"""Run fixed-split ABMIL training seeds sequentially and verify every result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import platform
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from masked_view_training import MaskedViewConfig
from class_balanced_training import ClassBalancedConfig, training_class_counts, weights_from_counts
from mil_extensions import EXTENSION_TYPES, experiment_label
from spatial_context import SpatialContextConfig
from run_baselines import (
    CONFIGS, ROOT, load_fixed_datasets, run_dataset, split_fingerprint,
    training_functions, write_json, write_suite_page,
)
from verify_and_summarize_baselines import verify_dataset

METRICS = ['test_accuracy_pct', 'test_macro_auroc', 'test_macro_f1', 'test_loss']
MATCH_FIELDS = [
    'dataset', 'n_class', 'D_feat', 'D_inner', 'D_attn', 'B', 'lr', 'wd',
    'droprate', 'train_epoch', 'n_shot',
]


def prepare_inputs(reference, keys, notebook, feature_root, metadata):
    """Check all inputs before training; inherit the recorded baseline settings."""
    _, _, cells = training_functions(notebook)
    source_hash = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
    prepared = {}
    for key in keys:
        folder = reference / key
        config = json.loads((folder / 'config.json').read_text())
        snapshot = json.loads((folder / 'source_snapshot.json').read_text())
        if snapshot['notebook_definitions_sha256'] != source_hash:
            raise ValueError(f'{key}: notebook definitions differ from the reference baseline.')
        if (config.get('dataset_key') != key or config['epochs'] != 30
                or config['train_epoch'] != 30 or config['run_mode'] != 'baseline'
                or config['optimizer'] != 'AdamW' or config['loss'] != 'CrossEntropyLoss'
                or config['scheduler'] is not None or config['n_shot'] != 0):
            raise ValueError(f'{key}: expected a full 30-epoch reference ABMIL baseline.')
        config['data_dir'] = str(feature_root / key / 'patch_feats_pretrain_medical_ssl.h5')
        config['dataset_csv_dir'] = str(metadata)
        conf = SimpleNamespace(**config)
        datasets = load_fixed_datasets(conf, folder / 'splits.json')
        if datasets[0].feature_dim != int(conf.D_feat):
            raise ValueError(f'{key}: reference feature dimension does not match inputs.')
        prepared[key] = (conf, datasets)
    return prepared, source_hash


def aggregate_results(rows, seeds, keys):
    """Require a complete, unique seed × dataset grid before estimating variability."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not keys or len(set(keys)) != len(keys):
        raise ValueError('Aggregation requires distinct datasets and at least two distinct seeds.')
    expected = {(seed, key) for seed in seeds for key in keys}
    pairs = [(row['training_seed'], row['dataset_key']) for row in rows]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError('Cannot summarize missing, duplicate, or unexpected seed/dataset runs.')
    records = []
    for key in keys:
        selected = [row for row in rows if row['dataset_key'] == key]
        summary = {'dataset_key': key, 'n_seeds': len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite {metric} for {key}.')
            summary[f'{metric}_mean'] = float(values.mean())
            summary[f'{metric}_std'] = float(values.std(ddof=1))
        records.append(summary)
    return records


def verify_run(output, reference_conf, seed, source_hash, extension=None, *, write_verification=True):
    result, checks = verify_dataset(output, expected_epochs=30)
    config = json.loads((output / 'config.json').read_text())
    splits = json.loads((output / 'splits.json').read_text())
    source = json.loads((output / 'source_snapshot.json').read_text())
    for field in MATCH_FIELDS:
        if config[field] != getattr(reference_conf, field):
            raise ValueError(f'Baseline setting changed: {field}.')
    if (config['seed'] != seed or config['training_seed'] != seed
            or config['split_seed'] != reference_conf.split_seed
            or config['split_sha256'] != reference_conf.split_sha256
            or split_fingerprint(splits) != reference_conf.split_sha256
            or source['notebook_definitions_sha256'] != source_hash):
        raise ValueError('Training seed, fixed splits, or model definitions changed.')
    if extension is not None:
        expected = extension.to_dict()
        if (config.get('method') != extension.name or result.get('method') != extension.name
                or config.get('training_extension') != expected
                or result.get('training_extension') != expected
                or config['loss'] != extension.loss_name
                or config['run_mode'] != 'extension' or result['run_mode'] != 'extension'):
            raise ValueError('Extension method, objective, or configuration changed.')
    if isinstance(extension, MaskedViewConfig):
        if config.get('mask_rng_seed') != seed:
            raise ValueError('Masked-view RNG seed changed.')
        saved_source = source['masked_view_training_source']
        saved_hash = hashlib.sha256(saved_source.encode()).hexdigest()
        current_hash = hashlib.sha256((ROOT / 'masked_view_training.py').read_bytes()).hexdigest()
        if saved_hash != source['masked_view_training_sha256'] or saved_hash != current_hash:
            raise ValueError('Masked-view training source differs from the verified implementation.')
        history = pd.read_csv(output / 'history.csv')
        weights = np.asarray([extension.weight_at_epoch(int(epoch)) for epoch in history.epoch])
        if extension.ramp_epochs:
            np.testing.assert_allclose(history.train_consistency_weight, weights, rtol=0, atol=1e-12)
            checks['consistency_schedule_verified'] = True
        np.testing.assert_allclose(
            history.train_loss,
            history.train_ce_loss + weights * history.train_consistency_loss,
            rtol=1e-5, atol=1e-7,
        )
        if (not history.train_retained_fraction.between(0, 1, inclusive='right').all()
                or (history.train_consistency_loss < -1e-6).any()
                or (history.train_ce_loss < 0).any()):
            raise ValueError('Invalid masked-view training diagnostics.')
        if ((extension.method == 'patch_dropout' or extension.consistency_weight == 0)
                and not np.allclose(history.train_consistency_loss, 0, atol=0)):
            raise ValueError('Unexpected consistency loss in a disabled consistency branch.')
        checks.update(method=extension.name, masked_view_source_sha256=saved_hash,
                      training_objective_verified=True, full_bag_evaluation=True)
    elif isinstance(extension, SpatialContextConfig):
        saved_source = source['spatial_context_source']
        saved_hash = hashlib.sha256(saved_source.encode()).hexdigest()
        current_hash = hashlib.sha256((ROOT / 'spatial_context.py').read_bytes()).hexdigest()
        if saved_hash != source['spatial_context_sha256'] or saved_hash != current_hash:
            raise ValueError('Spatial context source differs from the verified implementation.')
        checkpoint = torch.load(output / 'best_model.pt', map_location='cpu', weights_only=True)
        parameters = checkpoint['model_state_dict']
        if not {'context.0.weight', 'context.0.bias', 'context.2.weight',
                'context.2.bias'}.issubset(parameters):
            raise ValueError('Missing context-module parameters.')
        gate_keys = {'context_gate.weight', 'context_gate.bias'}
        if set(parameters).intersection(gate_keys) != (gate_keys if extension.gated else set()):
            raise ValueError('Context-gate parameters do not match the spatial configuration.')
        if sum(value.numel() for value in parameters.values()) != config['model_parameter_count']:
            raise ValueError('Spatial parameter count changed.')
        width = int(reference_conf.D_inner)
        hidden = width if extension.context_hidden is None else extension.context_hidden
        if (parameters['context.0.weight'].shape != (hidden, 2 * width)
                or parameters['context.2.weight'].shape != (width, hidden)):
            raise ValueError('Context hidden width does not match the saved configuration.')
        if extension.gated and not extension.gate_trainable:
            for name in ['best_model.pt', 'last_model.pt']:
                fixed = torch.load(output / name, map_location='cpu', weights_only=True)['model_state_dict']
                if (torch.count_nonzero(fixed['context_gate.weight']).item()
                        or not torch.equal(fixed['context_gate.bias'], torch.full_like(
                            fixed['context_gate.bias'],
                            np.log(extension.gate_initial / (1 - extension.gate_initial))))):
                    raise ValueError('Frozen gate changed during training.')
            checks['fixed_gate_unchanged'] = True
        manifest = json.loads((output / 'attention_panels/export.json').read_text())
        if (manifest.get('method') != extension.name
                or manifest.get('training_extension') != extension.to_dict()):
            raise ValueError('Spatial export used the wrong coordinate mode or settings.')
        profile = json.loads((output / 'resource_profile.json').read_text())
        if (profile != result.get('resource_profile') or profile['elapsed_seconds'] <= 0
                or profile['model_parameter_count'] != config['model_parameter_count']):
            raise ValueError('Invalid spatial resource profile.')
        checks.update(method=extension.name, spatial_context_source_sha256=saved_hash,
                      spatial_export_configuration_verified=True, resource_profile_verified=True,
                      full_bag_evaluation=True)
    elif isinstance(extension, ClassBalancedConfig):
        saved = source['class_balanced_training_source']
        digest = hashlib.sha256(saved.encode()).hexdigest()
        if (digest != source['class_balanced_training_sha256']
                or digest != hashlib.sha256((ROOT / 'class_balanced_training.py').read_bytes()).hexdigest()):
            raise ValueError('Class-balanced training source changed.')
        counts = training_class_counts(reference_conf.data_dir, splits['Train'])
        weights = weights_from_counts(counts, int(config['n_class'])).tolist()
        if (config.get('class_weights') != weights
                or config.get('class_weight_counts') != {str(k): v for k, v in counts.items()}
                or config.get('evaluation_loss') != 'CrossEntropyLoss'):
            raise ValueError('Class weights or evaluation objective changed.')
        history = pd.read_csv(output / 'history.csv')
        np.testing.assert_allclose(history.train_mean_class_weight, 1., rtol=0, atol=1e-6)
        if (history.train_loss < 0).any() or (history.train_ce_loss < 0).any():
            raise ValueError('Invalid class-weighted training diagnostics.')
        checks.update(class_weights_training_only_verified=True, b1_safe_weighting_verified=True,
                      unweighted_full_bag_evaluation=True, class_balanced_training_sha256=digest)
    checks.update(training_seed=seed, fixed_split_sha256=reference_conf.split_sha256)
    if write_verification:
        write_json(output / 'verification.json', {'status': 'passed', **checks})
    return {**result, 'training_seed': seed}, checks


def write_progress(output, state, rows):
    write_json(output / 'study_status.json', state)
    if rows:
        pd.DataFrame([{k: v for k, v in row.items() if k != 'warnings'} for row in rows]
                     ).to_csv(output / 'per_seed_results.csv', index=False)
    links = ''.join(
        f'<li><a href="seed_{row["training_seed"]}/{row["dataset_key"]}/index.html">'
        f'Seed {row["training_seed"]}: {escape(row["dataset"])}</a></li>' for row in rows
    )
    label = experiment_label(state.get('method', 'abmil_baseline'))
    (output / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>{escape(label)} seeds</title>'
        '<style>body{font:17px sans-serif;max-width:1400px;margin:32px}'
        'td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        f'<h1>Fixed-split {escape(label)} training-seed study</h1>'
        f'<p>Status: {escape(state["status"])}; completed '
        f'{len(rows)}/{state["total_runs"]}. Current: {escape(state["current_run"])}.</p>'
        '<p>Same saved slides and ordering; 30 epochs per run. Test results do not '
        'select checkpoints or seeds. This is not a split-uncertainty study.</p>'
        '<p><a href="protocol.json">Protocol</a> | '
        '<a href="study_status.json">Live status</a></p>'
        f'<ul>{links}</ul>'
        + (pd.DataFrame([{k: row[k] for k in ['training_seed', 'dataset_key', 'best_epoch', *METRICS]}
                         for row in rows]).to_html(index=False) if rows else '')
        + ('<p><a href="summary.md">Mean and sample standard deviation</a> | '
           '<a href="summary.csv">Summary CSV</a></p>' if state['status'] == 'complete' else '')
    )


def run_study(reference, output, *, seeds, keys, device='cuda', threads=2, dpi=140,
              notebook=ROOT / 'ECE1512_ProjectA_Working.ipynb',
              feature_root=ROOT / 'feature', metadata=ROOT / 'dataset_csv', extension=None):
    if extension is not None and not isinstance(extension, EXTENSION_TYPES):
        raise ValueError('extension must be a supported MIL extension configuration or None.')
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Use at least two distinct training seeds in [0, 2**32).')
    if not keys or len(set(keys)) != len(keys) or any(key not in CONFIGS for key in keys):
        raise ValueError('Use distinct supported datasets.')
    if threads < 1 or dpi < 1 or device not in {'cpu', 'cuda'}:
        raise ValueError('Invalid execution settings.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable.')
    if output.exists():
        raise FileExistsError(f'Output must be new: {output}')
    torch.set_num_threads(threads)
    prepared, source_hash = prepare_inputs(reference, keys, notebook, feature_root, metadata)
    output.mkdir(parents=True, exist_ok=False)
    state = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': seeds, 'datasets': keys, 'total_runs': len(seeds) * len(keys),
        'completed_runs': [], 'current_run': 'preflight complete', 'device': device,
        'method': extension.name if extension is not None else 'abmil_baseline',
    }
    protocol = {
        **state, 'reference_suite': str(reference), 'epochs_per_run': 30,
        'selection_rule': 'Max validation macro AUROC + macro F1; first epoch wins ties.',
        'notebook_definitions_sha256': source_hash,
        'split_sha256': {key: conf.split_sha256 for key, (conf, _) in prepared.items()},
        'baseline_settings': {
            key: {field: getattr(conf, field) for field in MATCH_FIELDS}
            for key, (conf, _) in prepared.items()
        },
        'split_seed': {key: conf.split_seed for key, (conf, _) in prepared.items()},
        'hostname': socket.gethostname(), 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'threads': threads,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'limitations': [
            'Training randomness only; fixed splits do not measure split uncertainty.',
            'A few seeds give a preliminary variability estimate, not a confidence interval.',
            'CAMELYON17 retains overlapping training/validation patients.',
            'All seeds are reported; none is selected using test performance.',
        ],
    }
    if extension is not None:
        protocol['training_extension'] = extension.to_dict()
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'code_sha256.json', {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / name for name in [
            'run_multiseed.py', 'run_baselines.py', 'dataloader.py', 'utils.py',
            'export_attention.py', 'wsi_visualization.py', 'verify_and_summarize_baselines.py',
            'mil_extensions.py', 'spatial_context.py',
        ]] + [notebook] + ([ROOT / 'masked_view_training.py', ROOT / 'run_consistency.py']
                          if isinstance(extension, MaskedViewConfig) else [])
        + ([ROOT / 'run_spatial.py'] if isinstance(extension, SpatialContextConfig) else [])
    })
    rows, checks = [], {}
    seed_state = None
    write_progress(output, state, rows)
    try:
        for seed in seeds:
            seed_output = output / f'seed_{seed}'
            seed_output.mkdir()
            seed_rows = []
            seed_state = {'status': 'running', 'training_seed': seed, 'current_dataset': 'none',
                          'method': state['method']}
            for key in keys:
                state['current_run'] = f'seed_{seed}/{key}'
                write_progress(output, state, rows)
                seed_state['current_dataset'] = key
                write_suite_page(seed_output, seed_rows, seed_state)
                conf, datasets = prepared[key]
                print(f'START {state["current_run"]}', flush=True)
                run_dataset(key, conf, datasets, seed_output / key, notebook,
                            epochs=30, device=device, dpi=dpi, training_seed=seed,
                            extension=extension)
                row, verification = verify_run(seed_output / key, conf, seed, source_hash,
                                               extension=extension)
                rows.append(row)
                seed_rows.append(row)
                checks[state['current_run']] = verification
                state['completed_runs'].append(state['current_run'])
                write_json(output / 'verification.json', {'status': 'partial', 'runs': checks})
                write_progress(output, state, rows)
                write_suite_page(seed_output, seed_rows, seed_state)
                print(f'VERIFIED {state["current_run"]}', flush=True)
            seed_state.update(status='complete', current_dataset='none')
            write_suite_page(seed_output, seed_rows, seed_state)
        summaries = aggregate_results(rows, seeds, keys)
        pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
        lines = [
            f'# Fixed-split {experiment_label(state["method"])} training-seed results', '',
            'Training seeds: ' + ', '.join(map(str, seeds)) + '. All runs trained for 30 epochs.',
            'The original split membership and ordering were fixed. Checkpoint selection used '
            'validation macro AUROC + macro F1; all seeds are reported.', '',
            '| Dataset | Seeds | Test accuracy (%) | Test macro AUROC | Test macro F1 |',
            '|---|---:|---:|---:|---:|',
        ]
        for row in summaries:
            cells = [f'{row[m + "_mean"]:.4f} ± {row[m + "_std"]:.4f}' for m in METRICS[:3]]
            lines.append(f'| {row["dataset_key"]} | {row["n_seeds"]} | ' + ' | '.join(cells) + ' |')
        lines.extend(['', 'Values are mean ± sample standard deviation (ddof=1), not confidence '
                      'intervals. Accuracy variation is in percentage points. This study does '
                      'not measure split variability or remove the CAMELYON17 patient-overlap '
                      'limitation. See per_seed_results.csv for every seed and selected epoch.', ''])
        if isinstance(extension, MaskedViewConfig):
            lines.extend([
                '## Training extension', '',
                f'- Method: `{extension.name}`.',
                f'- Objective: `{extension.loss_name}`.',
                f'- Patch retention: `{extension.keep_fraction}`; '
                'max(1, floor(N × retention)) real patches per training view.',
                f'- Consistency weight: `{extension.consistency_weight}`.',
                '- Validation, test, and attention export use full bags and ordinary cross-entropy.',
                '- These settings are experimental, not established optima. Compare with '
                'the matched baseline/comparator; do not infer improvement from this table alone.',
                '',
            ])
            if extension.ramp_epochs:
                lines.extend([
                    f'- Linear consistency-weight ramp over `{extension.ramp_epochs}` epochs: '
                    '`weight * min(one_based_epoch / ramp_epochs, 1)`.',
                    '- The history records the actual weight used at every epoch.', '',
                ])
        elif isinstance(extension, SpatialContextConfig):
            lines.extend([
                '## Spatial context extension', '',
                f'- Method: `{extension.name}`; coordinate mode `{extension.coordinate_mode}`.',
                f'- Up to `{extension.neighbors}` nearest other patches per patch; '
                'singleton neighbor mean is zero.',
                '- Residual context MLP before gated attention; final context layer initialized to zero.',
                f'- Fixed coordinate-shuffle seed: `{extension.shuffle_seed}` '
                '(used only in the shuffled control).',
                '- Full-bag cross-entropy for training and evaluation; no consistency loss.',
                '- Resource profiles record parameter count, end-to-end time, graph-cache statistics, '
                'and peak CUDA allocation when applicable.',
                '- Compare with baseline and the matched shuffled-coordinate control. '
                'Do not infer improvement or spatial benefit from this table alone.', '',
            ])
            if extension.gated:
                lines.extend([
                    '- A per-patch sigmoid linear gate scales the context residual; '
                    f'initial gate probability `{extension.gate_initial}`.',
                    '- The neighborhood definition is unchanged. Compare with the ungated '
                    'parent as well as ABMIL. No new gated shuffled control was trained.', '',
                ])
        (output / 'summary.md').write_text('\n'.join(lines))
        write_json(output / 'verification.json', {
            'status': 'passed', 'runs': checks, 'fixed_splits_verified': True,
            'total_validation_pngs': sum(item['validation_pngs_decoded'] for item in checks.values()),
        })
        state.update(status='complete', current_run='none',
                     finished_utc=datetime.now(timezone.utc).isoformat(), verified=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        if seed_state is not None and seed_state['status'] != 'complete':
            seed_state.update(status='failed', error=str(error))
            write_suite_page(seed_output, seed_rows, seed_state)
        raise
    finally:
        write_progress(output, state, rows)
    print(f'COMPLETE {output}', flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--datasets', choices=list(CONFIGS), nargs='+', default=list(CONFIGS))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dpi', type=int, default=140)
    args = parser.parse_args(argv)
    run_study(args.reference_suite.resolve(), args.output.resolve(), seeds=args.seeds,
              keys=args.datasets, device=args.device, threads=args.threads, dpi=args.dpi)


if __name__ == '__main__':
    main()
