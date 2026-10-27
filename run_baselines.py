"""Run the notebook ABMIL baseline, evaluate slides, and export validation panels."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
from html import escape
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader import HDF5FeatureDataset, build_HDF5_feat_dataset, mil_collate_fn
from execution_settings import make_loaders
from class_balanced_training import (
    ClassBalancedConfig, PerSlideWeightedCE, train_balanced_epoch, weights_from_counts,
)
from export_attention import class_names, export, load_notebook_model, parser as export_parser
from masked_view_training import MaskedViewConfig, train_masked_epoch
from mil_extensions import EXTENSION_TYPES, experiment_label
from spatial_context import SpatialContextConfig, evaluate_spatial, train_spatial_epoch
from utils import classification_metrics, infer_feature_dim, load_config, seed_everything

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    'C16': 'camelyon16_medical_ssl_config.yml',
    'C17': 'camelyon17_medical_ssl_config.yml',
    'BRACS': 'bracs_medical_ssl_config.yml',
}
SPLITS = ['Train', 'Validation', 'Test']


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def split_fingerprint(splits):
    """Hash split membership AND ordering, independently of JSON formatting."""
    return hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def load_fixed_datasets(conf, split_path):
    """Reuse saved slide lists without invoking any random split construction."""
    split_path = Path(split_path).resolve()
    saved = json.loads(split_path.read_text())
    if not isinstance(saved, dict) or set(saved) != set(SPLITS):
        raise ValueError('Saved splits must contain exactly Train, Validation, and Test.')
    for names in saved.values():
        if not isinstance(names, list) or not names or any(
            not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name)
            or name in {'.', '..'} for name in names
        ):
            raise ValueError('Each saved split must contain nonempty, safe slide IDs.')
    reference = json.loads(split_path.with_name('config.json').read_text())
    if reference['dataset'] != conf.dataset or int(reference['n_class']) != int(conf.n_class):
        raise ValueError('Saved splits belong to a different dataset or class task.')
    datasets = tuple(
        HDF5FeatureDataset(conf.data_dir, saved[name], conf) for name in SPLITS
    )
    audit_splits(datasets, conf)
    conf.split_source = str(split_path)
    conf.split_seed = int(reference.get('split_seed', reference['seed']))
    conf.split_sha256 = split_fingerprint(saved)
    return datasets


def training_functions(notebook_path):
    """Reuse the existing training/evaluation functions, without executing cells."""
    notebook = json.loads(notebook_path.read_text())
    sources = {}
    namespace = {
        'torch': torch, 'nn': nn, 'F': F, 'tqdm': tqdm,
        'classification_metrics': classification_metrics,
    }
    for cell_id, name in [('de7504f3', 'train_one_epoch'), ('4b2b81fd', 'evaluate')]:
        cell = next(cell for cell in notebook['cells'] if cell.get('id') == cell_id)
        source = ''.join(cell['source'])
        sources[cell_id] = source
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise ValueError(f'Expected one notebook definition for {name}.')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook_path), 'exec'), namespace)
    model_cell = next(cell for cell in notebook['cells'] if cell.get('id') == 'a8146687')
    sources['a8146687'] = ''.join(model_cell['source'])
    return namespace['train_one_epoch'], namespace['evaluate'], sources


def audit_splits(datasets, conf):
    """Check full slide coverage and record the actual split/class distribution."""
    names = [set(dataset.data_names) for dataset in datasets]
    if not all(names) or any(names[i] & names[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('Expected nonempty, disjoint train/validation/test slide splits.')
    audit = {'warnings': [], 'splits': {}}
    with h5py.File(conf.data_dir, 'r') as h5:
        if set.union(*names) != set(h5):
            raise ValueError('A full baseline must cover every available slide exactly once.')
        for key, dataset in zip(SPLITS, datasets):
            labels, sizes = [], []
            for slide_id in dataset.data_names:
                group = h5[slide_id]
                shape = group['feat'].shape
                if len(shape) != 2 or shape[0] < 1 or shape[1] != int(conf.D_feat):
                    raise ValueError(f'Invalid features for {slide_id}: {shape}.')
                if 'coords' not in group or group['coords'].shape != (shape[0], 2):
                    raise ValueError(f'Patch coordinates are required for {slide_id}.')
                label = int(group.attrs['label'])
                if label not in range(int(conf.n_class)):
                    raise ValueError(f'Out-of-range slide label for {slide_id}.')
                labels.append(label)
                sizes.append(shape[0])
            if set(labels) != set(range(int(conf.n_class))):
                raise ValueError(f'{key} must contain every class for macro-AUROC evaluation.')
            audit['splits'][key] = {
                'slides': len(labels), 'patches': sum(sizes), 'max_bag_patches': max(sizes),
                'class_counts': dict(sorted(Counter(labels).items())),
            }
    if str(conf.dataset).upper() == 'CAMELYON17':
        metadata = pd.read_csv(Path(conf.dataset_csv_dir) / 'camelyon17.csv').set_index('slide_id')
        cases = [set(metadata.loc[dataset.data_names, 'case_id']) for dataset in datasets]
        overlap = len(cases[0] & cases[1])
        audit['train_validation_case_overlap'] = overlap
        audit['train_test_case_overlap'] = len(cases[0] & cases[2])
        audit['validation_test_case_overlap'] = len(cases[1] & cases[2])
        if cases[2] & (cases[0] | cases[1]):
            raise ValueError('CAMELYON17 test patients overlap development patients.')
        audit['warnings'].append(
            'CAMELYON17 uses the supplied center-held-out test split and slide-level '
            '4-class labels, not official challenge patient-level staging.'
        )
        if overlap:
            audit['warnings'].append(
                f'The existing slide-level validation split shares {overlap} patients '
                'with training. Validation is not patient-independent; the baseline '
                'retains the existing split for reproducibility.'
            )
    return audit


def plot_learning_curves(history, best_epoch, output):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    if 'train_ce_loss' in frame:
        axes[0].plot(frame.epoch, frame.train_ce_loss, label='Train CE')
        axes[0].plot(frame.epoch, frame.train_loss, label='Train total objective', linestyle='--')
    else:
        axes[0].plot(frame.epoch, frame.train_loss, label='Train')
    axes[0].plot(frame.epoch, frame.val_loss, label='Validation')
    axes[0].set(title='Training objective / validation CE' if 'train_ce_loss' in frame
                else 'Cross-entropy loss', ylabel='Loss')
    axes[1].plot(frame.epoch, frame.train_acc, label='Train')
    axes[1].plot(frame.epoch, frame.val_acc, label='Validation')
    axes[1].set(title='Slide-level accuracy', ylabel='Accuracy (%)', ylim=(0, 100))
    axes[2].plot(frame.epoch, frame.val_auc, label='Macro AUROC')
    axes[2].plot(frame.epoch, frame.val_f1, label='Macro F1')
    axes[2].set(title='Validation checkpoint metrics', ylabel='Score', ylim=(0, 1.02))
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.axvline(best_epoch, color='gray', linestyle=':', label='Selected epoch')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    try:
        fig.savefig(output / 'learning_curves.png', dpi=140)
    finally:
        plt.close(fig)
    if 'train_consistency_loss' in frame:
        count = 3 if 'train_consistency_weight' in frame else 2
        fig, axes = plt.subplots(1, count, figsize=(5 * count, 4), layout='constrained')
        axes[0].plot(frame.epoch, frame.train_consistency_loss)
        axes[0].set(title='Unweighted training consistency KL', ylabel='KL per slide')
        axes[1].plot(frame.epoch, frame.train_retained_fraction)
        axes[1].set(title='Real patches retained in training views', ylabel='Fraction', ylim=(0, 1.02))
        if count == 3:
            axes[2].plot(frame.epoch, frame.train_consistency_weight)
            axes[2].set(title='Effective consistency weight', ylabel='Weight')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.grid(alpha=0.2)
        try:
            fig.savefig(output / 'masked_view_curves.png', dpi=140)
        finally:
            plt.close(fig)


def save_evaluation(split, metrics, probabilities, labels, slide_ids, conf, output):
    probabilities = probabilities.numpy()
    labels = labels.numpy()
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        'slide_id': slide_ids, 'label': labels, 'prediction': predictions,
        'correct': predictions == labels, 'confidence': probabilities.max(axis=1),
    })
    for index in range(int(conf.n_class)):
        frame[f'p_class_{index}'] = probabilities[:, index]
    frame.to_csv(output / f'{split}_predictions.csv', index=False)
    write_json(output / f'{split}_metrics.json', {
        **metrics, 'n_slides': len(labels), 'accuracy_unit': 'percent',
        'auc_average': 'macro_one_vs_rest', 'f1_average': 'macro',
    })
    names = class_names(conf)
    report = classification_report(
        labels, predictions, labels=list(range(int(conf.n_class))),
        target_names=names, output_dict=True, zero_division=0,
    )
    pd.DataFrame([
        {'class_id': index, 'class_name': name, **report[name]}
        for index, name in enumerate(names)
    ]).to_csv(output / f'{split}_per_class_metrics.csv', index=False)
    matrix = confusion_matrix(labels, predictions, labels=list(range(int(conf.n_class))))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        output / f'{split}_confusion_matrix.csv', index_label='true_class'
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    normalized = 100.0 * matrix / matrix.sum(axis=1, keepdims=True)
    for ax, values, heading in zip(
        axes, [matrix, normalized], ['Slide counts', 'Percentage within each true class']
    ):
        artist = ax.imshow(values, cmap='Blues', vmin=0)
        fig.colorbar(artist, ax=ax, shrink=0.85)
        ax.set(
            title=heading, xlabel='Predicted class', ylabel='True class',
            xticks=range(len(names)), yticks=range(len(names)),
            xticklabels=names, yticklabels=names,
        )
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
        for row in range(len(names)):
            for column in range(len(names)):
                value = values[row, column]
                text = f'{value:.0f}%' if ax is axes[1] else str(int(value))
                ax.text(column, row, text, ha='center', va='center', fontsize=8,
                        color='white' if value > values.max() / 2 else 'black')
    fig.suptitle(f'{conf.dataset} | {split} | accuracy {metrics["acc"]:.2f}%')
    try:
        fig.savefig(output / f'{split}_confusion_matrix.png', dpi=140)
    finally:
        plt.close(fig)
    return frame


def write_run_page(output, conf, validation_predictions, audit):
    cards = []
    names = class_names(conf)
    for row in validation_predictions.to_dict('records'):
        slide_id = row['slide_id']
        image = f'attention_panels/validation/{slide_id}.png'
        caption = (
            f'{slide_id}: true {names[row["label"]]}, predicted {names[row["prediction"]]} '
            f'({row["confidence"]:.1%})'
        )
        cards.append(
            f'<section><h3>{escape(caption)}</h3><a href="{escape(image)}">'
            f'<img loading="lazy" src="{escape(image)}" alt="{escape(caption)}"></a></section>'
        )
    warnings = ''.join(f'<li>{escape(warning)}</li>' for warning in audit['warnings'])
    label = experiment_label(getattr(conf, 'method', 'abmil_baseline'))
    extra_plot = ('<img src="masked_view_curves.png" alt="Consistency loss and patch retention">'
                  if getattr(conf, 'method', None) in {
                      'abmil_consistency', 'abmil_consistency_ramp', 'abmil_patch_dropout'} else '')
    (output / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>{escape(label)} results</title>'
        '<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;padding:0 16px}'
        'img{width:100%;height:auto}section{border-top:1px solid #ccc;margin-top:24px}</style>'
        f'<h1>{escape(conf.dataset)} {escape(label)}</h1>'
        '<p><a href="../index.html">All datasets</a> | '
        '<a href="history.csv">Training history CSV</a> | '
        '<a href="test_predictions.csv">Test predictions CSV</a> | '
        '<a href="test_metrics.json">Test metrics</a></p>'
        '<img src="learning_curves.png" alt="Training curves">'
        + extra_plot +
        '<img src="test_confusion_matrix.png" alt="Test confusion matrix">'
        f'<ul>{warnings}</ul><h2>Every validation WSI</h2>'
        '<p>Raw weights (linear) | cumulative attention | rank percentiles. '
        'Attention is not a patch tumor probability. Click an image for the PNG.</p>'
        + ''.join(cards)
    )


def run_dataset(key, conf, datasets, output, notebook_path, *, epochs, device='cpu',
                dpi=140, training_seed=None, extension=None, train_workers=0):
    if epochs < 1 or int(conf.n_shot) != 0:
        raise ValueError('Use positive epochs and the full training split (n_shot=0).')
    if extension is not None and not isinstance(extension, EXTENSION_TYPES):
        raise ValueError('extension must be a supported MIL extension configuration or None.')
    run_started = time.monotonic()
    conf = SimpleNamespace(**vars(conf))
    conf.split_seed = int(getattr(conf, 'split_seed', conf.seed))
    seed = int(conf.seed if training_seed is None else training_seed)
    if not 0 <= seed < 2**32:
        raise ValueError('Training seed must be in [0, 2**32).')
    conf.seed = conf.training_seed = seed
    conf.D_feat = infer_feature_dim(datasets[0])
    audit = audit_splits(datasets, conf)
    train_one_epoch, evaluate, sources = training_functions(notebook_path)
    if extension is not None:
        conf.method = extension.name
        conf.training_extension = extension.to_dict()
    if isinstance(extension, MaskedViewConfig):
        conf.mask_rng_seed = seed
        train_one_epoch = partial(
            train_masked_epoch, extension=extension,
            generator=torch.Generator(device='cpu').manual_seed(seed),
        )
    elif isinstance(extension, SpatialContextConfig):
        train_one_epoch, evaluate = train_spatial_epoch, evaluate_spatial
    elif isinstance(extension, ClassBalancedConfig):
        train_one_epoch = train_balanced_epoch
    seed_everything(int(conf.seed))
    model = load_notebook_model(conf, notebook_path).to(device)
    loaders = make_loaders(datasets, conf, device, train_workers)
    criterion = nn.CrossEntropyLoss()
    training_criterion = criterion
    if isinstance(extension, ClassBalancedConfig):
        weights = weights_from_counts(audit['splits']['Train']['class_counts'], int(conf.n_class))
        conf.class_weights = weights.tolist()
        conf.class_weight_counts = audit['splits']['Train']['class_counts']
        conf.evaluation_loss = 'CrossEntropyLoss'
        training_criterion = PerSlideWeightedCE(weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    run_mode = ('extension' if extension is not None else 'baseline'
                ) if epochs == int(conf.train_epoch) else 'smoke'
    config = {
        **vars(conf), 'dataset_key': key, 'run_mode': run_mode, 'epochs': epochs,
        'device': device, 'n_worker': train_workers, 'pin_memory': device == 'cuda',
        'class_names': class_names(conf),
        'split_sizes': {name: len(dataset) for name, dataset in zip(SPLITS, datasets)},
        'selection_rule': 'Max validation macro AUROC + macro F1; first epoch wins ties.',
        'optimizer': 'AdamW',
        'loss': extension.loss_name if extension is not None else 'CrossEntropyLoss',
        'scheduler': None,
        'unused_starter_fields': ['warmup_epoch', 'n_res'],
        'split_sha256': split_fingerprint({
            name: dataset.data_names for name, dataset in zip(SPLITS, datasets)
        }),
    }
    if isinstance(extension, SpatialContextConfig):
        config['model_parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=False)
    if isinstance(extension, SpatialContextConfig):
        model.set_neighbor_cache(output / 'spatial_neighbors', writable=True)
    write_json(output / 'config.json', config)
    write_json(output / 'splits.json', {
        name: dataset.data_names for name, dataset in zip(SPLITS, datasets)
    })
    write_json(output / 'split_audit.json', audit)
    snapshot = {
        'notebook_cells': sources,
        'notebook_definitions_sha256': hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        'runner_source': Path(__file__).read_text(),
        'exporter_source': (ROOT / 'export_attention.py').read_text(),
    }
    if isinstance(extension, MaskedViewConfig):
        source = (ROOT / 'masked_view_training.py').read_text()
        snapshot.update(masked_view_training_source=source,
                        masked_view_training_sha256=hashlib.sha256(source.encode()).hexdigest())
    elif isinstance(extension, SpatialContextConfig):
        source = (ROOT / 'spatial_context.py').read_text()
        snapshot.update(spatial_context_source=source,
                        spatial_context_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        extension_dispatch_source=(ROOT / 'mil_extensions.py').read_text())
    elif isinstance(extension, ClassBalancedConfig):
        source = (ROOT / 'class_balanced_training.py').read_text()
        snapshot.update(class_balanced_training_source=source,
                        class_balanced_training_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        extension_dispatch_source=(ROOT / 'mil_extensions.py').read_text())
    write_json(output / 'source_snapshot.json', snapshot)
    status = {'stage': 'training', 'completed_epochs': 0, 'total_epochs': epochs}
    write_json(output / 'status.json', status)
    history, best_score, best_epoch = [], float('-inf'), 0
    try:
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            epoch_args = {'epoch': epoch} if isinstance(extension, MaskedViewConfig) else {}
            training = train_one_epoch(model, training_criterion, loaders[0], optimizer, device,
                                       progress=False, **epoch_args)
            validation = evaluate(model, criterion, loaders[1], device, conf)
            if not all(np.isfinite(value) for value in [*training.values(), *validation.values()]):
                raise FloatingPointError('Non-finite training or validation metrics.')
            score = validation['auc'] + validation['f1']
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save({
                    'model_state_dict': model.state_dict(), 'config': config,
                    'epoch': epoch, 'validation_score': score,
                }, output / 'best_model.pt')
            history.append({
                'epoch': epoch, 'train_loss': training['loss'], 'train_acc': training['acc'],
                **{f'train_{name}': value for name, value in training.items()
                   if name not in {'loss', 'acc'}},
                **{f'val_{name}': value for name, value in validation.items()},
                'seconds': time.monotonic() - started,
            })
            pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
            torch.save({
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'config': config, 'epoch': epoch,
            }, output / 'last_model.pt')
            status.update(completed_epochs=epoch, best_epoch=best_epoch,
                          best_validation_score=best_score, latest_validation=validation)
            write_json(output / 'status.json', status)
            print(
                f'{key} {epoch:02d}/{epochs} | train loss {training["loss"]:.4f} '
                f'acc {training["acc"]:.2f}% | val acc {validation["acc"]:.2f}% '
                f'auc {validation["auc"]:.4f} f1 {validation["f1"]:.4f} '
                f'| best epoch {best_epoch} | {history[-1]["seconds"]:.1f}s', flush=True
            )
        status['stage'] = 'evaluation'
        write_json(output / 'status.json', status)
        checkpoint = torch.load(output / 'best_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        metrics_by_split = {}
        validation_frame = None
        for split, loader in [('validation', loaders[1]), ('test', loaders[2])]:
            metrics, probabilities, labels, slide_ids = evaluate(
                model, criterion, loader, device, conf, return_predictions=True
            )
            frame = save_evaluation(
                split, metrics, probabilities, labels, slide_ids, conf, output
            )
            metrics_by_split[split] = metrics
            if split == 'validation':
                validation_frame = frame
        plot_learning_curves(history, best_epoch, output)
        status['stage'] = 'validation_images'
        write_json(output / 'status.json', status)
        export_args = export_parser().parse_args([
            '--checkpoint', str(output / 'best_model.pt'), '--features', str(conf.data_dir),
            '--notebook', str(notebook_path), '--split', 'validation',
            '--output', str(output / 'attention_panels'), '--device', device,
            '--threads', str(torch.get_num_threads()), '--dpi', str(dpi),
        ])
        panels = export(export_args)
        image_manifest = json.loads((panels / 'export.json').read_text())
        if image_manifest['completed_pngs'] != len(datasets[1]):
            raise ValueError('Validation PNG count does not match the saved split.')
        write_run_page(output, conf, validation_frame, audit)
        result = {
            'dataset': conf.dataset, 'dataset_key': key, 'run_mode': run_mode,
            'classes': int(conf.n_class), 'epochs': epochs, 'best_epoch': best_epoch,
            'validation_slides': len(datasets[1]), 'test_slides': len(datasets[2]),
            'validation_accuracy_pct': metrics_by_split['validation']['acc'],
            'test_accuracy_pct': metrics_by_split['test']['acc'],
            'test_macro_auroc': metrics_by_split['test']['auc'],
            'test_macro_f1': metrics_by_split['test']['f1'],
            'test_loss': metrics_by_split['test']['loss'],
            'validation_pngs': image_manifest['completed_pngs'],
            'run_directory': str(output), 'warnings': audit['warnings'],
        }
        if extension is not None:
            result.update(method=extension.name, training_extension=extension.to_dict())
        if isinstance(extension, SpatialContextConfig):
            profile = {
                'model_parameter_count': config['model_parameter_count'],
                'elapsed_seconds': time.monotonic() - run_started,
                'peak_cuda_allocated_mib': (torch.cuda.max_memory_allocated() / 1024**2
                                            if device == 'cuda' else None),
                'train_eval_graph_cache': model.neighbor_cache.stats(),
                'note': 'Elapsed time includes audit, train/validation, final evaluation, and image export.',
            }
            write_json(output / 'resource_profile.json', profile)
            result['resource_profile'] = profile
        write_json(output / 'result.json', result)
        status['stage'] = 'complete'
        return result
    except BaseException as error:
        status.update(stage='failed', error=str(error))
        raise
    finally:
        write_json(output / 'status.json', status)


def write_suite_page(output, results, state):
    label = experiment_label(state.get('method', 'abmil_baseline'))
    rows = []
    for result in results:
        rows.append(
            f'<tr><td><a href="{result["dataset_key"]}/index.html">'
            f'{escape(result["dataset"])}</a></td>'
            f'<td>{result["best_epoch"]}/{result["epochs"]}</td>'
            f'<td>{result["validation_accuracy_pct"]:.2f}%</td>'
            f'<td>{result["test_accuracy_pct"]:.2f}%</td>'
            f'<td>{result["test_macro_auroc"]:.4f}</td>'
            f'<td>{result["test_macro_f1"]:.4f}</td><td>{result["validation_pngs"]}</td></tr>'
        )
    (output / 'index.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>{escape(label)} results</title>'
        '<style>body{font:17px sans-serif;margin:32px;max-width:1300px}'
        'table{border-collapse:collapse}th,td{padding:12px;border:1px solid #ccc}</style>'
        f'<h1>{escape(label)} results</h1>'
        f'<p>Status: {escape(state["status"])}. Current dataset: '
        f'{escape(state.get("current_dataset", "none"))}. Refresh for updates.</p>'
        '<p>Click a completed dataset for learning curves, confusion matrices, and every '
        'validation WSI image. Training progress is also in each dataset/status.json.</p>'
        '<table><tr><th>Dataset</th><th>Selected/trained epochs</th>'
        '<th>Validation accuracy</th><th>Test accuracy</th><th>Test macro AUROC</th>'
        '<th>Test macro F1</th><th>Validation PNGs</th></tr>'
        + ''.join(rows) + '</table>'
        '<p><a href="results.csv">Results CSV</a> | '
        '<a href="suite_status.json">Run status</a></p>'
        f'<p>One training seed ({state.get("training_seed", 42)}) in this suite. '
        'Predictions are slide-level, not '
        'patch labels or CAMELYON17 patient staging. Check split_audit.json for caveats.</p>'
    )
    if results:
        pd.DataFrame([{k: v for k, v in row.items() if k != 'warnings'}
                      for row in results]).to_csv(output / 'results.csv', index=False)
    write_json(output / 'suite_status.json', state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', choices=list(CONFIGS), nargs='+', default=list(CONFIGS))
    parser.add_argument('--output', type=Path, help='New directory; existing outputs are never overwritten.')
    parser.add_argument('--epochs', type=int, help='Debug override; nonconfigured budgets are labeled smoke.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dpi', type=int, default=140)
    parser.add_argument('--notebook', type=Path, default=ROOT / 'ECE1512_ProjectA_Working.ipynb')
    parser.add_argument('--training-seed', type=int,
                        help='Override training randomness, not split construction.')
    parser.add_argument('--splits-from', type=Path,
                        help='Reference suite containing <dataset>/splits.json and config.json.')
    args = parser.parse_args(argv)
    if args.threads < 1 or args.dpi < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('threads, dpi, and epochs must be positive.')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Do not repeat a dataset.')
    if args.training_seed is not None and not 0 <= args.training_seed < 2**32:
        parser.error('training-seed must be in [0, 2**32).')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or ROOT / 'results' / f'baseline_suite_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {
        'status': 'running', 'datasets': args.datasets, 'completed_datasets': [],
        'started_utc': datetime.now(timezone.utc).isoformat(), 'device': args.device,
        'training_seed': args.training_seed if args.training_seed is not None else 42,
    }
    results = []
    write_suite_page(output, results, state)
    print(f'Baseline suite: {output}', flush=True)
    try:
        for key in args.datasets:
            state['current_dataset'] = key
            write_suite_page(output, results, state)
            conf = load_config(str(ROOT / 'config' / CONFIGS[key]))
            conf.data_dir = str((ROOT / conf.data_dir).resolve())
            conf.dataset_csv_dir = str((ROOT / conf.dataset_csv_dir).resolve())
            if args.splits_from is None:
                datasets = build_HDF5_feat_dataset(conf.data_dir, conf)
            else:
                datasets = load_fixed_datasets(conf, args.splits_from / key / 'splits.json')
            epochs = int(conf.train_epoch) if args.epochs is None else args.epochs
            result = run_dataset(
                key, conf, datasets, output / key, args.notebook.resolve(),
                epochs=epochs, device=args.device, dpi=args.dpi,
                training_seed=args.training_seed,
            )
            results.append(result)
            state['completed_datasets'].append(key)
            write_suite_page(output, results, state)
            print(f'{key} complete: test accuracy {result["test_accuracy_pct"]:.2f}%.', flush=True)
        state.update(status='complete', current_dataset='none')
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        write_suite_page(output, results, state)
    print(f'All baselines complete: {output / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
