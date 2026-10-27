"""Export three-panel WSI attention PNGs for CAMELYON16, CAMELYON17, or BRACS."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dataloader import HDF5FeatureDataset
from spatial_context import SPATIAL_METHODS, SpatialContextABMIL, SpatialContextConfig
from wsi_visualization import plot_attention_triptych

ROOT = Path(__file__).resolve().parent
SPLITS = {'train': 'Train', 'validation': 'Validation', 'test': 'Test'}
CLASS_NAMES = ['Normal', 'Tumor']
DATASET_KEYS = {'CAMELYON16': 'C16', 'CAMELYON17': 'C17', 'BRACS': 'BRACS'}


def class_names(conf):
    if str(conf.dataset).upper() == 'CAMELYON16' and int(conf.n_class) == 2:
        return CLASS_NAMES
    return [f'Class {index}' for index in range(int(conf.n_class))]


def latest_checkpoint(project_dir=ROOT, dataset_key='C16'):
    candidates = list((project_dir / 'results' / dataset_key).glob('*/best_model.pt'))
    candidates.extend((project_dir / 'results').glob(f'*/{dataset_key}/best_model.pt'))
    if not candidates:
        raise FileNotFoundError(f'No saved {dataset_key} checkpoint. Run a baseline first.')
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_notebook_model(conf, notebook_path):
    """Load baseline definitions and an explicitly configured optional context layer."""
    notebook = json.loads(notebook_path.read_text())
    required = ['Classifier1FC', 'DimReduction', 'AttentionGated', 'ABMIL']
    model_cell = next(
        (cell for cell in notebook['cells'] if cell.get('id') == 'a8146687'), None
    )
    if model_cell is None:
        raise ValueError('Cannot find the working notebook ABMIL model-definition cell.')
    source = model_cell['source']
    source = ''.join(source) if isinstance(source, list) else source
    parsed = ast.parse(source)
    definitions = {
        node.name: node for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name in required
    }
    if set(definitions) != set(required):
        raise ValueError('The notebook does not contain the expected ABMIL classes.')
    module = ast.Module(body=[definitions[name] for name in required], type_ignores=[])
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(module, str(notebook_path), 'exec'), namespace)
    baseline = namespace['ABMIL'](conf)
    if getattr(conf, 'method', None) in SPATIAL_METHODS:
        options = SpatialContextConfig.from_dict(conf.training_extension)
        if options.name != conf.method:
            raise ValueError('Spatial method does not match the coordinate mode.')
        return SpatialContextABMIL(baseline, options)
    return baseline


def read_saved_splits(checkpoint_path, feature_path):
    split_path = checkpoint_path.parent / 'splits.json'
    saved = json.loads(split_path.read_text())
    flattened = []
    for name in SPLITS.values():
        names = saved.get(name)
        if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
            raise ValueError(f'Missing or invalid {name} slide IDs in {split_path}.')
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', s) or s in {'.', '..'} for s in names):
            raise ValueError('Slide IDs must be safe filenames.')
        flattened.extend(names)
    if len(set(flattened)) != len(flattened):
        raise ValueError('Saved splits contain duplicate or overlapping slide IDs.')
    with h5py.File(feature_path, 'r') as h5:
        if set(flattened) != set(h5):
            raise ValueError('The feature file does not match the saved full-dataset splits.')
    return saved


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--dataset', choices=list(DATASET_KEYS.values()), default='C16',
                        help='Dataset for automatic checkpoint discovery; default C16.')
    result.add_argument('--checkpoint', type=Path,
                        help='Explicit checkpoint determines the dataset; otherwise find latest saved run.')
    result.add_argument('--split', choices=[*SPLITS, 'all'], default='validation',
                        help='Default: all validation slides. Use all for every WSI.')
    result.add_argument('--output', type=Path, help='A new output directory; existing folders are never overwritten.')
    result.add_argument('--features', type=Path, help='Default: matching dataset in feature/.')
    result.add_argument('--notebook', type=Path, default=ROOT / 'ECE1512_ProjectA_Working.ipynb')
    result.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--dpi', type=int, default=140)
    return result


def export(args):
    if args.threads < 1 or args.dpi < 1:
        raise ValueError('threads and dpi must be positive.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu.')
    torch.set_num_threads(args.threads)
    checkpoint_path = (args.checkpoint or latest_checkpoint(dataset_key=args.dataset)).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    conf = SimpleNamespace(**checkpoint['config'])
    dataset_key = DATASET_KEYS.get(str(conf.dataset).upper())
    if dataset_key is None or int(conf.n_class) < 2:
        raise ValueError('Expected a supported MIL classification checkpoint.')
    features = (args.features or ROOT / 'feature' / dataset_key
                / 'patch_feats_pretrain_medical_ssl.h5').resolve()
    names = class_names(conf)
    conf.data_dir = str(features)
    saved = read_saved_splits(checkpoint_path, features)
    selected = list(SPLITS) if args.split == 'all' else [args.split]
    if any(not saved[SPLITS[name]] for name in selected):
        raise ValueError('A selected split has no slides.')
    model = load_notebook_model(conf, args.notebook.resolve()).to(args.device)
    if isinstance(model, SpatialContextABMIL):
        model.set_neighbor_cache(checkpoint_path.parent / 'spatial_neighbors', writable=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or checkpoint_path.parent / f'attention_panels_{args.split}_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    total = sum(len(saved[SPLITS[name]]) for name in selected)
    metadata = {
        'status': 'running', 'checkpoint': str(checkpoint_path),
        'dataset': conf.dataset, 'n_class': int(conf.n_class), 'class_names': names,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'run_mode': getattr(conf, 'run_mode', 'unknown'),
        'training_epochs': getattr(conf, 'epochs', None),
        'method': getattr(conf, 'method', 'abmil_baseline'),
        'training_extension': getattr(conf, 'training_extension', None),
        'display_coordinates': 'Original coordinates, including for the shuffled-coordinate control.',
        'features': str(features), 'notebook': str(args.notebook.resolve()),
        'split': args.split, 'split_counts': {name: len(saved[SPLITS[name]]) for name in selected},
        'expected_pngs': total, 'completed_pngs': 0,
        'raw_scale': 'Unchanged softmax weights; linear color scale from zero to slide maximum.',
        'percentile_scale': 'Within-slide zero-based midrank / (N-1), multiplied by 100.',
        'concentration': 'Descending raw-weight cumulative sum; x is percentage of patches.',
        'top_1pct': 'ceil(N * 0.01) patches; values are attention mass, not tumor probability.',
    }
    manifest = output / 'export.json'
    manifest.write_text(json.dumps(metadata, indent=2))
    print(f'Checkpoint: {checkpoint_path} (epoch {checkpoint["epoch"]})', flush=True)
    print(f'Exporting {total} slides to {output}; no training.', flush=True)
    try:
        with (output / 'summary.csv').open('x', newline='') as handle:
            writer = None
            with torch.inference_mode():
                for split in selected:
                    folder = output / split
                    folder.mkdir()
                    dataset = HDF5FeatureDataset(str(features), saved[SPLITS[split]], conf)
                    for item in dataset:
                        bag = item['input'].unsqueeze(0).to(args.device, dtype=torch.float32)
                        if bag.shape[-1] != int(conf.D_feat):
                            raise ValueError(f'{item["slide_id"]}: checkpoint feature dimension mismatch.')
                        spatial_inputs = (
                            {'coords': [item['coords']], 'slide_ids': [item['slide_id']]}
                            if isinstance(model, SpatialContextABMIL) else {}
                        )
                        logits, attention = model(bag, return_attention=True, **spatial_inputs)
                        probabilities = logits.softmax(dim=-1)[0].cpu().numpy()
                        if probabilities.shape != (int(conf.n_class),) or not np.isfinite(probabilities).all():
                            raise ValueError(f'{item["slide_id"]}: invalid model probabilities.')
                        label = int(item['label'])
                        if label not in range(int(conf.n_class)) or item['coords'] is None:
                            raise ValueError(f'{item["slide_id"]}: valid label and patch coordinates required.')
                        prediction = int(probabilities.argmax())
                        confidence = float(probabilities[prediction])
                        title = (
                            f'{conf.dataset} | {split} | {item["slide_id"]} | '
                            f'{getattr(conf, "run_mode", "saved")} checkpoint, epoch {checkpoint["epoch"]}\n'
                            f'True: {names[label]} | Predicted: {names[prediction]} '
                            f'({confidence:.1%}) | {bag.shape[1]:,} patches'
                        )
                        fig, stats = plot_attention_triptych(
                            item['coords'].numpy(), attention[0, 0].cpu().numpy(), title=title
                        )
                        image_path = folder / f'{item["slide_id"]}.png'
                        try:
                            fig.savefig(image_path, dpi=args.dpi, bbox_inches='tight')
                        finally:
                            plt.close(fig)
                        row = {
                            'split': split, 'slide_id': item['slide_id'], 'label': label,
                            'prediction': prediction, 'confidence': confidence,
                            'correct': label == prediction,
                            **{f'p_class_{i}': float(p) for i, p in enumerate(probabilities)},
                            **({'p_normal': float(probabilities[0]), 'p_tumor': float(probabilities[1])}
                               if dataset_key == 'C16' and int(conf.n_class) == 2 else {}),
                            **stats,
                            'png': str(image_path.relative_to(output)),
                        }
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        handle.flush()
                        metadata['completed_pngs'] += 1
                        print(f'[{metadata["completed_pngs"]}/{total}] {split}/{image_path.name}', flush=True)
                        # Release large slide tensors and cyclic Matplotlib canvases
                        # before fetching the next bag in memory-limited CPU exports.
                        del fig, bag, logits, attention, probabilities, item
                        gc.collect()
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'failed'
        metadata['error'] = str(error)
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2))
    print(f'Done: {total} PNGs, summary.csv, and export.json.', flush=True)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        export(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
