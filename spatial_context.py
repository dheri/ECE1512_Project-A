"""Coordinate-aware residual context before the unchanged gated ABMIL pooling."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import classification_metrics

SPATIAL_METHODS = {
    base + gate + compact + fixed
    for base in ['abmil_spatial', 'abmil_spatial_shuffled']
    for gate in ['', '_gated'] for compact in ['', '_compact']
    for fixed in ['', '_fixed'] if not fixed or gate
}


@dataclass(frozen=True)
class SpatialContextConfig:
    neighbors: int = 8
    coordinate_mode: str = 'real'
    shuffle_seed: int = 42
    cache_entries: int = 16
    gated: bool = False
    gate_initial: float = 0.1
    context_hidden: int | None = None
    gate_trainable: bool = True

    def __post_init__(self):
        if not isinstance(self.neighbors, Integral) or isinstance(self.neighbors, bool) or self.neighbors < 1:
            raise ValueError('neighbors must be a positive integer.')
        if self.coordinate_mode not in {'real', 'shuffled'}:
            raise ValueError('coordinate_mode must be real or shuffled.')
        if (not isinstance(self.shuffle_seed, Integral) or isinstance(self.shuffle_seed, bool)
                or not 0 <= self.shuffle_seed < 2**32):
            raise ValueError('shuffle_seed must be an integer in [0, 2**32).')
        if (not isinstance(self.cache_entries, Integral) or isinstance(self.cache_entries, bool)
                or self.cache_entries < 0):
            raise ValueError('cache_entries must be a nonnegative integer.')
        if not isinstance(self.gated, bool):
            raise ValueError('gated must be Boolean.')
        if (not isinstance(self.gate_initial, Real) or not math.isfinite(self.gate_initial)
                or not 0 < self.gate_initial < 1):
            raise ValueError('gate_initial must be finite and in (0, 1).')
        if not self.gated and self.gate_initial != 0.1:
            raise ValueError('A nondefault gate_initial requires gated=True.')
        if self.context_hidden is not None and (
                not isinstance(self.context_hidden, Integral)
                or isinstance(self.context_hidden, bool) or self.context_hidden < 1):
            raise ValueError('context_hidden must be a positive integer or None.')
        if not isinstance(self.gate_trainable, bool):
            raise ValueError('gate_trainable must be Boolean.')
        if not self.gated and not self.gate_trainable:
            raise ValueError('A frozen gate requires gated=True.')

    @property
    def name(self):
        base = 'abmil_spatial' if self.coordinate_mode == 'real' else 'abmil_spatial_shuffled'
        return (base + ('_gated' if self.gated else '')
                + ('_compact' if self.context_hidden is not None else '')
                + ('_fixed' if not self.gate_trainable else ''))

    @property
    def loss_name(self):
        return 'CrossEntropyLoss'

    def to_dict(self):
        saved = {
            'method': 'spatial_context', 'neighbors': int(self.neighbors),
            'coordinate_mode': self.coordinate_mode, 'shuffle_seed': int(self.shuffle_seed),
            'cache_entries': int(self.cache_entries), 'objective': self.loss_name,
            'context': 'h + Linear(ReLU(Linear(concat(h, mean_neighbor_h)))); hidden width D_inner.',
            'initialization': 'Zero final context layer; preserve baseline initialization RNG.',
            'graph': 'Canonical-coordinate cKDTree kNN, Euclidean input units, exclude self; schema 1.',
            'small_bags': 'min(k, N-1) neighbors; singleton neighbor mean is zero.',
            'duplicates': 'Reject duplicate coordinates rather than invent an ordering.',
            'shuffle': 'Fixed per slide ID and shuffle_seed, assigned in canonical coordinate order.',
            'evaluation': 'Full bags, with the same coordinate mode at training and inference.',
        }
        if self.gated:
            saved['context'] = (
                'h + sigmoid(Linear(concat(h, mean_neighbor_h))) * '
                'Linear(ReLU(Linear(concat(h, mean_neighbor_h)))); hidden width D_inner.'
            )
            saved['initialization'] = (
                'Zero final context layer; zero gate weights and logit(initial_probability) '
                'gate bias; preserve baseline initialization RNG.'
            )
            saved['gate'] = {'type': 'per_patch_sigmoid_linear',
                             'initial_probability': float(self.gate_initial)}
            if not self.gate_trainable:
                saved['gate']['trainable'] = False
                saved['initialization'] += ' Gate parameters are frozen.'
        if self.context_hidden is not None:
            saved['context_hidden'] = int(self.context_hidden)
            saved['context'] = saved['context'].replace(
                'hidden width D_inner.', f'hidden width {self.context_hidden}.')
        return saved

    @classmethod
    def from_dict(cls, saved):
        options = cls(**{name: saved[name] for name in
                         ['neighbors', 'coordinate_mode', 'shuffle_seed', 'cache_entries']},
                      gated='gate' in saved,
                      gate_initial=saved['gate']['initial_probability'] if 'gate' in saved else 0.1,
                      context_hidden=saved.get('context_hidden'),
                      gate_trainable=saved.get('gate', {}).get('trainable', True))
        if saved != options.to_dict():
            raise ValueError('Unsupported or inconsistent saved spatial configuration.')
        return options


def checked_coordinates(coords):
    if coords is None:
        raise ValueError('Spatial context requires patch coordinates.')
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    array = np.asarray(coords)
    if (array.ndim != 2 or array.shape[1] != 2 or len(array) < 1
            or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)):
        raise ValueError('Coordinates must be a nonempty numeric [N, 2] array.')
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('Coordinates must be finite.')
    return array


def build_neighbors(coords, options, slide_id=None):
    """O(N log N + Nk) lookup, never an N × N distance matrix."""
    coords = checked_coordinates(coords)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    canonical = coords[order]
    if len(coords) > 1 and np.any(np.all(canonical[1:] == canonical[:-1], axis=1)):
        raise ValueError('Duplicate patch coordinates are unsupported for spatial context.')
    if options.coordinate_mode == 'shuffled':
        if not isinstance(slide_id, str) or not slide_id:
            raise ValueError('The shuffled control requires a stable slide ID.')
        digest = hashlib.sha256(
            json.dumps([int(options.shuffle_seed), slide_id]).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        canonical = canonical[rng.permutation(len(canonical))]
    count = min(int(options.neighbors), len(coords) - 1)
    if count == 0:
        return np.empty((len(coords), 0), dtype=np.int64)
    # Canonical input order makes equidistant choices repeatable under row
    # permutations. Exact tie choices can still depend on the SciPy version.
    _, neighbors = cKDTree(canonical).query(canonical, k=count + 1, workers=1)
    result = np.empty((len(coords), count), dtype=np.int64)
    result[order] = order[neighbors[:, 1:]]
    return result


class SpatialNeighborCache:
    """Bounded CPU LRU plus optional per-run disk cache; never cache features."""

    def __init__(self, options, directory=None, writable=False):
        self.options = options
        self.directory = None if directory is None else Path(directory)
        self.writable = writable
        self.memory = OrderedDict()
        self.hits = self.disk_hits = self.builds = 0
        self.build_seconds = 0.

    def get(self, coords, slide_id=None):
        coords = checked_coordinates(coords)
        digest = hashlib.sha256()
        digest.update(json.dumps([self.options.to_dict(), slide_id, list(coords.shape)],
                                 sort_keys=True).encode())
        digest.update(coords.tobytes())
        key = digest.hexdigest()
        if key in self.memory:
            self.hits += 1
            self.memory.move_to_end(key)
            return self.memory[key]
        path = None if self.directory is None else self.directory / f'{key}.npy'
        shape = (len(coords), min(self.options.neighbors, len(coords) - 1))
        if path is not None and path.is_file():
            indices = np.load(path, allow_pickle=False)
            self.disk_hits += 1
        else:
            started = time.monotonic()
            indices = build_neighbors(coords, self.options, slide_id)
            self.build_seconds += time.monotonic() - started
            self.builds += 1
            if path is not None and self.writable:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle, indices, allow_pickle=False)
                temporary.replace(path)
        if indices.shape != shape or indices.dtype != np.int64:
            raise ValueError('Invalid spatial neighbor cache shape or dtype.')
        if indices.size and (indices.min() < 0 or indices.max() >= len(coords)
                             or np.any(indices == np.arange(len(coords))[:, None])):
            raise ValueError('Invalid spatial neighbor indices or self-neighbors.')
        # A graph first seen during inference must also be usable in later
        # gradient-enabled forwards, whose index operations save these indices.
        with torch.inference_mode(False):
            tensor = torch.from_numpy(indices)
        if self.options.cache_entries:
            self.memory[key] = tensor
            while len(self.memory) > self.options.cache_entries:
                self.memory.popitem(last=False)
        return tensor

    def stats(self):
        return {
            'memory_hits': self.hits, 'disk_hits': self.disk_hits, 'graphs_built': self.builds,
            'build_seconds': self.build_seconds, 'memory_entries': len(self.memory),
            'memory_bytes': sum(value.numel() * value.element_size() for value in self.memory.values()),
        }


class SpatialContextABMIL(nn.Module):
    def __init__(self, baseline, options):
        super().__init__()
        self.options = options
        self.dimreduction = baseline.dimreduction
        self.attention = baseline.attention
        self.classifier = baseline.classifier
        width = self.dimreduction.fc.out_features
        hidden = width if options.context_hidden is None else options.context_hidden
        # Do not shift baseline initialization or DataLoader RNG by adding a module.
        with torch.random.fork_rng(devices=[]):
            self.context = nn.Sequential(nn.Linear(2 * width, hidden), nn.ReLU(),
                                         nn.Linear(hidden, width))
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)
            if options.gated:
                self.context_gate = nn.Linear(2 * width, 1)
                nn.init.zeros_(self.context_gate.weight)
                nn.init.constant_(self.context_gate.bias,
                                  math.log(options.gate_initial / (1 - options.gate_initial)))
                self.context_gate.requires_grad_(options.gate_trainable)
        self.neighbor_cache = SpatialNeighborCache(options)

    def set_neighbor_cache(self, directory, writable=False):
        self.neighbor_cache = SpatialNeighborCache(self.options, directory, writable)

    def forward(self, x, return_attention=False, mask=None, *, coords=None, slide_ids=None):
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError('Features must be nonempty [B, N, D].')
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if (mask.shape != x.shape[:2] or mask.dtype != torch.bool
                or not mask.any(dim=1).all()):
            raise ValueError('mask must be Boolean [B, N] with a real patch in each bag.')
        if coords is None or len(coords) != len(x):
            raise ValueError('Supply one unpadded coordinate array per bag.')
        if slide_ids is not None and len(slide_ids) != len(x):
            raise ValueError('Supply one slide ID per bag.')
        h = self.dimreduction(x)
        delta = torch.zeros_like(h)
        for row in range(len(x)):
            valid = mask[row].nonzero(as_tuple=True)[0]
            row_coords = checked_coordinates(coords[row])
            if len(row_coords) != len(valid):
                raise ValueError('Coordinates must match real patches in their feature order.')
            name = None if slide_ids is None else slide_ids[row]
            indices = self.neighbor_cache.get(row_coords, name).to(x.device)
            own = h[row, valid]
            neighbor_mean = torch.zeros_like(own)
            # Sum one neighbor rank at a time, avoiding an [N, k, D] activation.
            for rank in range(indices.shape[1]):
                neighbor_mean = neighbor_mean + own.index_select(0, indices[:, rank])
            if indices.shape[1]:
                neighbor_mean = neighbor_mean / indices.shape[1]
            context_input = torch.cat([own, neighbor_mean], dim=-1)
            correction = self.context(context_input)
            if self.options.gated:
                correction = torch.sigmoid(self.context_gate(context_input)) * correction
            delta[row, valid] = correction
        context_h = h + delta
        scores = self.attention(context_h).masked_fill(~mask.unsqueeze(1), float('-inf'))
        attention = F.softmax(scores, dim=-1)
        logits = self.classifier(torch.bmm(attention, context_h).squeeze(1))
        return (logits, attention) if return_attention else logits


def spatial_batch_forward(model, data, device):
    return model(
        data['input'].to(device, dtype=torch.float32), mask=data['mask'].to(device),
        coords=data['coords'], slide_ids=data['slide_id'],
    )


def train_spatial_epoch(model, criterion, data_loader, optimizer, device, progress=True):
    model.train()
    total_loss = total_correct = total_examples = 0
    for data in tqdm(data_loader, desc='Training spatial slides', leave=False, disable=not progress):
        labels = data['label'].to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = spatial_batch_forward(model, data, device)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite spatial training loss.')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += len(labels)
    if total_examples == 0:
        raise ValueError('The training DataLoader is empty.')
    return {'loss': total_loss / total_examples, 'acc': 100. * total_correct / total_examples}


@torch.no_grad()
def evaluate_spatial(model, criterion, data_loader, device, conf, return_predictions=False):
    model.eval()
    total_loss = total_examples = 0
    probabilities, labels_all, slide_ids = [], [], []
    for data in data_loader:
        labels = data['label'].to(device, dtype=torch.long)
        logits = spatial_batch_forward(model, data, device)
        total_loss += criterion(logits, labels).item() * len(labels)
        total_examples += len(labels)
        probabilities.append(logits.softmax(dim=-1).cpu())
        labels_all.append(labels.cpu())
        slide_ids.extend(data['slide_id'])
    if total_examples == 0:
        raise ValueError('The evaluation DataLoader is empty.')
    probabilities, labels_all = torch.cat(probabilities), torch.cat(labels_all)
    metrics = classification_metrics(probabilities, labels_all, conf.n_class)
    metrics['loss'] = total_loss / total_examples
    if return_predictions:
        return metrics, probabilities, labels_all, slide_ids
    return metrics
