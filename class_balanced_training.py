"""Opt-in per-slide inverse-frequency CE, without within-batch normalization."""

from collections import Counter
from dataclasses import dataclass
from numbers import Integral

import h5py
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ClassBalancedConfig:
    @property
    def name(self):
        return 'abmil_class_balanced'

    @property
    def loss_name(self):
        return 'Mean(train_class_weight[label] * CrossEntropyLoss_per_slide)'

    def to_dict(self):
        return {
            'method': 'class_balanced_ce', 'objective': self.loss_name,
            'weights': 'N_train / (n_classes * train_class_count).',
            'normalization': 'Mean over slides, NOT division by batch label-weight sum.',
            'counts': 'Training slides only, after the frozen development partition.',
            'evaluation': 'Full bags; ordinary unweighted CrossEntropyLoss.',
            'architecture': 'Unchanged notebook ABMIL.',
        }


def weights_from_counts(counts, n_classes):
    if (not isinstance(n_classes, Integral) or isinstance(n_classes, bool)
            or n_classes < 2 or len(counts) != n_classes):
        raise ValueError('Require at least two classes and one count per class.')
    normalized = {}
    for label, count in counts.items():
        if (isinstance(label, bool) or str(label) not in {str(i) for i in range(n_classes)}
                or not isinstance(count, Integral) or isinstance(count, bool) or count < 1):
            raise ValueError('Training class counts must be positive integers for every class.')
        normalized[int(label)] = int(count)
    if set(normalized) != set(range(n_classes)):
        raise ValueError('Missing or duplicate training class IDs.')
    total = sum(normalized.values())
    return torch.tensor([total / (n_classes * normalized[c]) for c in range(n_classes)],
                        dtype=torch.float32)


def training_class_counts(features, train_names):
    if not train_names or len(train_names) != len(set(train_names)):
        raise ValueError('Require a nonempty list of unique training slides.')
    with h5py.File(features, 'r') as h5:
        return dict(sorted(Counter(int(h5[name].attrs['label']) for name in train_names).items()))


class PerSlideWeightedCE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32).detach().clone()
        if (weights.ndim != 1 or len(weights) < 2
                or not torch.isfinite(weights).all() or not (weights > 0).all()):
            raise ValueError('Require finite positive class weights.')
        self.register_buffer('weights', weights)

    def forward(self, logits, labels):
        if logits.ndim != 2 or logits.shape[1] != len(self.weights):
            raise ValueError('Logit class dimension must match class weights.')
        return (F.cross_entropy(logits, labels, reduction='none') * self.weights[labels]).mean()


def train_balanced_epoch(model, criterion, data_loader, optimizer, device, progress=True):
    if not isinstance(criterion, PerSlideWeightedCE):
        raise ValueError('Use the B1-safe per-slide weighted criterion.')
    model.train()
    total_loss = total_ce = total_weight = total_correct = examples = 0
    for data in tqdm(data_loader, desc='Training class-balanced slides', leave=False,
                     disable=not progress):
        bag = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device, dtype=torch.long)
        mask = data['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite class-weighted training loss.')
        loss.backward()
        optimizer.step()
        size = len(labels)
        total_loss += loss.item() * size
        total_ce += F.cross_entropy(logits.detach(), labels, reduction='sum').item()
        total_weight += criterion.weights[labels].sum().item()
        total_correct += (logits.detach().argmax(1) == labels).sum().item()
        examples += size
    if not examples:
        raise ValueError('The training DataLoader is empty.')
    return {'loss': total_loss / examples, 'ce_loss': total_ce / examples,
            'mean_class_weight': total_weight / examples, 'acc': 100. * total_correct / examples}
