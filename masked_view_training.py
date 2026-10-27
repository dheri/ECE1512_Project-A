"""Optional ABMIL patch-dropout and detached full-to-masked consistency training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _check_fraction(value):
    if not isinstance(value, Real) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('keep_fraction must be finite and in (0, 1].')


@dataclass(frozen=True)
class MaskedViewConfig:
    method: str = 'consistency'
    keep_fraction: float = 0.9
    consistency_weight: float = 1.0
    ramp_epochs: int = 0

    def __post_init__(self):
        if self.method not in {'consistency', 'patch_dropout'}:
            raise ValueError('method must be consistency or patch_dropout.')
        _check_fraction(self.keep_fraction)
        if (not isinstance(self.consistency_weight, Real)
                or not math.isfinite(self.consistency_weight)
                or self.consistency_weight < 0):
            raise ValueError('consistency_weight must be finite and nonnegative.')
        if self.method == 'patch_dropout' and self.consistency_weight != 0:
            raise ValueError('The patch_dropout comparator requires consistency_weight=0.')
        if (not isinstance(self.ramp_epochs, Integral) or isinstance(self.ramp_epochs, bool)
                or self.ramp_epochs < 0):
            raise ValueError('ramp_epochs must be a nonnegative integer.')
        if self.method == 'patch_dropout' and self.ramp_epochs:
            raise ValueError('A consistency ramp requires method=consistency.')

    @property
    def name(self):
        return f'abmil_{self.method}' + ('_ramp' if self.ramp_epochs else '')

    def weight_at_epoch(self, epoch):
        if not isinstance(epoch, Integral) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError('epoch must be a positive, one-based integer.')
        fraction = min(epoch / self.ramp_epochs, 1.0) if self.ramp_epochs else 1.0
        return self.consistency_weight * fraction

    @property
    def loss_name(self):
        if self.method == 'consistency':
            return 'CrossEntropyLoss(full) + weight * KL(detached full || masked)'
        return 'CrossEntropyLoss(masked)'

    def to_dict(self):
        saved = {
            'method': self.method, 'keep_fraction': float(self.keep_fraction),
            'consistency_weight': float(self.consistency_weight),
            'objective': self.loss_name, 'temperature': 1.0,
            'sampling': 'Uniform without replacement; max(1, floor(N * keep_fraction)).',
            'mask_rng': 'Dedicated CPU torch.Generator seeded with training_seed.',
            'evaluation': 'Full bags; no patch masking or consistency loss.',
        }
        if self.ramp_epochs:
            saved.update(ramp_epochs=int(self.ramp_epochs),
                         weight_schedule='weight * min(one_based_epoch / ramp_epochs, 1).')
        return saved


def experiment_label(method):
    return {
        'abmil_baseline': 'ABMIL baseline',
        'abmil_consistency': 'ABMIL masked-view consistency',
        'abmil_consistency_ramp': 'ABMIL masked-view consistency (short weight ramp)',
        'abmil_patch_dropout': 'ABMIL patch dropout',
    }[method]


def sample_retained_mask(mask, keep_fraction, generator):
    """Retain real patches only, with at least one per bag and independent RNG."""
    _check_fraction(keep_fraction)
    if (mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[0] < 1
            or not mask.any(dim=1).all()):
        raise ValueError('mask must be Boolean [B, N] with a real patch in every bag.')
    if generator.device.type != 'cpu':
        raise ValueError('Use a dedicated CPU generator for patch sampling.')
    original = mask.detach().cpu()
    retained = torch.zeros_like(original)
    for row in range(original.shape[0]):
        indices = original[row].nonzero(as_tuple=True)[0]
        count = max(1, math.floor(len(indices) * keep_fraction))
        if count == len(indices):
            selected = indices
        else:
            selected = indices[torch.randperm(len(indices), generator=generator)[:count]]
        retained[row, selected] = True
    return retained.to(mask.device)


def prediction_consistency_loss(full_logits, masked_logits):
    """Mean per-slide KL(full || masked); only the masked prediction gets gradients."""
    if (full_logits.ndim != 2 or full_logits.shape != masked_logits.shape
            or full_logits.shape[0] < 1 or full_logits.shape[1] < 2):
        raise ValueError('Both predictions must have the same nonempty [B, C] shape.')
    return F.kl_div(
        F.log_softmax(masked_logits, dim=-1),
        F.log_softmax(full_logits.detach(), dim=-1),
        reduction='batchmean', log_target=True,
    )


def train_masked_epoch(model, criterion, data_loader, optimizer, device, progress=True,
                       *, extension, generator, epoch=1):
    """One optimizer update per bag batch, retaining at most one forward graph."""
    model.train()
    weight = extension.weight_at_epoch(epoch)
    total_loss = total_ce = total_kl = 0.0
    total_correct = total_examples = total_real = total_retained = 0
    batches = tqdm(data_loader, desc='Training slides', leave=False, disable=not progress)
    for data in batches:
        bag = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device, dtype=torch.long)
        mask = data['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        retained = mask
        if extension.method == 'patch_dropout':
            retained = sample_retained_mask(mask, extension.keep_fraction, generator)
            logits = model(bag, mask=retained)
        else:
            logits = model(bag, mask=mask)
        ce = criterion(logits, labels)
        if not torch.isfinite(ce):
            raise FloatingPointError('Non-finite training cross-entropy.')
        target = logits.detach()
        ce_value = ce.item()
        # No optimizer step between passes: their gradients add to the same update.
        # Backpropagate CE first to avoid retaining two large slide graphs.
        ce.backward()
        kl_value = 0.0
        if extension.method == 'consistency' and weight > 0:
            retained = sample_retained_mask(mask, extension.keep_fraction, generator)
            masked_logits = model(bag, mask=retained)
            kl = prediction_consistency_loss(target, masked_logits)
            if not torch.isfinite(kl):
                raise FloatingPointError('Non-finite prediction consistency loss.')
            kl_value = kl.item()
            (weight * kl).backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_ce += ce_value * batch_size
        total_kl += kl_value * batch_size
        total_loss += (ce_value + weight * kl_value) * batch_size
        total_correct += (target.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size
        total_real += mask.sum().item()
        total_retained += retained.sum().item()
    if total_examples == 0:
        raise ValueError('The training DataLoader is empty.')
    metrics = {
        'loss': total_loss / total_examples,
        'acc': 100.0 * total_correct / total_examples,
        'ce_loss': total_ce / total_examples,
        'consistency_loss': total_kl / total_examples,
        'retained_fraction': total_retained / total_real,
    }
    if extension.ramp_epochs:
        metrics['consistency_weight'] = weight
    return metrics
