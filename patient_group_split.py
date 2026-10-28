"""Deterministic C17 development-only grouping; never repartition test slides."""

from collections import Counter
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_id(slide):
    match = re.fullmatch(r'patient_(\d+)_node_(\d+)', slide)
    if match is None:
        raise ValueError(f'Unsupported C17 slide ID: {slide}')
    return int(match.group(1))


def grouped_c17_split(original, development_labels, *, folds=10, seed=42):
    if set(original) != {'Train', 'Validation', 'Test'}:
        raise ValueError('Require the three original partitions.')
    pool = original['Train'] + original['Validation']
    all_names = pool + original['Test']
    if not all(original.values()) or len(set(all_names)) != len(all_names):
        raise ValueError('Original partitions must be nonempty and slide-disjoint.')
    if set(development_labels) != set(pool):
        raise ValueError('Use exactly development labels, not test labels.')
    labels = np.asarray([development_labels[name] for name in pool])
    if set(labels) != {0, 1, 2, 3}:
        raise ValueError('Development must contain exactly the four C17 classes.')
    groups = np.asarray([patient_id(name) for name in pool])
    test_groups = {patient_id(name) for name in original['Test']}
    if set(groups) & test_groups:
        raise ValueError('Original test patients overlap development patients.')
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    candidates = []
    for index, (train, validation) in enumerate(splitter.split(np.zeros(len(pool)), labels, groups)):
        counts = {'Train': dict(sorted(Counter(map(int, labels[train])).items())),
                  'Validation': dict(sorted(Counter(map(int, labels[validation])).items()))}
        covered = all(set(value) == {0, 1, 2, 3} for value in counts.values())
        candidates.append({'fold': index, 'class_counts': counts, 'all_classes': covered})
        if covered:
            # Explicit original-pool order, not an incidental splitter ordering.
            train_indices, validation_indices = set(train), set(validation)
            result = {
                'Train': [name for i, name in enumerate(pool) if i in train_indices],
                'Validation': [name for i, name in enumerate(pool) if i in validation_indices],
                'Test': list(original['Test']),
            }
            patients = {key: sorted({patient_id(name) for name in names})
                        for key, names in result.items()}
            if any(set(patients[a]) & set(patients[b]) for a, b in
                   [('Train', 'Validation'), ('Train', 'Test'), ('Validation', 'Test')]):
                raise ValueError('Patient separation failed.')
            return result, {
                'algorithm': 'StratifiedGroupKFold, first all-class development fold.',
                'n_splits': folds, 'seed': seed, 'selected_fold': index,
                'folds_examined': candidates, 'patients': patients,
                'patient_overlap': {'train_validation': 0, 'train_test': 0, 'validation_test': 0},
                'split_sizes': {key: len(names) for key, names in result.items()},
                'test_membership_and_order_unchanged': result['Test'] == original['Test'],
                'test_labels_used_for_partition_selection': False,
            }
    raise ValueError('No fixed-seed fold contains every class; stop without a seed search.')
