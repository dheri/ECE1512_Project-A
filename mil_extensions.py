"""Shared extension types and display names, separate from frozen experiment code."""

from masked_view_training import MaskedViewConfig, experiment_label as masked_label
from class_balanced_training import ClassBalancedConfig
from spatial_context import SPATIAL_METHODS, SpatialContextConfig

EXTENSION_TYPES = (MaskedViewConfig, SpatialContextConfig, ClassBalancedConfig)


def experiment_label(method):
    if method == 'abmil_class_balanced':
        return 'ABMIL class-balanced cross-entropy'
    if method == 'abmil_spatial':
        return 'ABMIL spatial context'
    if method == 'abmil_spatial_shuffled':
        return 'ABMIL spatial context (shuffled coordinates)'
    if method == 'abmil_spatial_gated':
        return 'ABMIL spatial context (learnable context gate)'
    if method == 'abmil_spatial_shuffled_gated':
        return 'ABMIL spatial context (shuffled coordinates, learnable context gate)'
    if method in SPATIAL_METHODS:
        details = []
        if '_shuffled' in method:
            details.append('shuffled coordinates')
        if '_compact' in method:
            details.append('compact context')
        if '_gated' in method:
            details.append('fixed context gate' if method.endswith('_fixed')
                           else 'learnable context gate')
        return 'ABMIL spatial context (' + ', '.join(details) + ')'
    return masked_label(method)
