"""Opt-in refinements: scheduled objectives, adaptive gates, and full workflows."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_attention import load_notebook_model
from masked_view_training import (
    MaskedViewConfig, prediction_consistency_loss, sample_retained_mask, train_masked_epoch,
)
from run_consistency import main as consistency_main
from run_multiseed import prepare_inputs, run_study, verify_run
from run_spatial import main as spatial_main
from spatial_context import SpatialContextABMIL, SpatialContextConfig, build_neighbors
import test_masked_view as fixtures


class RefinementUnitTests(unittest.TestCase):
    setUp = fixtures.MaskedViewUnitTests.setUp

    def test_ramp_schedule_and_legacy_serialization(self):
        old = MaskedViewConfig()
        self.assertNotIn('ramp_epochs', old.to_dict())
        self.assertNotIn('weight_schedule', old.to_dict())
        self.assertEqual(old.name, 'abmil_consistency')
        options = MaskedViewConfig(ramp_epochs=3)
        self.assertEqual(options.name, 'abmil_consistency_ramp')
        self.assertEqual([options.weight_at_epoch(i) for i in [1, 2, 3, 4, 30]],
                         [1 / 3, 2 / 3, 1, 1, 1])
        self.assertEqual(MaskedViewConfig(consistency_weight=.6, ramp_epochs=3)
                         .weight_at_epoch(1), .6 / 3)
        for invalid in [-1, True, 1.5]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MaskedViewConfig(ramp_epochs=invalid)
        for invalid in [0, -1, True, 1.5]:
            with self.subTest(epoch=invalid), self.assertRaises(ValueError):
                options.weight_at_epoch(invalid)
        with self.assertRaises(ValueError):
            MaskedViewConfig('patch_dropout', .9, 0, 3)

    def test_ramp_matches_joint_objective_at_each_phase(self):
        options = MaskedViewConfig(keep_fraction=.6, ramp_epochs=3)
        for epoch in [1, 2, 3, 4]:
            with self.subTest(epoch=epoch):
                actual, expected = copy.deepcopy(self.model), copy.deepcopy(self.model)
                optimizers = [torch.optim.AdamW(m.parameters(), lr=.001)
                              for m in [actual, expected]]
                retained = sample_retained_mask(self.batch['mask'], .6,
                                               torch.Generator().manual_seed(42))
                full = expected(self.batch['input'], mask=self.batch['mask'])
                masked = expected(self.batch['input'], mask=retained)
                weight = options.weight_at_epoch(epoch)
                objective = self.criterion(full, self.batch['label'])
                objective = objective + weight * prediction_consistency_loss(full, masked)
                objective.backward()
                optimizers[1].step()
                metrics = train_masked_epoch(
                    actual, self.criterion, [self.batch], optimizers[0], 'cpu', progress=False,
                    extension=options, generator=torch.Generator().manual_seed(42), epoch=epoch,
                )
                self.assertEqual(metrics['consistency_weight'], weight)
                self.assertAlmostEqual(metrics['loss'],
                                       metrics['ce_loss'] + weight * metrics['consistency_loss'])
                for a, b in zip(actual.parameters(), expected.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_zero_ramp_weight_preserves_baseline_updates_and_mask_rng(self):
        actual, expected = copy.deepcopy(self.model), copy.deepcopy(self.model)
        optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in [actual, expected]]
        self.criterion(expected(self.batch['input'], mask=self.batch['mask']),
                       self.batch['label']).backward()
        optimizers[1].step()
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        metrics = train_masked_epoch(
            actual, self.criterion, [self.batch], optimizers[0], 'cpu', progress=False,
            extension=MaskedViewConfig(consistency_weight=0, ramp_epochs=3),
            generator=generator, epoch=2,
        )
        self.assertTrue(torch.equal(before, generator.get_state()))
        self.assertEqual(metrics['consistency_weight'], 0)
        for a, b in zip(actual.parameters(), expected.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_gate_configuration_and_original_checkpoint_schema(self):
        old = SpatialContextConfig()
        self.assertNotIn('gate', old.to_dict())
        original = SpatialContextABMIL(copy.deepcopy(self.model), old)
        self.assertFalse(hasattr(original, 'context_gate'))
        self.assertFalse(any(name.startswith('context_gate.') for name in original.state_dict()))
        for mode in ['real', 'shuffled']:
            options = SpatialContextConfig(coordinate_mode=mode, gated=True)
            self.assertEqual(SpatialContextConfig.from_dict(options.to_dict()), options)
        for kwargs in [{'gated': 1}, {'gate_initial': .2},
                       *[{'gated': True, 'gate_initial': p}
                         for p in [0, 1, -.1, float('nan'), float('inf')]]]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpatialContextConfig(**kwargs)
        changed = SpatialContextConfig(gated=True).to_dict()
        changed['gate']['type'] = 'unsupported'
        with self.assertRaises(ValueError):
            SpatialContextConfig.from_dict(changed)

    def test_gate_initialization_preserves_rng_predictions_and_neighbors(self):
        options = SpatialContextConfig(gated=True)
        before = torch.get_rng_state().clone()
        model = SpatialContextABMIL(copy.deepcopy(self.model), options)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        coords = torch.arange(14).reshape(7, 2)
        bag = self.batch['input'][:1]
        logits, attention = model(bag, coords=[coords], return_attention=True)
        expected, expected_attention = self.model(bag, return_attention=True)
        self.assertTrue(torch.equal(logits, expected))
        self.assertTrue(torch.equal(attention, expected_attention))
        gate = torch.sigmoid(model.context_gate(torch.randn(7, 8)))
        torch.testing.assert_close(gate, torch.full((7, 1), .1))
        self.assertEqual(model.context_gate.weight.shape, (1, 8))
        np.testing.assert_array_equal(build_neighbors(coords, options),
                                      build_neighbors(coords, SpatialContextConfig()))

    def test_gate_gets_gradient_after_zero_context_begins_learning(self):
        model = SpatialContextABMIL(copy.deepcopy(self.model), SpatialContextConfig(gated=True))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        bag = self.batch['input'][:1]
        coords = [torch.arange(14).reshape(7, 2)]
        initial = model.context_gate.bias.detach().clone()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            self.criterion(model(bag, coords=coords), self.batch['label'][:1]).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            magnitude = model.context_gate.bias.grad.abs().sum().item()
            if step == 0:
                self.assertEqual(magnitude, 0)
                self.assertGreater(model.context[-1].bias.grad.abs().sum().item(), 0)
            else:
                self.assertGreater(magnitude, 0)
            optimizer.step()
        self.assertFalse(torch.equal(initial, model.context_gate.bias))

    def test_gate_can_ignore_context_or_reproduce_parent_and_respects_permutation(self):
        options = SpatialContextConfig(neighbors=3, gated=True)
        model = SpatialContextABMIL(copy.deepcopy(self.model), options)
        with torch.no_grad():
            model.context[-1].weight.normal_(0, .3)
            model.context[-1].bias.normal_(0, .1)
            model.context_gate.bias.fill_(-100)
        bag = self.batch['input'][:1]
        coords = torch.arange(14).reshape(7, 2)
        torch.testing.assert_close(model(bag, coords=[coords]), self.model(bag), rtol=0, atol=0)
        parent = SpatialContextABMIL(copy.deepcopy(self.model), SpatialContextConfig(neighbors=3))
        parent.context.load_state_dict(model.context.state_dict())
        with torch.no_grad():
            model.context_gate.bias.fill_(100)
        torch.testing.assert_close(model(bag, coords=[coords]), parent(bag, coords=[coords]))
        with torch.no_grad():
            model.context_gate.weight.normal_(0, .3)
            model.context_gate.bias.fill_(-2.)
        permutation = torch.tensor([6, 1, 4, 2, 0, 3, 5])
        torch.testing.assert_close(
            model(bag, coords=[coords]),
            model(bag[:, permutation], coords=[coords[permutation]]), rtol=1e-5, atol=1e-6,
        )

    def test_refinement_cli_options_dispatch_without_changing_defaults(self):
        with patch('run_consistency.run_study') as run:
            consistency_main(['--reference-suite', '/tmp/reference', '--output', '/tmp/new',
                              '--ramp-epochs', '3'])
            self.assertEqual(run.call_args.kwargs['extension'], MaskedViewConfig(ramp_epochs=3))
        with patch('run_spatial.run_study') as run:
            spatial_main(['--reference-suite', '/tmp/reference', '--output', '/tmp/new', '--gated'])
            self.assertEqual(run.call_args.kwargs['extension'], SpatialContextConfig(gated=True))


class RefinementWorkflowTests(unittest.TestCase):
    setUp = fixtures.MaskedViewWorkflowTests.setUp
    make_reference = fixtures.MaskedViewWorkflowTests.make_reference

    def test_full_studies_reload_export_and_reject_schedule_tampering(self):
        self.make_reference()
        prepared, source_hash = prepare_inputs(
            self.reference, ['C16'], self.notebook, self.root / 'feature', self.root,
        )
        for options in [MaskedViewConfig(ramp_epochs=3), SpatialContextConfig(gated=True)]:
            with self.subTest(method=options.name):
                output = self.root / options.name
                with contextlib.redirect_stdout(io.StringIO()):
                    run_study(
                        self.reference, output, seeds=[42, 43], keys=['C16'], device='cpu', dpi=30,
                        notebook=self.notebook, feature_root=self.root / 'feature', metadata=self.root,
                        extension=options,
                    )
                self.assertEqual(json.loads((output / 'study_status.json').read_text())['status'],
                                 'complete')
                for seed in [42, 43]:
                    folder = output / f'seed_{seed}/C16'
                    saved = torch.load(folder / 'best_model.pt', weights_only=True)
                    model = load_notebook_model(SimpleNamespace(**saved['config']), self.notebook)
                    model.load_state_dict(saved['model_state_dict'], strict=True)
                    manifest = json.loads((folder / 'attention_panels/export.json').read_text())
                    self.assertEqual(manifest['method'], options.name)
                    self.assertEqual(manifest['completed_pngs'], 2)
                    self.assertEqual(manifest['training_extension'], options.to_dict())
                    history = pd.read_csv(folder / 'history.csv')
                    if isinstance(options, MaskedViewConfig):
                        np.testing.assert_allclose(history.train_consistency_weight,
                                                   [1 / 3, 2 / 3] + [1] * 28)
                        self.assertIn('masked_view_curves.png', (folder / 'index.html').read_text())
                        self.assertTrue(json.loads((folder / 'verification.json').read_text())
                                        ['consistency_schedule_verified'])
                    else:
                        self.assertIn('context_gate.weight', saved['model_state_dict'])
                        self.assertNotIn('train_consistency_weight', history)
                if isinstance(options, MaskedViewConfig):
                    folder = output / 'seed_42/C16'
                    history = pd.read_csv(folder / 'history.csv')
                    history.loc[0, 'train_consistency_weight'] = 1
                    history.to_csv(folder / 'history.csv', index=False)
                    with self.assertRaises(AssertionError):
                        verify_run(folder, prepared['C16'][0], 42, source_hash, options)


if __name__ == '__main__':
    unittest.main()
