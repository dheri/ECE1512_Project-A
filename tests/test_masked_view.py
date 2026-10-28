"""Synthetic checks for masking, consistency gradients, and verified study outputs."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader import mil_collate_fn
from export_attention import load_notebook_model
from masked_view_training import (
    MaskedViewConfig, prediction_consistency_loss, sample_retained_mask, train_masked_epoch,
)
from run_baselines import load_fixed_datasets, run_dataset, training_functions
from run_consistency import main
from run_multiseed import prepare_inputs, run_study, verify_run


class RecordingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.masks = []

    def forward(self, bag, mask=None):
        self.masks.append(mask.detach().clone())
        return self.model(bag, mask=mask)


class MaskedViewUnitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.notebook = ROOT / 'ECE1512_ProjectA_Working.ipynb'
        self.conf = SimpleNamespace(D_feat=3, D_inner=4, D_attn=2, n_class=3, droprate=0.)
        self.model = load_notebook_model(self.conf, self.notebook)
        self.batch = {
            'input': torch.randn(2, 7, 3), 'label': torch.tensor([0, 2]),
            'mask': torch.tensor([[True] * 7, [True] * 4 + [False] * 3]),
        }
        self.criterion = nn.CrossEntropyLoss()

    def test_configuration_validation(self):
        for fraction in [0, -0.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                MaskedViewConfig(keep_fraction=fraction)
        for weight in [-1, float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                MaskedViewConfig(consistency_weight=weight)
        with self.assertRaises(ValueError):
            MaskedViewConfig(method='other')
        with self.assertRaisesRegex(ValueError, 'requires consistency_weight=0'):
            MaskedViewConfig(method='patch_dropout')
        self.assertEqual(MaskedViewConfig('patch_dropout', 0.9, 0).name, 'abmil_patch_dropout')

    def test_mask_counts_padding_singletons_and_input_preservation(self):
        mask = torch.tensor([
            [True] * 7, [True, False, True, False, True, False, True],
            [False, False, True, False, False, False, False],
        ])
        original = mask.clone()
        sampled = sample_retained_mask(mask, 0.5, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(sampled.sum(dim=1), torch.tensor([3, 2, 1])))
        self.assertFalse((sampled & ~mask).any())
        self.assertTrue(torch.equal(mask, original))
        self.assertTrue(torch.equal(
            sample_retained_mask(mask, 1, torch.Generator().manual_seed(42)), mask,
        ))
        for invalid in [mask.float(), mask[0], torch.zeros_like(mask), mask[:0]]:
            with self.assertRaises(ValueError):
                sample_retained_mask(invalid, .9, torch.Generator())

    def test_sampling_is_reproducible_and_does_not_consume_global_rng(self):
        mask = torch.ones(2, 40, dtype=torch.bool)
        state = torch.get_rng_state().clone()
        samples = [
            sample_retained_mask(mask, .5, torch.Generator().manual_seed(seed))
            for seed in [42, 43, 42]
        ]
        self.assertTrue(torch.equal(samples[0], samples[2]))
        self.assertFalse(torch.equal(samples[0], samples[1]))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_kl_direction_reduction_and_detached_teacher(self):
        p = torch.tensor([[.8, .2], [.1, .9]], dtype=torch.float64)
        q = torch.tensor([[.6, .4], [.3, .7]], dtype=torch.float64)
        full = p.log().requires_grad_()
        masked = q.log().requires_grad_()
        loss = prediction_consistency_loss(full, masked)
        expected = (p * (p.log() - q.log())).sum(dim=1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(full.grad)
        self.assertTrue(torch.isfinite(masked.grad).all())
        self.assertGreater(masked.grad.abs().sum().item(), 0)
        extreme = torch.tensor([[1000., -1000.]])
        self.assertTrue(torch.isfinite(prediction_consistency_loss(extreme, -extreme)))
        self.assertAlmostEqual(prediction_consistency_loss(extreme, extreme).item(), 0.)
        with self.assertRaises(ValueError):
            prediction_consistency_loss(torch.ones(2, 3), torch.ones(2, 2))

    def test_sequential_backprop_matches_joint_objective(self):
        actual = RecordingModel(copy.deepcopy(self.model))
        expected = copy.deepcopy(self.model)
        options = MaskedViewConfig(keep_fraction=.6, consistency_weight=.7)
        actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=.001, weight_decay=1e-5)
        expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=.001, weight_decay=1e-5)
        retained = sample_retained_mask(
            self.batch['mask'], .6, torch.Generator().manual_seed(42),
        )
        full = expected(self.batch['input'], mask=self.batch['mask'])
        masked = expected(self.batch['input'], mask=retained)
        ce = self.criterion(full, self.batch['label'])
        kl = prediction_consistency_loss(full, masked)
        (ce + .7 * kl).backward()
        expected_optimizer.step()
        with patch.object(actual_optimizer, 'step', wraps=actual_optimizer.step) as step:
            metrics = train_masked_epoch(
                actual, self.criterion, [self.batch], actual_optimizer, 'cpu', progress=False,
                extension=options, generator=torch.Generator().manual_seed(42),
            )
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(actual.masks), 2)
        self.assertTrue(torch.equal(actual.masks[0], self.batch['mask']))
        self.assertTrue(torch.equal(actual.masks[1], retained))
        for actual_parameter, expected_parameter in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(metrics['loss'], metrics['ce_loss'] + .7 * metrics['consistency_loss'])
        self.assertAlmostEqual(metrics['retained_fraction'], 6 / 11)

    def test_zero_weight_matches_baseline_updates_and_preserves_mask_rng(self):
        baseline = copy.deepcopy(self.model)
        extended = RecordingModel(copy.deepcopy(self.model))
        original_train, _, _ = training_functions(self.notebook)
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
            for model in [baseline, extended]
        ]
        batches = [self.batch, self.batch]
        original = original_train(baseline, self.criterion, batches, optimizers[0],
                                  'cpu', progress=False)
        generator = torch.Generator().manual_seed(42)
        before = generator.get_state().clone()
        actual = train_masked_epoch(
            extended, self.criterion, batches, optimizers[1], 'cpu', progress=False,
            extension=MaskedViewConfig(consistency_weight=0), generator=generator,
        )
        for expected_parameter, actual_parameter in zip(baseline.parameters(), extended.parameters()):
            self.assertTrue(torch.equal(expected_parameter, actual_parameter))
        self.assertEqual(actual['loss'], original['loss'])
        self.assertEqual(actual['acc'], original['acc'])
        self.assertEqual(actual['consistency_loss'], 0)
        self.assertEqual(actual['retained_fraction'], 1)
        self.assertEqual(len(extended.masks), len(batches))
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_patch_dropout_uses_only_masked_ce_and_one_forward(self):
        model = RecordingModel(self.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        options = MaskedViewConfig('patch_dropout', .5, 0)
        with patch('masked_view_training.prediction_consistency_loss') as consistency:
            metrics = train_masked_epoch(
                model, self.criterion, [self.batch], optimizer, 'cpu', progress=False,
                extension=options, generator=torch.Generator().manual_seed(42),
            )
        consistency.assert_not_called()
        self.assertEqual(len(model.masks), 1)
        self.assertTrue(torch.equal(model.masks[0].sum(dim=1), torch.tensor([3, 2])))
        self.assertEqual(metrics['loss'], metrics['ce_loss'])
        self.assertEqual(metrics['consistency_loss'], 0)

    def test_empty_loader_and_nonfinite_training_fail(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        kwargs = dict(progress=False, extension=MaskedViewConfig(),
                      generator=torch.Generator().manual_seed(42))
        with self.assertRaisesRegex(ValueError, 'empty'):
            train_masked_epoch(self.model, self.criterion, [], optimizer, 'cpu', **kwargs)
        bad = {**self.batch, 'input': torch.full_like(self.batch['input'], float('nan'))}
        with self.assertRaises(FloatingPointError):
            train_masked_epoch(self.model, self.criterion, [bad], optimizer, 'cpu', **kwargs)


class MaskedViewWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.notebook = ROOT / 'ECE1512_ProjectA_Working.ipynb'
        self.reference = self.root / 'reference'
        _, _, cells = training_functions(self.notebook)
        self.source_hash = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()

    def make_reference(self, key='C16', classes=2, dataset='CAMELYON16'):
        feature = self.root / f'feature/{key}/patch_feats_pretrain_medical_ssl.h5'
        feature.parent.mkdir(parents=True)
        splits = {
            split: [f'{split.lower()}_{label}' for label in range(classes)]
            for split in ['Train', 'Validation', 'Test']
        }
        rows = []
        with h5py.File(feature, 'w') as h5:
            for index, (split, names) in enumerate(splits.items()):
                for label, name in enumerate(names):
                    rng = np.random.default_rng(index * 100 + label)
                    size = 5 + label
                    group = h5.create_group(name)
                    group.create_dataset('feat', data=(
                        rng.normal(size=(size, 3)) * .6 + label
                    ).astype(np.float32))
                    group.create_dataset('coords', data=np.arange(size * 2).reshape(size, 2))
                    group.attrs['label'] = label
                    rows.append({'slide_id': name, 'case_id': f'{split}_{label}'})
        if key == 'C17':
            pd.DataFrame(rows).to_csv(self.root / 'camelyon17.csv', index=False)
        conf = SimpleNamespace(
            dataset=dataset, dataset_key=key, n_class=classes,
            D_feat=3, D_inner=4, D_attn=2, B=1, seed=42, n_shot=0,
            lr=1e-4, wd=1e-5, droprate=0., train_epoch=30, epochs=30,
            optimizer='AdamW', loss='CrossEntropyLoss', scheduler=None,
            run_mode='baseline', data_dir=str(feature), dataset_csv_dir=str(self.root),
        )
        folder = self.reference / key
        folder.mkdir(parents=True)
        (folder / 'config.json').write_text(json.dumps(vars(conf)))
        (folder / 'splits.json').write_text(json.dumps(splits))
        (folder / 'source_snapshot.json').write_text(json.dumps({
            'notebook_definitions_sha256': self.source_hash,
        }))
        return conf, splits

    def test_complete_consistency_study_and_tamper_checks(self):
        conf, splits = self.make_reference()
        prepared, source_hash = prepare_inputs(
            self.reference, ['C16'], self.notebook, self.root / 'feature', self.root,
        )
        options = MaskedViewConfig()
        output = self.root / 'consistency'
        with contextlib.redirect_stdout(io.StringIO()):
            rows = run_study(
                self.reference, output, seeds=[42, 43], keys=['C16'], device='cpu', dpi=30,
                notebook=self.notebook, feature_root=self.root / 'feature', metadata=self.root,
                extension=options,
            )
        self.assertEqual([row['method'] for row in rows], ['abmil_consistency'] * 2)
        self.assertEqual(json.loads((output / 'study_status.json').read_text())['status'], 'complete')
        protocol = json.loads((output / 'protocol.json').read_text())
        self.assertEqual(protocol['training_extension'], options.to_dict())
        self.assertEqual(protocol['baseline_settings']['C16']['lr'], conf.lr)
        summary = (output / 'summary.md').read_text()
        self.assertIn('masked-view consistency', summary)
        self.assertIn('sample standard deviation', summary)
        self.assertIn('full bags', summary)
        self.assertIn('abmil_consistency', pd.read_csv(output / 'per_seed_results.csv').method.tolist())
        for seed in [42, 43]:
            folder = output / f'seed_{seed}/C16'
            actual = json.loads((folder / 'config.json').read_text())
            self.assertEqual(actual['run_mode'], 'extension')
            self.assertEqual(actual['mask_rng_seed'], seed)
            self.assertEqual(actual['split_seed'], 42)
            self.assertEqual(json.loads((folder / 'splits.json').read_text()), splits)
            verification = json.loads((folder / 'verification.json').read_text())
            self.assertTrue(verification['training_objective_verified'])
            self.assertTrue(verification['full_bag_evaluation'])
            history = pd.read_csv(folder / 'history.csv')
            self.assertEqual(len(history), 30)
            self.assertGreater(history.train_consistency_loss.max(), 0)
            self.assertTrue((history.train_retained_fraction < 1).all())
            self.assertTrue((folder / 'masked_view_curves.png').is_file())
            self.assertIn('masked-view consistency', (folder / 'index.html').read_text())
        folder = output / 'seed_42/C16'
        config_path = folder / 'config.json'
        original = config_path.read_text()
        actual = json.loads(original)
        actual['training_extension']['consistency_weight'] = 2.
        config_path.write_text(json.dumps(actual))
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            verify_run(folder, prepared['C16'][0], 42, source_hash, options)
        config_path.write_text(original)
        snapshot_path = folder / 'source_snapshot.json'
        snapshot = json.loads(snapshot_path.read_text())
        snapshot['masked_view_training_source'] += '\n# tampered\n'
        snapshot_path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, 'training source'):
            verify_run(folder, prepared['C16'][0], 42, source_hash, options)
        with self.assertRaises(FileExistsError):
            run_study(self.reference, output, seeds=[42, 43], keys=['C16'],
                      device='cpu', extension=options)

    def test_two_four_and_seven_class_smoke_runs_keep_full_bag_evaluation(self):
        for key, classes, dataset in [
            ('C16', 2, 'CAMELYON16'), ('C17', 4, 'CAMELYON17'), ('BRACS', 7, 'BRACS'),
        ]:
            with self.subTest(key=key):
                conf, splits = self.make_reference(key, classes, dataset)
                datasets = load_fixed_datasets(conf, self.reference / key / 'splits.json')
                output = self.root / 'smoke' / key
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_dataset(
                        key, conf, datasets, output, self.notebook, epochs=1, dpi=30,
                        extension=MaskedViewConfig(),
                    )
                self.assertEqual(result['run_mode'], 'smoke')
                self.assertEqual(result['validation_pngs'], classes)
                self.assertEqual(len(pd.read_csv(output / 'test_predictions.csv')), classes)
                # Recompute full-bag CE from the actual selected model.
                saved = torch.load(output / 'best_model.pt', weights_only=True)
                model = load_notebook_model(SimpleNamespace(**saved['config']), self.notebook)
                model.load_state_dict(saved['model_state_dict'])
                _, evaluate, _ = training_functions(self.notebook)
                loader = DataLoader(datasets[2], batch_size=1, collate_fn=mil_collate_fn)
                metrics = evaluate(model, nn.CrossEntropyLoss(), loader, 'cpu', conf)
                self.assertAlmostEqual(metrics['loss'], result['test_loss'])
                self.assertEqual(json.loads((output / 'splits.json').read_text()), splits)
                self.assertFalse(hasattr(conf, 'training_extension'))

    def test_patch_dropout_study_is_labeled_and_verified(self):
        self.make_reference()
        output = self.root / 'dropout'
        with contextlib.redirect_stdout(io.StringIO()):
            run_study(
                self.reference, output, seeds=[42, 43], keys=['C16'], device='cpu', dpi=30,
                feature_root=self.root / 'feature', metadata=self.root,
                extension=MaskedViewConfig('patch_dropout', .9, 0),
            )
        summary = (output / 'summary.md').read_text()
        self.assertIn('ABMIL patch dropout', summary)
        history = pd.read_csv(output / 'seed_42/C16/history.csv')
        self.assertTrue((history.train_consistency_loss == 0).all())
        self.assertTrue((history.train_retained_fraction < 1).all())

    def test_cli_defaults_comparator_and_invalid_settings(self):
        arguments = ['--reference-suite', str(self.reference),
                     '--output', str(self.root / 'output')]
        with patch('run_consistency.run_study') as study:
            main(arguments)
            self.assertEqual(study.call_args.kwargs['extension'], MaskedViewConfig())
            main(arguments + ['--method', 'patch_dropout'])
            self.assertEqual(study.call_args.kwargs['extension'],
                             MaskedViewConfig('patch_dropout', .9, 0))
        for extra in [['--keep-fraction', '0'], ['--consistency-weight', 'nan'],
                      ['--method', 'patch_dropout', '--consistency-weight', '1']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments + extra)
        self.assertFalse((self.root / 'output').exists())

    def test_extension_controller_failure_is_recorded(self):
        self.make_reference()
        output = self.root / 'failed'
        with patch('run_multiseed.run_dataset', side_effect=RuntimeError('synthetic failure')):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                run_study(
                    self.reference, output, seeds=[42, 43], keys=['C16'], device='cpu',
                    feature_root=self.root / 'feature', metadata=self.root,
                    extension=MaskedViewConfig(),
                )
        status = json.loads((output / 'study_status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['completed_runs'], [])
        self.assertEqual(status['method'], 'abmil_consistency')


if __name__ == '__main__':
    unittest.main(verbosity=2)
