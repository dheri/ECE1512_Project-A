"""Spatial graphs, residual context, export parity, and fixed-split study checks."""

import contextlib
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader import HDF5FeatureDataset, mil_collate_fn
from export_attention import export, load_notebook_model, parser as export_parser
from run_baselines import load_fixed_datasets, run_dataset
from run_multiseed import prepare_inputs, run_study, verify_run
from run_spatial import main
from spatial_context import (
    SpatialContextABMIL, SpatialContextConfig, SpatialNeighborCache, build_neighbors,
    evaluate_spatial, train_spatial_epoch,
)
import test_masked_view as fixtures


class SpatialUnitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.notebook = ROOT / 'ECE1512_ProjectA_Working.ipynb'
        self.conf = SimpleNamespace(D_feat=3, D_inner=4, D_attn=2, n_class=3,
                                    droprate=0., seed=42)
        self.coords = torch.tensor([[x, y] for x in range(3) for y in range(3)])
        self.bag = torch.randn(1, 9, 3)

    def model(self, options=None, nonzero=False):
        baseline = load_notebook_model(self.conf, self.notebook)
        model = SpatialContextABMIL(baseline, options or SpatialContextConfig(neighbors=3))
        if nonzero:
            with torch.no_grad():
                model.context[-1].weight.normal_(0, .4)
                model.context[-1].bias.normal_(0, .1)
        return model

    def test_configuration_and_saved_metadata_validation(self):
        for kwargs in [
            {'neighbors': 0}, {'neighbors': 1.5}, {'neighbors': True},
            {'coordinate_mode': 'other'}, {'shuffle_seed': -1},
            {'shuffle_seed': 2**32}, {'cache_entries': -1},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpatialContextConfig(**kwargs)
        for mode in ['real', 'shuffled']:
            options = SpatialContextConfig(coordinate_mode=mode)
            self.assertEqual(SpatialContextConfig.from_dict(options.to_dict()), options)
        saved = SpatialContextConfig().to_dict()
        saved['context'] = 'changed'
        with self.assertRaises(ValueError):
            SpatialContextConfig.from_dict(saved)

    def test_neighbors_match_bruteforce_without_self(self):
        coords = np.array([[0., 0.], [1., .2], [4., .3], [10., .8], [18., .1]])
        actual = build_neighbors(coords, SpatialContextConfig(neighbors=2))
        distances = ((coords[:, None] - coords[None])**2).sum(axis=-1)
        np.fill_diagonal(distances, np.inf)
        expected = np.argsort(distances, axis=1)[:, :2]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(build_neighbors(coords[:2], SpatialContextConfig()).shape, (2, 1))
        self.assertEqual(build_neighbors(coords[:1], SpatialContextConfig()).shape, (1, 0))

    def test_invalid_missing_duplicate_and_nonfinite_coordinates(self):
        options = SpatialContextConfig()
        for coords in [None, [], np.ones((3, 3)), [[0, float('nan')]],
                       [[float('inf'), 1]], [[0, 0], [0, 0]]]:
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                build_neighbors(coords, options)
        with self.assertRaisesRegex(ValueError, 'slide ID'):
            build_neighbors(self.coords, SpatialContextConfig(coordinate_mode='shuffled'))

    def test_graph_equivariance_including_distance_ties(self):
        permutation = np.array([8, 2, 5, 0, 7, 3, 1, 6, 4])
        coords = self.coords.numpy()
        for mode in ['real', 'shuffled']:
            options = SpatialContextConfig(neighbors=3, coordinate_mode=mode)
            original = build_neighbors(coords, options, 'slide_a')
            reordered = build_neighbors(coords[permutation], options, 'slide_a')
            np.testing.assert_array_equal(permutation[reordered], original[permutation])

    def test_shuffle_is_fixed_per_slide_seed_and_does_not_consume_rng(self):
        coords = np.arange(60).reshape(30, 2)
        options = SpatialContextConfig(neighbors=3, coordinate_mode='shuffled')
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        first = build_neighbors(coords, options, 'slide_a')
        np.testing.assert_array_equal(first, build_neighbors(coords, options, 'slide_a'))
        self.assertFalse(np.array_equal(first, build_neighbors(coords, options, 'slide_b')))
        self.assertFalse(np.array_equal(first, build_neighbors(
            coords, SpatialContextConfig(3, 'shuffled', 43), 'slide_a',
        )))
        after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(numpy_state, after)))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_disk_cache_lru_readonly_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'graphs'
            options = SpatialContextConfig(neighbors=3, cache_entries=1)
            cache = SpatialNeighborCache(options, directory, writable=True)
            original = cache.get(self.coords, 'a')
            self.assertTrue(torch.equal(original, cache.get(self.coords, 'a')))
            self.assertEqual(cache.stats()['memory_hits'], 1)
            cache.get(self.coords, 'b')
            self.assertEqual(cache.stats()['memory_entries'], 1)
            reread = SpatialNeighborCache(options, directory, writable=False)
            self.assertTrue(torch.equal(original, reread.get(self.coords, 'a')))
            self.assertEqual(reread.stats()['disk_hits'], 1)
            missing = Path(temporary) / 'readonly_missing'
            SpatialNeighborCache(options, missing).get(self.coords, 'c')
            self.assertFalse(missing.exists())
            for path in directory.glob('*.npy'):
                broken = np.load(path, allow_pickle=False)
                broken[0, 0] = 0
                np.save(path, broken, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'neighbor indices'):
                SpatialNeighborCache(options, directory).get(self.coords, 'a')

    def test_large_graph_has_only_n_times_k_indices(self):
        coords = np.column_stack([np.arange(10000), np.arange(10000) % 37])
        graph = build_neighbors(coords, SpatialContextConfig())
        self.assertEqual(graph.shape, (10000, 8))
        self.assertEqual(graph.nbytes, 10000 * 8 * 8)

    def test_identity_initialization_preserves_baseline_and_global_rng(self):
        baseline = load_notebook_model(self.conf, self.notebook)
        state = torch.get_rng_state().clone()
        model = SpatialContextABMIL(copy.deepcopy(baseline), SpatialContextConfig())
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        baseline_logits, baseline_attention = baseline(self.bag, return_attention=True)
        logits, attention = model(self.bag, coords=[self.coords], return_attention=True)
        self.assertTrue(torch.equal(logits, baseline_logits))
        self.assertTrue(torch.equal(attention, baseline_attention))
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(extra, 3 * self.conf.D_inner**2 + 2 * self.conf.D_inner)

    def test_inference_warmed_cache_can_be_used_for_training(self):
        model = self.model(nonzero=True)
        with torch.inference_mode():
            model(self.bag, coords=[self.coords])
        loss = model(self.bag, coords=[self.coords]).square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(model.context[-1].weight.grad).all())
        self.assertEqual(model.neighbor_cache.stats()['memory_hits'], 1)

    def test_context_uses_coordinates_and_respects_patch_permutation(self):
        permutation = torch.tensor([8, 2, 5, 0, 7, 3, 1, 6, 4])
        for mode in ['real', 'shuffled']:
            model = self.model(SpatialContextConfig(3, mode), nonzero=True)
            original = model(self.bag, coords=[self.coords], slide_ids=['slide_a'])
            reordered = model(self.bag[:, permutation], coords=[self.coords[permutation]],
                              slide_ids=['slide_a'])
            torch.testing.assert_close(original, reordered, rtol=1e-5, atol=1e-6)
        model = self.model(nonzero=True)
        original = model(self.bag, coords=[self.coords])
        moved = model(self.bag, coords=[self.coords[permutation]])
        self.assertFalse(torch.allclose(original, moved, atol=1e-6))

    def test_padding_singletons_and_coordinate_shape_checks(self):
        model = self.model(nonzero=True)
        padded = torch.cat([self.bag, torch.randn_like(self.bag)], dim=0)
        mask = torch.tensor([[True] * 9, [True] * 4 + [False] * 5])
        padded[1, 4:] = 10000
        logits, attention = model(
            padded, mask=mask, coords=[self.coords, self.coords[:4]], return_attention=True,
        )
        expected = torch.cat([
            model(padded[:1], coords=[self.coords]),
            model(padded[1:2, :4], coords=[self.coords[:4]]),
        ])
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(attention[1, :, 4:], torch.zeros_like(attention[1, :, 4:])))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 1))
        self.assertTrue(torch.isfinite(model(self.bag[:, :1], coords=[self.coords[:1]])).all())
        for kwargs in [
            {}, {'coords': [None]}, {'coords': [self.coords[:4]]},
            {'coords': [self.coords], 'mask': torch.zeros(1, 9, dtype=torch.bool)},
            {'coords': [self.coords], 'mask': torch.ones(1, 9)},
            {'coords': [self.coords], 'slide_ids': []},
        ]:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                model(self.bag, **kwargs)

    def test_gradient_flow_and_empty_loader_errors(self):
        model = self.model(nonzero=True)
        coords = self.coords.float().requires_grad_()
        batch = {
            'input': self.bag, 'mask': torch.ones(1, 9, dtype=torch.bool),
            'label': torch.tensor([1]), 'coords': [coords], 'slide_id': ['slide_a'],
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        original = model.context[0].weight.detach().clone()
        metrics = train_spatial_epoch(model, nn.CrossEntropyLoss(), [batch], optimizer,
                                      'cpu', progress=False)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertFalse(torch.equal(original, model.context[0].weight))
        self.assertIsNone(coords.grad)
        for layer in [model.dimreduction.fc, model.context[0], model.context[-1],
                      model.attention.attention_weights_w, model.classifier.fc]:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        evaluation_batches = [
            {**batch, 'label': torch.tensor([label]), 'slide_id': [f'slide_{label}']}
            for label in range(3)
        ]
        result, probabilities, labels, names = evaluate_spatial(
            model, nn.CrossEntropyLoss(), evaluation_batches, 'cpu', self.conf, return_predictions=True,
        )
        self.assertEqual(names, ['slide_0', 'slide_1', 'slide_2'])
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()))
        for function, args in [
            (train_spatial_epoch, (model, nn.CrossEntropyLoss(), [], optimizer, 'cpu')),
            (evaluate_spatial, (model, nn.CrossEntropyLoss(), [], 'cpu', self.conf)),
        ]:
            with self.assertRaisesRegex(ValueError, 'empty'):
                function(*args, **({'progress': False} if function is train_spatial_epoch else {}))


class SpatialWorkflowTests(unittest.TestCase):
    # Reuse only the synthetic HDF5/reference setup, not the masked-view test cases.
    setUp = fixtures.MaskedViewWorkflowTests.setUp
    make_reference = fixtures.MaskedViewWorkflowTests.make_reference

    def test_complete_studies_for_real_and_shuffled_modes(self):
        self.make_reference()
        prepared, source_hash = prepare_inputs(
            self.reference, ['C16'], self.notebook, self.root / 'feature', self.root,
        )
        for mode in ['real', 'shuffled']:
            with self.subTest(mode=mode):
                options = SpatialContextConfig(neighbors=2, coordinate_mode=mode)
                output = self.root / mode
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = run_study(
                        self.reference, output, seeds=[42, 43], keys=['C16'], device='cpu', dpi=30,
                        feature_root=self.root / 'feature', metadata=self.root, extension=options,
                    )
                self.assertEqual([row['method'] for row in rows], [options.name] * 2)
                self.assertEqual(json.loads((output / 'study_status.json').read_text())['status'], 'complete')
                self.assertIn('Spatial context extension', (output / 'summary.md').read_text())
                self.assertIn('coordinate mode', (output / 'summary.md').read_text())
                for seed in [42, 43]:
                    folder = output / f'seed_{seed}/C16'
                    config = json.loads((folder / 'config.json').read_text())
                    self.assertNotIn('mask_rng_seed', config)
                    self.assertEqual(config['loss'], 'CrossEntropyLoss')
                    self.assertEqual(config['run_mode'], 'extension')
                    history = pd.read_csv(folder / 'history.csv')
                    self.assertEqual(len(history), 30)
                    self.assertNotIn('train_consistency_loss', history.columns)
                    self.assertFalse((folder / 'masked_view_curves.png').exists())
                    self.assertNotIn('masked_view_curves.png', (folder / 'index.html').read_text())
                    check = json.loads((folder / 'verification.json').read_text())
                    self.assertTrue(check['spatial_export_configuration_verified'])
                    self.assertTrue(check['resource_profile_verified'])
                    self.assertEqual(len(list((folder / 'spatial_neighbors').glob('*.npy'))), 6)
                    # Fresh export must reproduce probabilities using the saved mode.
                    args = export_parser().parse_args([
                        '--checkpoint', str(folder / 'best_model.pt'), '--features',
                        prepared['C16'][0].data_dir, '--output', str(folder / 'reexport'), '--dpi', '30',
                    ])
                    with contextlib.redirect_stdout(io.StringIO()):
                        export(args)
                    original = pd.read_csv(folder / 'attention_panels/summary.csv')
                    repeated = pd.read_csv(folder / 'reexport/summary.csv')
                    np.testing.assert_allclose(original.p_class_0, repeated.p_class_0, atol=1e-7)
                folder = output / 'seed_42/C16'
                path = folder / 'config.json'
                original = path.read_text()
                changed = json.loads(original)
                changed['training_extension']['coordinate_mode'] = 'other'
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    verify_run(folder, prepared['C16'][0], 42, source_hash, options)
                path.write_text(original)
                snapshot_path = folder / 'source_snapshot.json'
                snapshot = json.loads(snapshot_path.read_text())
                snapshot['spatial_context_source'] += '\n# tampered\n'
                snapshot_path.write_text(json.dumps(snapshot))
                with self.assertRaisesRegex(ValueError, 'Spatial context source'):
                    verify_run(folder, prepared['C16'][0], 42, source_hash, options)
                with self.assertRaises(FileExistsError):
                    run_study(self.reference, output, seeds=[42, 43], keys=['C16'],
                              device='cpu', extension=options)

    def test_two_four_and_seven_class_workflows(self):
        for key, classes, dataset in [
            ('C16', 2, 'CAMELYON16'), ('C17', 4, 'CAMELYON17'), ('BRACS', 7, 'BRACS'),
        ]:
            with self.subTest(key=key):
                conf, splits = self.make_reference(key, classes, dataset)
                datasets = load_fixed_datasets(conf, self.reference / key / 'splits.json')
                folder = self.root / 'smoke' / key
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_dataset(
                        key, conf, datasets, folder, self.notebook, epochs=1, dpi=30,
                        extension=SpatialContextConfig(),
                    )
                self.assertEqual(result['run_mode'], 'smoke')
                self.assertEqual(result['validation_pngs'], classes)
                self.assertEqual(json.loads((folder / 'splits.json').read_text()), splits)
                self.assertFalse(hasattr(conf, 'training_extension'))

    def test_cli_configuration_and_invalid_settings(self):
        args = ['--reference-suite', str(self.reference), '--output', str(self.root / 'new')]
        with patch('run_spatial.run_study') as study:
            main(args)
            self.assertEqual(study.call_args.kwargs['extension'], SpatialContextConfig())
            main(args + ['--coordinate-mode', 'shuffled', '--neighbors', '3'])
            self.assertEqual(study.call_args.kwargs['extension'],
                             SpatialContextConfig(neighbors=3, coordinate_mode='shuffled'))
        for extra in [['--neighbors', '0'], ['--shuffle-seed', '-1'], ['--cache-entries', '-1']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + extra)
        self.assertFalse((self.root / 'new').exists())

    def test_duplicate_coordinates_fail_with_recorded_status(self):
        conf, _ = self.make_reference()
        datasets = load_fixed_datasets(conf, self.reference / 'C16/splits.json')
        with h5py.File(conf.data_dir, 'r+') as h5:
            group = h5[datasets[0].data_names[0]]
            coords = group['coords'][:]
            coords[0] = coords[1]
            group['coords'][:] = coords
        output = self.root / 'bad'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Duplicate'):
            run_dataset('C16', conf, datasets, output, self.notebook, epochs=1, dpi=30,
                        extension=SpatialContextConfig())
        self.assertEqual(json.loads((output / 'status.json').read_text())['stage'], 'failed')

    def test_real_training_bag_forward_for_all_datasets(self):
        if not all((ROOT / 'feature' / key / 'patch_feats_pretrain_medical_ssl.h5').is_file()
                   for key in ['C16', 'C17', 'BRACS']):
            self.skipTest('Optional course feature files are not bundled; see DATA.md.')
        for key in ['C16', 'C17', 'BRACS']:
            with self.subTest(key=key):
                reference = ROOT / 'reference_splits' / key
                config = json.loads((reference / 'config.json').read_text())
                config['data_dir'] = str(ROOT / 'feature' / key / 'patch_feats_pretrain_medical_ssl.h5')
                config['dataset_csv_dir'] = str(ROOT / 'dataset_csv')
                options = SpatialContextConfig()
                config.update(method=options.name, training_extension=options.to_dict())
                conf = SimpleNamespace(**config)
                names = json.loads((reference / 'splits.json').read_text())['Train']
                item = HDF5FeatureDataset(conf.data_dir, names[:1], conf)[0]
                batch = mil_collate_fn([item])
                model = load_notebook_model(conf, self.notebook)
                with torch.inference_mode():
                    logits, attention = model(
                        batch['input'], mask=batch['mask'], coords=batch['coords'],
                        slide_ids=batch['slide_id'], return_attention=True,
                    )
                self.assertEqual(logits.shape, (1, conf.n_class))
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(attention.sum(), torch.tensor(1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
