"""Check the packaged scientific evidence without course features or new training."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import classification_metrics

METHODS = ["baseline", "consistency90", "spatial_gated"]
KEYS = ["C16", "C17", "BRACS"]
METRICS = ["test_accuracy_pct", "test_macro_auroc", "test_macro_f1", "test_loss"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(splits):
    return hashlib.sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    torch.set_num_threads(2)
    primary = ROOT / "results/primary"
    grid = pd.read_csv(primary / "all_runs.csv")
    expected = set(itertools.product(METHODS, range(42, 52), KEYS))
    found = list(zip(grid.variant, grid.training_seed, grid.dataset_key))
    assert len(found) == len(set(found)) == 90 and set(found) == expected
    comparisons = pd.read_csv(primary / "paired_comparison.csv")
    assert len(comparisons) == 60
    splits = {}
    for key in KEYS:
        ref = ROOT / "reference_splits" / key
        splits[key] = json.loads((ref / "splits.json").read_text())
        config = json.loads((ref / "config.json").read_text())
        assert fingerprint(splits[key]) == config["split_sha256"]
        flattened = sum(splits[key].values(), [])
        assert len(flattened) == len(set(flattened))
    metadata = pd.read_csv(ROOT / "dataset_csv/camelyon17.csv").set_index("slide_id")
    patients = [set(metadata.loc[splits["C17"][name], "case_id"]) for name in splits["C17"]]
    assert [len(p) for p in patients] == [54, 6, 40]
    assert all(not a & b for a, b in itertools.combinations(patients, 2))
    notebook = nbformat.read(ROOT / "ECE1512_ProjectA_Working.ipynb", as_version=4)
    nbformat.validate(notebook)
    cells = {name: next(c.source for c in notebook.cells if c.id == name)
             for name in ["de7504f3", "4b2b81fd", "a8146687"]}
    source_hash = hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()
    provenance = json.loads((ROOT / "provenance/artifact_sources.json").read_text())
    assert source_hash == provenance["notebook_definitions_sha256"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"notebook:{cell.id}", "exec")
            assert not cell.outputs and cell.execution_count is None and "TODO(" not in cell.source
    for filename, expected_hash in provenance["source_files"].items():
        assert sha(ROOT / filename) == expected_hash, filename
    for filename, record in provenance["primary_artifacts"].items():
        assert sha(primary / filename) == record["packaged_sha256"], filename

    max_metric_error = 0.
    for row in grid.itertuples(index=False):
        folder = primary / row.variant / f"seed_{row.training_seed}" / row.dataset_key
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        history = pd.read_csv(folder / "history.csv")
        assert list(history.epoch) == list(range(1, 31))
        assert int(np.argmax(history.val_auc + history.val_f1)) + 1 == row.best_epoch == result["best_epoch"]
        assert config["split_sha256"] == fingerprint(splits[row.dataset_key])
        for split, name in [("test", "Test"), ("validation", "Validation")]:
            frame = pd.read_csv(folder / f"{split}_predictions.csv")
            assert list(frame.slide_id) == splits[row.dataset_key][name]
            probabilities = frame[[f"p_class_{i}" for i in range(row.classes)]].to_numpy(dtype=np.float32, copy=True)
            labels = frame.label.to_numpy(dtype=np.int64, copy=True)
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-5)
            np.testing.assert_array_equal(probabilities.argmax(1), frame.prediction)
            values = classification_metrics(torch.from_numpy(probabilities), torch.from_numpy(labels), row.classes)
            recorded = json.loads((folder / f"{split}_metrics.json").read_text())
            for metric in ["acc", "auc", "f1"]:
                error = abs(values[metric] - recorded[metric])
                max_metric_error = max(max_metric_error, error)
                assert error <= 1e-5, (folder, split, metric, error)
            ce = -np.log(probabilities[np.arange(len(labels)), labels].astype(float)).mean()
            assert abs(ce - recorded["loss"]) <= 1e-5
            if split == "test":
                for metric in METRICS:
                    assert abs(getattr(row, metric) - result[metric]) < 1e-12
                for ours, metric in [("acc", "test_accuracy_pct"), ("auc", "test_macro_auroc"),
                                     ("f1", "test_macro_f1"), ("loss", "test_loss")]:
                    assert abs(recorded[ours] - result[metric]) < 1e-5
    for method in METHODS:
        summary = pd.read_csv(primary / method / "summary.csv")
        for row in summary.itertuples(index=False):
            values = grid[(grid.variant == method) & (grid.dataset_key == row.dataset_key)]
            for metric in METRICS:
                assert np.isclose(values[metric].mean(), getattr(row, metric + "_mean"), atol=1e-12)
                assert np.isclose(values[metric].std(ddof=1), getattr(row, metric + "_std"), atol=1e-12)
    indexed = grid.set_index(["variant", "training_seed", "dataset_key"])
    for row in comparisons.itertuples(index=False):
        for metric in METRICS[:3]:
            delta = indexed.loc[(row.variant, row.training_seed, row.dataset_key), metric] - \
                    indexed.loc[("baseline", row.training_seed, row.dataset_key), metric]
            assert abs(delta - getattr(row, "delta_" + metric)) < 1e-12
    historical = pd.read_csv(ROOT / "results/historical/all_runs.csv")
    assert len(historical) == len(historical.drop_duplicates(["arm", "training_seed", "dataset_key"])) == 108
    for row in historical.itertuples(index=False):
        folder = ROOT / "results/historical" / row.arm / f"seed_{row.training_seed}" / row.dataset_key
        frame = pd.read_csv(folder / "test_predictions.csv")
        assert list(frame.slide_id) == splits[row.dataset_key]["Test"]
        assert abs(100 * (frame.label == frame.prediction).mean() - row.test_accuracy_pct) < 1e-5
        classes = {"C16": 2, "C17": 4, "BRACS": 7}[row.dataset_key]
        probabilities = frame[[f"p_class_{i}" for i in range(classes)]].to_numpy(dtype=np.float32, copy=True)
        labels = frame.label.to_numpy(dtype=np.int64, copy=True)
        values = classification_metrics(torch.from_numpy(probabilities), torch.from_numpy(labels), classes)
        for ours, metric in [("acc", "test_accuracy_pct"), ("auc", "test_macro_auroc"),
                             ("f1", "test_macro_f1")]:
            assert abs(values[ours] - getattr(row, metric)) <= 1e-5
        ce = -np.log(probabilities[np.arange(len(labels)), labels].astype(float)).mean()
        assert abs(ce - row.test_loss) <= 1e-5
    examples = json.loads((ROOT / "report/attention_selection.json").read_text())["examples"]
    assert len(examples) == 6
    for example in examples:
        image = ROOT / "report/figures" / example["figure"]
        assert sha(image) == example["source_sha256"]
        with Image.open(image) as im:
            im.verify()
        frame = pd.read_csv(primary / example["method"] / "seed_42" / example["dataset"] /
                            "validation_predictions.csv")
        first = frame[frame.label == int(example["label"])].iloc[0]
        assert first.slide_id == example["slide_id"]
    checkpoint_manifest = ROOT / "checkpoints/manifest.json"
    checkpoints = json.loads(checkpoint_manifest.read_text())
    assert len(checkpoints) == 9
    for name, record in checkpoints.items():
        assert sha(ROOT / "checkpoints" / name) == record["packaged_sha256"]
        checkpoint = torch.load(ROOT / "checkpoints" / name, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == record["selected_epoch"]
        assert all(torch.isfinite(t).all() for t in checkpoint["model_state_dict"].values())
    report = {
        "status": "passed", "primary_runs": 90, "primary_completed_epochs": 2700,
        "historical_supporting_runs": 108, "paired_rows": 60,
        "patient_counts_C17": [54, 6, 40], "C17_patient_overlap": 0,
        "prediction_metric_sets_recomputed": 180, "representative_checkpoints": 9,
        "historical_test_metric_sets_recomputed": 108,
        "attention_examples_decoded": 6, "notebook_definitions_sha256": source_hash,
        "max_prediction_metric_absolute_error": max_metric_error,
        "external_features_required": False, "new_scientific_training": False,
    }
    (ROOT / "checks/result_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
