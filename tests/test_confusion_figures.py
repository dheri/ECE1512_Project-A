"""Saved-confusion aggregation and chart labels, without training or course features."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_confusion_figures as figures


class ConfusionFigureTests(unittest.TestCase):
    def test_average_counts_do_not_multiply_unique_support(self):
        info = figures.summarize(
            [np.array([[3, 1], [1, 1]]), np.array([[4, 0], [0, 2]])],
            ["Normal", "Tumor"], [42, 43])
        self.assertEqual(info["support"], [4, 2])
        self.assertEqual(info["unique_test_slides"], 6)
        np.testing.assert_allclose(info["row_percentages"], [[87.5, 12.5], [25, 75]])
        self.assertAlmostEqual(info["mean_accuracy_pct"], 100 * 5 / 6)

    def test_changed_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same nonempty"):
            figures.summarize(
                [np.array([[3, 1], [1, 1]]), np.array([[4, 1], [0, 1]])],
                ["Normal", "Tumor"], [42, 43])

    def test_all_primary_saved_runs_and_class_supports(self):
        data = figures.collect()
        expected = {"C16": [80, 49], "C17": [121, 16, 18, 45],
                    "BRACS": [6, 16, 9, 11, 12, 12, 20]}
        for dataset, support in expected.items():
            for method in figures.METHODS:
                with self.subTest(dataset=dataset, method=method):
                    info = data[dataset][method]
                    self.assertEqual(info["support"], support)
                    self.assertEqual(info["training_seeds"], list(range(42, 52)))
                    np.testing.assert_allclose(np.sum(info["row_percentages"], axis=1), 100)
        self.assertAlmostEqual(data["C16"]["baseline"]["diagonal_recall_pct"][1],
                               data["C16"]["spatial_gated"]["diagonal_recall_pct"][1])

    def test_support_labels_use_unique_slides(self):
        fig, ax = figures.plt.subplots()
        try:
            labels = figures.class_support_labels(
                ax, "C17", {"support": [121, 16, 18, 45]}, compact=True)
            self.assertEqual(labels, ["0 (n=121)", "1 (n=16)", "2 (n=18)", "3 (n=45)"])
        finally:
            figures.plt.close(fig)

    def test_tampered_confusion_is_rejected(self):
        source = figures.DATA / "baseline/seed_42/C16"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cm = pd.read_csv(source / "test_confusion_matrix.csv", index_col=0)
            cm.iloc[0, 0] -= 1
            cm.iloc[0, 1] += 1
            cm.to_csv(directory / "test_confusion_matrix.csv")
            (directory / "test_predictions.csv").write_bytes((source / "test_predictions.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "disagrees"):
                figures.read_run(directory, 2)

    def test_tampered_probability_argmax_is_rejected(self):
        source = figures.DATA / "baseline/seed_42/C16"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = pd.read_csv(source / "test_predictions.csv")
            predicted = int(predictions.loc[0, "prediction"])
            predictions.loc[0, f"p_class_{predicted}"] = 0
            predictions.loc[0, f"p_class_{1 - predicted}"] = 1
            predictions.to_csv(directory / "test_predictions.csv", index=False)
            (directory / "test_confusion_matrix.csv").write_bytes(
                (source / "test_confusion_matrix.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "argmax"):
                figures.read_run(directory, 2)


if __name__ == "__main__":
    unittest.main()
