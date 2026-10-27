"""Portable submission setup checks; no course features or scientific training."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import main


class SubmissionTests(unittest.TestCase):
    def test_training_requires_explicit_confirmation(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["--output", str(ROOT / "results/reproduced_test")])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((ROOT / "results/reproduced_test").exists())

    def test_duplicate_grid_is_rejected_before_training(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["--output", str(ROOT / "results/reproduced_test"), "--confirm-training",
                  "--methods", "baseline", "baseline"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((ROOT / "results/reproduced_test").exists())

    def test_notebook_uses_frozen_final_splits(self):
        notebook = json.loads((ROOT / "ECE1512_ProjectA_Working.ipynb").read_text())
        cell = next(c for c in notebook["cells"] if c.get("id") == "41b18f7b")
        source = "".join(cell["source"])
        self.assertIn("load_fixed_datasets", source)
        self.assertIn("'reference_splits'", source)
        setup = next(c for c in notebook["cells"] if c.get("id") == "65ae9a49")
        self.assertNotIn("subprocess", "".join(setup["source"]))

    def test_help_commands_do_not_need_data(self):
        for script in ["reproduce.py", "scripts/export_representative.py", "scripts/build_report.py"]:
            result = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
