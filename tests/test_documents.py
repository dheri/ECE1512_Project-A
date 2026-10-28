"""Document sources and builders, without course features or neural inference."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_poster
import build_report


class DocumentTests(unittest.TestCase):
    def test_report_has_eight_deliberate_pages(self):
        definition, pages = build_report.definition()
        self.assertEqual(pages, 8)
        self.assertEqual(sum(n.get("pageBreak") == "before" for n in definition["content"]), 7)
        self.assertNotIn("watermark", definition)
        self.assertEqual(definition["info"]["author"], "Parteek Dheri")

    def test_report_follows_lecture_sections(self):
        text = (ROOT / "report/report_lecture.md").read_text()
        titles = ["## 1. Literature review", "## 2. Complete the provided notebook",
                  "## 3. Propose two research ideas"]
        self.assertEqual(sorted(text.index(t) for t in titles), [text.index(t) for t in titles])
        for method in ["ABMIL", "CLAM", "DSMIL", "TransMIL", "DTFD-MIL"]:
            self.assertIn(method, text)

    def test_figure_and_caption_stay_together(self):
        nodes = build_report.parse("![Caption](figures/C17_baseline_seed42_validation.png)")
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes[0]["unbreakable"])
        self.assertTrue(nodes[0]["stack"][0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(nodes[0]["stack"][1]["text"][0]["text"], "Caption")

    def test_poster_uses_lecture_aspect_ratio_and_embedded_images(self):
        definition = build_poster.definition(ROOT / "presentation/poster.md")
        self.assertEqual(definition["pageSize"], {"width": 1280, "height": 720})
        self.assertNotIn("watermark", definition)
        rendered = json.dumps(definition)
        for title in ["Introduction", "Methods", "Results", "Conclusions"]:
            self.assertIn(title, rendered)
        self.assertEqual(rendered.count("data:image/png;base64,"), 2)

    def test_poster_table_matches_audited_results(self):
        rows = build_poster.accuracy_table()["table"]["body"]
        self.assertEqual(rows[1:], [
            ["C16", "87.67 ± 3.38", "89.69 ± 2.81"],
            ["C17", "77.45 ± 4.65", "79.75 ± 2.72"],
            ["BRACS", "42.21 ± 2.74", "43.37 ± 3.38"],
        ])

    def test_duplicate_poster_sections_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poster.md"
            path.write_text("# Title\n## Methods\none\n## Methods\ntwo\n")
            with self.assertRaisesRegex(ValueError, "Duplicate poster section"):
                build_poster.sections(path)

    def test_poster_png_header_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poster.png"
            header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
            path.write_bytes(header + struct.pack(">II", 5120, 2880))
            self.assertEqual(build_poster.validate_png(path), (5120, 2880))
            path.write_bytes(header + struct.pack(">II", 1024, 768))
            with self.assertRaisesRegex(ValueError, "Unexpected poster image size"):
                build_poster.validate_png(path)
            path.write_bytes(b"not a PNG")
            with self.assertRaisesRegex(ValueError, "not a PNG"):
                build_poster.validate_png(path)

    def test_poster_help_needs_no_machine_learning_environment(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/build_poster.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
