"""Document sources and builders, without course features or neural inference."""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_report
import build_submission
import package_manifest


class DocumentTests(unittest.TestCase):
    def test_report_has_ten_deliberate_pages(self):
        definition, pages = build_report.definition()
        self.assertEqual(pages, 10)
        self.assertEqual(sum(n.get("pageBreak") == "before" for n in definition["content"]), 9)
        self.assertNotIn("watermark", definition)
        self.assertEqual(definition["info"]["author"], "Parteek Dheri")

    def test_report_follows_lecture_sections(self):
        text = (ROOT / "report/report.md").read_text()
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

    def test_literature_table_gives_contribution_most_space(self):
        definition, _ = build_report.definition()
        table = next(node["table"] for node in definition["content"] if "table" in node)
        widths = table["widths"]
        self.assertAlmostEqual(sum(widths), build_report.WIDTH - 24)
        for width, fraction in zip(widths, [.20, .25, .55]):
            self.assertAlmostEqual(width / sum(widths), fraction)
        ordinary = build_report.parse("| One | Two |\n| --- | --- |\n| a | b |")
        self.assertEqual(ordinary[0]["table"]["widths"], ["*", "*"])

    def test_invalid_table_widths_are_rejected(self):
        table = "\n| One | Two |\n| --- | --- |\n| a | b |"
        for widths in ["20,25", "0,100", "20,25,55"]:
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                build_report.parse(f"<!-- table-widths: {widths} -->" + table)
        with self.assertRaises(ValueError):
            build_report.parse("<!-- table-widths: 40,60 -->\nNot a table")

    def test_report_architecture_images_and_extension_sources(self):
        text = (ROOT / "report/report.md").read_text()
        for method in ["baseline", "consistency", "spatial"]:
            path = ROOT / f"report/figures/architecture_{method}.png"
            self.assertIn(f"(figures/{path.name})", text)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", path.read_bytes()[16:24])
            self.assertGreaterEqual(width, 2000)
            self.assertGreater(height, 500)
        for reference in range(11, 16):
            self.assertEqual(text.count(f"\n[{reference}] "), 1)
        for title in ["Unsupervised data augmentation", "masked hard instance mining",
                      "2D point clouds", "Inductive representation", "Highway networks"]:
            self.assertIn(title, text)

    def test_references_use_readable_compact_type(self):
        reference = build_report.parse("[11] A primary reference.")[0]
        self.assertEqual(reference["fontSize"], 9)
        ordinary = build_report.parse("A body paragraph.")[0]
        self.assertNotIn("fontSize", ordinary)

    def test_report_includes_nine_confusions_and_unique_supports(self):
        text = (ROOT / "report/report.md").read_text()
        self.assertIn("(figures/all_nine_confusions.png)", text)
        self.assertIn("n labels count unique test slides", text)
        summary = json.loads((ROOT / "report/tables/test_confusion_summary.json").read_text())
        expected = {"C16": [80, 49], "C17": [121, 16, 18, 45],
                    "BRACS": [6, 16, 9, 11, 12, 12, 20]}
        for dataset, support in expected.items():
            for method in ["baseline", "consistency90", "spatial_gated"]:
                self.assertEqual(summary[dataset][method]["support"], support)
        for name in ["all_nine", "C16", "C17", "BRACS"]:
            path = ROOT / f"report/figures/{name}_confusions.png"
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_submission_builder_help_needs_no_machine_learning_environment(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/build_submission.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_and_code_only_scope(self):
        self.assertEqual([p.name for p in (ROOT / "report").glob("*.md")], ["report.md"])
        self.assertFalse((ROOT / "presentation").exists())
        names = [name for name, _ in package_manifest.files()]
        build_submission.validate_submission_scope(names)

    def test_scope_guard_rejects_presentation_and_legacy_source(self):
        for name in ["presentation/poster.md", "scripts/build_poster.py",
                     "checks/poster_validation.json", "report/report_lecture.md"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Out-of-scope"):
                build_submission.validate_submission_scope([name])

    def test_report_assets_have_an_active_purpose(self):
        text = (ROOT / "report/report.md").read_text()
        images = {Path(name).name for name in re.findall(r"!\[.*?\]\((figures/[^)]+)\)", text)}
        selection = json.loads((ROOT / "report/attention_selection.json").read_text())
        images.update(example["figure"] for example in selection["examples"])
        images.update(f"{dataset}_confusions.png" for dataset in ["C16", "C17", "BRACS"])
        self.assertEqual(images, {p.name for p in (ROOT / "report/figures").glob("*.png")})
        tables = {name + ".md" for name in re.findall(r"\{\{table:(\w+)\}\}", text)}
        tables.add("test_confusion_summary.json")
        self.assertEqual(tables, {p.name for p in (ROOT / "report/tables").iterdir() if p.is_file()})


if __name__ == "__main__":
    unittest.main()
