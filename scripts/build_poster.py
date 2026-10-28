"""Build the editable one-page poster and its 5120 x 2880 submission image."""
import argparse
import base64
import csv
from pathlib import Path
import shutil
import statistics
import struct
import subprocess

from build_report import ROOT, render_document, rich

WIDTH, HEIGHT = 1280, 720
BLUE = "#126479"
INK = "#182b36"


def sections(path):
    result, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            result["Title"] = line[2:].strip()
        elif line.startswith("## "):
            current = line[3:].strip()
            if current in result:
                raise ValueError(f"Duplicate poster section: {current}")
            result[current] = []
        elif current and line.strip():
            result[current].append(line.strip())
    return result


def heading(text):
    return {"text": text, "fontSize": 17, "bold": True, "color": BLUE,
            "margin": [0, 7, 0, 5]}


def prose(lines, size=12.6):
    nodes = []
    for line in lines:
        if line.startswith("- "):
            nodes.append({"columns": [{"width": 9, "text": "•", "color": BLUE},
                                      {"width": "*", "text": rich(line[2:])}],
                          "fontSize": size, "margin": [0, 0, 0, 4]})
        else:
            nodes.append({"text": rich(line), "fontSize": size, "margin": [0, 0, 0, 3]})
    return nodes


def picture(path, width):
    return {"image": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
            "width": width, "margin": [0, 4, 0, 2]}


def accuracy_table():
    with (ROOT / "results/primary/all_runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    body = [[{"text": t, "bold": True, "color": BLUE} for t in
             ["Data", "ABMIL", "Gated spatial"]]]
    for dataset in ["C16", "C17", "BRACS"]:
        cells = [dataset]
        for method in ["baseline", "spatial_gated"]:
            selected = [r for r in rows if r["variant"] == method and r["dataset_key"] == dataset]
            if sorted(int(r["training_seed"]) for r in selected) != list(range(42, 52)):
                raise ValueError(f"Expected ten unique seeds: {dataset}/{method}.")
            values = [float(r["test_accuracy_pct"]) for r in selected]
            cells.append(f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}")
        body.append(cells)
    return {"table": {"headerRows": 1, "widths": [48, "*", "*"], "body": body,
                      "dontBreakRows": True},
            "fontSize": 12, "layout": "lightHorizontalLines", "margin": [0, 0, 0, 5]}


def definition(source):
    s = sections(source)
    left, middle, right = 300, 510, 370
    pipeline = [{"text": rich(line), "fontSize": 11.9, "margin": [6, 2, 6, 2],
                 "fillColor": "#eef5f7"} for line in s["Pipeline"]]
    pipeline_node = {"table": {"widths": ["*"], "body": [[node] for node in pipeline]},
                     "layout": "noBorders", "margin": [0, 0, 0, 8]}
    columns = [
        {"width": left, "stack": [
            heading("Introduction"), *prose(s["Introduction"]),
            heading("Methods"), pipeline_node, *prose(s["Methods"]),
            heading("Two research ideas"), *prose(s["Research ideas"]),
        ]},
        {"width": middle, "stack": [
            heading("Results"), accuracy_table(), *prose(s["Accuracy caption"], 10.5),
            picture(ROOT / "presentation/figures/spatial_accuracy_differences.png", middle),
            *prose(s["Difference caption"], 10.5),
            {"text": "Attention diagnostic", "fontSize": 13, "bold": True,
             "color": BLUE, "margin": [0, 4, 0, 3]},
            picture(ROOT / "report/figures/C17_baseline_seed42_validation.png", middle),
            *prose(s["Attention caption"], 10.5),
        ]},
        {"width": right, "stack": [
            heading("Conclusions"), *prose(s["Conclusions"]),
            heading("Tradeoffs"), *prose(s["Tradeoffs"]),
            heading("Limitations"), *prose(s["Limitations"]),
            heading("References"), *prose(s["References"], 9.3),
        ]},
    ]
    return {
        "pageSize": {"width": WIDTH, "height": HEIGHT},
        "pageMargins": [28, 18, 28, 25],
        "defaultStyle": {"font": "Roboto", "fontSize": 12.6, "lineHeight": 1.06, "color": INK},
        "content": [
            {"text": s["Title"], "fontSize": 31, "bold": True, "color": INK,
             "margin": [0, 0, 0, 3]},
            {"text": " ".join(s["Author"]), "fontSize": 12.8, "margin": [0, 0, 0, 5]},
            {"table": {"widths": ["*"], "body": [[{
                "text": " ".join(s["Takeaway"]), "fontSize": 16, "bold": True,
                "fillColor": "#e8f2f5", "color": BLUE, "margin": [9, 5, 9, 5]}]]},
             "layout": "noBorders", "margin": [0, 0, 0, 1]},
            {"columns": columns, "columnGap": 22},
        ],
        "footer": {"text": " ".join(s["Footer"]), "fontSize": 9, "color": "#50616b",
                   "alignment": "center", "margin": [28, 3, 28, 0]},
        "info": {"title": s["Title"], "author": "Parteek Dheri",
                 "subject": "ECE1512 Project A final poster",
                 "creator": "ECE1512 Project A", "producer": "ECE1512 Project A"},
        "compress": True, "language": "en",
    }


def validate_png(path):
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Poster image is not a PNG.")
    dimensions = struct.unpack(">II", data[16:24])
    if dimensions != (5120, 2880):
        raise ValueError(f"Unexpected poster image size: {dimensions}.")
    if len(data) >= 10_000_000:
        raise ValueError("Poster exceeds the conservative 10 MB submission limit.")
    return dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "presentation/poster.md")
    parser.add_argument("--node", default=shutil.which("node"))
    parser.add_argument("--pdfmake", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path.cwd() / "tmp/poster_build")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "outputs/ECE1512_ProjectA_Poster.pdf")
    args = parser.parse_args()
    if not args.node or not shutil.which("pdftoppm"):
        parser.error("Node and pdftoppm are required; use a separate rendering environment.")
    output = args.output.resolve()
    render_document(definition(args.source), 1, node=args.node, pdfmake=args.pdfmake,
                    workdir=args.workdir, output=output, number_pages=False)
    prefix = output.with_suffix("")
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", "5120",
                    "-scale-to-y", "2880", str(output), str(prefix)], check=True)
    png = Path(str(prefix) + ".png")
    validate_png(png)
    for file in [output, png]:
        packaged = ROOT / "presentation" / file.name
        if file.resolve() != packaged.resolve():
            shutil.copyfile(file, packaged)
    print(f"Built {png} (5120 x 2880; {png.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
