"""Build the editable report as a vector PDF; requires Node and a PDF library."""
import argparse
import base64
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 516


def rich(text):
    # Roboto lacks several mathematical subscript/operator glyphs. Keep indices
    # explicit instead of emitting unreadable missing-character boxes.
    text = text.translate(str.maketrans({
        "ᵢ": "_i", "ₚ": "_p", "ᵀ": "^T", "ᵥ": "_v", "ᵤ": "_u", "ₛ": "_s",
        "ⱼ": "_j", "꜀": "_c", "₁": "_1", "₂": "_2", "⊙": "*",
        "∥": "||", "∈": "in", "→": " -> ",
    }))
    pieces = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|https?://[^\s]+)", text)
    runs = []
    for p in pieces:
        if not p:
            continue
        if p.startswith("http"):
            label = ("CVF publisher page" if "thecvf.com" in p else
                     "PMLR publisher page" if "mlr.press" in p else
                     "BRACS publisher page" if "oup.com" in p else
                     "arXiv:" + p.rsplit("/", 1)[-1] if "arxiv.org" in p else
                     "Zenodo:" + p.rsplit("/", 1)[-1] if "zenodo.org" in p else
                     "CAMELYON17 challenge" if "grand-challenge.org" in p else p)
            runs.append({"text": label, "link": p, "color": "#222222", "decoration": "underline"})
        elif p.startswith("`"):
            runs.append({"text": p[1:-1], "color": "#333333"})
        elif p.startswith("**"):
            runs.append({"text": p[2:-2], "bold": True})
        else:
            runs.append({"text": p})
    return runs


def parse(text, width=WIDTH):
    lines = text.strip().splitlines()
    nodes = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("<!-- columns -->"):
            j = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == "<!-- /columns -->")
            block = "\n".join(lines[i + 1:j]).split("<!-- column -->")
            if len(block) != 2:
                raise ValueError("Each column block must contain exactly two columns.")
            nodes.append({"columns": [{"width": (width - 18) / 2, "stack": parse(b, (width - 18) / 2)}
                                      for b in block], "columnGap": 18})
            i = j + 1
            continue
        match = re.fullmatch(r"\{\{table:([\w_]+)\}\}", line)
        if match:
            table_text = (ROOT / "report/tables" / (match[1] + ".md")).read_text()
            nodes.extend(parse(table_text, width))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [s.strip() for s in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[: -]+", s) for s in row):
                    rows.append(row)
                i += 1
            body = [[{"text": rich(t), "bold": ri == 0, "fontSize": 8 if len(row) > 4 else 9,
                      "fillColor": "#eeeeee" if ri == 0 else "#ffffff", "margin": [0, 2, 0, 2]}
                     for t in row] for ri, row in enumerate(rows)]
            nodes.append({"table": {"headerRows": 1, "widths": ["*"] * len(rows[0]), "body": body,
                                   "dontBreakRows": True},
                          "layout": "lightHorizontalLines", "margin": [0, 3, 0, 11]})
            continue
        match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if match:
            image = (ROOT / "report" / match[2]).resolve()
            data = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
            image_width = width * (.92 if "validation.png" in match[2] else
                                   .89 if "paired_seed" in match[2] else .97)
            nodes.append({"image": data, "width": image_width, "alignment": "center",
                          "margin": [0, 4, 0, 3]})
            nodes.append({"text": rich(match[1]), "fontSize": 8, "lineHeight": 1.02,
                          "margin": [0, 0, 0, 7]})
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            nodes.append({"text": rich(line.lstrip("# ")), "style": f"h{level}"})
            i += 1
            continue
        if line.startswith("- ") or re.match(r"^\d+\. ", line):
            items = []
            ordered = not line.startswith("- ")
            while i < len(lines) and (lines[i].strip().startswith("- ") or
                                      re.match(r"^\d+\. ", lines[i].strip())):
                items.append({"text": rich(re.sub(r"^(?:- |\d+\. )", "", lines[i].strip()))})
                i += 1
            nodes.append({"ol" if ordered else "ul": items, "margin": [10, 1, 0, 7]})
            continue
        if line.startswith("> "):
            nodes.append({"text": rich(line[2:]), "fontSize": 9.4,
                          "margin": [4, 5, 0, 8], "color": "#111111"})
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            paragraph.append(lines[i].strip())
            i += 1
        nodes.append({"text": rich(" ".join(paragraph)), "alignment": "justify",
                      "margin": [0, 0, 0, 5]})
    return nodes


def definition():
    pages = (ROOT / "report/report.md").read_text().split("<!-- page -->")
    content = []
    for number, page in enumerate(pages, 1):
        nodes = parse(page)
        if number > 1:
            nodes[0]["pageBreak"] = "before"
        if number == 1:
            for node in nodes:
                if "columns" in node:
                    break
                node["alignment"] = "center"
        if number == 2:
            for node in nodes:
                if "columns" in node:
                    node.update(fontSize=9.1, lineHeight=1.04)
        content.extend(nodes)
    return {
        "pageSize": "LETTER", "pageMargins": [48, 40, 48, 42],
        "defaultStyle": {"font": "Roboto", "fontSize": 9.4, "lineHeight": 1.04},
        "styles": {
            "h1": {"fontSize": 17, "bold": True, "alignment": "center", "margin": [0, 0, 0, 7]},
            "h2": {"fontSize": 12.5, "bold": True, "margin": [0, 0, 0, 9]},
            "h3": {"fontSize": 10, "bold": True, "margin": [0, 5, 0, 5]},
        },
        "content": content,
        "footer": {"text": "ECE1512 Project A  |  Parteek Dheri", "fontSize": 7.5,
                   "color": "#666666", "alignment": "center", "margin": [0, 2, 0, 0]},
        "info": {"title": "ECE1512 Project A: Whole-Slide Classification",
                 "author": "Parteek Dheri",
                 "subject": "ABMIL baseline and two research extensions",
                 "creator": "ECE1512 Project A", "producer": "ECE1512 Project A"},
        "compress": True, "language": "en",
    }, len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=shutil.which("node"))
    parser.add_argument("--pdfmake", type=Path, required=True, help="Installed PDF library directory.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd() / "tmp/report_build")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "outputs/ECE1512_ProjectA_Report.pdf")
    args = parser.parse_args()
    if not args.node:
        parser.error("Node is required. Install the renderer separately, not in the training environment.")
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec, expected_pages = definition()
    source = args.workdir / "project_a.pdf.json"
    source.write_text(json.dumps(spec, ensure_ascii=False))
    script = args.workdir / "render_report.cjs"
    script.write_text("""
const fs = require('fs');
const path = require('path');
const [library, source, output] = process.argv.slice(2);
const Printer = require(library);
const vfs = require(path.join(library, 'build/vfs_fonts.js'));
const buffer = name => Buffer.from(vfs[name], 'base64');
const printer = new Printer({Roboto: {
  normal: buffer('Roboto-Regular.ttf'), bold: buffer('Roboto-Medium.ttf'),
  italics: buffer('Roboto-Italic.ttf'), bolditalics: buffer('Roboto-MediumItalic.ttf')
}});
const definition = JSON.parse(fs.readFileSync(source, 'utf8'));
const fontkit = require(require.resolve('@foliojs-fork/fontkit', {paths: [library]}));
const font = fontkit.create(buffer('Roboto-Regular.ttf'));
function checkGlyphs(node) {
  if (Array.isArray(node)) return node.forEach(checkGlyphs);
  if (!node || typeof node !== 'object') return;
  if (typeof node.text === 'string') {
    for (const c of node.text) {
      if (c.codePointAt(0) > 127 && !font.hasGlyphForCodePoint(c.codePointAt(0)))
        throw new Error('Unsupported report font glyph: ' + c);
    }
  }
  Object.values(node).forEach(checkGlyphs);
}
checkGlyphs(definition.content);
const doc = printer.createPdfKitDocument(definition, {bufferPages: true});
const range = doc.bufferedPageRange();
for (let i = range.start; i < range.start + range.count; i++) {
  doc.switchToPage(i);
  const bottom = doc.page.margins.bottom;
  doc.page.margins.bottom = 0;
  doc.fontSize(8).fillColor('#333333').text(String(i + 1), 48, 767,
                                         {width: 516, align: 'center', lineBreak: false});
  doc.page.margins.bottom = bottom;
}
const out = fs.createWriteStream(output);
doc.pipe(out); doc.end();
out.on('error', e => { console.error(e); process.exitCode = 1; });
""")
    subprocess.run([args.node, str(script.resolve()), str(args.pdfmake.resolve()),
                    str(source.resolve()), str(args.output.resolve())], check=True)
    assert args.output.stat().st_size > 1000
    with args.output.open("rb") as f:
        assert f.read(5) == b"%PDF-"
    if shutil.which("pdfinfo"):
        info = subprocess.check_output(["pdfinfo", str(args.output)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)[1])
        if pages != expected_pages or pages > 20:
            raise ValueError(f"Expected {expected_pages} deliberate pages, rendered {pages}. Adjust layout.")
    script.unlink()
    packaged = ROOT / "report/ECE1512_ProjectA_Report.pdf"
    if args.output.resolve() != packaged.resolve():
        shutil.copyfile(args.output, packaged)
    print(f"Built {args.output.resolve()} ({args.output.stat().st_size:,} bytes; {expected_pages} pages).")


if __name__ == "__main__":
    main()
