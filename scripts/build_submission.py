"""Build and verify separate report and code submission files without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from package_manifest import ROOT, MANIFEST, files


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    saved = json.loads(MANIFEST.read_text())
    current = {name: sha256(path) for name, path in files()}
    if current != saved:
        raise ValueError("Package differs from its manifest. Review changes before refreshing it.")
    validation = json.loads((ROOT / "checks/pdf_validation.json").read_text())
    source_pdf = ROOT / validation["pdf"]
    if validation["status"] != "passed" or sha256(source_pdf) != validation["sha256"]:
        raise ValueError("Final report does not match its PDF validation record.")
    output.mkdir(parents=True, exist_ok=True)
    report = output / "ECE1512_ProjectA_Report.pdf"
    archive = output / "ECE1512_ProjectA_Code.zip"
    checksum = output / "SUBMISSION_FILES.sha256"
    record = output / "submission_validation.json"
    for path in [report, archive, checksum, record]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output directory.")
    members = {**saved, str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST)}
    prefix = "ECE1512_ProjectA_Submission/"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        zip_path = temporary / archive.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for name in sorted(members):
                handle.write(ROOT / name, prefix + name)
        with zipfile.ZipFile(zip_path) as handle:
            if handle.testzip() is not None:
                raise ValueError("ZIP integrity check failed.")
            if set(handle.namelist()) != {prefix + name for name in members}:
                raise ValueError("ZIP member list differs from the verified package.")
            for name, expected in members.items():
                if hashlib.sha256(handle.read(prefix + name)).hexdigest() != expected:
                    raise ValueError(f"ZIP content hash mismatch: {name}")
        shutil.copyfile(source_pdf, temporary / report.name)
        (temporary / report.name).replace(report)
        zip_path.replace(archive)
    checksums = {path.name: sha256(path) for path in [report, archive]}
    checksum.write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    result = {
        "status": "passed", "code_archive": archive.name, "zip_members": len(members),
        "all_zip_members_match_source_sha256": True, "zip_crc_check": "passed",
        "report": report.name, "report_pages": validation["pages"],
        "sha256": checksums, "includes_git_or_raw_features": False,
        "new_scientific_training": False, "uploaded_or_pushed": False,
    }
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "outputs/submission")
    args = parser.parse_args()
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
