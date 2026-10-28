"""Write or verify a SHA-256 manifest of the review package."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/package_sha256.json"
EXCLUDE = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "node_modules",
           "feature", "slide", "tmp", "outputs"}


def files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == MANIFEST or set(relative.parts) & EXCLUDE:
            continue
        if any(p.startswith("reproduced") for p in relative.parts) or path.suffix == ".pyc":
            continue
        yield str(relative), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh after an intentional review edit.")
    args = parser.parse_args()
    current = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files()}
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n")
        print(f"Wrote {len(current)} file fingerprints.")
    else:
        saved = json.loads(MANIFEST.read_text())
        changed = sorted(name for name in set(current) | set(saved) if current.get(name) != saved.get(name))
        if changed:
            raise SystemExit("Package differs from its manifest:\n" + "\n".join(changed))
        print(f"Verified {len(current)} file fingerprints.")


if __name__ == "__main__":
    main()
