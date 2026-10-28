"""Export attention from one packaged representative checkpoint, without training."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["baseline", "consistency90", "spatial_gated"])
    parser.add_argument("--dataset", required=True, choices=["C16", "C17", "BRACS"])
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="validation")
    args = parser.parse_args(argv)
    if not args.features.is_file():
        parser.error("Feature file not found; see DATA.md.")
    if args.output.exists():
        parser.error("Output must be a new directory.")
    from export_attention import export, parser as export_parser
    temp_root = ROOT / "tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="attention_", dir=temp_root) as folder:
        folder = Path(folder)
        shutil.copyfile(ROOT / "checkpoints" / f"{args.method}_seed42_{args.dataset}.pt",
                        folder / "best_model.pt")
        shutil.copyfile(ROOT / "reference_splits" / args.dataset / "splits.json",
                        folder / "splits.json")
        request = export_parser().parse_args([
            "--checkpoint", str(folder / "best_model.pt"),
            "--features", str(args.features.resolve()), "--output", str(args.output.resolve()),
            "--device", args.device, "--split", args.split,
            "--notebook", str(ROOT / "ECE1512_ProjectA_Working.ipynb"),
        ])
        export(request)


if __name__ == "__main__":
    main()
