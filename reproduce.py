"""Portable entry point for the fixed-split, three-method study."""
import argparse
from pathlib import Path

from masked_view_training import MaskedViewConfig
from spatial_context import SpatialContextConfig

ROOT = Path(__file__).resolve().parent
METHODS = {
    "baseline": None,
    "consistency90": MaskedViewConfig(),
    "spatial_gated": SpatialContextConfig(gated=True, gate_initial=.1),
}
KEYS = ["C16", "C17", "BRACS"]
SEEDS = list(range(42, 52))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=ROOT / "feature")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=KEYS, default=KEYS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_training:
        parser.error("Training is expensive. Add --confirm-training to start new runs.")
    for values in [args.methods, args.datasets, args.seeds]:
        if len(values) != len(set(values)):
            parser.error("Methods, datasets and seeds must be distinct.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32).")

    import pandas as pd
    import torch
    from run_baselines import run_dataset, write_json
    from run_multiseed import aggregate_results, prepare_inputs, verify_run

    output = args.output.resolve()
    if output.exists():
        parser.error("Output must be a new directory; existing results are never overwritten.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select --device cpu explicitly if appropriate.")
    torch.set_num_threads(2)
    prepared, source_hash = prepare_inputs(
        ROOT / "reference_splits", args.datasets,
        ROOT / "ECE1512_ProjectA_Working.ipynb",
        args.feature_root.resolve(), ROOT / "dataset_csv",
    )
    for conf, _ in prepared.values():
        if hasattr(conf, "reference_template_only"):
            del conf.reference_template_only
    output.mkdir(parents=True)
    rows = []
    write_json(output / "protocol.json", {
        "seeds": args.seeds, "datasets": args.datasets, "methods": args.methods,
        "epochs": 30, "jobs": 1, "threads": 2, "train_workers": 0,
        "order": "seed, dataset, method", "device": args.device,
        "selection": "First maximum validation macro AUROC + macro F1.",
        "notebook_definitions_sha256": source_hash,
    })
    for seed in args.seeds:
        for key in args.datasets:
            conf, datasets = prepared[key]
            for method in args.methods:
                extension = METHODS[method]
                folder = output / method / f"seed_{seed}" / key
                run_dataset(key, conf, datasets, folder,
                            ROOT / "ECE1512_ProjectA_Working.ipynb",
                            epochs=30, device=args.device, training_seed=seed,
                            extension=extension, train_workers=0)
                row, checks = verify_run(folder, conf, seed, source_hash, extension)
                row.update(variant=method)
                rows.append(row)
                write_json(folder / "verification.json", checks)
                pd.DataFrame(rows).to_csv(output / "all_runs.csv", index=False)
                torch.cuda.empty_cache()
    if len(args.seeds) >= 2:
        for method in args.methods:
            summary = aggregate_results([r for r in rows if r["variant"] == method],
                                        args.seeds, args.datasets)
            pd.DataFrame(summary).to_csv(output / method / "summary.csv", index=False)
    print(f"Completed and verified {len(rows)} runs in {output}")


if __name__ == "__main__":
    main()
