"""Run proposal 1 (or its patch-dropout comparator) on fixed baseline splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from masked_view_training import MaskedViewConfig
from run_multiseed import CONFIGS, ROOT, run_study


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--reference-suite', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True, help='New output directory.')
    result.add_argument('--method', choices=['consistency', 'patch_dropout'], default='consistency')
    result.add_argument('--keep-fraction', type=float, default=0.9)
    result.add_argument('--ramp-epochs', type=int, default=0,
                        help='Optional linear weight ramp; 0 preserves the original objective.')
    result.add_argument('--consistency-weight', type=float,
                        help='Default 1 for consistency, 0 for patch_dropout; must be 0 for patch_dropout.')
    result.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    result.add_argument('--datasets', choices=list(CONFIGS), nargs='+', default=list(CONFIGS))
    result.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--dpi', type=int, default=140)
    result.add_argument('--notebook', type=Path, default=ROOT / 'ECE1512_ProjectA_Working.ipynb')
    result.add_argument('--feature-root', type=Path, default=ROOT / 'feature')
    result.add_argument('--metadata', type=Path, default=ROOT / 'dataset_csv')
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    weight = args.consistency_weight
    if weight is None:
        weight = 1.0 if args.method == 'consistency' else 0.0
    try:
        extension = MaskedViewConfig(args.method, args.keep_fraction, weight, args.ramp_epochs)
    except ValueError as error:
        argument_parser.error(str(error))
    return run_study(
        args.reference_suite.resolve(), args.output.resolve(), seeds=args.seeds,
        keys=args.datasets, device=args.device, threads=args.threads, dpi=args.dpi,
        notebook=args.notebook.resolve(), feature_root=args.feature_root.resolve(),
        metadata=args.metadata.resolve(), extension=extension,
    )


if __name__ == '__main__':
    main()
