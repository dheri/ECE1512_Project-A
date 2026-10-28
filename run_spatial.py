"""Run spatial-context ABMIL or its matched shuffled-coordinate control."""

import argparse
from pathlib import Path

from run_multiseed import CONFIGS, ROOT, run_study
from spatial_context import SpatialContextConfig


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--reference-suite', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True, help='New output directory.')
    result.add_argument('--coordinate-mode', choices=['real', 'shuffled'], default='real')
    result.add_argument('--neighbors', type=int, default=8)
    result.add_argument('--shuffle-seed', type=int, default=42)
    result.add_argument('--cache-entries', type=int, default=16)
    result.add_argument('--gated', action='store_true', help='Learn a per-patch sigmoid context gate.')
    result.add_argument('--gate-initial', type=float, default=0.1)
    result.add_argument('--context-hidden', type=int, help='Optional context-MLP hidden width.')
    result.add_argument('--fixed-gate', action='store_true', help='Freeze the initial context gate.')
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
    try:
        extension = SpatialContextConfig(args.neighbors, args.coordinate_mode,
                                         args.shuffle_seed, args.cache_entries,
                                         args.gated, args.gate_initial,
                                         args.context_hidden, not args.fixed_gate)
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
