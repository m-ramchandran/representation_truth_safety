"""Build or redraw the cross-collection paper figures.

Build the core analyses after both collection jobs finish:

    python -m classifiers.analysis_scripts.paper

GPU boundary artifacts are optional and requested explicitly:

    python -m classifiers.analysis_scripts.paper --stages boundaries
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..load_results import RESULTS
from .build import COLLECTIONS, CORE_STAGES, STAGES, build
from .redraw import redraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", nargs="+", choices=COLLECTIONS,
                        default=list(COLLECTIONS))
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(CORE_STAGES))
    parser.add_argument("--redraw", action="store_true",
                        help="redraw from paper tables without recomputing")
    parser.add_argument("--skip-input-check", action="store_true",
                        help="allow a build without run_all manifests")
    args = parser.parse_args()
    output_root = args.output_root or args.results_root / "paper"
    if args.redraw:
        count = redraw(output_root)
        print(f"paper redraw: {count} figures -> {output_root / 'figures'}")
        return
    manifest = build(
        results_root=args.results_root,
        output_root=output_root,
        collections=tuple(args.data),
        stages=tuple(args.stages),
        check_inputs=not args.skip_input_check,
    )
    print(f"paper build: {manifest}")


if __name__ == "__main__":
    main()
