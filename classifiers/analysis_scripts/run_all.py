"""Run the complete post-hoc analysis pipeline for one collection.

The four collections remain separate cluster jobs; this module only gives
them one shared command manifest so their analysis coverage cannot drift.
Layers come from the collection (Gemma 17/22/29/33, Qwen 14/18/24/27).

    python -m classifiers.analysis_scripts.run_all --data gemma_original
    python -m classifiers.analysis_scripts.run_all --data gemma_tf_a
    python -m classifiers.analysis_scripts.run_all --data qwen_original
    python -m classifiers.analysis_scripts.run_all --data qwen_tf_a
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from ..data import COLLECTIONS
from .load_results import RESULTS, add_data_args, resolve_root
from .paper.inputs import mark_incomplete, write_manifest


# Layers are per collection — Gemma runs 17/22/29/33 and Qwen the
# depth-matched 14/18/24/27, so this cannot be a module constant or
# every Qwen analysis silently finds no results. --layers still
# overrides.
EXPERIMENTS = ("per_dataset", "leave_out", "merged")
STAGES = ("per-layer", "cross-layer", "redraw", "paper-inputs")


@dataclass(frozen=True)
class Command:
    label: str
    module: str
    args: tuple[str, ...] = ()

    def argv(self, data: str, root: Path) -> list[str]:
        command = [sys.executable, "-m", self.module, *self.args,
                   "--data", data]
        default_root = RESULTS / data
        if root != default_root:
            command.extend(("--root", str(root)))
        return command


def commands(layers: tuple[int, ...],
             stages: tuple[str, ...] = STAGES) -> list[Command]:
    """The canonical ordered command list for either collection."""
    selected = set(stages)
    planned = []
    if "per-layer" in selected:
        for layer in layers:
            layer_arg = str(layer)
            for experiment in EXPERIMENTS:
                planned.append(Command(
                    f"errors {experiment} layer {layer}",
                    "classifiers.analysis_scripts.errors",
                    ("--experiment", experiment, "--tag", "raw_last",
                     "--layer", layer_arg),
                ))
            for experiment in EXPERIMENTS:
                planned.append(Command(
                    f"margins {experiment} layer {layer}",
                    "classifiers.analysis_scripts.margins",
                    ("--experiment", experiment, "--tag", "raw_last",
                     "--layer", layer_arg),
                ))
            planned.append(Command(
                f"prompts layer {layer}",
                "classifiers.analysis_scripts.prompts",
                ("--tag", "raw_last", "--layer", layer_arg),
            ))
            for module in ("coef_norms", "coef_similarity"):
                for holdout_kind in ("dataset", "task"):
                    planned.append(Command(
                        f"{module} layer {layer} holdout {holdout_kind}",
                        f"classifiers.analysis_scripts.{module}",
                        ("--tag", "raw_last", "--layer", layer_arg,
                         "--holdout-kind", holdout_kind),
                    ))
            planned.append(Command(
                f"rs_directions layer {layer}",
                "classifiers.analysis_scripts.rs_directions",
                ("--tag", "raw_last", "--layer", layer_arg),
            ))

    if "cross-layer" in selected:
        for experiment in ("per_dataset", "merged"):
            planned.append(Command(
                f"per_slice {experiment}",
                "classifiers.analysis_scripts.per_slice",
                ("--experiment", experiment, "--tag", "raw_last"),
            ))
        for label, module, args in (
                ("gates", "gates", ("--tag", "raw_last")),
                ("coef_pca", "coef_pca", ("--tag", "raw_last")),
                ("rs_holdout", "rs_holdout", ("--tag", "raw_last")),
                ("cross_methods", "cross_methods",
                 ("--tag", "raw_last")),
                ("hyperparams", "hyperparams", ()),
                ("complexity", "complexity", ()),
                ("coef_sparsity", "coef_sparsity", ())):
            planned.append(Command(
                label, f"classifiers.analysis_scripts.{module}", args))

    if "redraw" in selected:
        planned.append(Command(
            "redraw runner AUC figures",
            "classifiers.analysis_scripts.redraw", ("--only", "auc")))
    return planned


def run_commands(planned: list[Command], *, data: str, root: Path,
                 dry_run: bool, keep_going: bool) -> list[str]:
    failures = []
    for number, command in enumerate(planned, start=1):
        argv = command.argv(data, root)
        print(f"[{number}/{len(planned)}] {command.label}", flush=True)
        if dry_run:
            print("  " + " ".join(argv), flush=True)
            continue
        result = subprocess.run(argv, check=False)
        if result.returncode == 0:
            continue
        failures.append(f"{command.label} (exit {result.returncode})")
        if not keep_going:
            break
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_args(parser)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Scope layers (default: the collection's "
                             "own, data.COLLECTIONS[...].layers)")
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="run later analyses after a failure, then exit nonzero")
    args = parser.parse_args()
    root = resolve_root(args)
    stages = tuple(args.stages)
    args.layers = args.layers or list(COLLECTIONS[args.data].layers)
    planned = commands(tuple(args.layers), stages)
    if "paper-inputs" in stages and not args.dry_run:
        mark_incomplete(root, args.data)
    failures = run_commands(
        planned, data=args.data, root=root,
        dry_run=args.dry_run, keep_going=args.keep_going)

    if failures:
        if "paper-inputs" in stages and not args.dry_run:
            mark_incomplete(root, args.data, failures)
        formatted = "\n  - ".join(failures)
        raise SystemExit(f"{len(failures)} analysis command(s) failed:\n"
                         f"  - {formatted}")

    if "paper-inputs" in stages:
        if args.dry_run:
            print(f"paper inputs: would inventory {root}")
        else:
            manifest = write_manifest(root, args.data)
            print(f"paper inputs: {manifest}")


if __name__ == "__main__":
    main()
