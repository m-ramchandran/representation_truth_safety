"""Run ON compute cluster (GPU node): the merged pooled run with the pool expanded
to include claude_vds and wildjailbreak_adversarial (blood stays out).

Registers a derived collection "<base>_expanded_pool" (identical except
pool_excluded) and invokes the standard run_merged CLI on it, so results
land under classifiers/results/<base>_expanded_pool/merged/ without
touching the real trees or the repo.

    python run_merged_expanded_pool.py <base_collection> <layer> [methods_csv]

The optional third argument restricts the run to a comma-separated
method subset (default: the full METHODS tuple below); the runner's
resume record is (features, layer, method), so subsets can fill one
result tree in parallel.
"""
from __future__ import annotations

import dataclasses
import runpy
import sys

import classifiers.data as data

METHODS = ("diff", "l2reg", "mlp", "xgboost",
           "random_slopes", "random_slopes_task")


def main() -> None:
    base, layer = sys.argv[1], sys.argv[2]
    methods = (sys.argv[3].split(",") if len(sys.argv) > 3
               else list(METHODS))
    name = base + "_expanded_pool"
    if name not in data.COLLECTIONS:
        data.COLLECTIONS[name] = dataclasses.replace(
            data.COLLECTIONS[base],
            pool_excluded=("claude_blood_test_ranges",))
    sys.argv = ["run_merged", "--data", name, "--raw", "last",
                "--features", "raw", "--layers", layer,
                "--methods", *methods, "--n-jobs", "5"]
    runpy.run_module("classifiers.scripts.run_merged", run_name="__main__")


if __name__ == "__main__":
    main()
