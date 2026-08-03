"""Hyperparameter stability across the merged run's outer folds.

The merged experiment saves the hyperparameters chosen inside each
outer fold (`params` is a per-fold list) plus the whole-corpus fit's
choice (`full_params`). Folds that disagree flag a flat or noisy tuning
landscape — worth knowing before reading much into that method's
coefficients — and a whole-corpus choice outside the folds' range is
its own warning. Platt constants (platt_a / platt_b) are fitted
calibration values, not chosen hyperparameters, and are excluded.

    python -m classifiers.analysis.hyperparams [--root ...]

Scans every merged npz under results/ (all tags, layers, methods) and
writes <root>/merged/analysis/hyperparam_stability.csv: one row per
(features, layer, method, hyperparameter) with the per-fold values, the
number of distinct values, the whole-corpus value, and a `stable` flag
(all folds agree).
"""

from __future__ import annotations

import argparse

import pandas as pd

from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction, npz_path, resolve_root)

EXCLUDE = ("platt_a", "platt_b")


def stability_rows(d: dict, features: str, layer: int,
                   method: str) -> list[dict]:
    folds = d["params"] if isinstance(d["params"], list) else [d["params"]]
    full = d.get("full_params", {})
    keys = sorted({k for f in folds for k in f} - set(EXCLUDE))
    rows = []
    for k in keys:
        vals = [f.get(k) for f in folds]
        rows.append({"features": features, "layer": layer,
                     "method": method, "param": k,
                     "fold_values": " ".join(str(v) for v in vals),
                     "n_unique": len({str(v) for v in vals}),
                     "full_value": full.get(k, ""),
                     "stable": len({str(v) for v in vals}) == 1})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    m = load_metrics("merged", args.root)
    runs = m[["features", "layer", "method"]].drop_duplicates()
    rows = []
    for r in runs.itertuples():
        d = load_prediction(npz_path("merged", r.features, r.layer,
                                     r.method, root=args.root), keys=())
        rows += stability_rows(d, r.features, r.layer, r.method)
    if not rows:
        raise FileNotFoundError(f"no merged runs under {args.root}")

    tables, _ = analysis_dirs(args.root, "merged")
    df = pd.DataFrame(rows)
    df.to_csv(tables / "hyperparam_stability.csv", index=False)
    unstable = df[~df["stable"]]
    print(f"hyperparams: {len(df)} rows, {len(unstable)} unstable "
          f"({sorted(unstable['method'].unique())}) -> "
          f"{tables / 'hyperparam_stability.csv'}")


if __name__ == "__main__":
    main()
