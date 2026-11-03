"""Norm accounting: whose geometry shapes the pooled direction?

An earlier sum-vs-merged run found the plain sum of raw-space directions
dominated by a few weakly regularized fits (norms ~30x the rest) — a
norm artifact, not signal. This module puts the numbers behind that
check for 4B, per (tag, layer, method):

  norm         cache-unit norm of each dataset's own direction, with the
               chosen C alongside (the artifact reads as norm tracking C)
  cos_merged   cosine of the merged whole-corpus direction to each
               dataset's (unit) direction — the naive per-dataset
               attribution of the pooled direction
  delta_leave_out
               1 - cos(merged, leave-out-D pool direction): how far the
               pooled direction moves when dataset D is dropped

All from saved npz only (no activations), so any tag works — sparse
supports are aligned pairwise. Merged / leave-out columns are NaN where
those runs don't exist. The least-squares decomposition of the merged
direction over the singles lives in coef_pca's merged section; the
same cosines under Mahalanobis geometry live in coef_similarity's mls
table.

    python -m classifiers.analysis_scripts.coef_norms --tag raw_last --layer 9 \
        [--methods l2reg diff ...] [--holdout-kind dataset|task] [--root ...]

Output: <root>/coefficients/coef_norms_<tag>_layer<L>_<holdout kind>.csv,
one row per (method, dataset).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .directions import (DIRECTION_KEYS, LINEAR_METHODS,
                         leave_out_name, load_direction, pair_cos,
                         raw_direction)
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction,
                           npz_path, resolve_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--methods", nargs="+",
                    default=list(LINEAR_METHODS),
                    choices=LINEAR_METHODS)
    ap.add_argument("--holdout-kind", default="dataset",
                    choices=("dataset", "task"))
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    m = load_metrics("per_dataset", args.root)
    m = m[(m["features"] == args.tag) & (m["layer"] == args.layer)]
    rows = []
    for method in args.methods:
        try:
            merged = load_direction("merged", args.tag, args.layer,
                                    method, root=args.root)
        except (FileNotFoundError, KeyError, ValueError):
            merged = None
        for ds in sorted(m[m["method"] == method]
                         ["train_dataset"].unique()):
            try:
                d = load_prediction(
                    npz_path("per_dataset", args.tag, args.layer, method,
                             ds, args.root), keys=DIRECTION_KEYS)
                single = raw_direction(d, method)
                params = d["params"]
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  {method} {ds}: skipped ({e})")
                continue
            try:
                leave = load_direction(
                    "leave_out", args.tag, args.layer, method,
                    leave_out_name(ds, args.holdout_kind, args.data),
                    root=args.root)
            except (FileNotFoundError, KeyError):
                leave = None
            rows.append({
                "method": method, "dataset": ds,
                "norm": round(float(np.linalg.norm(single[0])), 4),
                "C": params.get("C", ""),
                "cos_merged": round(pair_cos(merged, single), 4)
                              if merged is not None else np.nan,
                "delta_leave_out": round(1 - pair_cos(merged, leave), 4)
                                   if merged is not None
                                   and leave is not None else np.nan,
            })
    if not rows:
        raise FileNotFoundError(f"no fits for tag={args.tag!r} "
                                f"layer={args.layer}")
    tables, _ = analysis_dirs(args.root)
    path = tables / (f"coef_norms_{args.tag}_layer{args.layer}"
                     f"_{args.holdout_kind}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    spread = df.groupby("method")["norm"].agg(["min", "max"])
    print(f"coef_norms: {len(df)} rows -> {path}\n"
          f"norm spread per method:\n{spread.round(3).to_string()}")


if __name__ == "__main__":
    main()
