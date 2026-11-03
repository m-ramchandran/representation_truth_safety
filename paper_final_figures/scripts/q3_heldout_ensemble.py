"""Q3b: ensembling and agreement on genuinely held-out targets,
from the per-sample held-out predictions extracted on compute cluster.

For every (run, layer, kind, holdout, target dataset) cell we build the
samples x methods probability matrix and compute:
  - each method's AUC (sanity: matches the metrics csvs)
  - the mean-probability ensemble over the 10 roster methods, the 5
    linear, the 5 nonlinear
  - a rank-average ensemble (robust to calibration differences)
  - best single method (oracle) and median single method
Writes figure3_heldout_method_ensemble.csv.

Alignment within a cell is by stored order; verified by checking that
every method's y sequence is identical (else the cell is skipped).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import common as C

HELD = C.DATA / "heldout"


def cells(df: pd.DataFrame):
    for (layer, kind, holdout, dataset), g in df.groupby(
            ["layer", "kind", "holdout", "dataset"], observed=True):
        mats = {}
        y_ref = None
        ok = True
        for method, gm in g.groupby("method", observed=True):
            gm = gm.sort_values("idx")
            if y_ref is None:
                y_ref = gm.y.to_numpy()
            elif len(gm) != len(y_ref) or not np.array_equal(gm.y.to_numpy(), y_ref):
                ok = False
                break
            mats[method] = gm.proba.to_numpy()
        if ok and y_ref is not None and len(np.unique(y_ref)) == 2:
            yield layer, kind, holdout, dataset, y_ref, mats


def main() -> None:
    rows = []
    for (model, pert), coll in C.COLLECTIONS.items():
        path = HELD / f"{coll}_heldout.parquet"
        if not path.exists():
            print(f"{coll}: parquet missing, skipped")
            continue
        df = pd.read_parquet(path)
        df = df[df.method.isin(C.METHODS)]
        for layer, kind, holdout, dataset, y, mats in cells(df):
            methods = [m for m in C.METHODS if m in mats]
            if len(methods) < 8:
                continue
            P = np.column_stack([mats[m] for m in methods])
            aucs = {m: roc_auc_score(y, P[:, i]) for i, m in enumerate(methods)}
            lin = [i for i, m in enumerate(methods) if m in C.LINEAR_METHODS]
            non = [i for i, m in enumerate(methods) if m in C.NONLINEAR_METHODS]
            ranks = pd.DataFrame(P).rank(axis=0).to_numpy()
            rows.append({
                "model": model, "perturbation": pert, "layer": layer,
                "depth": C.DEPTH_OF[(model, layer)], "kind": str(kind),
                "holdout": str(holdout), "dataset": str(dataset),
                "n": len(y),
                "best_single": max(aucs.values()),
                "median_single": float(np.median(list(aucs.values()))),
                "ens_mean_all": roc_auc_score(y, P.mean(1)),
                "ens_rank_all": roc_auc_score(y, ranks.mean(1)),
                "ens_mean_linear": roc_auc_score(y, P[:, lin].mean(1)),
                "ens_mean_nonlinear": roc_auc_score(y, P[:, non].mean(1)),
                "auc_diff": aucs.get("diff", np.nan),
                "auc_l2reg": aucs.get("l2reg", np.nan),
            })
        print(f"{coll}: done ({len(rows)} cells so far)")
    out = pd.DataFrame(rows)
    out.to_csv(C.DATA / "figure3_heldout_method_ensemble.csv", index=False)

    pd.set_option("display.width", 250)
    tgt = out[(out.kind.isin(["leave_dataset", "leave_task"])
               & (out.holdout == out.dataset))
              | (out.kind == "merged_eval")]
    tgt = tgt.copy()
    tgt["ens_minus_best"] = tgt.ens_mean_all - tgt.best_single
    tgt["ens_minus_median"] = tgt.ens_mean_all - tgt.median_single
    tgt["rank_minus_mean"] = tgt.ens_rank_all - tgt.ens_mean_all
    print("== held-out targets: ensemble vs singles ==")
    print(tgt.groupby("kind")[["ens_minus_best", "ens_minus_median",
                               "rank_minus_mean"]].describe()
          .loc[:, (slice(None), ["mean", "50%", "min", "max"])].round(3))
    print("\nper dataset (median):")
    print(tgt.groupby("dataset")[["best_single", "ens_mean_all",
                                  "ens_minus_best", "ens_minus_median"]]
          .median().round(3).sort_values("ens_minus_best"))


if __name__ == "__main__":
    main()
