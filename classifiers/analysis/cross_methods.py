"""Every way we score a dataset, side by side.

One row per dataset, one column per *training regime*, so a dataset's
whole story reads across a single row:

  in_domain_fit   trained on D, scored on D with NO held-out split —
                  the optimistic ceiling ("overfit" reference). Not in
                  any runner csv: it is recomputed here by scoring the
                  per_dataset full-fit's saved direction on D's own
                  activations, so it exists only for the methods with a
                  coefficient in cache units (l1reg / l2reg /
                  svm_linear / lda / diff) and only for raw-space tags.
                  Needs the activation caches; --no-in-domain skips it.
  within_cv       nested CV inside D (per_dataset, kind=cv) — the
                  honest single-dataset number
  merged_cv       the pooled model's nested-CV predictions restricted
                  to D's samples (merged, kind=per_dataset / eval_only)
  leave_one_out   D held out of the pool entirely (leave_out,
                  holdout_kind=dataset)
  leave_task_out  D's whole task held out (leave_out,
                  holdout_kind=task) — the strictest test

Reading a row left to right is the generalization gradient: how much a
dataset's AUC decays as the classifier is allowed to know less about
it. A dataset whose merged_cv holds up near within_cv is served by the
pooled model; one that collapses at leave_one_out has little the other
datasets share.

    python -m classifiers.analysis.cross_methods --tag raw_last \\
        [--layers 17 22 29 33] [--methods ...] [--no-in-domain]
        [--data ...] [--root ...]

Outputs under <root>/cross_methods/ (its own section, beside
per_dataset / leave_out / merged): tables/regimes_<tag>.csv (long form,
every method and layer), one figure per method,
figures/regimes_<method>_<tag>.png — a panel per layer, datasets x
regimes — and figures/regimes_best_<tag>.png, that same grid with the
methods collapsed: each cell holds the highest AUC any method reached
there, labelled underneath with the method that reached it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import COLLECTIONS
from . import directions
from .directions import (LINEAR_METHODS, RAW_SPACE, load_direction)
from .load_results import (add_data_args, analysis_dirs, auc_or_nan,
                           load_metrics, resolve_root)
from .plots import layer_panels, save

# column order = decreasing knowledge of the evaluated dataset
REGIMES = ("in_domain_fit", "within_cv", "merged_cv", "leave_one_out",
           "leave_task_out")


def from_metrics(root, tag: str, layer: int) -> pd.DataFrame:
    """The four csv-backed regimes as a long frame (method, dataset,
    regime, auc). Experiments that have not run yet are skipped."""
    rows = []

    def add(df, regime, dataset_col):
        for r in df.itertuples():
            rows.append({"method": r.method,
                         "dataset": getattr(r, dataset_col),
                         "regime": regime, "auc": r.auc})

    try:
        m = load_metrics("per_dataset", root)
        m = m[(m["features"] == tag) & (m["layer"] == layer)
              & (m["kind"] == "cv")]
        add(m, "within_cv", "train_dataset")
    except FileNotFoundError:
        print("  per_dataset: no metrics, within_cv skipped")
    try:
        m = load_metrics("merged", root)
        m = m[(m["features"] == tag) & (m["layer"] == layer)
              & (m["kind"].isin(("per_dataset", "eval_only")))]
        add(m, "merged_cv", "test_dataset")
    except FileNotFoundError:
        print("  merged: no metrics, merged_cv skipped")
    try:
        m = load_metrics("leave_out", root)
        m = m[(m["features"] == tag) & (m["layer"] == layer)]
        add(m[m["holdout_kind"] == "dataset"], "leave_one_out",
            "test_dataset")
        add(m[m["holdout_kind"] == "task"], "leave_task_out",
            "test_dataset")
    except FileNotFoundError:
        print("  leave_out: no metrics, holdout regimes skipped")
    return pd.DataFrame(rows)


def in_domain_rows(root, tag: str, layer: int, methods, datasets,
                   data: str) -> list:
    """The no-split ceiling: each per_dataset full fit scored on its own
    training data. Recomputed from the saved direction (the runners
    never wrote this number), so it covers the coefficient methods on
    raw-space tags only."""
    if tag.rsplit("_", 1)[0] not in RAW_SPACE:
        print(f"  in_domain_fit: {tag} directions have no cache space, "
              "skipped")
        return []
    usable = [m for m in methods if m in LINEAR_METHODS]
    if not usable:
        return []
    units = directions.load_units(layer, tag.rsplit("_", 1)[1], ("raw",),
                                  data)
    rows = []
    for method in usable:
        for ds in datasets:
            if ds not in units:
                continue
            try:
                w, _ = load_direction("per_dataset", tag, layer, method,
                                      ds, root=root)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            X, y = units[ds]["blocks"]["raw"], units[ds]["y"]
            rows.append({"method": method, "dataset": ds,
                         "regime": "in_domain_fit",
                         "auc": auc_or_nan(y, np.asarray(X) @ w)})
    return rows


def stacked(name: str, width: int = 11) -> str:
    """A method name for the cell under an AUC, broken at its
    underscores when it is wider than the cell (random_slopes_task) so
    it cannot run into the neighboring column."""
    lines = []
    for part in name.split("_"):
        if lines and len(lines[-1]) + 1 + len(part) <= width:
            lines[-1] += "_" + part
        else:
            lines.append(part)
    return "\n".join(lines)


def best_by_cell(long: pd.DataFrame, order) -> tuple[dict, dict]:
    """Per layer, the winner of every (dataset, regime) cell: the
    highest AUC any method reached there, and the name of that method.
    Methods are averaged over their rows first, exactly as the
    per-method panels' pivot_table does, so a method with repeated rows
    cannot win a cell on its luckiest one. Ties go to the
    alphabetically first method. Returns (aucs, winners), both keyed by
    layer and aligned cell for cell."""
    aucs, winners = {}, {}
    for layer, gl in long.groupby("layer"):
        mean = (gl.groupby(["method", "dataset", "regime"], as_index=False)
                ["auc"].mean().dropna(subset=["auc"]))
        if mean.empty:
            continue
        win = mean.loc[mean.groupby(["dataset", "regime"])["auc"].idxmax()]
        cols = [r for r in REGIMES if r in set(win["regime"])]
        auc = (win.pivot(index="dataset", columns="regime", values="auc")
               .reindex(index=order).reindex(columns=cols)
               .dropna(how="all"))
        if auc.empty:
            continue
        aucs[layer] = auc
        winners[layer] = (win.pivot(index="dataset", columns="regime",
                                    values="method")
                          .reindex(index=auc.index).reindex(columns=cols)
                          .map(lambda v: stacked(v)
                               if isinstance(v, str) else ""))
    return aucs, winners


def figures_from_long(long: pd.DataFrame, order, tag: str,
                      figures) -> int:
    """Every figure this module draws, from the same long frame it
    saves: one regime grid per method, then the best-of grid. redraw
    re-enters here with that csv read back off disk, so the two paths
    cannot drift."""
    n = 0
    for method, g in long.groupby("method"):
        by_layer = {}
        for layer, gl in g.groupby("layer"):
            wide = (gl.pivot_table(index="dataset", columns="regime",
                                   values="auc")
                    .reindex(index=order)
                    .reindex(columns=[r for r in REGIMES
                                      if r in set(gl["regime"])]))
            by_layer[layer] = wide.dropna(how="all")
        by_layer = {k: v for k, v in by_layer.items() if not v.empty}
        if not by_layer:
            continue
        save(layer_panels(by_layer, f"AUC by training regime — {method}",
                          f"{tag} — columns left to right know less "
                          f"and less about the evaluated dataset",
                          vmin=0.5, vmax=1.0),
             figures / f"regimes_{method}_{tag}.png")
        n += 1

    # the same grid once more, collapsed over methods: what the best
    # tool available reaches in each cell, and which tool that is. No
    # method count in the subtitle — the pool differs by column
    # (in_domain_fit exists only for the coefficient methods), and the
    # label under each AUC says who won it anyway
    aucs, winners = best_by_cell(long, order)
    if aucs:
        save(layer_panels(aucs, "AUC by training regime — best method",
                          f"{tag} — the best method in each cell, named "
                          f"under its AUC",
                          vmin=0.5, vmax=1.0, sublabels=winners),
             figures / f"regimes_best_{tag}.png")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="default: the collection's layers")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="which methods get an in_domain_fit row (that "
                         "recompute is the slow part); the csv and the "
                         "figures always cover every method in the "
                         "metrics. Default: all of them")
    ap.add_argument("--no-in-domain", action="store_true",
                    help="skip the in_domain_fit column (it needs the "
                         "activation caches)")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    c = COLLECTIONS[args.data]
    layers = args.layers or list(c.layers)

    frames = []
    for layer in layers:
        df = from_metrics(args.root, args.tag, layer)
        if df.empty:
            print(f"  layer {layer}: no metrics, skipped")
            continue
        methods = args.methods or sorted(df["method"].unique())
        if not args.no_in_domain:
            df = pd.concat([df, pd.DataFrame(in_domain_rows(
                args.root, args.tag, layer, methods,
                sorted(df["dataset"].unique()), args.data))],
                ignore_index=True)
        frames.append(df.assign(layer=layer))
    if not frames:
        raise FileNotFoundError(f"no metrics for tag={args.tag!r} under "
                                f"{args.root}")
    long = pd.concat(frames, ignore_index=True)
    tables, figures = analysis_dirs(args.root, "cross_methods")
    long.round(4).to_csv(tables / f"regimes_{args.tag}.csv", index=False)

    order = [d for d in c.datasets if d in set(long["dataset"])]
    n = figures_from_long(long, order, args.tag, figures)
    print(f"cross_methods: {len(long)} rows, {n} figures under "
          f"{tables.parent}")


if __name__ == "__main__":
    main()
