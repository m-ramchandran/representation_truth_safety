"""What the l1 penalty keeps.

Scans every per_dataset l1reg fit under results/ (all feature sets and
layers present) and reports the support structure of the saved
coefficients: how many features each training dataset's fit kept, and
which features recur across datasets — in the same direction or not.

"Nonzero" is |coef| > tol in the fit's own (standardized) space, the
space the penalty acted in; cuML's L1 can leave near-zeros rather than
exact zeros, hence the tolerance. Recurrence tables exist only for the
feature sets where column i is the same feature in every fit (raw /
sae / transcoder, the latter two mapped to dictionary indices via
kept_features); pca bases and bin edges are per-fit, so those tags get
support sizes only. Reported coefficient values are in cache units
(via directions.py), signs unchanged by the mapping.

    python -m classifiers.analysis_scripts.coef_sparsity [--tol 1e-8] [--root ...]

Outputs under <root>/coefficients/:
  support_sizes.csv     one row per (feature set, layer, training
                        dataset): total features, nonzero count and
                        fraction, chosen C
  recurrence_<tag>_layer<L>.csv
                        features kept by >= 2 datasets: dictionary /
                        dimension index, how many datasets keep it (and
                        with which sign), each dataset's value
  support_sizes.png     support sizes as bars, one panel per (tag, layer)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .directions import (DIRECTION_KEYS, raw_direction,
                         stack_directions)
from .plots import FS, panel_grid, save
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction,
                           npz_path, resolve_root)

COMPARABLE = ("raw", "sae", "transcoder")   # column i = the same feature


def load_l1_fits(root: Path) -> dict:
    """(tag, layer) -> {train dataset: npz dict} for every per_dataset
    l1reg fit whose npz has a coefficient (older runs are skipped with a
    note)."""
    m = load_metrics("per_dataset", root)
    runs = m[m["method"] == "l1reg"][["features", "layer", "train_dataset"]] \
        .drop_duplicates()
    fits = {}
    for r in runs.itertuples():
        d = load_prediction(npz_path("per_dataset", r.features, r.layer,
                                     "l1reg", r.train_dataset, root),
                            keys=DIRECTION_KEYS)
        if "coef" not in d:
            print(f"  {r.features} layer{r.layer} {r.train_dataset}: "
                  "no coef in npz (pre-2026-07-20 run), skipped")
            continue
        fits.setdefault((r.features, r.layer), {})[r.train_dataset] = d
    return fits


def support_sizes(fits: dict, tol: float) -> pd.DataFrame:
    """One row per fit: total features, nonzero count/fraction, chosen C."""
    rows = []
    for (tag, layer), group in sorted(fits.items()):
        for dataset, d in sorted(group.items()):
            coef = np.asarray(d["coef"]).ravel()
            nz = int((np.abs(coef) > tol).sum())
            rows.append({"features": tag, "layer": layer, "dataset": dataset,
                         "n_features": coef.size, "n_nonzero": nz,
                         "frac_nonzero": round(nz / coef.size, 4),
                         "C": d["params"].get("C", "")})
    return pd.DataFrame(rows)


def recurrence(group: dict, tol: float) -> pd.DataFrame:
    """Features kept by >= 2 of the group's datasets, with each dataset's
    cache-unit value (0 = not kept). The support mask is taken on the
    stored coef (where the penalty acted), then values are mapped to
    cache units — the mapping divides by a positive scale, so it cannot
    change which entries are zero or their signs."""
    names = sorted(group)
    dirs = []
    for ds in names:
        d = group[ds]
        w, idx = raw_direction(d, "l1reg")
        w = np.where(np.abs(np.asarray(d["coef"]).ravel()) > tol, w, 0.0)
        dirs.append((w, idx))
    W, union = stack_directions(dirs)
    kept = W != 0
    counts = kept.sum(axis=0)
    cols = np.flatnonzero(counts >= 2)
    df = pd.DataFrame({
        "feature": union[cols] if union is not None else cols,
        "n_datasets": counts[cols],
        "n_pos": (W[:, cols] > 0).sum(axis=0),
        "n_neg": (W[:, cols] < 0).sum(axis=0),
        **{ds: W[i, cols].round(4) for i, ds in enumerate(names)},
    })
    order = np.lexsort((-np.abs(W[:, cols]).mean(axis=0),
                        -counts[cols].astype(float)))
    return df.iloc[order].reset_index(drop=True)


def size_panels(sizes: pd.DataFrame):
    """Support-size bars, one panel per (tag, layer), sized for the
    rcParams text (the pre-2026-07-23 version hardcoded tiny fonts)."""
    groups = list(sizes.groupby(["features", "layer"]))
    fig, axes = panel_grid(len(groups), 5.6, 4.6, ncols=4)
    for ax, ((tag, layer), g) in zip(axes, groups):
        g = g.sort_values("dataset")
        ax.bar(range(len(g)), g["n_nonzero"], color="tab:blue")
        ax.set_xticks(range(len(g)), g["dataset"], rotation=45,
                      ha="right", fontsize=FS["tick"] - 3)
        ax.set_title(f"{tag} layer{layer} (of {g['n_features'].max()})",
                     fontsize=FS["subtitle"])
    fig.suptitle("l1reg support sizes")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="|coef| above this counts as nonzero")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    fits = load_l1_fits(args.root)
    if not fits:
        raise FileNotFoundError(f"no per_dataset l1reg fits with "
                                f"coefficients under {args.root}")
    tables, figures = analysis_dirs(args.root)

    sizes = support_sizes(fits, args.tol)
    sizes.to_csv(tables / "support_sizes.csv", index=False)

    save(size_panels(sizes), figures / "support_sizes.png")

    n_rec = 0
    for (tag, layer), group in sorted(fits.items()):
        if tag.rsplit("_", 1)[0] not in COMPARABLE or len(group) < 2:
            continue
        recurrence(group, args.tol).to_csv(
            tables / f"recurrence_{tag}_layer{layer}.csv", index=False)
        n_rec += 1
    print(f"coef_sparsity: wrote support_sizes csv + figure and {n_rec} "
          f"recurrence csvs under {tables.parent}")


if __name__ == "__main__":
    main()
