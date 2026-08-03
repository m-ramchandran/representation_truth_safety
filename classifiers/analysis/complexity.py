"""Model complexity per run: trained parameters and total rule size.

Two counts per run, both at the hyperparameters the run actually chose
(mlp's hidden width, moe's expert count, svm_rbf's landmark count):

  n_trained   numbers *estimated* from the training data — by gradient
              descent, a convex solver, or closed form (class means,
              covariances, the diff direction). knn trains nothing;
              svm_rbf's trained part is its linear weights on the
              Nystrom features.
  n_total     everything the fitted rule needs to score a new sample:
              n_trained plus memorized data — knn's stored training
              set, svm_rbf's landmark rows and whitening map. For every
              other method the two counts are equal.

Standardization, featurization (PCA basis, bin edges, kept-feature
masks), and the Platt (a, b) are excluded: the first two belong to the
feature set, not the method, and Platt adds the same 2 numbers wherever
it applies. Merged runs are counted at the whole-corpus fit (the
`full_` arrays / `full_params`).

The rules, with d = input dimension after featurization:

  method          n_trained            n_total
  l1reg/l2reg/svm_linear   d + 1       same   (lda's means + pooled
        /lda/diff                       covariance collapse to d + 1)
  qda             d(d+1)/2 + d + 1     same   (the collapsed quadratic
        rule; two full class covariances are estimated to get there)
  knn             0                    n_train * (d + 1)  (memorized
        training set + labels)
  svm_rbf         m' + 1               m*d + m*m' + m' + 1  (m landmark
        rows, the m x m' whitening, linear weights on m' features)
  mlp             (d + 1)*h + h + 1    same
  random_slopes   (G + 1)*(d + 1)      same   (G pooled datasets — or
        pooled tasks for random_slopes_task; xgboost sits this analysis
        out: tree structure isn't saved in the npz) (an
        unknown-dataset sample is scored with the shared d + 1 only)
  moe             2K*(d + 1)           same   (K linear experts + gate)

    python -m classifiers.analysis.complexity [--data ...] [--root ...]
        [--experiment per_dataset leave_out merged]

Writes <root>/<experiment>/analysis/tables/complexity.csv — one row per
run with d, n_train, the size-determining hyperparameters, both counts,
and `n_nonzero` (nonzero coefficients, where a coef in input space
exists — l1reg's effective size) — and one log-scale grouped bar chart
per feature set under analysis/figures/, one row of bars per layer
(bars = median over the experiment's fits, dots = the individual
fits' totals).
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction, npz_path, resolve_root)
from .plots import FS, save

COMPLEXITY_KEYS = frozenset(
    p + k for p in ("", "full_")
    for k in ("coef", "standardize_scale", "group_labels",
              "expert_w")) | {"cv_y"}

METHOD_ORDER = ("diff", "knn", "l1reg", "l2reg", "svm_linear", "svm_rbf",
                "lda", "qda", "mlp", "random_slopes", "random_slopes_task",
                "moe")

# xgboost has no comparable parameter count: tree structure isn't saved
# in the npz (only the gain importances), so it sits this analysis out.
NO_COMPLEXITY_RULE = ("xgboost",)


def _input_dim(d: dict, prefix: str) -> int:
    """Post-featurization input dimension: the standardizer's width, or
    the coefficient's for diff (which has no standardizer)."""
    for key in (f"{prefix}standardize_scale", f"{prefix}coef"):
        if key in d:
            return int(np.asarray(d[key]).ravel().shape[0])
    raise KeyError("no standardize_scale or coef to read the input "
                   "dimension from (an old run without model arrays?)")


def complexity_row(d: dict, method: str) -> dict:
    """One npz's size accounting: d, n_train, the size-determining
    hyperparameters, n_trained / n_total, and n_nonzero (see module
    docstring)."""
    prefix = "full_" if any(k.startswith("full_") for k in d) else ""
    params = d.get(f"{prefix}params", d["params"])
    if isinstance(params, list):                 # merged without full_params
        params = params[0]
    dim = _input_dim(d, prefix)
    n_train = len(d["cv_y"])
    hyper = ""
    if method in ("l1reg", "l2reg", "svm_linear", "lda", "diff"):
        trained = total = dim + 1
    elif method == "qda":
        trained = total = dim * (dim + 1) // 2 + dim + 1
    elif method == "knn":
        trained, total = 0, n_train * (dim + 1)
        if "n_neighbors" in params:
            hyper = f"k={params['n_neighbors']}"
    elif method == "svm_rbf":
        m = int(params["n_landmarks"])
        mk = int(np.asarray(d[f"{prefix}coef"]).ravel().shape[0])
        trained = mk + 1
        total = m * dim + m * mk + mk + 1
        hyper = f"m={m}"
    elif method == "mlp":
        h = int(params["hidden"])
        trained = total = (dim + 1) * h + h + 1
        hyper = f"h={h}"
    elif method in ("random_slopes", "random_slopes_task"):
        g = len(d[f"{prefix}group_labels"])   # datasets, or tasks
        trained = total = (g + 1) * (dim + 1)
        hyper = f"G={g}"
    elif method == "moe":
        k = (int(np.asarray(d[f"{prefix}expert_w"]).shape[0])
             if f"{prefix}expert_w" in d else int(params["K"]))
        trained = total = 2 * k * (dim + 1)
        hyper = f"K={k}"
    else:
        raise KeyError(f"no complexity rule for method {method!r}")
    row = {"d": dim, "n_train": n_train, "size_hyper": hyper,
           "n_trained": trained, "n_total": total, "n_nonzero": ""}
    coef = d.get(f"{prefix}coef")
    if coef is not None and np.asarray(coef).ravel().shape[0] == dim:
        row["n_nonzero"] = int(np.count_nonzero(coef))
    return row


def _runs(m: pd.DataFrame, experiment: str):
    """(features, layer, method, npz dataset key, label) per run."""
    if experiment == "per_dataset":
        cols = ["features", "layer", "method", "train_dataset"]
        for r in m[cols].drop_duplicates().itertuples():
            yield (r.features, r.layer, r.method, r.train_dataset,
                   r.train_dataset)
    elif experiment == "leave_out":
        cols = ["features", "layer", "method", "holdout_kind", "holdout"]
        for r in m[cols].drop_duplicates().itertuples():
            yield (r.features, r.layer, r.method,
                   f"{r.holdout_kind}_{r.holdout}", r.holdout)
    else:
        for r in m[["features", "layer", "method"]].drop_duplicates() \
                .itertuples():
            yield r.features, r.layer, r.method, None, "merged"


def _fmt(n: float) -> str:
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= cut:
            return f"{n / cut:.3g}{suffix}"
    return f"{n:.0f}"


def complexity_bars(ax, df: pd.DataFrame, title: str,
                    legend: bool = False) -> None:
    """Log-scale grouped bars per method on one axis — trained
    parameters next to the total the rule carries (labeled medians over
    the group's fits; they differ only for knn and svm_rbf) — with each
    individual fit's total a dot."""
    med = (df.groupby("method")[["n_trained", "n_total"]].median()
           .reindex([m for m in METHOD_ORDER
                     if m in set(df["method"])]).dropna())
    x = np.arange(len(med), dtype=float)
    ax.bar(x - 0.19, med["n_trained"], 0.36, color="#265fa8", zorder=2,
           label="trained parameters")
    ax.bar(x + 0.19, med["n_total"], 0.36, color="#9ec2e8", zorder=2,
           label="total parameters")
    first = True
    for xi, method in zip(x, med.index):
        pts = df.loc[df["method"] == method, "n_total"]
        ax.scatter(np.full(len(pts), xi + 0.19), pts, s=8, color="k",
                   zorder=3, label="individual fits" if first else None)
        first = False
    ax.set_yscale("log")
    ymin = ax.get_ylim()[0]
    for xi, method in zip(x, med.index):
        for off, col in ((-0.19, "n_trained"), (0.19, "n_total")):
            v = med.loc[method, col]
            ax.annotate(_fmt(v) if v > 0 else "0", (xi + off, max(v, ymin)),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=FS["cell"])
    ax.set_xticks(x, med.index, rotation=45, ha="right",
                  fontsize=FS["tick"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.set_ylabel("parameters (log)", fontsize=FS["label"])
    ax.set_title(title, fontsize=FS["title"] - 4)
    if legend:   # to the right of the axes — inside, it sits on the bars
        ax.legend(fontsize=FS["legend"], loc="upper left",
                  bbox_to_anchor=(1.02, 1), borderaxespad=0)


def complexity_figure(df: pd.DataFrame, experiment: str, features: str):
    """One figure per (experiment, feature set): a row of bars per
    layer, so the layers are subfigures rather than separate files."""
    layers = sorted(df["layer"].unique())
    n_methods = df["method"].nunique()
    fig, axes = plt.subplots(len(layers), 1,
                             figsize=(2.1 * n_methods + 3,
                                      5.6 * len(layers)),
                             squeeze=False, layout="constrained")
    for ax, layer in zip(axes[:, 0], layers):
        complexity_bars(ax, df[df["layer"] == layer], f"layer {layer}",
                        legend=(layer == layers[0]))
    fig.suptitle(f"{experiment} {features} — model size",
                 fontsize=FS["suptitle"])
    return fig


def run_experiment(experiment: str, root) -> pd.DataFrame:
    m = load_metrics(experiment, root)
    rows, missing = [], 0
    for features, layer, method, ds_key, label in _runs(m, experiment):
        if method in NO_COMPLEXITY_RULE:
            continue
        path = npz_path(experiment, features, layer, method, ds_key,
                        root=root)
        if not path.exists():        # metrics synced without this npz
            missing += 1
            continue
        rows.append({"features": features, "layer": layer,
                     "method": method, "fit": label,
                     **complexity_row(load_prediction(
                         path, keys=COMPLEXITY_KEYS), method)})
    if missing:
        print(f"{experiment}: {missing} runs in the metrics tables have "
              f"no local npz, skipped")
    if not rows:
        raise FileNotFoundError(f"no prediction npz under "
                                f"{root / experiment / 'predictions'}")
    df = pd.DataFrame(rows).sort_values(
        ["features", "layer", "method", "fit"])
    tables, figures = analysis_dirs(root, experiment)
    df.to_csv(tables / "complexity.csv", index=False)
    for features, g in df.groupby("features"):
        save(complexity_figure(g, experiment, features),
             figures / f"complexity_{features}.png")
    print(f"{experiment}: {len(df)} runs, "
          f"{df['features'].nunique()} charts -> {tables.parent}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_args(ap)
    ap.add_argument("--experiment", nargs="+",
                    default=["per_dataset", "leave_out", "merged"],
                    choices=["per_dataset", "leave_out", "merged"])
    args = ap.parse_args()
    args.root = resolve_root(args)

    done = 0
    for experiment in args.experiment:
        try:
            run_experiment(experiment, args.root)
            done += 1
        except FileNotFoundError as e:
            print(f"{experiment}: skipped ({e})")
    if not done:
        raise FileNotFoundError(f"no experiment results under {args.root}")


if __name__ == "__main__":
    main()
