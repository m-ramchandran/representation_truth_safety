"""How similar are the directions different training datasets learn?

Per method, the dataset x dataset similarity of the per_dataset
coefficient directions (in cache units, via directions.py), under up to
three measures:

  plain     cosine similarity (the Sigma = I baseline)
  mahalanobis_reg_cov_pool  Mahalanobis cosine whitened by the pooled corpus covariance
  mahalanobis_reg_cov_test  Mahalanobis cosine with column D whitened by dataset D's own
            covariance — asymmetric: how similar do two directions look
            in the evaluated dataset's geometry?

Covariances are Ledoit-Wolf-shrunk estimates on the tag's raw
activations, never the raw sample covariance, so the
mahalanobis_reg_cov_* measures exist only for feature sets whose
directions live in raw activation space (raw, and pca back-projected);
sae / transcoder tags get plain cosine only, aligned on the union of
their kept features. Signs are kept everywhere: anti-alignment is a
finding, not noise. The estimates are cached under
<root>/covariances/<tag>_layer<L>.npz (see cached_covs) — reruns skip
the activation loading, the slow part of this module; the cache
re-estimates itself when the dataset roster changes. Like
predictions/, this directory is big and is not meant to be copied
around.

Also written: a summary per method and measure — mean within-task and
cross-task similarity, and the Spearman rank correlation of
similarity(A, B) against transfer AUC(A -> B) (does direction geometry
predict transfer?) — and the M/L/S triangle: per dataset D, the
pairwise similarities of the Merged whole-corpus fit, the Leave-out
pool fit that held out D, and D's own Single fit. The 1B headline to
check: M and L agree under the pooled geometry while, in D's own
geometry (mahalanobis_reg_cov_test), the leave-out direction can diverge from both.

    python -m classifiers.analysis.coef_similarity --tag raw_last --layer 9 \
        [--methods l2reg diff ...] [--holdout-kind dataset|task] \
        [--plain-only] [--root ...]

Outputs under <root>/coefficients/: similarity_<method>_<tag>_layer<L>.csv
(long form), similarity_summary_<tag>_layer<L>.csv,
mls_<tag>_layer<L>_<holdout kind>.csv with ONE heatmap figure per
(layer, kind) — a panel per method, columns = the three pairs grouped
measure-major — and one similarity heatmap figure per measure (a panel
per method). Missing npz — methods or experiments not yet run — are
skipped with a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data import task_of
from . import directions
from ..fig_style import heatmap_size
from .directions import (LINEAR_METHODS, RAW_SPACE, leave_out_name,
                         load_direction, stack_directions)
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           resolve_root)
from .plots import (heatmap, method_panels, panel_grid, save,
                    subtitle_corner)

MLS_PAIRS = (("M_L", "merged", "leave_out"), ("M_S", "merged", "single"),
             ("L_S", "leave_out", "single"))


def mls_figure(mls_df: pd.DataFrame, stem: str, kind: str):
    """One figure for the whole MLS table: a panel per method, columns
    = the three pairs grouped measure-major. Also called by redraw.py on
    the saved csv."""
    mls_df = mls_df.copy()
    pair_order = [p for p, *_ in MLS_PAIRS]
    cols = [f"{p} ({m})" for m in mls_df["measure"].unique()
            for p in pair_order]
    mls_df["col"] = mls_df["pair"] + " (" + mls_df["measure"] + ")"
    methods = sorted(mls_df["method"].unique())
    mats = {m: mls_df[mls_df["method"] == m]
            .pivot(index="dataset", columns="col", values="value")
            .reindex(columns=cols) for m in methods}
    n_ds = max(len(df) for df in mats.values())
    fig, axes = panel_grid(len(methods), *heatmap_size(n_ds, len(cols)))
    for ax, m in zip(axes, methods):
        heatmap(mats[m], vmin=-1, vmax=1, cmap="coolwarm",
                title=m, ax=ax)
    fig.suptitle("Merged / Leave-out / Single direction cosines")
    subtitle_corner(fig, f"{stem} — held-out {kind}")
    return fig


def lw_cov(X) -> np.ndarray:
    """Ledoit-Wolf-shrunk covariance of one activation matrix
    (sklearn). Memory note: sklearn holds several full-size float64
    temporaries — ~14 GB at pooled-corpus size — so the job that
    estimates (rather than cache-hits) covariances needs the memory
    headroom of ~24 cores. That is the deliberate trade: sklearn's
    maintained estimator plus a fatter one-time job, over a hand-rolled
    chunked implementation."""
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(np.asarray(X, dtype=np.float64)).covariance_


def cached_covs(root: Path, tag: str, layer: int, names: list[str],
                data: str) -> dict:
    """name -> Ledoit-Wolf covariance, plus "pool": estimating these is
    the slow part of this module (it loads every dataset's activation
    cache), so they are cached under <root>/covariances/ — one npz per
    (tag, layer), reused whenever it covers exactly the requested
    datasets and recomputed (and rewritten) whenever the roster
    changes. Stored float64: the estimates may be inverted later, so
    keep the full estimation precision — ~52 MB per matrix at d=2560.
    A float32 cache fails the dtype check below and is re-estimated.
    Like predictions/, this directory is not meant to be copied
    around."""
    names = sorted(names)
    path = root / "covariances" / f"{tag}_layer{layer}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if list(d["pool_names"]) == names \
                and d["cov_pool"].dtype == np.float64:
            print(f"  covariances from cache ({path.name})")
            return {**{n: d[f"cov_{n}"] for n in names},
                    "pool": d["cov_pool"]}
        print("  covariance cache stale (roster or dtype), re-estimating")
    units = directions.load_units(layer, tag.rsplit("_", 1)[1],
                                  ("raw",), data)
    X = {n: units[n]["blocks"]["raw"] for n in names}
    covs = {n: lw_cov(X[n]) for n in names}
    covs["pool"] = lw_cov(np.concatenate([X[n] for n in names]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pool_names=np.array(names),
                        cov_pool=covs["pool"],
                        **{f"cov_{n}": covs[n] for n in names})
    print(f"  covariances estimated and cached -> {path}")
    return covs


def cos_plain(W: np.ndarray) -> np.ndarray:
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    return Wn @ Wn.T


def cos_sigma(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Mahalanobis cosine of every row pair under one covariance."""
    M = (W @ S) @ W.T
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def cos_sigma_test(W: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Asymmetric matrix: column j whitened by covs[j] (dataset j's own
    geometry)."""
    C = np.empty((len(W), len(W)))
    for j, S in enumerate(covs):
        V = W @ S                                  # (n, d)
        d = np.sqrt((V * W).sum(axis=1))           # w_i' S w_i
        C[:, j] = (V @ W[j]) / (d * d[j])
    return C


def similarity_matrices(W: np.ndarray, names: list[str],
                        covs: dict | None) -> dict[str, pd.DataFrame]:
    """measure -> dataset x dataset DataFrame. mah_* only when
    covariances are given (raw-space tags)."""
    out = {"plain": cos_plain(W)}
    if covs is not None:
        out["mahalanobis_reg_cov_pool"] = cos_sigma(W, covs["pool"])
        out["mahalanobis_reg_cov_test"] = cos_sigma_test(W, [covs[n] for n in names])
    return {m: pd.DataFrame(v, index=names, columns=names)
            for m, v in out.items()}


def transfer_aucs(metrics: pd.DataFrame, method: str) -> dict:
    """(train, test) -> transfer AUC for one method."""
    m = metrics[(metrics["method"] == method)
                & (metrics["train_dataset"] != metrics["test_dataset"])]
    return {(r.train_dataset, r.test_dataset): r.auc for r in m.itertuples()}


def summarize(mats: dict, method: str, aucs: dict,
              data: str = "original") -> list[dict]:
    """Per measure: within/cross-task mean similarity and the Spearman
    correlation of similarity(A,B) with transfer AUC(A->B)."""
    rows = []
    for measure, df in mats.items():
        pairs = [(a, b) for a in df.index for b in df.columns if a != b]
        within = [df.loc[a, b] for a, b in pairs
                  if task_of(a, data) == task_of(b, data)]
        cross = [df.loc[a, b] for a, b in pairs
                 if task_of(a, data) != task_of(b, data)]
        scored = [(df.loc[a, b], aucs[(a, b)]) for a, b in pairs
                  if (a, b) in aucs]
        sims, aucs_ = (np.array(v) for v in zip(*scored)) if scored \
            else (np.array([]), np.array([]))
        rho = spearmanr(sims, aucs_)[0] if len(sims) > 2 \
            and sims.std() > 1e-12 and aucs_.std() > 1e-12 else np.nan
        rows.append({"method": method, "measure": measure,
                     "within_task": np.mean(within) if within else np.nan,
                     "cross_task": np.mean(cross) if cross else np.nan,
                     "spearman_vs_transfer_auc": rho,
                     "n_pairs_with_auc": len(scored)})
    return rows


def mls_rows(tag: str, layer: int, method: str, singles: dict,
             holdout_kind: str, covs: dict | None, root: Path,
             data: str = "original") -> list[dict]:
    """The Merged / Leave-out / Single triangle per dataset. `singles`
    holds the already-loaded per_dataset (w, idx) directions; leave-out
    loads are memoized per holdout. Datasets whose leave-out or merged
    fit is missing are skipped with a note."""
    try:
        merged = load_direction("merged", tag, layer, method, root=root)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  mls {method}: no merged fit ({e})")
        return []
    rows, leaves = [], {}
    for ds in singles:
        name = leave_out_name(ds, holdout_kind, data)
        try:
            if name not in leaves:
                leaves[name] = load_direction("leave_out", tag, layer,
                                              method, name, root=root)
            fits = {"merged": merged, "leave_out": leaves[name],
                    "single": singles[ds]}
        except (FileNotFoundError, KeyError) as e:
            print(f"  mls {method} {ds}: skipped ({e})")
            continue
        for pair, a, b in MLS_PAIRS:
            W, _ = stack_directions([fits[a], fits[b]])
            vals = {"plain": cos_plain(W)[0, 1]}
            if covs is not None:
                vals["mahalanobis_reg_cov_pool"] = cos_sigma(W, covs["pool"])[0, 1]
                vals["mahalanobis_reg_cov_test"] = cos_sigma(W, covs[ds])[0, 1]
            rows += [{"dataset": ds, "method": method, "pair": pair,
                      "measure": m, "value": round(v, 4)}
                     for m, v in vals.items()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--methods", nargs="+",
                    default=list(LINEAR_METHODS),
                    choices=LINEAR_METHODS)
    ap.add_argument("--holdout-kind", default="dataset",
                    choices=("dataset", "task"))
    ap.add_argument("--plain-only", action="store_true",
                    help="skip the covariance-based measures (no "
                         "activation loading)")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    metrics = load_metrics("per_dataset", args.root)
    metrics = metrics[(metrics["features"] == args.tag)
                      & (metrics["layer"] == args.layer)]
    tables, figures = analysis_dirs(args.root)
    stem = f"{args.tag}_layer{args.layer}"

    # directions per method (datasets with a saved fit for that method)
    per_method = {}
    for method in args.methods:
        dirs, names = [], []
        for ds in sorted(metrics[metrics["method"] == method]
                         ["train_dataset"].unique()):
            try:
                dirs.append(load_direction("per_dataset", args.tag,
                                           args.layer, method, ds,
                                           root=args.root))
                names.append(ds)
            except (FileNotFoundError, KeyError) as e:
                print(f"  {method} {ds}: skipped ({e})")
        if len(names) >= 2:
            per_method[method] = (names, dict(zip(names, dirs)),
                                  stack_directions(dirs))
        else:
            print(f"  {method}: fewer than 2 fits with coefficients, "
                  "skipped")
    if not per_method:
        raise FileNotFoundError(f"no per_dataset fits for tag={args.tag!r} "
                                f"layer={args.layer}")

    # covariances (raw-space tags only, unless --plain-only)
    covs = None
    if args.tag.rsplit("_", 1)[0] in RAW_SPACE and not args.plain_only:
        all_names = sorted({n for names, _, _ in per_method.values()
                            for n in names})
        covs = cached_covs(args.root, args.tag, args.layer, all_names,
                           args.data)

    summary, mls, panels = [], [], {}
    for method, (names, singles, (W, idx)) in per_method.items():
        mats = similarity_matrices(W, names,
                                   covs if idx is None else None)
        long = pd.concat([df.rename_axis("dataset_a")
                            .reset_index()
                            .melt("dataset_a", var_name="dataset_b",
                                  value_name="value")
                            .assign(measure=m)
                          for m, df in mats.items()])
        long.round(4).to_csv(tables / f"similarity_{method}_{stem}.csv",
                             index=False)
        summary += summarize(mats, method, transfer_aucs(metrics, method),
                             args.data)
        for m, df in mats.items():
            panels[(method, m)] = df.rename_axis(
                index="fit", columns="dataset").stack()
        mls += mls_rows(args.tag, args.layer, method, singles,
                        args.holdout_kind, covs if idx is None else None,
                        args.root, args.data)

    pd.DataFrame(summary).round(3).to_csv(
        tables / f"similarity_summary_{stem}.csv", index=False)
    if mls:
        mls_df = pd.DataFrame(mls)
        mls_df.to_csv(tables / f"mls_{stem}_{args.holdout_kind}.csv",
                      index=False)
        save(mls_figure(mls_df, stem, args.holdout_kind),
             figures / f"mls_{stem}_{args.holdout_kind}.png")

    written = []
    for measure in {m for _, m in panels}:
        s = pd.concat({meth: series for (meth, m), series in panels.items()
                       if m == measure}, names=["method"])
        fig = method_panels(s, vmin=-1, vmax=1, cmap="coolwarm",
                            title=f"Direction similarity ({measure})",
                            subtitle=stem)
        save(fig, figures / f"similarity_{measure}_{stem}.png")
        written.append(measure)
    print(f"coef_similarity: wrote {len(per_method)} per-method csvs, "
          f"summary, {'mls, ' if mls else ''}and {written} heatmaps "
          f"under {tables.parent}")


if __name__ == "__main__":
    main()
