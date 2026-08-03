"""Do the per-dataset directions share a low-dimensional subspace, and
does the merged direction live in it?

Each per-dataset direction (cache units via directions.py) is unit-normalized
— so every dataset has equal say regardless of its fit's weight
magnitude, which varies with C and carries no signal — then the rows are
mean-centered and PCA'd. Every reconstruction includes the mean, so k=0
is the bare mean direction and the PCs rank deviations from it; at full
rank the reconstruction is exact and recovers the original fit's AUC
(the built-in sanity check). AUCs are threshold-free (X @ w), so the
dropped intercepts don't matter.

Four analyses per (tag, layer, method):
  spectrum        explained variance per PC
  reconstruction  each dataset's direction approximated by mean + top-k
                  PCs, AUC on its own data for k = 1..n-1, with the
                  original fit's AUC as the ceiling
  individual PCs  AUC of (mean + that PC's term alone) per (dataset, PC)
  held-out        PCA on the OTHER datasets' directions only; k=0 is
                  their bare mean applied to the held-out dataset (pure
                  transfer), higher k adds its coordinates within the
                  others' subspace; the gap to its own fit's AUC is its
                  private, orthogonal part

Plus the pooled sections (when those fits exist):
  merged      the whole-corpus full_coef direction over TWO bases, side
              by side: the single-dataset
              directions, and the leave-out pool directions (one per
              held-out dataset / task, --holdout-kind). Per basis:
              cosine to the basis mean, energy inside the span, and the
              least-squares weighted sum with its pooled AUC next to
              the merged fit's own and the mean's (closing the 1B
              sum-vs-merged question: plain sum 0.61-0.68, unit sum
              0.73-0.76, merged ~0.92). The leave-out directions are
              each nearly-pooled fits, so the gap between the two
              spans measures what pooling adds beyond any combination
              of single-dataset classifiers. Figure: one bar panel of
              weights per basis, that basis's scalars as its subtitle.
  leave-out   per held-out dataset D (--holdout-kind dataset, or D's
              whole task): the leave-out pool direction's cosine to D's
              own single direction and to the pool singles' mean, its
              energy inside the pool singles' span, and its AUC on D.

Raw-space feature sets only (raw, pca): sae/transcoder supports and
bin/PC bases differ per fit, so their directions share no common basis.

    python -m classifiers.analysis.coef_pca --tag raw_last \
        [--layers 17 22 29 33] [--method l2reg] \
        [--holdout-kind dataset|task] [--root ...]

Outputs under <root>/coefficients/: csvs per layer (suffixed
_<method>_<tag>_layer<L>, merged/leave-out also by holdout kind);
the reconstruction / individual / heldout / leave-out figures hold
all requested layers as panels (suffixed _<method>_<tag>[_<kind>]),
while pca_merged keeps one two-basis figure per layer.
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from ..data import COLLECTIONS
from . import directions
from .directions import (LINEAR_METHODS, RAW_SPACE, cosine,
                         leave_out_name, load_direction,
                         stack_directions, unit_rows)
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           resolve_root)
from .plots import (FS_SMALL, layer_panels, line_titles, panel_grid,
                    save, subtitle_corner)


def auc_of(w: np.ndarray, X, y) -> float:
    return float(roc_auc_score(y, np.asarray(X) @ w))


def auc_lines_panel(ax, wide: pd.DataFrame, title: str) -> None:
    """One line per dataset over k on one axis, the 'original' column
    as the dashed ceiling."""
    ks = [c for c in wide.columns if c != "original"]
    for ds, row in wide.iterrows():
        line, = ax.plot(range(len(ks)), row[ks], marker="o", ms=4,
                        label=ds)
        ax.axhline(row["original"], color=line.get_color(), lw=.8,
                   ls="--", alpha=.5)
    ax.set_xticks(range(len(ks)), ks, fontsize=FS_SMALL["tick"])
    ax.tick_params(axis="y", labelsize=FS_SMALL["tick"])
    ax.set_xlabel("components used", fontsize=FS_SMALL["label"])
    ax.set_ylabel("AUC", fontsize=FS_SMALL["label"])
    ax.set_title(title, fontsize=FS_SMALL["title"])
    ax.grid(True, alpha=.3)


def layer_line_figure(by_layer: dict, title: str, subtitle: str):
    """A panel of auc lines per layer, two per row, one shared legend
    beside the figure — all layers in one figure."""
    fig, axes = panel_grid(len(by_layer), 7.2, 4.6, ncols=2)
    for ax, (layer, wide) in zip(axes, sorted(by_layer.items())):
        auc_lines_panel(ax, wide, f"layer {layer}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper",
               fontsize=FS_SMALL["legend"])
    fig.suptitle(title, fontsize=FS_SMALL["suptitle"])
    subtitle_corner(fig, subtitle)
    return fig


def reconstruction(pca: PCA, scores, names, data) -> pd.DataFrame:
    """AUC of mean + top-k reconstruction per dataset, k = 1..rank."""
    out = pd.DataFrame(index=pd.Index(names, name="dataset"))
    for k in range(1, pca.n_components_ + 1):
        recon = pca.mean_ + scores[:, :k] @ pca.components_[:k]
        out[f"k{k}"] = [auc_of(recon[i], *data[n])
                        for i, n in enumerate(names)]
    return out


def individual_pcs(pca: PCA, scores, names, data) -> pd.DataFrame:
    """AUC of (mean + a single PC's term) per (dataset, PC)."""
    out = pd.DataFrame(index=pd.Index(names, name="dataset"))
    for j in range(pca.n_components_):
        out[f"PC{j + 1}"] = [
            auc_of(pca.mean_ + scores[i, j] * pca.components_[j], *data[n])
            for i, n in enumerate(names)]
    return out


def heldout(Wn: np.ndarray, names, data) -> pd.DataFrame:
    """For each dataset: PCA on the others' directions, AUC of
    mean + first-k coordinates of the held-out direction, k = 0..rank."""
    rows = {}
    for i, name in enumerate(names):
        others = np.delete(Wn, i, axis=0)
        p = PCA(n_components=len(others) - 1, svd_solver="full").fit(others)
        c = p.transform(Wn[i:i + 1])[0]
        rows[name] = [auc_of(p.mean_ + c[:k] @ p.components_[:k],
                             *data[name])
                      for k in range(p.n_components_ + 1)]
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "dataset"
    out.columns = [f"k{k}" for k in out.columns]
    return out


def basis_rows(wm, B: np.ndarray, names, pooled_auc, basis: str) -> list:
    """The merged direction over one basis of directions: cosine to the
    basis mean, energy inside the span, the least-squares weights, and
    the pooled AUCs of merged / lstsq sum / bare mean."""
    mean_dir = B.mean(axis=0)
    beta = np.linalg.lstsq(B.T, wm, rcond=None)[0]
    w_ls = B.T @ beta
    # merged's unit energy split three ways: along the basis mean, the
    # rest of the span, and outside it
    cos_mean = cosine(wm, mean_dir)
    in_span = float(np.linalg.norm(w_ls) ** 2)
    return [(basis, "cos_merged_mean_dir", "", cos_mean),
            (basis, "energy_in_span", "", in_span),
            (basis, "energy_along_mean", "", cos_mean ** 2),
            (basis, "energy_rest_of_span", "", in_span - cos_mean ** 2),
            (basis, "energy_outside_span", "", 1.0 - in_span),
            (basis, "pooled_auc_merged", "", pooled_auc(wm)),
            (basis, "pooled_auc_mean_dir", "", pooled_auc(mean_dir)),
            (basis, "pooled_auc_lstsq_sum", "", pooled_auc(w_ls)),
            *((basis, "lstsq_weight", n, float(b))
              for n, b in zip(names, beta))]


def merged_section(Wn: np.ndarray, names, data, tag, layer, method,
                   lo_dirs, kind, root,
                   covs: dict = None) -> pd.DataFrame | None:
    """The merged full_coef over two bases:
    the single-dataset directions, and the leave-out pool directions
    (one per held-out dataset or task, preloaded in `lo_dirs`). The
    leave-out directions are each nearly-pooled fits, so their span
    should hold the merged direction almost entirely — the gap to the
    singles' span measures what pooling adds beyond any combination of
    single-dataset classifiers. Long form: basis / quantity / dataset /
    value."""
    try:
        wm, _ = load_direction("merged", tag, layer, method, root=root)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  merged section skipped ({e})")
        return None
    wm = wm / np.linalg.norm(wm)
    yp = np.concatenate([data[n][1] for n in names])

    def pooled_auc(w):    # scores per dataset — no pooled matrix copy
        return float(roc_auc_score(yp, np.concatenate(
            [np.asarray(data[n][0]) @ w for n in names])))

    rows = basis_rows(wm, Wn, names, pooled_auc, "single")
    # cos(merged, each dataset's own direction) — the left panel of the
    # 1B-style figure; the Mahalanobis version rides along whenever the
    # covariance cache is already warm (never estimated here)
    for n, w in zip(names, Wn):
        rows.append(("single", "cos_merged_dataset_plain", n,
                     cosine(wm, w)))
    if covs is not None:
        S = covs["pool"]
        for n, w in zip(names, Wn):
            num = float(wm @ S @ w)
            den = float(np.sqrt(wm @ S @ wm) * np.sqrt(w @ S @ w))
            rows.append(("single", "cos_merged_dataset_mahalanobis", n,
                         num / den))
    loaded = {k.split("_", 1)[1]: wi for k, wi in sorted(lo_dirs.items())
              if wi is not None}
    if len(loaded) >= 2:
        L = unit_rows(np.stack([w for w, _ in loaded.values()]))
        rows += basis_rows(wm, L, list(loaded), pooled_auc,
                           f"leave_out_{kind}")
    return pd.DataFrame(rows,
                        columns=("basis", "quantity", "dataset", "value"))


BASIS_LABELS = {"single": "single-dataset coefficients",
                "leave_out_dataset": "leave-out directions (dataset)",
                "leave_out_task": "leave-out directions (task)"}


def merged_figure(merged: pd.DataFrame, stem: str, sizes: dict = None):
    """Merged against the single-dataset subspace. Preferred to a bar
    chart of least-squares weights because it PLOTS the scalars instead
    of burying them in a subtitle:

    left    cos(merged, each dataset's own direction), grouped bars,
            plain vs Mahalanobis (pooled covariance) where the cached
            covariances made the second series available. Datasets are
            ordered by n descending with n in the tick label, so
            "the pooled fit follows the big datasets" reads at a glance;
            a zero line makes sign flips obvious.
    right   where the merged direction lives: one stacked bar per basis
            splitting its unit energy into along the basis mean / the
            rest of the span / outside the span, with the in-span
            fraction annotated above and the pooled AUCs underneath.
    """
    per_ds = merged[merged["quantity"].str.startswith("cos_merged_dataset")]
    scal = merged[merged["dataset"] == ""]
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(max(13.0, 0.62 * per_ds["dataset"].nunique() + 9), 6.8),
        width_ratios=(2.2, 1.0), layout="constrained")

    names = list(per_ds[per_ds["quantity"] == "cos_merged_dataset_plain"]
                 ["dataset"])
    if sizes:                      # biggest datasets first (1B ordering)
        names = sorted(names, key=lambda n: -sizes.get(n, 0))
    x = np.arange(len(names), dtype=float)
    series = [("cos_merged_dataset_plain", "plain cosine", "tab:gray"),
              ("cos_merged_dataset_mahalanobis",
               "Mahalanobis (pooled covariance)", "tab:red")]
    present = [s for s in series
               if (per_ds["quantity"] == s[0]).any()]
    width = 0.8 / len(present)
    for k, (q, label, color) in enumerate(present):
        vals = (per_ds[per_ds["quantity"] == q]
                .set_index("dataset")["value"].reindex(names))
        ax1.bar(x + (k - (len(present) - 1) / 2) * width, vals, width,
                color=color, label=label, zorder=2)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x, [f"{n}\n" + (f"n={sizes[n]:,}" if sizes and n in sizes
                                   else "") for n in names],
                   rotation=45, ha="right", fontsize=FS_SMALL["tick"])
    ax1.tick_params(axis="y", labelsize=FS_SMALL["tick"])
    ax1.set_ylabel("cos(merged, dataset's own direction)",
                   fontsize=FS_SMALL["label"])
    ax1.grid(True, axis="y", alpha=.3)
    ax1.legend(fontsize=FS_SMALL["legend"], loc="best")
    line_titles(ax1, "Alignment of the merged direction with each dataset",
                "" if sizes else None, FS_SMALL)

    bases = list(dict.fromkeys(merged["basis"]))
    parts = [("energy_along_mean", "along the basis mean", "tab:red"),
             ("energy_rest_of_span", "rest of the span", "tab:orange"),
             ("energy_outside_span", "outside the span", "tab:blue")]
    bx = np.arange(len(bases), dtype=float)
    bottom = np.zeros(len(bases))
    for q, label, color in parts:
        vals = np.array([float(scal[(scal["basis"] == b)
                                    & (scal["quantity"] == q)]["value"]
                               .iloc[0]) for b in bases])
        ax2.bar(bx, vals, 0.55, bottom=bottom, color=color, label=label,
                zorder=2)
        bottom += vals
    for i, b in enumerate(bases):
        get = lambda q: float(scal[(scal["basis"] == b)
                                   & (scal["quantity"] == q)]["value"].iloc[0])
        # every scalar above its own bar — nothing below the axes, where
        # it would collide with the tick labels
        # short lines: a single wide line runs into the next bar's note
        ax2.text(i, 1.03, f"in-span {get('energy_in_span'):.3f}\n"
                          f"AUC {get('pooled_auc_merged'):.3f}\n"
                          f"lstsq {get('pooled_auc_lstsq_sum'):.3f}",
                 ha="center", va="bottom", fontsize=FS_SMALL["cell"])
    ax2.set_xticks(bx, ["\n".join(textwrap.wrap(BASIS_LABELS.get(b, b), 18))
                        for b in bases], fontsize=FS_SMALL["tick"])
    ax2.tick_params(axis="y", labelsize=FS_SMALL["tick"])
    ax2.set_ylim(0, 1.42)
    ax2.set_ylabel("fraction of merged's unit energy",
                   fontsize=FS_SMALL["label"])
    # outside the axes — a stacked bar fills 0..1, so any in-axes legend
    # sits on top of the data
    ax2.legend(fontsize=FS_SMALL["legend"], loc="upper left",
               bbox_to_anchor=(1.02, 1), borderaxespad=0)
    line_titles(ax2, "Where the merged direction lives", None, FS_SMALL)

    fig.suptitle(f"Merged direction vs the single-dataset fits — {stem}",
                 fontsize=FS_SMALL["suptitle"])
    return fig


def leave_out_section(Wn: np.ndarray, names, data, lo_dirs, kind,
                      collection) -> pd.DataFrame | None:
    """Each leave-out pool direction (preloaded in `lo_dirs`) against
    the single directions it was trained beside: per held-out dataset
    D, the cosine of the leave-out direction to D's own single
    direction, its cosine to the pool singles' mean direction, the
    energy inside the pool singles' span, and its AUC on D. Task kind
    drops the whole task's singles from the pool, mirroring the
    runner."""
    rows = []
    for i, ds in enumerate(names):
        name = leave_out_name(ds, kind, collection)
        entry = lo_dirs.get(name)
        if entry is None:
            continue
        wl = entry[0] / np.linalg.norm(entry[0])
        pool = [n for n in names
                if leave_out_name(n, kind, collection) != name]
        if len(pool) < 2:
            print(f"  leave-out {ds}: pool of {len(pool)} singles, skipped")
            continue
        P = Wn[[names.index(n) for n in pool]]
        beta = np.linalg.lstsq(P.T, wl, rcond=None)[0]
        rows.append({
            "dataset": ds,
            "cos_single": cosine(wl, Wn[i]),
            "cos_pool_mean_dir": cosine(wl, P.mean(axis=0)),
            "energy_in_pool_span": float(np.linalg.norm(P.T @ beta) ** 2),
            "auc_on_heldout": auc_of(wl, *data[ds]),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("dataset").round(4)


def warm_covs(args, layer: int, names) -> dict | None:
    """The Ledoit-Wolf covariances IF coef_similarity already cached them
    for this (tag, layer) — never estimated here, so this module keeps
    its light activation-only cost."""
    path = args.root / "covariances" / f"{args.tag}_layer{layer}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    stored = list(d["pool_names"])
    if not set(names) <= set(stored):
        return None
    return {"pool": d["cov_pool"], **{n: d[f"cov_{n}"] for n in names}}


def run_layer(args, m: pd.DataFrame, layer: int, out: dict) -> None:
    """One layer: all csvs, the per-layer pca_merged figure, and the
    frames the cross-layer figures assemble later."""
    ml = m[m["layer"] == layer]
    dirs, names = [], []
    for ds in sorted(ml["train_dataset"].unique()):
        try:
            dirs.append(load_direction("per_dataset", args.tag, layer,
                                       args.method, ds, root=args.root))
            names.append(ds)
        except (FileNotFoundError, KeyError) as e:
            print(f"  L{layer} {ds}: skipped ({e})")
    if len(names) < 3:
        raise FileNotFoundError(f"need >= 3 fits with coefficients, have "
                                f"{len(names)}")
    W, _ = stack_directions(dirs)
    Wn = unit_rows(W)

    units = directions.load_units(layer, args.tag.rsplit("_", 1)[1],
                                  ("raw",), args.data)
    data = {n: (units[n]["blocks"]["raw"], units[n]["y"]) for n in names}
    original = pd.Series({n: auc_of(W[i], *data[n])
                          for i, n in enumerate(names)}, name="original")

    # the leave-out pool directions, loaded once per layer and shared
    # by the merged and leave-out sections
    lo_dirs = {}
    for ds in names:
        name = leave_out_name(ds, args.holdout_kind, args.data)
        if name in lo_dirs:
            continue
        try:
            lo_dirs[name] = load_direction("leave_out", args.tag, layer,
                                           args.method, name,
                                           root=args.root)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  leave-out {name}: skipped ({e})")
            lo_dirs[name] = None

    tables, figures = analysis_dirs(args.root)
    stem = f"{args.method}_{args.tag}_layer{layer}"

    pca = PCA(n_components=len(names) - 1, svd_solver="full").fit(Wn)
    scores = pca.transform(Wn)
    evr = pca.explained_variance_ratio_
    pd.DataFrame({"pc": [f"PC{j + 1}" for j in range(len(evr))],
                  "explained_variance_ratio": evr.round(4),
                  "cumulative": np.cumsum(evr).round(4)}) \
        .to_csv(tables / f"pca_spectrum_{stem}.csv", index=False)

    recon = reconstruction(pca, scores, names, data).join(original)
    recon.round(3).to_csv(tables / f"pca_reconstruction_{stem}.csv")
    out["recon"][layer] = recon

    ind = individual_pcs(pca, scores, names, data)
    ind.round(3).to_csv(tables / f"pca_individual_{stem}.csv")
    out["individual"][layer] = ind

    held = heldout(Wn, names, data).join(original)
    held.round(3).to_csv(tables / f"pca_heldout_{stem}.csv")
    out["heldout"][layer] = held

    merged = merged_section(Wn, names, data, args.tag, layer,
                            args.method, lo_dirs, args.holdout_kind,
                            args.root, covs=warm_covs(args, layer, names))
    if merged is not None:
        merged.round(4).to_csv(
            tables / f"pca_merged_{stem}_{args.holdout_kind}.csv",
            index=False)
        sizes = {n: len(data[n][1]) for n in names}
        save(merged_figure(merged, stem, sizes),
             figures / f"pca_merged_{stem}_{args.holdout_kind}.png")

    # csv only — the figure was dropped because its columns duplicate
    # the MLS triangle's L_S and cross_methods' leave-out AUC
    lo = leave_out_section(Wn, names, data, lo_dirs, args.holdout_kind,
                           args.data)
    if lo is not None:
        lo.to_csv(tables / f"pca_leave_out_{stem}_{args.holdout_kind}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="default: the collection's layers")
    ap.add_argument("--method", default="l2reg", choices=LINEAR_METHODS)
    ap.add_argument("--holdout-kind", default="dataset",
                    choices=("dataset", "task"))
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    if args.tag.rsplit("_", 1)[0] not in RAW_SPACE:
        raise ValueError(f"{args.tag}: directions of this feature set "
                         "share no common basis (raw-space tags only)")
    layers = args.layers or list(COLLECTIONS[args.data].layers)

    m = load_metrics("per_dataset", args.root)
    m = m[(m["features"] == args.tag) & (m["method"] == args.method)]
    out = {"recon": {}, "individual": {}, "heldout": {}}
    for layer in layers:
        try:
            run_layer(args, m, layer, out)
        except FileNotFoundError as e:
            print(f"  layer {layer}: skipped ({e})")

    # cross-layer figures, a panel per layer
    _, figures = analysis_dirs(args.root)
    fstem = f"{args.method}_{args.tag}"
    if out["recon"]:
        save(layer_line_figure(
                 out["recon"], "Rank-k reconstruction AUC",
                 f"PCA fit on single-dataset coefficients — {fstem}"),
             figures / f"pca_reconstruction_{fstem}.png")
    if out["heldout"]:
        save(layer_line_figure(
                 out["heldout"], "Held-out subspace AUC",
                 f"PCA fit on the other datasets' single coefficients — "
                 f"{fstem}"),
             figures / f"pca_heldout_{fstem}.png")
    if out["individual"]:
        vmin = min(.5, min(float(v.min().min())
                           for v in out["individual"].values()))
        save(layer_panels(
                 out["individual"], "Single-PC reconstruction AUC",
                 f"PCA fit on single-dataset coefficients — {fstem}",
                 vmin=vmin, vmax=1),
             figures / f"pca_individual_{fstem}.png")
    done = [k for k, v in out.items() if v]
    layers_done = sorted(set().union(*(v.keys() for v in out.values())))
    print(f"coef_pca: layers {layers_done} -> {', '.join(done)} figures "
          f"under {figures.parent}")


if __name__ == "__main__":
    main()
