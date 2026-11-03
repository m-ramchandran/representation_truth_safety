"""All figures for the merged clustering run, in four groups.

    python -m clustering.scripts.make_figures [group ...]     # default: all groups

Figures go under results/merged_last/figures/, and each figure's companion
csv goes to the mirror path under results/merged_last/tables/.

  model_selection  figures/L<layer>/model_selection_gmm.png
  ledger           figures/across_layers/variability_ledger.png
  pca_stability    figures/across_layers/pca_stability.png
  clusters_on_pcs  figures/L<layer>/clusters_on_pcs/<run>.png
  ari_stability    figures/L<layer>/ari_stability.png (+ tables csv)
  method_agreement figures/L<layer>/method_agreement.png (+ tables csv)
  composition      figures/composition/<method>.png
  cross_layer_ari  figures/across_layers/cross_layer_ari.png (+ tables csv)
  embedding_maps   figures/L<layer>/embedding_maps.png

Everything runs from saved files; nothing re-fits. The first three groups
need only the local light artifacts (metrics.csv + saved PCA npz files).
clusters_on_pcs needs each run's assignments.npz and model.joblib (on a compute cluster,
not rsync'd) and the activations; the ARI/composition groups need
assignments.npz, and embedding_maps the saved embedding coords + stability
files. Each group is skipped with a message where its files are missing.

AIC/BIC are only comparable at a fixed number of PCs (the likelihood lives in
a different space at each PC count), so the model-selection grid gives each
PC count its own row/y-scale, and the ledger's reference GMM setting is the
best seed-mean silhouette over (covariance, PC count). BIC's choice of k hits
the k=40 grid edge everywhere, so the ledger reports the silhouette-best fit.

clusters_on_pcs shows every fit (seed 0; GMM and k-means-on-PCs at 15 and 50
PCs only), styled after the 1B standard_GMM component atlas
(1B/experiments/standard_GMM/atlas_data.py). The workhorse figures are one
PC1-PC2 grid per method (per cov for GMM): rows = the setting that isn't k
(PC count / space / factors / neighbors), columns = k (or resolution / min
cluster size). GMM/MFA get two color versions -- points by source dataset
and points by cluster assignment -- both with each component's fitted 2-sd
ellipse and population-sized mean dot drawn in black; k-means/Leiden/HDBSCAN
are colored by cluster. Alongside, one full atlas (all six pairs of the
first 4 PCs) per setting group at its seed-averaged silhouette-best fit.
Datasets are drawn largest-first so small ones stay visible on top.
"""

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from sklearn.metrics import adjusted_rand_score

from . import plots
from .load_data import load_merged
from .pca import PCAComponents, compare_fits

ROOT = Path(__file__).parents[1]
LAYERS = [9, 17, 22, 29]
PCS = [15, 25, 40, 50, 60]
LAYER_COLOR = {L: plt.cm.tab10(i) for i, L in enumerate(LAYERS)}

DATASETS = ["geometry_truth", "harmful_harmless", "pubmedqa_artificial",
            "pubmedqa_labeled", "social_iqa", "wildjailbreak", "xstest"]
VARIANT = "last"
SHOW_PCS = (15, 50)          # GMM / k-means-on-PCs settings shown
PC_PAIRS = list(combinations(range(4), 2))
NSD = 2


def load_metrics(layer):
    df = pd.read_csv(ROOT / f"results/merged_last/tables/L{layer}/metrics.csv")
    df["pc"] = df["run"].str.extract(r"pc=(\d+)").astype(float)
    df["boot"] = df["run"].str.extract(r"boot(\d+)").astype(float)
    df["space"] = df["run"].str.extract(r"space=([a-z]+)")
    return df


# ---------------------------------------------------------------- model_selection

def gmm_model_selection(df, layer, out):
    gmm = df[df["method"] == "gmm"].sort_values("pc").copy()
    gmm["line"] = gmm["covariance_type"]
    gmm["row"] = "pc=" + gmm["pc"].astype(int).astype(str)
    seeds = gmm[gmm["boot"].isna()].copy()
    seeds["rep"] = seeds["seed"]
    boots = gmm[gmm["boot"].notna()].copy()
    boots["rep"] = boots["boot"]
    return plots.model_selection_panel(
        {"seeds": seeds, "PCA bootstraps": boots}, "row", "line", "rep",
        [("aic", "min", False), ("bic", "min", False), ("silhouette", "max", True)],
        title=f"GMM stability — layer {layer}, merged datasets (N=163,249)", out=out)


# ---------------------------------------------------------------- ledger

def _best(sub, grid=("k",)):
    """(silhouette, cluster count) of the silhouette-best fit over the grid.

    `sub` is one replicate (one seed or bootstrap) of one family; `grid`
    names the setting columns searched (k for GMM/k-means, resolution ×
    n_neighbors for Leiden, ...). The count is the fit's n_clusters, so it
    works the same for methods that choose their own number of clusters.
    """
    g = sub.groupby(list(grid))[["silhouette", "n_clusters"]].mean()
    idx = g["silhouette"].idxmax()
    return g["silhouette"].max(), g.loc[idx, "n_clusters"]


def reference_setting(seeds):
    """Best (covariance, pc) by seed-mean best-over-k silhouette."""
    g = seeds.groupby(["covariance_type", "pc", "k"])["silhouette"].mean()
    cov, pc, _ = g.idxmax()
    return cov, int(pc)


def ledger_rows(df):
    """Ordered [(row label, group, [(silhouette, count) per replicate])] for
    one layer. Group 'gmm' = the reference GMM with one choice changed;
    group 'other' = a different algorithm at its own best settings."""
    gmm = df[(df["method"] == "gmm") & df["boot"].isna()]
    boots = df[(df["method"] == "gmm") & df["boot"].notna()]
    cov, pc = reference_setting(gmm)
    other_cov = "full" if cov == "diag" else "diag"

    def at(d, c, p):
        return d[(d["covariance_type"] == c) & (d["pc"] == p)]

    def per(sub, by="seed", grid=("k",)):
        return [_best(s, grid) for _, s in sub.groupby(by, dropna=False)]

    def per_gmm(c, p):  # 10 replicates: 5 seeds + 5 PCA bootstraps
        return per(at(gmm, c, p)) + per(at(boots, c, p), by="boot")

    rows = [
        (f"GMM: {cov} cov, {pc} PCs", "gmm", per_gmm(cov, pc)),
        (f"GMM: {other_cov} cov, {pc} PCs", "gmm", per_gmm(other_cov, pc)),
    ]
    rows += [(f"GMM: {cov} cov, {p} PCs", "gmm", per_gmm(cov, p))
             for p in PCS if p != pc]
    others = [
        ("k-means: on PCs", df[(df["method"] == "kmeans") & (df["space"] == "pca")], ("pc", "k")),
        ("k-means: raw activations", df[(df["method"] == "kmeans") & (df["space"] == "raw")], ("k",)),
        ("MFA: raw activations", df[df["method"] == "mfa"], ("n_factors", "k")),
        (f"Leiden: graph on {PCS[0]} PCs", df[df["method"] == "leiden"], ("resolution", "n_neighbors")),
        (f"HDBSCAN: {PCS[0]} PCs", df[df["method"] == "hdbscan"], ("min_cluster_size",)),
    ]
    rows += [(label, "other", per(sub, grid=grid)) for label, sub, grid in others if len(sub)]
    return rows, (cov, pc)


def variability_ledger(per_layer, out):
    """per_layer: {layer: (ledger_rows list, (cov, pc) reference)}."""
    plots.set_style()
    labels = [r[0] for r in per_layer[LAYERS[0]][0]]
    groups = [r[1] for r in per_layer[LAYERS[0]][0]]
    n = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(19, 0.85 * n + 3), sharey=True)
    for p, (ax, xlabel) in enumerate(zip(axes, (
            "best silhouette (each fit's own space)",
            "number of clusters of the silhouette-best fit"))):
        for i in range(n):
            for j, L in enumerate(LAYERS):
                vals = [rep[p] for rep in per_layer[L][0][i][2]]
                yy = (n - 1 - i) + (1.5 - j) * 0.14
                ax.plot([min(vals), max(vals)], [yy, yy], "-", lw=3, alpha=0.6,
                        color=LAYER_COLOR[L])
                ax.plot([np.mean(vals)], [yy], "o", ms=9, color=LAYER_COLOR[L])
        first_other = groups.index("other")
        ax.axhline(n - first_other - 0.5, color="gray", ls="--", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.6, n - 0.4)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels[::-1], fontsize=16)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=LAYER_COLOR[L], label=f"L{L}")
               for L in LAYERS]
    leg = fig.legend(handles=handles, loc="outside right upper", fontsize=16,
                     title="dot = mean of replicates, bar = min to max\n"
                           "GMM rows: 5 seeds + 5 PCA resamples\n"
                           "other methods: 5 seeds\n"
                           "k chosen by silhouette\n(BIC hits the k=40 grid edge)")
    leg.get_title().set_fontsize(13)
    fig.suptitle("Stability using silhouette score to determine number of clusters")
    return plots._finish(fig, out)


# ---------------------------------------------------------------- pca_stability

def load_stability(layer, pc):
    ldir = ROOT / f"results/merged_last/runs/L{layer}"
    fits = {"full": PCAComponents.load(ldir / f"pca__pc={pc}.npz")}
    kind = {"full": "reference"}
    for p in sorted(ldir.glob(f"pca__pc={pc}__boot*.npz")):
        name = p.stem.split("__")[-1]
        fits[name], kind[name] = PCAComponents.load(p), "bootstrap"
    return compare_fits(fits, kind)


def pca_stability_overview(out):
    plots.set_style()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(24, 6.5))
    for L in LAYERS:
        lo, mid, hi = [], [], []
        for pc in PCS:
            r = load_stability(L, pc)
            ov = [r.overlap_to_ref[n] for n in r.fit_names if n != "full"]
            lo.append(min(ov)); mid.append(np.mean(ov)); hi.append(max(ov))
        a1.plot(PCS, mid, "o-", color=LAYER_COLOR[L], label=f"L{L}")
        a1.fill_between(PCS, lo, hi, color=LAYER_COLOR[L], alpha=0.15)

        for ax, pc in ((a2, PCS[0]), (a3, PCS[-1])):
            r = load_stability(L, pc)
            M = np.stack([r.cos_bestmatch[n] for n in r.fit_names if n != "full"])
            comp = np.arange(1, M.shape[1] + 1)
            ax.plot(comp, M.mean(0), "-", color=LAYER_COLOR[L], label=f"L{L}")
            ax.fill_between(comp, M.min(0), M.max(0), color=LAYER_COLOR[L], alpha=0.15)
    a1.set_xlabel("# PCs"); a1.set_ylabel("subspace overlap to full-data PCA")
    a1.set_title("PCA subspace stability under bootstrap"); a1.set_xticks(PCS); a1.legend()
    for ax, pc in ((a2, PCS[0]), (a3, PCS[-1])):
        ax.set_xlabel("component"); ax.set_ylabel("|cos| to full-data PC (best match)")
        ax.set_title(f"per-component agreement at pc={pc}"); ax.set_ylim(-0.02, 1.02); ax.legend()
    return plots._finish(fig, out)


# ---------------------------------------------------------------- clusters_on_pcs

def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}", flush=True)


def _panels():
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    return fig, axes.ravel()


def _axis_labels(ax, a, b, ev):
    ax.set_xlabel(f"PC{a + 1} (var {100 * ev[a]:.0f}%)", fontsize=15)
    ax.set_ylabel(f"PC{b + 1} (var {100 * ev[b]:.0f}%)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_aspect("equal")


def _ds_palette(ds):
    """Stable color per dataset name; draw order largest-first so the small
    datasets end up on top."""
    names = sorted(set(ds))
    colors = {d: plt.cm.tab10(i) for i, d in enumerate(names)}
    order = sorted(names, key=lambda d: np.sum(ds == d), reverse=True)
    return colors, order


def _points_by_dataset(ax, Z, ds, colors, order, a=0, b=1):
    for d in order:
        m = ds == d
        ax.scatter(Z[m, a], Z[m, b], s=3.5, color=colors[d],
                   alpha=0.45, lw=0, zorder=1)


def _points_by_cluster(ax, Z, comp, a=0, b=1):
    """tab10 per cluster, repeating past 10; -1 = noise in light gray below."""
    for i, c in enumerate(np.unique(comp)):
        m = comp == c
        ax.scatter(Z[m, a], Z[m, b], s=3.5, alpha=0.45, lw=0,
                   color="0.8" if c < 0 else plt.cm.tab10(i % 10),
                   zorder=1 if c < 0 else 2)


def _component_outlines(ax, means, covs, frac, a=0, b=1):
    """Each component's fitted 2-sd ellipse and population-sized mean dot,
    in black so they read over any point coloring."""
    for c in range(len(means)):
        mu = means[c][[a, b]]
        S = covs[c][np.ix_([a, b], [a, b])]
        evals, evecs = np.linalg.eigh(S)
        ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        ax.add_patch(Ellipse(mu, 2 * NSD * np.sqrt(evals[1]),
                             2 * NSD * np.sqrt(evals[0]), angle=ang,
                             facecolor="none", edgecolor="k", lw=1.4,
                             zorder=3))
        ax.scatter(*mu, s=max(600 * frac[c], 20), marker="o", facecolor="k",
                   edgecolor="w", lw=0.6, zorder=4)


def _dataset_legend(fig, colors):
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=v, label=k)
                        for k, v in colors.items()],
               loc="upper left", bbox_to_anchor=(1.0, 0.80), fontsize=13,
               title="dataset", title_fontsize=14)


def _frac(comp, means):
    return np.array([np.mean(comp == c) for c in range(len(means))])


def _grid_spec(runs):
    """One workhorse figure per method (per cov for GMM): rows = the setting
    that isn't k (PC count / space / factors / neighbors), columns = k (or
    resolution / min cluster size). Returns {figure key: spec}."""
    specs = {}
    for r in runs:
        p = _params(r)
        if r.startswith("gmm"):
            key, title = f"gmm__cov={p['cov']}", f"GMM: {p['cov']} cov"
            row, rk = f"{p['pc']} PCs", float(p["pc"])
            col, ck = f"k = {p['k']}", float(p["k"])
        elif r.startswith("kmeans"):
            key, title = "kmeans", "k-means"
            raw = p["space"] == "raw"
            row, rk = ("raw space", 0.0) if raw else (f"{p['pc']} PCs", float(p["pc"]))
            col, ck = f"k = {p['k']}", float(p["k"])
        elif r.startswith("mfa"):
            key, title = "mfa", "MFA"
            row, rk = f"{p['nf']} factors", float(p["nf"])
            col, ck = f"k = {p['k']}", float(p["k"])
        elif r.startswith("leiden"):
            key, title = "leiden", "Leiden"
            row, rk = f"{p['nn']} neighbors", float(p["nn"])
            col, ck = f"res {p['res']}", float(p["res"])
        else:
            key, title = "hdbscan", "HDBSCAN"
            row, rk = "", 0.0
            col, ck = f"min cluster size {p['mcs']}", float(p["mcs"])
        s = specs.setdefault(key, {"title": title, "rows": {}, "cols": {},
                                   "cells": {}})
        s["rows"][row], s["cols"][col] = rk, ck
        s["cells"][(row, col)] = r
    return specs


def hyperparam_grid(spec, fits, Z, ev, by, out, ds=None, colors=None, order=None):
    """A method's whole grid in one figure of PC1-PC2 panels (missing cells,
    e.g. k = 40 for raw k-means, are left blank). fits maps run ->
    (labels, means, covs); means/covs are None for methods without Gaussians."""
    rows = sorted(spec["rows"], key=spec["rows"].get)
    cols = sorted(spec["cols"], key=spec["cols"].get)
    head = {clab: min(i for i, r in enumerate(rows) if (r, clab) in spec["cells"])
            for clab in cols}   # first populated row per column carries its title
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(4.6 * len(cols), 4.5 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for i, rlab in enumerate(rows):
        for j, clab in enumerate(cols):
            ax = axes[i][j]
            run = spec["cells"].get((rlab, clab))
            if run is None:
                ax.axis("off")
                continue
            comp, means, covs = fits[run]
            if by == "dataset":
                _points_by_dataset(ax, Z, ds, colors, order)
            else:
                _points_by_cluster(ax, Z, comp)
            if means is not None:
                _component_outlines(ax, means, covs, _frac(comp, means))
            if run.startswith(("leiden", "hdbscan")):
                n = len(np.unique(comp[comp >= 0]))
                ax.set_title(f"{clab} ({n} clusters)", fontsize=15)
            elif i == head[clab]:
                ax.set_title(clab, fontsize=15)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=13)
            if i == len(rows) - 1:
                ax.set_xlabel(f"PC1 (var {100 * ev[0]:.0f}%)", fontsize=15)
            if j == 0:
                lab = f"PC2 (var {100 * ev[1]:.0f}%)"
                ax.set_ylabel(f"{rlab}\n{lab}" if rlab else lab, fontsize=15)
    title = spec["title"] + (" (colored by cluster)" if by == "cluster" else "")
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if by == "dataset":
        _dataset_legend(fig, colors)
    _save(fig, out)


def pc_atlas(Z, comp, means, covs, ev, by, title, out,
             ds=None, colors=None, order=None):
    """One fit on all six pairs of the first 4 PCs."""
    fig, axes = _panels()
    frac = None if means is None else _frac(comp, means)
    for ax, (a, b) in zip(axes, PC_PAIRS):
        if by == "dataset":
            _points_by_dataset(ax, Z, ds, colors, order, a, b)
        else:
            _points_by_cluster(ax, Z, comp, a, b)
        if means is not None:
            _component_outlines(ax, means, covs, frac, a, b)
        _axis_labels(ax, a, b, ev)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    if by == "dataset":
        _dataset_legend(fig, colors)
    _save(fig, out)


def gmm_first4(model):
    """A fitted GMM's means and covariances, marginalized to the first 4 PCs."""
    means = np.asarray(model.means_)[:, :4]
    if model.covariance_type == "diag":
        covs = np.stack([np.diag(v[:4]) for v in model.covariances_])
    else:
        covs = np.asarray(model.covariances_)[:, :4, :4]
    return means, covs


def mfa_first4(model, comps):
    """An MFA (fit in raw space) marginalized to the first 4 PCs: means and
    covariances Lam Lam^T + Psi mapped through the standardize-then-PCA
    linear map. Only the 4 x 4 projected covariance is ever formed."""
    mu = model.mu.detach().cpu().numpy()
    Lam = model.Lam.detach().cpu().numpy()
    Psi = model.Psi.detach().cpu().numpy()
    A = comps.components[:4] / comps.scaler_scale[None, :]   # (4, D)
    means = comps.transform(mu)[:, :4]
    APsiAt = (A * Psi) @ A.T
    covs = np.stack([(A @ Lam[c]) @ (A @ Lam[c]).T + APsiAt
                     for c in range(len(mu))])
    return means, covs


def _params(run):
    return dict(t.split("=") for t in run.split("__") if "=" in t)


def shown(run):
    """seed-0 fit of every setting; GMM / k-means-on-PCs at SHOW_PCS only."""
    if not run.startswith(("gmm__", "mfa__", "kmeans__", "leiden__", "hdbscan__")):
        return False
    if "boot" in run or not (run.endswith("__seed0") or run.startswith("hdbscan__")):
        return False
    p = _params(run)
    return "pc" not in p or int(p["pc"]) in SHOW_PCS


def group_of(run):
    """The workhorse figure a run belongs to: its setting without k."""
    p = _params(run)
    if run.startswith("gmm"):
        return f"gmm__cov={p['cov']}__pc={p['pc']}"
    if run.startswith("kmeans"):
        return ("kmeans__space=raw" if p["space"] == "raw"
                else f"kmeans__space=pca__pc={p['pc']}")
    return run.split("__")[0]                     # mfa / leiden / hdbscan


def group_title(group):
    p = _params(group)
    if group.startswith("gmm"):
        return f"GMM: {p['cov']} cov, {p['pc']} PCs"
    if group.startswith("kmeans"):
        return ("k-means: raw space" if p["space"] == "raw"
                else f"k-means: {p['pc']} PCs")
    return {"mfa": "MFA", "leiden": "Leiden", "hdbscan": "HDBSCAN"}[group]


def panel_title(run, comp):
    p = _params(run)
    if run.startswith(("gmm", "kmeans")):
        return f"k = {p['k']}"
    if run.startswith("mfa"):
        return f"{p['nf']} factors, k = {p['k']}"
    n = len(np.unique(comp[comp >= 0]))
    if run.startswith("leiden"):
        return f"res {p['res']}, {p['nn']} neighbors ({n} clusters)"
    return f"min cluster size {p['mcs']} ({n} clusters)"


def best_run(df, runs):
    """The group's seed-averaged silhouette-best setting (its seed-0 run)."""
    d = df[df["boot"].isna()].copy()
    d["key"] = d["run"].str.replace(r"__seed\d+$", "", regex=True)
    sil = d.groupby("key")["silhouette"].mean()
    return max(runs, key=lambda r: sil.get(r.replace("__seed0", ""), -np.inf))


def clusters_on_pcs(layer, out_dir):
    ldir = ROOT / f"results/merged_last/runs/L{layer}"
    runs = [r.name for r in sorted(ldir.iterdir())
            if r.is_dir() and shown(r.name) and (r / "assignments.npz").exists()]
    if not runs:
        print(f"L{layer}: no assignments.npz here, clusters_on_pcs skipped", flush=True)
        return
    df = load_metrics(layer)
    ds = np.load(ldir / "samples.npz", allow_pickle=True)["dataset_id"]
    comps = PCAComponents.load(ldir / f"pca__pc={SHOW_PCS[0]}.npz")
    Z = comps.transform(load_merged(DATASETS, VARIANT, layer)["X"])[:, :4]
    ev = comps.explained_variance_ratio[:4]
    colors, order = _ds_palette(ds)

    fits = {}
    for r in runs:
        comp = np.load(ldir / r / "assignments.npz")["labels"]
        means = covs = None
        if r.startswith(("gmm", "mfa")):
            model = joblib.load(ldir / r / "model.joblib")
            means, covs = (gmm_first4(model) if r.startswith("gmm")
                           else mfa_first4(model, comps))
        fits[r] = (comp, means, covs)

    # workhorse grids: one figure per method (per cov for GMM)
    for key, spec in _grid_spec(runs).items():
        if key.startswith(("gmm", "mfa")):
            hyperparam_grid(spec, fits, Z, ev, "dataset",
                            out_dir / f"{key}__by=dataset.png", ds, colors, order)
            hyperparam_grid(spec, fits, Z, ev, "cluster",
                            out_dir / f"{key}__by=cluster.png")
        else:
            hyperparam_grid(spec, fits, Z, ev, "cluster", out_dir / f"{key}.png")

    # full atlas at each setting group's silhouette-best fit
    groups = {}
    for r in runs:
        groups.setdefault(group_of(r), []).append(r)
    for group, rs in groups.items():
        br = best_run(df, rs)
        comp, means, covs = fits[br]
        atlas_title = f"{group_title(group)}, {panel_title(br, comp)}"
        key = br.replace("__seed0", "")
        if br.startswith(("gmm", "mfa")):
            pc_atlas(Z, comp, means, covs, ev, "dataset", atlas_title,
                     out_dir / f"atlas__{key}__by=dataset.png", ds, colors, order)
            pc_atlas(Z, comp, means, covs, ev, "cluster",
                     f"{atlas_title} (colored by cluster)",
                     out_dir / f"atlas__{key}__by=cluster.png")
        else:
            pc_atlas(Z, comp, None, None, ev, "cluster", atlas_title,
                     out_dir / f"atlas__{key}.png")


# ---------------------------------------------------------------- ARI groups

K_GRID = [4, 6, 8, 10, 12, 16, 20, 26, 32, 40]
MFA_K, MFA_NF = [6, 10, 16], [6, 9, 12]
LEIDEN_RES, LEIDEN_NN = [0.5, 1.0, 2.0], [15, 30]
SEEDS, N_BOOT = range(5), 5
AGREE_PC = 50        # PCA-space methods enter the agreement figures at this PC count


def _labels_of(layer, run):
    p = ROOT / f"results/merged_last/runs/L{layer}/{run}/assignments.npz"
    return np.load(p)["labels"] if p.exists() else None


def _load_pool(layer, runs):
    return [l for l in (_labels_of(layer, r) for r in runs) if l is not None]


def _ari(a, b):
    """ARI on the points both fits assign (drops HDBSCAN's noise points)."""
    m = (a >= 0) & (b >= 0)
    return adjusted_rand_score(a[m], b[m])


def _stat(vals):
    """(mean, min, max, count) of a list of pairwise ARIs; None if empty."""
    return (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)),
            len(vals)) if vals else None


def _within(pool):
    return _stat([_ari(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]])


def _between(pool_a, pool_b):
    return _stat([_ari(a, b) for a in pool_a for b in pool_b])


def _gmm_runs(cov, pc, k, arm):
    if arm == "seeds":
        return [f"gmm__cov={cov}__pc={pc}__k={k}__seed{s}" for s in SEEDS]
    return [f"gmm__cov={cov}__pc={pc}__k={k}__seed0__boot{b}" for b in range(N_BOOT)]


def _ari_csv(path, rows):
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _tables(fig_dir):
    """The tables/ dir mirroring a figures/ subdir (figures/L9 -> tables/L9)."""
    parts = [p if p != "figures" else "tables" for p in fig_dir.parts]
    return Path(*parts)


def _stab_curve(ax, xs, stats, color, label, ls="-"):
    """Mean line + min-max band, skipping x values with no pairs."""
    pts = [(x, s) for x, s in zip(xs, stats) if s]
    if not pts:
        return
    xs, st = zip(*pts)
    ax.plot(xs, [s[0] for s in st], marker="o", ls=ls, ms=4, color=color, label=label)
    ax.fill_between(xs, [s[1] for s in st], [s[2] for s in st], color=color, alpha=0.12)


# ---------------------------------------------------------------- ari_stability

def ari_stability(layer, out_dir):
    """Same setting, different replicates: mean pairwise ARI across the 5 seeds
    per setting (for GMM also across the 5 PCA-bootstrap refits), band =
    min-max over the pairs. HDBSCAN is deterministic and has no replicates."""
    if _labels_of(layer, f"gmm__cov=diag__pc={AGREE_PC}__k=4__seed0") is None:
        print(f"L{layer}: no assignments.npz here, ari_stability skipped", flush=True)
        return
    rows = []

    def curve(family, arm, setting, xs, runs_of):
        stats = []
        for x in xs:
            pool = _load_pool(layer, runs_of(x))
            stats.append(_within(pool) if len(pool) > 1 else None)
        for x, s in zip(xs, stats):
            if s:
                rows.append({"family": family, "arm": arm, "setting": setting,
                             "x": x, "mean_ari": round(s[0], 4),
                             "min_ari": round(s[1], 4), "max_ari": round(s[2], 4),
                             "n_pairs": s[3]})
        return stats

    plots.set_style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), sharey=True)
    (ax_d, ax_f, ax_kp), (ax_kr, ax_m, ax_l) = axes
    for ax, cov in ((ax_d, "diag"), (ax_f, "full")):
        for i, pc in enumerate(PCS):
            st = curve(f"GMM {cov}", "seeds", f"pc={pc}", K_GRID,
                       lambda k: _gmm_runs(cov, pc, k, "seeds"))
            _stab_curve(ax, K_GRID, st, f"C{i}", f"{pc} PCs")
            st = curve(f"GMM {cov}", "PCA bootstraps", f"pc={pc}", K_GRID,
                       lambda k: _gmm_runs(cov, pc, k, "boots"))
            _stab_curve(ax, K_GRID, st, f"C{i}", None, ls="--")
        ax.set_title(f"GMM, {cov} covariance", fontsize=19)
    for i, pc in enumerate(PCS):
        st = curve("k-means PCs", "seeds", f"pc={pc}", K_GRID,
                   lambda k: [f"kmeans__space=pca__pc={pc}__k={k}__seed{s}" for s in SEEDS])
        _stab_curve(ax_kp, K_GRID, st, f"C{i}", f"{pc} PCs")
    ax_kp.set_title("k-means on PCs", fontsize=19)
    st = curve("k-means raw", "seeds", "", K_GRID,
               lambda k: [f"kmeans__space=raw__k={k}__seed{s}" for s in SEEDS])
    _stab_curve(ax_kr, K_GRID, st, "C0", None)
    ax_kr.set_title("k-means on raw activations", fontsize=19)
    for i, nf in enumerate(MFA_NF):
        st = curve("MFA", "seeds", f"nf={nf}", MFA_K,
                   lambda k: [f"mfa__nf={nf}__k={k}__seed{s}" for s in SEEDS])
        _stab_curve(ax_m, MFA_K, st, f"C{i}", f"{nf} factors")
    ax_m.set_title("MFA (raw activations)", fontsize=19)
    for i, nn in enumerate(LEIDEN_NN):
        st = curve("Leiden", "seeds", f"nn={nn}", LEIDEN_RES,
                   lambda r: [f"leiden__res={r}__nn={nn}__seed{s}" for s in SEEDS])
        _stab_curve(ax_l, LEIDEN_RES, st, f"C{i}", f"{nn} neighbors")
    ax_l.set_title(f"Leiden (graph on {PCS[0]} PCs)", fontsize=19)
    ax_l.set_xlabel("resolution", fontsize=17)
    for ax in (ax_d, ax_f, ax_kp, ax_kr, ax_m):
        ax.set_xlabel("k", fontsize=17)
        ax.set_xticks(K_GRID)
    for ax in axes.ravel():
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=15)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=14)
    for ax in (ax_d, ax_kr):
        ax.set_ylabel("mean pairwise ARI", fontsize=17)
    fig.suptitle(f"Agreement between replicate fits of the same setting — layer {layer} "
                 "(line = mean pairwise ARI, band = min-max; "
                 "dashed = across the 5 PCA-bootstrap refits)", fontsize=21)
    _save(fig, out_dir / "ari_stability.png")
    _ari_csv(_tables(out_dir) / "ari_stability.csv", rows)


# ---------------------------------------------------------------- method_agreement

FAMILIES = ["GMM diag", "GMM full", "k-means PCs", "k-means raw", "MFA", "Leiden", "HDBSCAN"]


def _family_pools(layer):
    """{family: {grid k: [run names]}}. Leiden/HDBSCAN fits enter at their
    nearest grid k, and only when their found cluster count is within 15%
    (at least 1) of it; the fixed-k methods enter at the k they were asked for."""
    pools = {
        "GMM diag": {k: _gmm_runs("diag", AGREE_PC, k, "seeds")
                        + _gmm_runs("diag", AGREE_PC, k, "boots") for k in K_GRID},
        "GMM full": {k: _gmm_runs("full", AGREE_PC, k, "seeds")
                        + _gmm_runs("full", AGREE_PC, k, "boots") for k in K_GRID},
        "k-means PCs": {k: [f"kmeans__space=pca__pc={AGREE_PC}__k={k}__seed{s}"
                            for s in SEEDS] for k in K_GRID},
        "k-means raw": {k: [f"kmeans__space=raw__k={k}__seed{s}" for s in SEEDS]
                        for k in K_GRID},
        "MFA": {k: [f"mfa__nf={nf}__k={k}__seed{s}" for nf in MFA_NF for s in SEEDS]
                for k in MFA_K},
    }
    met = load_metrics(layer)
    for family, method in (("Leiden", "leiden"), ("HDBSCAN", "hdbscan")):
        d = {}
        for _, r in met[met["method"] == method].iterrows():
            k = min(K_GRID, key=lambda g: abs(g - r["n_clusters"]))
            if abs(k - r["n_clusters"]) <= max(1, 0.15 * k):
                d.setdefault(k, []).append(r["run"])
        pools[family] = d
    return pools


def method_agreement(layer, out_dir):
    """ARI between every pair of methods as a function of k, over all
    cross-pairs of the two methods' replicate fits. Lower-triangle grid of
    panels, one per method pair."""
    if _labels_of(layer, f"gmm__cov=diag__pc={AGREE_PC}__k=4__seed0") is None:
        print(f"L{layer}: no assignments.npz here, method_agreement skipped", flush=True)
        return
    pools = _family_pools(layer)
    stats, rows = {}, []
    for k in K_GRID:
        loaded = {f: _load_pool(layer, pools[f].get(k, [])) for f in FAMILIES}
        for i, fa in enumerate(FAMILIES):
            for fb in FAMILIES[i + 1:]:
                s = _between(loaded[fa], loaded[fb])
                if s:
                    stats.setdefault((fa, fb), {})[k] = s
                    rows.append({"family_a": fa, "family_b": fb, "k": k,
                                 "mean_ari": round(s[0], 4), "min_ari": round(s[1], 4),
                                 "max_ari": round(s[2], 4), "n_pairs": s[3]})

    plots.set_style()
    n = len(FAMILIES)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(3.0 * (n - 1), 2.8 * (n - 1)),
                             sharex=True, sharey=True)
    for i in range(1, n):
        for j in range(n - 1):
            ax = axes[i - 1, j]
            if j >= i:
                ax.axis("off")
                continue
            d = stats.get((FAMILIES[j], FAMILIES[i]), {})
            ks = sorted(d)
            _stab_curve(ax, ks, [d[k] for k in ks], "C0", None)
            if not ks:
                ax.text(0.5, 0.5, "no k in common", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color="gray")
            if i == j + 1:
                ax.set_title(FAMILIES[j], fontsize=18)
            if j == 0:
                ax.set_ylabel(FAMILIES[i], fontsize=16)
            if i == n - 1:
                ax.set_xlabel("k", fontsize=16)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([4, 12, 20, 32, 40])
            ax.tick_params(labelsize=14)
    fig.suptitle(f"Agreement between methods (ARI) at matched numbers of clusters — layer {layer}",
                 fontsize=22)
    fig.text(0.5, -0.01,
             f"PCA-space methods at {AGREE_PC} PCs. Pools per k: 5 seeds "
             "(GMM also its 5 PCA-bootstrap refits; MFA all factor settings); "
             f"Leiden/HDBSCAN (graph on {PCS[0]} PCs) enter at the nearest k within 15% "
             "(at least 1) of their found cluster count. Points a fit labels noise are dropped from "
             "its pairs. Line = mean over all cross-pairs, band = min-max.",
             ha="center", fontsize=13)
    _save(fig, out_dir / "method_agreement.png")
    _ari_csv(_tables(out_dir) / "method_agreement.csv", rows)


# ---------------------------------------------------------------- composition

MIN_SEG = 0.10       # smallest (dataset, label) share drawn as its own segment
DATASET_HOME = 0.5   # a cluster holding this much of a dataset always shows it
TOPIC_PURE = 0.80    # a segment this dominated by one topic gets the topic's name
HATCHES = ["", "///", "xxx"]


def featured_settings():
    """One representative run per method family, the same at every layer:
    BIC-best k for the model-based methods, silhouette-best setting for the
    rest (criterion averaged over seeds and layers; PCA methods at AGREE_PC)."""
    df = pd.concat([load_metrics(L) for L in LAYERS])
    df = df[df["boot"].isna()]

    def best(sub, by, crit, take_max):
        g = sub.groupby(by)[crit].mean()
        return g.idxmax() if take_max else g.idxmin()

    gmm = df[(df["method"] == "gmm") & (df["pc"] == AGREE_PC)]
    k_d = int(best(gmm[gmm["covariance_type"] == "diag"], "k", "bic", False))
    k_f = int(best(gmm[gmm["covariance_type"] == "full"], "k", "bic", False))
    nf, mk = best(df[df["method"] == "mfa"], ["n_factors", "k"], "bic", False)
    kp = df[(df["method"] == "kmeans") & (df["space"] == "pca") & (df["pc"] == AGREE_PC)]
    kpk = int(best(kp, "k", "silhouette", True))
    krk = int(best(df[(df["method"] == "kmeans") & (df["space"] == "raw")],
                   "k", "silhouette", True))
    res, nn = best(df[df["method"] == "leiden"], ["resolution", "n_neighbors"],
                   "silhouette", True)
    mcs = int(best(df[df["method"] == "hdbscan"], "min_cluster_size", "silhouette", True))
    return {
        "GMM diag": f"gmm__cov=diag__pc={AGREE_PC}__k={k_d}__seed0",
        "GMM full": f"gmm__cov=full__pc={AGREE_PC}__k={k_f}__seed0",
        "k-means PCs": f"kmeans__space=pca__pc={AGREE_PC}__k={kpk}__seed0",
        "k-means raw": f"kmeans__space=raw__k={krk}__seed0",
        "MFA": f"mfa__nf={int(nf)}__k={int(mk)}__seed0",
        "Leiden": f"leiden__res={res}__nn={int(nn)}__seed0",
        "HDBSCAN": f"hdbscan__mcs={mcs}",
    }


def _segments(m, ds, y, topic, keep=()):
    """One component's (share, dataset, label, text) for every (dataset, label)
    cell holding >= MIN_SEG of its points, largest first — plus every cell of
    the datasets in `keep` (the ones this cluster is home to), however small.
    The text is the label, plus the topic when one topic dominates the cell
    and the dataset actually has topics."""
    n = m.sum()
    segs = []
    for d in np.unique(ds[m]):
        for l in np.unique(y[m & (ds == d)]):
            cell = m & (ds == d) & (y == l)
            share = cell.sum() / n
            if share < MIN_SEG and d not in keep:
                continue
            text = str(l)
            tv, tc = np.unique(topic[cell], return_counts=True)
            if len(np.unique(topic[ds == d])) > 1 and tc.max() / tc.sum() >= TOPIC_PURE:
                text += f"\n{tv[tc.argmax()]}"
            segs.append((share, d, str(l), text))
    return sorted(segs, key=lambda s: -s[0])


def _bar_text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def composition(feats, out_dir):
    """Stacked composition bar per cluster: colored segments = the (dataset,
    label) cells that each hold >= MIN_SEG of the cluster (color = dataset,
    hatching separates labels within a dataset), grey = everything smaller,
    pooled ('mixed'). One figure per method, rows = layers."""
    for family, run in feats.items():
        per_layer = []
        for L in LAYERS:
            comp = _labels_of(L, run)
            if comp is None:
                continue
            smp = np.load(ROOT / f"results/merged_last/runs/L{L}/samples.npz", allow_pickle=True)
            per_layer.append((L, comp, smp["dataset_id"], smp["labels"], smp["topic"]))
        if not per_layer:
            print(f"{family}: no assignments.npz here, composition skipped", flush=True)
            continue
        colors, _ = _ds_palette(per_layer[0][2])
        maxn = max(len(np.unique(c)) for _, c, *_ in per_layer)
        fig, axes = plt.subplots(len(per_layer), 1,
                                 figsize=(max(0.45 * maxn + 3, 11), 4.4 * len(per_layer)),
                                 squeeze=False)
        for ax, (L, comp, ds, y, topic) in zip(axes[:, 0], per_layer):
            rank = {d: {l: i for i, l in enumerate(sorted(np.unique(y[ds == d])))}
                    for d in np.unique(ds)}
            used = np.unique(comp)
            dom = {c: max(np.unique(ds[comp == c]),
                          key=lambda d: np.sum(ds[comp == c] == d)) for c in used}
            order = sorted(used, key=lambda c: (dom[c], -np.sum(comp == c)))
            ds_total = {d: np.sum(ds == d) for d in np.unique(ds)}
            for x, c in enumerate(order):
                m = comp == c
                cov = {d: np.sum(m & (ds == d)) / ds_total[d] for d in np.unique(ds[m])}
                homes = {d for d, v in cov.items() if v >= DATASET_HOME}
                bottom = 0.0
                for share, d, l, text in _segments(m, ds, y, topic, homes):
                    ax.bar(x, share, bottom=bottom, color=colors[d],
                           hatch=HATCHES[rank[d][l]], edgecolor="white", lw=0.4)
                    if share >= 0.15:
                        ax.text(x, bottom + share / 2, text, ha="center", va="center",
                                fontsize=7.5, rotation=90, color=_bar_text_color(colors[d]))
                    bottom += share
                if bottom < 0.999:
                    ax.bar(x, 1 - bottom, bottom=bottom, color="0.85",
                           edgecolor="white", lw=0.4)
                # a home dataset too thin to see gets called out on the bar
                small = sorted(d for d in homes if np.sum(m & (ds == d)) / m.sum() < 0.05)
                if small:
                    ax.text(x, 0.99, "\n".join(f"{cov[d]:.0%} {d}" for d in small),
                            ha="center", va="top", rotation=90, fontsize=8,
                            color=colors[small[0]] if len(small) == 1 else "black",
                            bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1))
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(["noise" if c < 0 else f"c{c}" for c in order],
                               rotation=90, fontsize=9)
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_ylabel("fraction of cluster", fontsize=16)
            ax.set_title(f"L{L}", fontsize=16)
        title = f"{group_title(group_of(run))}, {panel_title(run, per_layer[0][1])}"
        fig.suptitle(f"{title} — cluster composition by dataset x label", fontsize=20)
        fig.tight_layout()
        fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in colors.items()]
                   + [Patch(facecolor="0.85", label=f"mixed (all cells < {MIN_SEG:.0%})")],
                   loc="upper left", bbox_to_anchor=(1.0, 0.88), fontsize=13,
                   title="dataset", title_fontsize=14)
        fig.text(1.005, 0.60,
                 f"segments = (dataset, label) cells\n>= {MIN_SEG:.0%} of the cluster,\n"
                 f"plus any dataset with\n>= {DATASET_HOME:.0%} of its points here\n"
                 "(called out on the bar\nwhen too thin to see)\n"
                 "text = label (and topic\nwhere one dominates)\n"
                 "hatching separates labels\nwithin a dataset",
                 ha="left", va="top", fontsize=12,
                 transform=fig.transFigure)
        _save(fig, out_dir / f"{family.replace(' ', '_').replace('-', '_').lower()}.png")


# ---------------------------------------------------------------- cross_layer_ari

def cross_layer_ari(feats, out_dir):
    """Partition similarity between layers (ARI), one panel per method at its
    featured setting; the merged set has the same samples in the same order
    at every layer. Also reports each layer vs the dataset partition."""
    ids = {L: np.load(ROOT / f"results/merged_last/runs/L{L}/samples.npz",
                      allow_pickle=True)["sample_id"] for L in LAYERS}
    for L in LAYERS[1:]:
        assert np.array_equal(ids[L], ids[LAYERS[0]]), "merged sample order differs"
    ds = np.load(ROOT / f"results/merged_last/runs/L{LAYERS[0]}/samples.npz",
                 allow_pickle=True)["dataset_id"]

    rows, drawn = [], False
    fig, axes = plt.subplots(1, len(feats), figsize=(4.3 * len(feats), 4.8))
    for ax, (family, run) in zip(axes, feats.items()):
        labs = {L: _labels_of(L, run) for L in LAYERS}
        if any(v is None for v in labs.values()):
            ax.text(0.5, 0.5, "assignments\nnot local", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            continue
        drawn = True
        nl = len(LAYERS)
        M = np.eye(nl)
        for i, a in enumerate(LAYERS):
            for j, b in enumerate(LAYERS):
                if j > i:
                    M[i, j] = M[j, i] = _ari(labs[a], labs[b])
                    rows.append({"family": family, "a": f"L{a}", "b": f"L{b}",
                                 "ari": round(M[i, j], 4)})
            m = labs[a] >= 0
            rows.append({"family": family, "a": f"L{a}", "b": "datasets",
                         "ari": round(adjusted_rand_score(ds[m], labs[a][m]), 4)})
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(nl):
            for j in range(nl):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14,
                        color="black" if M[i, j] > 0.6 else "white")
        ax.set_xticks(range(nl), [f"L{L}" for L in LAYERS], fontsize=14)
        ax.set_yticks(range(nl), [f"L{L}" for L in LAYERS], fontsize=14)
        ax.grid(False)
        counts = "/".join(str(len(np.unique(l[l >= 0]))) for l in labs.values())
        ax.set_title(f"{family}\n({counts} clusters)", fontsize=16)
    if not drawn:
        print("no assignments.npz here, cross_layer_ari skipped", flush=True)
        plt.close(fig)
        return
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=13)
    fig.suptitle("cluster similarity across layers (ARI), each method at its featured setting",
                 fontsize=21)
    _save(fig, out_dir / "cross_layer_ari.png")
    _ari_csv(_tables(out_dir) / "cross_layer_ari.csv", rows)
    vs = {r["family"]: [] for r in rows}
    for r in rows:
        if r["b"] == "datasets":
            vs[r["family"]].append(f"{r['a']} {r['ari']:.3f}")
    for family, v in vs.items():
        if v:
            print(f"{family} vs the dataset partition: " + ", ".join(v), flush=True)


# ---------------------------------------------------------------- embedding_maps

EMBEDDINGS = (("tsne", "t-SNE", "perp", "perplexity"), ("umap", "UMAP", "nn", "n_neighbors"))


def _recommended_gcp(curve):
    """Paper's automated GCP rule (Ma et al. 2025, Sec 4.3): scan the
    connectivity values low to high and recommend the smallest one that either
    exceeds global stability 0.9 or fails to beat the previous value by >=5% (a
    plateau); if neither ever triggers, the largest value. Returns (gcp, gs)."""
    vs = sorted(curve)
    for prev, cur in zip(vs, vs[1:]):
        if curve[prev] > 0.9 or curve[cur] <= 1.05 * curve[prev]:
            return prev, curve[prev]
    return vs[-1], curve[vs[-1]]


def _best_embedding(emb_dir, method, tag):
    """(global stability, pc, connectivity value) of the NESS-recommended
    setting. The connectivity value follows the paper's GCP rule per PC curve
    (see `_recommended_gcp`); the PC count — which the paper has no analogue for
    — is then the one whose recommended embedding has the highest stability.
    Curves come from the saved per-PC json, rebuilt from the per-setting files
    when a curve json is missing."""
    best = None
    for pc in PCS:
        curve_file = emb_dir / f"ness_{method}__pc={pc}.json"
        if curve_file.exists():
            curve = {int(v): g for v, g in json.loads(curve_file.read_text()).items()}
        else:
            curve = {int(p.stem.split("=")[-1]): float(np.load(p)["local_stability"].mean())
                     for p in emb_dir.glob(f"ness_{method}__pc={pc}__{tag}=*.npz")}
        if not curve:
            continue
        v, g = _recommended_gcp(curve)
        if best is None or g > best[0]:
            best = (g, pc, v)
    return best


def embedding_maps(layer, out_dir):
    """The NESS-recommended t-SNE and UMAP of the layer, each shown twice:
    colored by dataset, and by per-point stability across the seed ensemble.
    The map drawn is seed 0 of the recommended (PC count, connectivity) setting."""
    emb_dir = ROOT / f"results/merged_last/runs/L{layer}/embeddings"
    if not emb_dir.exists():
        print(f"L{layer}: no embeddings here, embedding_maps skipped", flush=True)
        return
    ds = np.load(ROOT / f"results/merged_last/runs/L{layer}/samples.npz",
                 allow_pickle=True)["dataset_id"]
    colors, order = _ds_palette(ds)
    fig, axes = plt.subplots(2, 2, figsize=(17, 15.5))
    im = None
    for row, (method, name, tag, param) in enumerate(EMBEDDINGS):
        best = _best_embedding(emb_dir, method, tag)
        if best is None:
            for ax in axes[row]:
                ax.axis("off")
            continue
        g, pc, v = best
        coords = np.load(emb_dir / f"{method}__pc={pc}__{tag}={v}__seed0.npz")["coords"]
        stab = np.load(emb_dir / f"ness_{method}__pc={pc}__{tag}={v}.npz")["local_stability"]
        _points_by_dataset(axes[row, 0], coords, ds, colors, order)
        axes[row, 0].set_title(f"{name}: {pc} PCs, {param} {v} — by dataset", fontsize=18)
        im = axes[row, 1].scatter(coords[:, 0], coords[:, 1], c=stab, cmap="viridis",
                                  vmin=0, vmax=1, s=3.5, alpha=0.45, lw=0)
        axes[row, 1].set_title(f"{name} — per-point stability across seeds "
                               f"(mean {g:.3f})", fontsize=18)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    fig.suptitle(f"NESS-recommended embeddings — layer {layer} (map shown = seed 0)",
                 fontsize=22)
    fig.tight_layout()
    if im is not None:
        cb = fig.colorbar(im, ax=axes[:, 1], fraction=0.03, pad=0.02)
        cb.set_label("local neighborhood stability", fontsize=15)
        cb.ax.tick_params(labelsize=13)
    _dataset_legend(fig, colors)
    _save(fig, out_dir / "embedding_maps.png")


# ---------------------------------------------------------------- driver

GROUPS = ("model_selection", "ledger", "pca_stability", "clusters_on_pcs",
          "ari_stability", "method_agreement", "composition", "cross_layer_ari",
          "embedding_maps")


def main(groups=GROUPS):
    figdir = ROOT / "results/merged_last/figures"
    across = figdir / "across_layers"
    if {"model_selection", "ledger"} & set(groups):
        per_layer = {}
        for L in LAYERS:
            df = load_metrics(L)
            d = figdir / f"L{L}"
            d.mkdir(parents=True, exist_ok=True)
            if "model_selection" in groups:
                gmm_model_selection(df, L, d / "model_selection_gmm.png")
            if "ledger" in groups:
                rows, ref = ledger_rows(df)
                per_layer[L] = (rows, ref)
                sil = [rep[0] for rep in rows[0][2]]
                print(f"L{L}: reference {ref[0]}, pc={ref[1]}; seed silhouette "
                      f"{min(sil):.3f}..{max(sil):.3f}; silhouette-best k per seed "
                      f"{sorted(int(rep[1]) for rep in rows[0][2])}")
        if "ledger" in groups:
            across.mkdir(parents=True, exist_ok=True)
            variability_ledger(per_layer, across / "variability_ledger.png")
    if "pca_stability" in groups:
        across.mkdir(parents=True, exist_ok=True)
        pca_stability_overview(across / "pca_stability.png")
    if "clusters_on_pcs" in groups:
        for L in LAYERS:
            clusters_on_pcs(L, figdir / f"L{L}" / "clusters_on_pcs")
    if "ari_stability" in groups:
        for L in LAYERS:
            ari_stability(L, figdir / f"L{L}")
    if "method_agreement" in groups:
        for L in LAYERS:
            method_agreement(L, figdir / f"L{L}")
    if {"composition", "cross_layer_ari"} & set(groups):
        feats = featured_settings()
        for family, run in feats.items():
            print(f"featured {family}: {run}", flush=True)
        if "composition" in groups:
            composition(feats, figdir / "composition")
        if "cross_layer_ari" in groups:
            cross_layer_ari(feats, across)
    if "embedding_maps" in groups:
        for L in LAYERS:
            embedding_maps(L, figdir / f"L{L}")
    print(f"figures written to {figdir}")


if __name__ == "__main__":
    bad = set(sys.argv[1:]) - set(GROUPS)
    if bad:
        sys.exit(f"unknown figure group(s) {sorted(bad)}; choose from {GROUPS}")
    main(tuple(sys.argv[1:]) or GROUPS)
