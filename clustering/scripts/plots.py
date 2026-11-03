"""Figures for a clustering run — few, dense, large text.

Consolidated per the plan:
  pc_scatter_grid       PC1-4 three ways (cluster / label / dataset) in one figure
  scree                 PCA eigenvalue spectrum
  model_selection_panel AIC, BIC, silhouette vs k; AIC/BIC-best marked
  composition_heatmap   one heatmap: clusters (with size) x [labels | topics]
  pca_stability_panel   subspace overlap + per-component agreement + pairwise fits
  method_ari_matrix     agreement across methods/configs
  embedding_plot        t-SNE/UMAP, NESS-unstable points greyed
  mds_centroid_map      inter-cluster distance map + the numbers
  across_layers         a metric vs layer, per dataset

Everything reads saved artifacts / arrays; nothing re-fits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def set_style():
    plt.rcParams.update({
        "figure.constrained_layout.use": True,
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "figure.titlesize": 19, "axes.grid": True, "grid.alpha": 0.3,
    })


def _scatter(ax, Z, color_by, title):
    cats = np.unique(color_by)
    for c in cats:
        m = color_by == c
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, label=str(c))
    ax.set_title(title)
    if len(cats) <= 12:
        ax.legend(markerscale=2, framealpha=0.9)


def pc_scatter_grid(Z, labelings: dict, pairs=((0, 1), (2, 3)), out=None):
    """Rows = colorings (cluster/label/dataset), columns = PC pairs."""
    set_style()
    rows, cols = list(labelings), pairs
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6 * len(cols), 5 * len(rows)),
                             squeeze=False)
    for i, name in enumerate(rows):
        for j, (a, b) in enumerate(cols):
            ax = axes[i][j]
            _scatter(ax, Z[:, [a, b]], np.asarray(labelings[name]), f"{name}: PC{a+1} vs PC{b+1}")
            ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}")
    return _finish(fig, out)


def scree(explained_variance_ratio, out=None):
    set_style()
    ev = np.asarray(explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), "o-", label="cumulative")
    ax.bar(np.arange(1, len(ev) + 1), ev, alpha=0.4, label="per component")
    ax.axhline(0.95, ls="--", color="gray"); ax.set_xlabel("component"); ax.set_ylabel("explained variance")
    ax.legend()
    return _finish(fig, out)


def model_selection_panel(sources: dict, row_col, line_col, rep_col, metrics,
                          x="k", share_rows=("silhouette",), title=None, out=None):
    """Model-selection grid: rows = values of df[row_col] (e.g. pc count),
    columns = metric × variability source (`sources`: {name: df}, e.g.
    {"seeds": ..., "PCA bootstraps": ...}).

    Within a panel: one line per df[line_col] = mean over replicates
    (df[rep_col]), a min-max band across replicates, a "|" tick at each
    replicate's chosen x, and — when flagged — a star at the best mean point,
    named in the title. metrics: list of (column, 'min'|'max', mark_best).

    The source panels of each (row, metric) pair share a y-scale, so band
    widths (seed vs bootstrap variability) compare directly. Metrics whose
    scale is comparable across rows (`share_rows`) share one y-scale
    everywhere; AIC/BIC are not (the likelihood lives in a different space
    at each pc count), so those rescale per row.
    """
    set_style()
    src_names = list(sources)
    rows = list(dict.fromkeys(r for s in sources.values() for r in s[row_col]))
    lines = sorted(set().union(*[set(s[line_col]) for s in sources.values()]))
    colors = {n: plt.cm.tab10(i % 10) for i, n in enumerate(lines)}
    ncols = len(metrics) * len(src_names)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(5.5 * ncols, 4.2 * len(rows)),
                             squeeze=False)
    shared = {m: [] for m, _, _ in metrics}  # metric -> axes, for share_rows
    for i, rv in enumerate(rows):
        for j, (m, how, mark_best) in enumerate(metrics):
            pair = []
            for si, sname in enumerate(src_names):
                ax = axes[i][j * len(src_names) + si]
                pair.append(ax); shared[m].append(ax)
                sub = sources[sname]
                sub = sub[sub[row_col] == rv]
                best = None  # (value, x, line name)
                for name in lines:
                    c = colors[name]
                    piv = sub[sub[line_col] == name].pivot_table(index=x, columns=rep_col, values=m)
                    if piv.empty or piv.isna().all().all():
                        continue
                    mean = piv.mean(axis=1)
                    ax.plot(mean.index, mean.values, "o-", ms=4, color=c, label=str(name))
                    ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=c, alpha=0.2)
                    chosen = piv.idxmin() if how == "min" else piv.idxmax()  # per replicate
                    ax.plot(chosen.values, mean.loc[chosen.values], "|", ms=16, mew=2.5, color=c)
                    xb = mean.idxmin() if how == "min" else mean.idxmax()
                    v = mean.loc[xb]
                    if best is None or (v < best[0] if how == "min" else v > best[0]):
                        best = (v, xb, name)
                panel_title = f"{m} — {sname}\n" if i == 0 else ""
                if mark_best and best is not None:
                    val = f" = {best[0]:.3f}" if m == "silhouette" else ""
                    panel_title += f"{how} at {x}={best[1]:g} ({best[2]}){val}"
                    ax.plot([best[1]], [best[0]], "*", color="crimson", ms=20, zorder=5)
                ax.set_title(panel_title)
                xvals = sorted(sub[x].unique())
                ax.set_xticks(xvals)
                ax.tick_params(axis="x", labelsize=13)
                ax.tick_params(axis="y", labelsize=14)
                if i == len(rows) - 1:
                    ax.set_xlabel(x)
            axes[i][0].set_ylabel(str(rv))
            lo = min(a.get_ylim()[0] for a in pair)
            hi = max(a.get_ylim()[1] for a in pair)
            for a in pair:
                a.set_ylim(lo, hi)
                if max(abs(lo), abs(hi)) >= 1e6:  # ticks in millions, not "1e7" offset
                    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    for m, _, _ in metrics:
        if m in share_rows and shared[m]:
            lo = min(a.get_ylim()[0] for a in shared[m])
            hi = max(a.get_ylim()[1] for a in shared[m])
            for a in shared[m]:
                a.set_ylim(lo, hi)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside right upper", fontsize=26, markerscale=3,
               handlelength=3, borderpad=0.8)
    if title:
        fig.suptitle(title, fontsize=30)
    return _finish(fig, out)


def composition_heatmap(labels, categories: dict, out=None):
    """One heatmap: clusters (size in tick labels) x stacked category blocks.

    categories: {'label': label_array, 'topic': topic_array} — each becomes a
    column block, drawn side by side with a divider.
    """
    from .metrics import composition
    set_style()
    labels = np.asarray(labels)
    clusters = np.unique(labels[labels >= 0])
    blocks, col_labels, bounds = [], [], []
    for name, cat in categories.items():
        _, cats, _, frac, _ = composition(labels, np.asarray(cat))
        blocks.append(frac)
        col_labels += [f"{name}={c}" for c in cats]
        bounds.append(sum(b.shape[1] for b in blocks))
    M = np.hstack(blocks)
    sizes = np.array([(labels == c).sum() for c in clusters])

    fig, ax = plt.subplots(figsize=(1.1 * M.shape[1] + 3, 0.6 * len(clusters) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"C{c} (n={s})" for c, s in zip(clusters, sizes)])
    for b in bounds[:-1]:
        ax.axvline(b - 0.5, color="white", lw=3)
    fig.colorbar(im, ax=ax, label="fraction of cluster")
    return _finish(fig, out)


def pca_stability_panel(result, out=None):
    """result: pca.StabilityResult."""
    set_style()
    names = [n for n in result.fit_names if n != "full"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    kinds = sorted(set(result.fit_kind[n] for n in names))
    for kind in kinds:
        sub = [n for n in names if result.fit_kind[n] == kind]
        axes[0].scatter([kind] * len(sub), [result.overlap_to_ref[n] for n in sub], s=80)
    axes[0].set_title("subspace overlap vs full"); axes[0].set_ylabel("overlap"); axes[0].set_ylim(0, 1.02)

    for n in names:
        axes[1].plot(range(1, result.m + 1), result.cos_bestmatch[n], "o-", alpha=0.6,
                     label=result.fit_kind[n])
    axes[1].set_title("per-component agreement"); axes[1].set_xlabel("component"); axes[1].set_ylabel("|cos|")
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())

    im = axes[2].imshow(result.pairwise_overlap, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(result.fit_names))); axes[2].set_yticks(range(len(result.fit_names)))
    axes[2].set_xticklabels(result.fit_names, rotation=90); axes[2].set_yticklabels(result.fit_names)
    axes[2].set_title("pairwise subspace overlap"); fig.colorbar(im, ax=axes[2])
    return _finish(fig, out)


def method_ari_matrix(label_sets, names, out=None):
    from .agreement import pairwise_ari
    set_style()
    M = pairwise_ari(label_sets)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3, 1.0 * len(names) + 2))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=11)
    ax.set_title("clustering agreement (ARI)"); fig.colorbar(im, ax=ax)
    return _finish(fig, out)


def embedding_plot(coords, labels, unstable=None, title="embedding", out=None):
    set_style()
    coords, labels = np.asarray(coords), np.asarray(labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    stable = np.ones(len(coords), bool)
    if unstable is not None:
        stable[unstable] = False
        ax.scatter(coords[~stable, 0], coords[~stable, 1], s=10, color="lightgray", label="unstable")
    for c in np.unique(labels):
        m = (labels == c) & stable
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.7, label=str(c))
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.legend(markerscale=2)
    return _finish(fig, out)


def mds_centroid_map(clusters, dist, coords, out=None):
    """Faithful inter-cluster distance map + the distance numbers."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(coords[:, 0], coords[:, 1], s=200)
    for i, c in enumerate(clusters):
        ax1.annotate(f"C{c}", coords[i], fontsize=15, ha="center", va="center")
    ax1.set_title("cluster centroids (MDS)"); ax1.set_xlabel("dim 1"); ax1.set_ylabel("dim 2")

    im = ax2.imshow(dist, cmap="viridis")
    ax2.set_xticks(range(len(clusters))); ax2.set_yticks(range(len(clusters)))
    ax2.set_xticklabels([f"C{c}" for c in clusters]); ax2.set_yticklabels([f"C{c}" for c in clusters])
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            ax2.text(j, i, f"{dist[i, j]:.1f}", ha="center", va="center", color="white", fontsize=11)
    ax2.set_title("centroid distances"); fig.colorbar(im, ax=ax2)
    return _finish(fig, out)


def across_layers(layer_values: dict, ylabel, title, out=None):
    """layer_values: {series_name: {layer: value}} -> one line per series."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, d in layer_values.items():
        layers = sorted(d)
        ax.plot(layers, [d[l] for l in layers], "o-", label=name)
    ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    return _finish(fig, out)


def _finish(fig, out):
    if out is not None:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import tempfile, os
    from sklearn.datasets import make_blobs
    from .methods import fit_gmm
    from . import pca, embeddings

    d = tempfile.mkdtemp()
    X, y = make_blobs(n_samples=500, n_features=40, centers=4, random_state=0)
    topic = np.array(["a", "b", "c"])[np.random.default_rng(0).integers(0, 3, len(X))]
    Z = pca.fit(X, 6).transform(X)
    res = fit_gmm(Z, 4, "full", 0)

    pc_scatter_grid(Z, {"cluster": res.labels, "label": y, "topic": topic}, out=f"{d}/scatter.png")
    composition_heatmap(res.labels, {"label": y, "topic": topic}, out=f"{d}/composition.png")
    clusters, _, dist, coords = embeddings.centroid_distance_map(Z, res.labels)
    mds_centroid_map(clusters, dist, coords, out=f"{d}/mds.png")
    print("wrote:", os.listdir(d), "in", d)
    print(d)
