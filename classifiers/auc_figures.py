"""The runners' AUC figures, import-safe anywhere (matplotlib + csv
only — no cuML), so they can be drawn by the GPU runners AND redrawn
locally from the metrics csvs (analysis_scripts.redraw) when only
styling changed. runner_common re-exports everything here for the
runners' use.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from .fig_style import FS, install_rcparams

install_rcparams()

# Inches per heatmap cell. imshow keeps cells SQUARE, so the width per
# column and the height per row must match: allocating more width than
# height per cell (a 1.5-wide / 0.75-tall split) makes
# the image fill only part of its panel and dumps the remainder into the
# gaps between panels. Matching them keeps the cells large — bigger than
# that effective 0.75in — with no wasted space.
CELL_IN = 0.85

# canonical display order for the classifier methods
METHOD_ORDER = ("l1reg", "l2reg", "svm_linear", "svm_rbf", "diff", "lda",
                "qda", "knn", "mlp", "xgboost", "random_slopes",
                "random_slopes_task", "moe", "moe_hard")


def dataset_order(names, datasets) -> list:
    """Sort dataset names into the collection's roster order, unknown
    names after in alphabetical order, and "all" (the merged run's pooled
    row) first."""
    def key(d):
        if d == "all":
            return (-1, "")
        return (datasets.index(d) if d in datasets else len(datasets), d)
    return sorted(names, key=key)


def read_metric_rows(csv_path: Path, tag: str, layer: int) -> list[dict]:
    """This (tag, layer)'s rows of a metrics csv, [] if the file is absent."""
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return [r for r in csv.DictReader(f)
                if r["features"] == tag and r["layer"] == str(layer)]


def box_cells(ax, cells, color: str = "red", lw: float = 2.5) -> None:
    """Outline the given (row, col) cells — the best performer per
    column / per dataset."""
    from matplotlib.patches import Rectangle

    for i, j in cells:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=color, linewidth=lw, zorder=5))


def best_per_column(m: np.ndarray) -> list:
    """(row, col) of the highest finite value in each column."""
    out = []
    for j in range(m.shape[1]):
        col = m[:, j]
        if np.isfinite(col).any():
            out.append((int(np.nanargmax(np.where(np.isfinite(col), col,
                                                  np.nan))), j))
    return out


def draw_auc_matrix(ax, m: np.ndarray, row_labels, col_labels, *,
                    xlabel: str = None, ylabel: str = None):
    """An AUC matrix on one axis: viridis image on [0.5, 1], each finite
    cell annotated with its full value ("0.947", "1.000" — always a
    leading digit). Returns the image (for a colorbar)."""
    im = ax.imshow(m, vmin=0.5, vmax=1.0, cmap="viridis")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center",
                        va="center", fontsize=FS["cell"],
                        color="w" if m[i, j] < 0.85 else "k")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right",
                  fontsize=FS["tick"])
    ax.set_yticks(range(len(row_labels)), row_labels,
                  fontsize=FS["tick"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"])
    return im


def plot_method_by_dataset(tables_dir: Path, csv_name, tag: str, layers,
                           methods, datasets, *, xlabel: str, title: str,
                           out_path: Path) -> None:
    """The methods x test-datasets AUC figure shared by run_leave_out
    and run_merged: one panel per layer, two panels per row, with all of a
    feature set's layers in a single figure. The suptitle carries the run
    information, while each panel just
    names its layer). `csv_name(layer)` gives the layer's metrics csv
    inside `tables_dir`; layers whose csv is missing are skipped, so a
    Slurm task that finished its own layer draws whatever layers exist
    so far and the last task to finish completes the figure."""
    import matplotlib.pyplot as plt

    per_layer = {}
    for layer in layers:
        rows = read_metric_rows(tables_dir / csv_name(layer), tag, layer)
        if rows:
            per_layer[layer] = rows
    if not per_layer:
        return
    meths = [m for m in methods
             if any(r["method"] == m for rows in per_layer.values()
                    for r in rows)]
    tests = dataset_order({r["test_dataset"] for rows in per_layer.values()
                           for r in rows}, datasets)
    n_cols = min(2, len(per_layer))
    n_rows = math.ceil(len(per_layer) / n_cols)
    # square cells at CELL_IN; the additive terms are only room for the
    # tick labels and colorbar
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=((CELL_IN * len(tests) + 3.2) * n_cols,
                 (CELL_IN * len(meths) + 2.6) * n_rows),
        squeeze=False, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.02,
                                wspace=0.02)
    for ax in axes.flat[len(per_layer):]:
        ax.axis("off")
    for i, (ax, (layer, rows)) in enumerate(zip(axes.flat,
                                                sorted(per_layer.items()))):
        m = np.full((len(meths), len(tests)), np.nan)
        for r in rows:
            if r["method"] in meths:
                m[meths.index(r["method"]),
                  tests.index(r["test_dataset"])] = float(r["auc"])
        im = draw_auc_matrix(
            ax, m, meths, tests,
            xlabel=xlabel if i + n_cols >= len(per_layer) else None)
        box_cells(ax, best_per_column(m))     # best method per dataset
        ax.set_title(f"layer {layer}", fontsize=FS["title"])
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=FS["tick"])
    fig.suptitle(f"{title} — red box = best method per column\n",
                 fontsize=FS["suptitle"])
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmaps(csv_path: Path, tag: str, layer: int, methods,
                  train, tests, results: Path) -> None:
    """run_per_dataset's figure for one (tag, layer): a panel per method
    with the AUC matrix, training datasets x the run's datasets
    (diagonal = CV)."""
    import matplotlib.pyplot as plt

    rows = read_metric_rows(csv_path, tag, layer)
    if not rows:
        return
    mats = {}
    for method in methods:
        m = np.full((len(train), len(tests)), np.nan)
        for r in rows:
            if r["method"] == method and r["train_dataset"] in train \
                    and r["test_dataset"] in tests:
                m[train.index(r["train_dataset"]),
                  tests.index(r["test_dataset"])] = float(r["auc"])
        mats[method] = m
    # which method wins each (train, test) cell — that panel's cell gets
    # the red box marks the best method per dataset
    stack = np.stack([mats[m] for m in methods])
    best = np.full(stack.shape[1:], -1)
    finite = np.isfinite(stack).any(axis=0)
    best[finite] = np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf),
                                axis=0)[finite]

    # square cells at CELL_IN; the additive terms are only room for the
    # tick labels and colorbar
    n_cols = min(3, len(methods))
    n_rows = math.ceil(len(methods) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=((CELL_IN * len(tests) + 3.4) * n_cols,
                 (CELL_IN * len(train) + 3.0) * n_rows),
        squeeze=False, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.03,
                                wspace=0.03)
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    for i, (ax, method) in enumerate(zip(axes.flat, methods)):
        # axis labels on the grid's outer edges only — an inner panel's
        # xlabel gets pushed below its long rotated tick names, straight
        # into the next row's title
        im = draw_auc_matrix(
            ax, mats[method], train, tests,
            xlabel="test dataset" if i + n_cols >= len(methods) else None,
            ylabel="train dataset" if i % n_cols == 0 else None)
        box_cells(ax, np.argwhere(best == i))
        ax.set_title(method, fontsize=FS["title"])
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=FS["tick"])
    fig.suptitle(f"AUC, {tag} layer {layer} (diagonal = CV, off-diagonal "
                 f"= train row -> test column; red box = best method for "
                 f"that cell)\n", fontsize=FS["suptitle"])
    out = results / "figures" / f"auc_{tag}_layer{layer}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
