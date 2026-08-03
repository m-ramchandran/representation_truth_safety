"""Shared figure helpers: annotated heatmaps from DataFrames. Text
sizes and the rcParams install live in classifiers.fig_style (shared
with scripts/runner_common.py)."""

from __future__ import annotations

import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..fig_style import FS, FS_SMALL, heatmap_size, install_rcparams

PANEL_COLS = 3          # panels per row in multi-panel figures

install_rcparams()


def _wrap(title: str, width_in: float) -> str:
    return "\n".join(textwrap.wrap(title, max(20, int(12 * width_in))))


def cell_text_color(cmap: str, frac: float) -> str:
    """Black on light cells, white on dark ones, from the luminance of
    the actual colormap color at `frac` (the 0-1 position in [vmin,
    vmax]) — a fixed midpoint threshold picks invisible white on
    colormaps that are light in the middle (coolwarm)."""
    r, g, b, _ = matplotlib.colormaps[cmap](float(np.clip(frac, 0, 1)))
    return "k" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "w"


def heatmap(df, *, vmin: float, vmax: float, cmap: str = "viridis",
            title: str = "", subtitle: str = "", ax=None,
            colorbar: bool = True, fs: dict = None, sublabels=None):
    """df's values as a colored matrix, each finite cell annotated (text
    color follows the cell's brightness); row and column labels from the
    index and columns, plus a colorbar legend. `subtitle` is a smaller
    line under the title (wrapped) for detail that would otherwise bloat
    it. `sublabels` is an aligned frame of short strings (may contain
    newlines) drawn a step smaller under each cell's number, for a value
    that names where the number came from — the winning method in
    cross_methods' best-of panel. `fs` swaps the text sizes (FS_SMALL
    for the single-axes coefficient plots). Returns the axis."""
    fs = fs or FS
    if ax is None:
        _, ax = plt.subplots(figsize=heatmap_size(len(df),
                                                  len(df.columns)))
    m = df.to_numpy(dtype=float)
    im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                frac = (m[i, j] - vmin) / (vmax - vmin) if vmax > vmin else .5
                color = cell_text_color(cmap, frac)
                sub = None if sublabels is None else sublabels.iat[i, j]
                sub = sub if isinstance(sub, str) and sub else None
                # the number rides up to leave the lower half of the cell
                # for the sublabel, further the more lines it wraps to
                dy = 0 if sub is None else 0.16 + 0.10 * sub.count("\n")
                ax.text(j, i - dy, f"{m[i, j]:.2f}", ha="center",
                        va="center", fontsize=fs["cell"], color=color)
                if sub:
                    ax.text(j, i + 0.20, sub, ha="center", va="center",
                            fontsize=fs["cell"] - 2, color=color)
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=45,
                  ha="right", fontsize=fs["tick"])
    ax.set_yticks(range(len(df.index)), df.index, fontsize=fs["tick"])
    width_in = 0.8 * len(df.columns) + 3.6
    line_titles(ax, _wrap(title, width_in), subtitle, fs=fs,
                wrap_in=width_in)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=fs["tick"])
    return ax


def panel_grid(n: int, panel_w: float, panel_h: float,
               ncols: int = PANEL_COLS):
    """A wrapped (nrows x ncols) grid for n same-sized panels — extra
    axes switched off — with constrained layout doing the spacing.
    Returns (fig, the n live axes)."""
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False, layout="constrained")
    # room for panel titles + the subtitle line above each axes
    fig.get_layout_engine().set(h_pad=0.35, w_pad=0.2)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, list(axes.flat[:n])


def subtitle_corner(fig, subtitle: str) -> None:
    """The run stem (or other detail) as a small gray line in a
    multi-panel figure's top-left corner — the suptitle stays a clean
    short title, with no run stems inside it."""
    if subtitle:
        fig.text(0.005, 0.998, subtitle, ha="left", va="top",
                 fontsize=FS["subtitle"], color="0.35")


def line_titles(ax, title: str, subtitle: str = "", fs: dict = None,
                wrap_in: float = None) -> None:
    """Title + smaller gray subtitle above one axes — the one subtitle
    mechanism for heatmaps and line/bar plots alike. `wrap_in` (the
    panel width in inches) wraps the subtitle to the panel at ~9
    chars/inch so long method lists cannot spill into neighboring
    panels; the title pad grows with the subtitle's line count so the
    two never collide."""
    fs = fs or FS_SMALL
    if subtitle and wrap_in:
        subtitle = "\n".join(textwrap.wrap(subtitle,
                                           max(20, int(9 * wrap_in))))
    pad = 8 + 21 * (subtitle.count("\n") + 1) if subtitle else 6
    ax.set_title(title, fontsize=fs["title"], pad=pad)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=fs["subtitle"], color="0.35")


def layer_panels(by_layer: dict, title: str, subtitle: str, *,
                 vmin: float, vmax: float, cmap: str = "viridis",
                 fs: dict = None, sublabels: dict = None):
    """One heatmap panel per layer, two per row, sized from the largest
    frame — the shared shape of every layers-in-one-figure heatmap
    (per_slice, gates, coef_pca all draw through here). `sublabels`, if
    given, maps the same layers to frames of per-cell sublabels."""
    fs = fs or FS_SMALL
    n_col = max(len(v.columns) for v in by_layer.values())
    n_row = max(len(v) for v in by_layer.values())
    fig, axes = panel_grid(len(by_layer), *heatmap_size(n_row, n_col),
                           ncols=2)
    for ax, (layer, df) in zip(axes, sorted(by_layer.items())):
        heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap,
                title=f"layer {layer}", ax=ax, fs=fs,
                sublabels=None if sublabels is None
                else sublabels.get(layer))
    fig.suptitle(title, fontsize=fs["suptitle"])
    subtitle_corner(fig, subtitle)
    return fig


def method_panels(values, *, vmin: float, vmax: float,
                  cmap: str = "viridis", title: str = "",
                  subtitle: str = ""):
    """One heatmap panel per method from a Series indexed by
    (method, fit, dataset): rows = fit, columns = evaluated dataset,
    panels wrapped PANEL_COLS per row, a colorbar legend on each.
    `subtitle` (the run stem) goes to the top-left corner. Methods with
    nothing to show (all NaN — knn in the margin stats) are left out,
    and when every fit evaluates exactly its own dataset (leave-out in
    dataset mode) the diagonal panels collapse to a single
    method x dataset heatmap."""
    values = values.dropna()
    idx = values.index
    if (idx.get_level_values("fit") == idx.get_level_values("dataset")).all():
        ax = heatmap(values.droplevel("fit").unstack("dataset"),
                     vmin=vmin, vmax=vmax, cmap=cmap, title=title,
                     subtitle=subtitle)
        return ax.figure
    methods = list(idx.get_level_values("method").unique())
    n_fit = idx.get_level_values("fit").nunique()
    n_col = idx.get_level_values("dataset").nunique()
    # imshow keeps cells square, so panel width tracks the height scale
    # (plus room for row labels and the colorbar) — a wider box would
    # only add whitespace between panels
    fig, axes = panel_grid(len(methods), *heatmap_size(n_fit, n_col))
    for ax, method in zip(axes, methods):
        heatmap(values.xs(method).unstack("dataset"), vmin=vmin, vmax=vmax,
                cmap=cmap, title=method, ax=ax)
    fig.suptitle(f"{title} — rows = fit, columns = evaluated dataset",
                 fontsize=FS["suptitle"])
    subtitle_corner(fig, subtitle)
    return fig


def save(fig_or_ax, path) -> None:
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    # bbox_inches="tight" keeps artists anchored outside the axes (the
    # bar charts' side legends) inside the saved image instead of
    # clipping them at the figure edge
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
