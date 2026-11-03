"""Steering figure, projected-coefficient version: figure6_steering_projected.

A row per model (Gemma-3-4B layer 29, Qwen2.5-7B layer 24), a column per
dataset. In each cell the x axis is how many principal components of the
coefficient vector's own training rows were kept (2, 10, 50) and the y
axis is the answer shift per steering step of size 1: how far the model's
preference for "True" over "False" moved, in log-odds, per push of one
True-minus-False class separation of that dataset. Every vector is unit
length, so at a given step every vector is pushed the same distance.

Five lines per cell, one per coefficient vector: own-dataset regression,
pooled regression, pooled hierarchical fit (solid), and the family-removed
regression and hierarchical fit (dashed). Two horizontal reference lines
are the class-mean differences the earlier figure compared against: the
dataset's own and the family-removed one. Grey bands are what random
vectors do after the same projection (median of 8 draws): the darker band
on the dataset's own components, the lighter band on the family-removed
pool's. A line inside its band is indistinguishable from an arbitrary
vector in that same subspace. Each cell carries the baseline AUC of the
unedited model's True/False answer on that dataset.

Inputs: data/figure6_projected_effects.csv (written by
distill_projected_steering.py from the gemmaproj / qwenproj sweeps).
The original figure6_steering is left as it was."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import fig_common as fc

NAVY, LIGHT = "#0B3B8C", "#6D9BF1"
KS = [2, 10, 50]
X = np.arange(len(KS))
# legend wording follows the other figures: the method, then the rows it
# was fitted on (target / pool / domain-absent pool)
SOURCES = [
    ("L2 reg on target", "own_l2", fc.ORANGE, "-"),
    ("L2 reg on pool", "pooled_l2", fc.TEAL, "-"),
    ("random slopes on pool", "pooled_rs", fc.PURPLE, "-"),
    ("L2 reg on domain-absent pool", "famout_l2", fc.TEAL, "dashed"),
    ("random slopes on domain-absent pool", "famout_rs", fc.PURPLE, "dashed"),
]
DASH = (0, (1.8, 1.3))      # dash length, gap, in line widths: short enough
                            # to show several dashes in a normal legend handle
REFS = [("diff-in-means on target", "own_diff", NAVY),
        ("diff-in-means on domain-absent pool", "famout_diff", LIGHT)]
BANDS = [("rand_own", 0.38, "random vectors, target components"),
         ("rand_famout", 0.20, "random vectors, domain-absent components")]
RUNS = [("gemma", 29), ("qwen", 24)]
RUN_TITLE = {("gemma", 29): "Gemma-3-4B", ("qwen", 24): "Qwen2.5-7B"}
SHORT_TARGET = {"common_claim": "CommonClaim", "claude_vds": "VDS",
                "pubmedqa_labeled": "PubMedQA-lab", "xstest": "XSTest",
                "wildjailbreak_vanilla": "WJB vanilla",
                "wildjailbreak_adversarial": "WJB adversarial"}
TICK_PT, ANNOT_PT, LABEL_PT, CELL_TITLE_PT, LEGEND_PT = 17.0, 16.0, 18.0, 17.0, 16.5
ROW_TITLE_PT = 19.0
YCOL = "per_step"


def line(g: pd.DataFrame, source: str) -> np.ndarray:
    return np.array([g[(g.source == source) & (g.k == k)][YCOL].mean()
                     for k in KS])


def band(g: pd.DataFrame, source: str) -> np.ndarray:
    return np.array([g[(g.source == source) & (g.k == k)][YCOL].abs().median()
                     for k in KS])


def ref(g: pd.DataFrame, source: str) -> float:
    return float(g[g.source == source][YCOL].mean())


def cell(ax, g: pd.DataFrame, auc: float, ylim: tuple[float, float]) -> None:
    for source, alpha, _ in BANDS:
        b = band(g, source)
        ax.fill_between(X, 0, b, color=fc.GRAY, alpha=alpha, zorder=1, lw=0)
    for _, source, color in REFS:
        v = ref(g, source)
        if np.isfinite(v):
            ax.axhline(v, color=color, lw=1.6, ls="-", zorder=3)
    for _, source, color, ls in SOURCES:
        y = line(g, source)
        if np.isfinite(y).any():
            ax.plot(X, y, color=color, lw=1.8, ls=DASH if ls == "dashed" else "-",
                    marker="o", ms=3.8, zorder=4)
    ax.axhline(0, color=fc.INK, lw=0.6, zorder=2)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.35, len(KS) - 0.65)
    ax.set_xticks(X)
    ax.set_xticklabels([str(k) for k in KS])
    ax.tick_params(labelsize=TICK_PT)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # above the frame, right-aligned: outside the plotting area, so no
    # line can cross it whatever the row's scale turns out to be
    ax.text(1.0, 1.03, f"AUC {auc:.2f}", transform=ax.transAxes,
            fontsize=ANNOT_PT, ha="right", va="bottom", color=fc.MUTED)


def main() -> None:
    fc.setup()
    eff = pd.read_csv(C.DATA / "figure6_projected_effects.csv")
    datasets = [d for d in C.DATASETS if d in SHORT_TARGET]
    col = {d: i for i, d in enumerate(datasets)}
    n_d = len(datasets)

    # one y scale per model row, shared by its six cells: a little over
    # the highest line or band the row needs, and below zero only as far
    # as a line actually goes
    ylims = {}
    for m, l in RUNS:
        run = eff[(eff.model == m) & (eff.layer == l)]
        lo, hi = 0.0, 0.0
        for t in datasets:
            g = run[run.target == t]
            if not len(g):
                continue
            vals = [line(g, s) for _, s, _, _ in SOURCES] \
                + [np.array([ref(g, s)]) for _, s, _ in REFS] \
                + [band(g, s) for s, _, _ in BANDS]
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            lo, hi = min(lo, v.min()), max(hi, v.max())
        ylims[(m, l)] = (lo * 1.12 if lo < 0 else -0.04 * hi, hi * 1.12)

    # a wide, short canvas: two rows of low cells, one compact legend block
    # Taller rows keep the cells readable once LaTeX scales the figure to
    # the column; the header and legend keep their
    # absolute size, so the added height all goes to the two rows
    fig = plt.figure(figsize=(14.8, 6.75))
    gs = fig.add_gridspec(2, n_d, hspace=0.30, wspace=0.16,
                          left=0.10, right=0.985, top=0.905, bottom=0.275)
    grid_axes, row_left = [], {}
    for row, (m, l) in enumerate(RUNS):
        run = eff[(eff.model == m) & (eff.layer == l)]
        last = row == len(RUNS) - 1
        for t in [d for d in datasets if d in set(run.target)]:
            ax = fig.add_subplot(gs[row, col[t]])
            grid_axes.append(ax)
            g = run[run.target == t]
            cell(ax, g, float(g.baseline_auc.iloc[0]), ylims[(m, l)])
            if row == 0:
                # one line above the AUC label, which sits just over the frame
                ax.set_title(SHORT_TARGET[t], loc="left",
                             fontsize=CELL_TITLE_PT, pad=1.55 * ANNOT_PT)
            if not last:
                ax.tick_params(labelbottom=False)
            if col[t] == 0:
                row_left[row] = ax
            else:
                ax.tick_params(labelleft=False)

    for row, ax in row_left.items():
        pos = ax.get_position()
        fig.text(pos.x0 - 0.092, pos.y0 + pos.height / 2, RUN_TITLE[RUNS[row]],
                 rotation=90, ha="center", va="center", fontsize=ROW_TITLE_PT)
    boxes = [a.get_position() for a in row_left.values()]
    fig.text(min(b.x0 for b in boxes) - 0.054,
             (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2,
             "answer shift per\nsteering step (log-odds)", rotation=90,
             ha="center", va="center", fontsize=LABEL_PT, linespacing=1.0)
    blank = lambda: plt.Line2D([], [], ls="", marker="")   # noqa: E731
    ref_h = [plt.Line2D([], [], color=c, lw=2.0) for _, _, c in REFS]
    src_h = [plt.Line2D([], [], color=c, lw=2.2,
                        ls=DASH if ls == "dashed" else "-", marker="o", ms=4.5)
             for _, _, c, ls in SOURCES]
    band_h = [plt.Rectangle((0, 0), 1, 1, facecolor=fc.GRAY, alpha=a)
              for _, a, _ in BANDS]
    # three columns of three rows, no blanks, filled down each column:
    # the in-pool probes; everything fitted on the domain-absent pool; the
    # target reference line and the two noise bands. Measured after
    # drawing: it must stay inside the canvas, or the saved figure widens
    # and the type shrinks on the page.
    handles = src_h[:3] + src_h[3:] + [ref_h[1]] + [ref_h[0]] + band_h
    labels = ([lbl for lbl, _, _, _ in SOURCES] + [REFS[1][0]]
              + [REFS[0][0]] + [lbl for _, _, lbl in BANDS])
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, 0.155), ncols=3, frameon=False,
                     handlelength=2.0, handleheight=1.0, labelspacing=0.35,
                     columnspacing=0.9, fontsize=LEGEND_PT)
    fig.canvas.draw()
    ext = leg.get_window_extent()
    width_in = (ext.x1 - ext.x0) / fig.dpi
    if width_in > fig.get_size_inches()[0]:
        raise SystemExit(f"legend is {width_in:.1f} in wide, wider than the "
                         f"{fig.get_size_inches()[0]:.1f} in canvas")
    print(f"  legend {width_in:.2f} in wide (canvas {fig.get_size_inches()[0]:.1f} in)")
    # the x-axis label sits midway between the bottom row's tick labels
    # and the top of the legend, clear of both
    inv = fig.transFigure.inverted()
    y_top_row = inv.transform((0, leg.get_window_extent().y1))[1]
    bottom_axes = [a for a in grid_axes if a.get_position().y0
                   == min(b.get_position().y0 for b in grid_axes)]
    tick_bottom = min(inv.transform((0, t.get_window_extent().y0))[1]
                      for a in bottom_axes for t in a.get_xticklabels()
                      if t.get_text())
    left = min(a.get_position().x0 for a in grid_axes)
    right = max(a.get_position().x1 for a in grid_axes)
    fig.text((left + right) / 2, (tick_bottom + y_top_row) / 2,
             "leading principal components of the training data used for projection",
             ha="center", va="center", fontsize=LABEL_PT)
    fc.save(fig, "figure6_steering_projected",
            fc.caption_text("figure6_steering_projected"))


if __name__ == "__main__":
    main()
