"""Q12 figures: pooled training, every dataset, one figure per pool.

figure4_pooled_training      the expanded pool: per-dataset AUC under pooled
                           training beside the alignment of the pooled
                           random-slopes coefficients with the per-dataset
                           ones, measured by the regularized Mahalanobis
                           cosine (the plain
                           cosine it replaced is kept in the same file). Absorbs the top panel of the former
                           random-slopes figure.
Supersedes the separate q9_conflict scatter: the within-dataset ceiling it
compared against is drawn here per dataset and per run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import fig_common as F

DEPTH = 3                     # 88% of the stack: gemma 29, qwen 24
RUNS = [("gemma", "true_false"), ("qwen", "true_false")]
# the pooled difference in class means is drawn in the navy fig_steering
# uses for the same direction family, so the two figures read against each
# other: it is the direction that moves the model's answer there
NAVY = "#0B3B8C"
AUC_MARKS = [("l2reg", "Pooled L2", F.BLUE, "o", False),
             ("diff", "Pooled diff-in-means", NAVY, "s", False),
             ("random_slopes_task", "Random slopes (domain)", F.TEAL, "^",
              True),
             ("random_slopes", "Random slopes (dataset)", F.TEAL, "D", False),
             ("within", "Within-dataset fit", F.INK, "|", False),
             ("shared_cv", "Shared direction only", F.GRAY, "o", True)]
# text a step above the deep dive's standard: this figure is dense and
# the legend, pair headers, model subtitles and axis titles carry it
LEGEND_PT, HEADER_PT, SUBTITLE_PT, XLABEL_PT = 17.0, 19.0, 18.0, 19.0
MEASURE = "mahalanobis_reg_cov_pool"   # see coefficient_alignment
COS_MARKS = [("shared", "Shared direction", F.GRAY, "o", False),
             ("pooled_l2", "Pooled L2", F.BLUE, "s", True),
             ("dev", "Dataset deviation", F.PURPLE, "o", False),
             ("shared_plus_dev", "Shared + deviation", F.TEAL, "D", False)]


# ----------------------------------------------------------------- data
def shared_only_auc() -> pd.Series:
    """Cross-validated AUC of the pooled random-slopes SHARED direction on
    its own, from the same expanded-pool fit panel a scores: each outer
    fold refit at its stored hyperparameters, the held-out rows scored
    with the shared weights alone, and the five folds' out-of-fold scores
    pooled (figure4_shared_direction_cv.csv, written by q12_rs_shared_cv.py from the
    rs_shared_cv.py refits)."""
    cv = pd.read_csv(C.DATA / "figure4_shared_direction_cv.csv")
    cv = cv[(cv.fold == "all") & (cv.depth == DEPTH)]
    return cv.set_index(["model", "perturbation", "dataset"]).auc_shared


def pooled_auc() -> pd.Series:
    """Per-dataset AUC with the target in the expanded pool; the input is
    already the 88%-depth layer, and the shared-only rung comes from the
    fold refits of the same fit (shared_only_auc)."""
    exp = pd.read_csv(C.DATA / "figure4_expanded_pool_auc.csv")
    fits = (exp[exp.method.isin([m[0] for m in AUC_MARKS
                                 if m[0] not in ("within", "shared_cv")])]
            [["model", "perturbation", "dataset", "method", "auc"]]
            .rename(columns={"method": "series", "auc": "value"}))
    ceiling = (exp.groupby(["model", "perturbation", "dataset"],
                           as_index=False).within_cv_l2.first()
               .rename(columns={"within_cv_l2": "value"}).assign(series="within"))
    shared = (shared_only_auc().rename("value").reset_index()
              .assign(series="shared_cv"))
    both = pd.concat([fits, ceiling, shared], ignore_index=True)
    return both.groupby(["model", "perturbation", "dataset",
                         "series"]).value.median()


def coefficient_alignment() -> pd.Series:
    """Regularized Mahalanobis cosines for panel b.

    The random-slopes components and the pooled L2 direction are compared
    with the coefficients fitted on each dataset alone. The comparison is
    made in the geometry of the activations rather than in raw
    coordinates: every inner product is taken through the Ledoit-Wolf
    regularized covariance of the pooled activations,

        cos_S(a, b) = a' S b / sqrt((a' S a) (b' S b)),

    so two coefficient vectors count as aligned when they order the data
    the same way, not when their raw coordinates happen to agree. Plain
    cosines understate alignment badly here, because the directions the
    fits disagree about are mostly directions the activations barely vary
    along. This is the measure coef_similarity calls
    mahalanobis_reg_cov_pool and supplementary figure 4 already uses for
    the dataset-pair similarities. The direct cosine between the shared
    random-slopes direction and pooled L2 is stored in the same file with
    an empty dataset and reported in the caption; it is one value per
    model rather than one per dataset, so it is not repeated on every row
    of the panel.

    Read from the expanded-pool fit that panel a scores, so both panels
    describe one experiment and VDS
    and WJB-adversarial -- pool members there but not in the base pool --
    have deviations of their own. figure4_mahalanobis_alignment.csv is
    written by mahalanobis_alignment.py, which runs on compute cluster because the
    covariance estimates live there; it also carries the plain cosines
    panel b used to draw (measure "plain") and a variant whitened by each
    dataset's own covariance ("mahalanobis_reg_cov_test").

    Roster caveat: the cached pooled covariance was estimated over 13
    datasets, Blood ranges included, while the pool and the panel are the
    other 12. Only the whitening is affected, and only through one
    dataset's share of the corpus.
    """
    coll = {v: k for k, v in C.COLLECTIONS.items()}
    m = pd.read_csv(C.DATA / "figure4_mahalanobis_alignment.csv")
    m = m[(m.measure == MEASURE) & m.dataset.notna()
          & m.comparison.isin([c[0] for c in COS_MARKS])]
    m["model"] = m.collection.map(lambda c: coll[c][0])
    m["perturbation"] = m.collection.map(lambda c: coll[c][1])
    return m.groupby(["model", "perturbation", "dataset",
                      "comparison"]).value.median()


# -------------------------------------------------------------- drawing
def _pair(fig, inner, table, marks, xlim, model_labels, order, ypos,
          xticks, first_col, zero_line):
    """One pair of panels: same dataset rows, one panel per model."""
    axes = []
    for m, (model, pert) in enumerate(RUNS):
        ax = fig.add_subplot(inner[0, m])
        axes.append(ax)
        ax.set_yticks(ypos)
        vals = {ser: np.array([table.get((model, pert, d, ser), np.nan)
                               for d in order]) for ser, *_ in marks}
        stack = np.vstack(list(vals.values()))
        ax.hlines(ypos, np.nanmin(stack, 0), np.nanmax(stack, 0),
                  color=F.GRID, lw=1.8, zorder=1)
        for ser, label, colour, marker, hollow in marks:
            style = ({"color": colour, "linewidths": 1.8} if marker == "|"
                     else {"facecolors": "none" if hollow else colour,
                           "edgecolors": colour, "linewidths": 1.5})
            ax.scatter(vals[ser], ypos, s=150 if marker == "|" else 44,
                       marker=marker, zorder=3,
                       label=label if m == 0 else None, **style)
        if zero_line:
            ax.axvline(0, color=F.MUTED, lw=0.8, zorder=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xticks(xticks)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(model_labels[model], loc="left",
                     fontsize=SUBTITLE_PT, pad=13)
        ax.set_yticklabels([C.DATASET_LABEL[d] for d in order]
                           if first_col and m == 0 else [])
    return axes


def main() -> None:
    F.setup()
    auc = pooled_auc()
    cos = coefficient_alignment()
    order = (auc.xs("l2reg", level="series").groupby("dataset").min()
             .sort_values().index.tolist())
    ypos = np.arange(len(order))

    # narrower than the deep dive's canvas so the text, which keeps its
    # point size, reads larger once the figure is scaled to the column
    exp_fig = plt.figure(figsize=(13.0, 7.2))
    outer = exp_fig.add_gridspec(1, 2, wspace=0.135, top=0.795, bottom=0.30,
                                 left=0.105, right=0.985)
    pairs = [("Pooled training, target in the pool", auc, AUC_MARKS,
              (0.58, 1.012), [0.6, 0.8, 1.0], False,
              "AUC on the target dataset"),
             ("Alignment of the coefficients", cos, COS_MARKS, (-0.22, 1.0),
              [0.0, 0.5, 1.0], True,
              "Mahalanobis cos with each dataset's direction")]
    first_axes = []
    for b, (header, table, marks, xlim, xticks, zero, xlab) in enumerate(pairs):
        axes = _pair(exp_fig, outer[0, b].subgridspec(1, 2, wspace=0.145),
                     table, marks, xlim, C.MODEL_LABEL, order, ypos, xticks,
                     b == 0, zero)
        first_axes.append((axes, header, xlab))

    exp_fig.canvas.draw()
    for letter, (axes, header, xlab) in zip("ab", first_axes):
        box = axes[0].get_position()
        centre = (box.x0 + axes[-1].get_position().x1) / 2
        # Left-align each pair header immediately after its panel marker.
        panel_y = 0.875
        exp_fig.text(box.x0 - 0.016, panel_y, f"{letter})", ha="right",
                     va="center", fontsize=F.pt("title"), fontweight="bold",
                     color=F.INK)
        exp_fig.text(box.x0, panel_y, header, ha="left", va="center",
                     fontsize=HEADER_PT, color=F.INK)
        # the x-label belongs to the pair, not to one panel of it
        exp_fig.text(centre, 0.205, xlab, ha="center", va="center",
                     fontsize=XLABEL_PT, color=F.INK)
        h, l = axes[0].get_legend_handles_labels()
        # nudged apart so the two legends never touch
        exp_fig.legend(h, l, loc="upper center",
                       bbox_to_anchor=(centre + (-0.05 if letter == "a"
                                                else 0.014), 0.165),
                       ncol=2, frameon=False,
                       fontsize=LEGEND_PT, handletextpad=0.5,
                       columnspacing=1.0)

    F.save(exp_fig, "figure4_pooled_training", F.caption_text("figure4_pooled_training"))


if __name__ == "__main__":
    main()
