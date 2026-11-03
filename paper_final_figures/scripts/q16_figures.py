"""Q16 + Q2/Q3 figures: probe similarity, and transfer by label type.

figure3_similarity_and_labels, one row, two panels:
  a  agreement between two methods' per-sample predictions, per training
     condition (figure3_prediction_similarity.csv)
  b  what averaging buys, as ensemble minus reference AUC: the mean of
     the ten methods against one method (left; figure3_method_ensemble.csv,
     figure3_heldout_method_ensemble.csv) and combinations of the other
     datasets' classifiers against the target's own (right;
     figure3_heldout_direction_ensemble.csv)

supplementary_figure5_label_types, one matrix per model:
     single-source diff-in-means transfer AUC by source and target label
     type (supplementary_figure5_label_transfer.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import fig_common as fc
from q16_label_semantics import (GROUP_LABEL, GROUP_ORDER, LABEL_GROUP,
                                 MID_DEPTHS)

DEAD_TARGET = "claude_blood_test_ranges"      # within-CV ceiling 0.45/0.59
KIND_COLOR = {"within_linear": fc.BLUE, "cross_family": fc.PURPLE,
              "within_nonlinear": fc.ORANGE}
KIND_LABEL = {"within_linear": "linear ↔ linear",
              "cross_family": "linear ↔ nonlinear",
              "within_nonlinear": "nonlinear ↔ nonlinear"}
REGIMES = ["within_cv", "merged_included", "leave_dataset_out",
           "leave_domain_out"]
RSHORT = {"within_cv": "Within\ntarget", "merged_included": "Pooled,\ntarget in",
          "leave_dataset_out": "Target\nabsent", "leave_domain_out": "Domain\nabsent"}
SHORT = {"claude_anatomy": "Anatomy", "wildjailbreak_adversarial": "WJB-adv",
         "claude_vds": "VDS"}
ANNOT_DY = {"wildjailbreak_adversarial": 8, "claude_vds": -8,
            "claude_anatomy": 9}
TRANSFER_REGIMES = ["single_source", "leave_domain_out",
                    "leave_dataset_out"]
TRANSFER_LABEL = {"single_source": "One source",
                  "leave_domain_out": "Domain absent",
                  "leave_dataset_out": "Target absent"}
LEGEND_PT = 14.5
SUBTITLE_PT = 14


# ------------------------------------------------------------ panel a
def agreement_panel(ax) -> None:
    """Spearman between two methods' per-sample predictions, split by
    whether the pair is linear/linear, mixed, or nonlinear/nonlinear."""
    corr = pd.read_csv(C.DATA / "figure3_prediction_similarity.csv")
    width = 0.24
    for k, kind in enumerate(KIND_COLOR):
        sel = corr[corr.pair_kind == kind]
        med = sel.groupby("regime")["corr"].median().reindex(REGIMES)
        q1 = sel.groupby("regime")["corr"].quantile(.25).reindex(REGIMES)
        q3 = sel.groupby("regime")["corr"].quantile(.75).reindex(REGIMES)
        x = np.arange(len(REGIMES)) + (k - 1) * width
        ax.bar(x, med, width=width * 0.9, color=KIND_COLOR[kind],
               label=KIND_LABEL[kind], zorder=3)
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none",
                    ecolor=fc.INK, elinewidth=1, capsize=2, zorder=4)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([RSHORT[r] for r in REGIMES])
    ax.set_ylim(0.5, 1.0)
    # one line: at this panel height that needs a size below the axis
    # label standard, else the text overruns the axes at both ends
    ax.set_ylabel("Correlation between predictions", fontsize=14.5)


# ------------------------------------------------------------ panel b
def _ensemble_by_regime() -> dict:
    """One frame per regime, keyed the way fig01 keys them.

    within_cv is the only regime with no per-sample scores, so it keeps the
    stored ensemble column — safe there, because the per-dataset runs fit
    only the ten roster methods, so that column can only be their mean. The
    merged runs also fit the four pooled-only methods, and their stored
    column averages all fourteen; the other three regimes therefore come
    from the per-sample scores, where the roster is ours to choose.

    merged_eval is the pooled fit scored on the datasets kept permanently
    outside the pool (WJB adversarial and VDS once Blood ranges is dropped),
    so those cells are target-absent, not pooled-target-in. leave_task cells are matched by domain, not by name:
    matching on name only ever caught social_iqa, which is both a dataset
    and a domain.
    """
    cv = pd.read_csv(C.DATA / "figure3_method_ensemble.csv")
    h = pd.read_csv(C.DATA / "figure3_heldout_method_ensemble.csv")
    # paper roster: Blood ranges is dropped everywhere (it reaches this
    # table as a merged_eval target and inside the medical leave_task rows)
    cv = cv[cv.dataset.isin(C.PAPER_DATASETS)]
    h = h[h.dataset.isin(C.PAPER_DATASETS)]
    h["vs_best"] = h.ens_mean_all - h.best_single
    h["vs_median"] = h.ens_mean_all - h.median_single
    lt = h[h.kind == "leave_task"].copy()
    lt["target_domain"] = lt.dataset.map(C.DOMAIN_OF)
    lt["holdout_domain"] = lt.holdout.map(C.DOMAIN_RENAME)

    w = cv[cv.regime == "within_cv"]
    return {
        "within_cv": pd.DataFrame({"vs_best": w.ens_minus_best,
                                   "vs_median": w.ens_minus_median}),
        "merged_included": h[h.kind == "merged_cv"],
        "leave_dataset_out": pd.concat(
            [h[(h.kind == "leave_dataset") & (h.holdout == h.dataset)],
             h[h.kind == "merged_eval"]]),
        "leave_domain_out": lt[lt.target_domain == lt.holdout_domain],
    }


def _rebuild_table() -> pd.DataFrame:
    ph = pd.read_csv(C.DATA / "figure3_heldout_direction_ensemble.csv")
    kcols = [c for c in ph.columns if c.startswith("k") and c[1:].isdigit()]
    ph["k_best"] = ph[kcols].max(axis=1)
    # the leftover is a per-run quantity: take the median of the gaps, not
    # the gap between two medians (they differ by up to 0.03)
    ph["gap"] = ph.original - ph.k_best
    med = ph.groupby("dataset")[["k0", "k_best", "original", "gap"]].median()
    return med.reindex([d for d in C.PAPER_DATASETS if d in med.index])


def ensembles_panel(fig, slot) -> tuple:
    """Two sub-axes on one 'ensemble − reference AUC' scale: averaging
    the ten methods (left) and averaging the other datasets' classifiers
    (right)."""
    inner = slot.subgridspec(1, 2, wspace=0.10, width_ratios=[1.90, 1.05])
    axm = fig.add_subplot(inner[0, 0])
    axd = fig.add_subplot(inner[0, 1], sharey=axm)
    rng = np.random.default_rng(3)

    by_regime = _ensemble_by_regime()
    for i, regime in enumerate(REGIMES):
        g = by_regime[regime]
        for j, (col, color, lab) in enumerate(
                [("vs_best", fc.INK, "best single method"),
                 ("vs_median", fc.TEAL, "median single method")]):
            v = g[col]
            x = i + (j - 0.5) * 0.36
            axm.scatter(np.full(len(v), x) + rng.uniform(-0.06, 0.06, len(v)),
                        v, s=9, alpha=0.25, color=color, zorder=2)
            axm.hlines(v.median(), x - 0.13, x + 0.13, color=color, lw=3,
                       zorder=4, label=lab if i == 0 else None)
    axm.axhline(0, color=fc.MUTED, lw=0.9)
    axm.set_xticks(range(len(REGIMES)))
    axm.set_xticklabels([RSHORT[r] for r in REGIMES])
    axm.set_xlim(-0.55, len(REGIMES) - 0.45)
    axm.set_ylabel("Ensemble − target AUC")

    med = _rebuild_table()
    diffs = [med.k0 - med.original, med.k_best - med.original]
    for j, vals in enumerate(diffs):
        x = np.full(len(vals), j) + rng.uniform(-0.10, 0.10, len(vals))
        axd.scatter(x, vals, s=26, alpha=0.9, zorder=3,
                    color=fc.GRAY if j == 0 else fc.PURPLE,
                    edgecolors="white", linewidths=0.4)
        axd.hlines(vals.median(), j - 0.2, j + 0.2, color=fc.INK, lw=3,
                   zorder=4)
    # VDS and WJB-adversarial sit 0.015 apart, so their labels are nudged
    # apart vertically rather than printed on top of one another
    for name, label in SHORT.items():
        axd.annotate(label, (1, diffs[1][name]), ha="right", va="center",
                     xytext=(-8, ANNOT_DY.get(name, 0)),
                     textcoords="offset points", fontsize=13.5, color=fc.INK)
    axd.axhline(0, color=fc.MUTED, lw=0.9)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Average\nof others", "Best\nweighting"])
    axd.set_xlim(-0.45, 1.45)   # tight enough that the two 2-line labels clear
    plt.setp(axd.get_yticklabels(), visible=False)
    axm.set_ylim(-0.37, 0.15)
    return axm, axd


def similarity_figure() -> None:
    # two panels on a narrow, short canvas, panel a about square; the
    # gutter is an explicit column because b's y-axis label sits in it
    fig = plt.figure(figsize=(12.3, 5.0))
    # the gutter column carries b's y-axis label, so it is wide enough
    # for that label to clear panel a
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 0.32, 1.62],
                          wspace=0.0, left=0.075, right=0.985,
                          top=0.85, bottom=0.29)
    axa = fig.add_subplot(gs[0, 0])
    agreement_panel(axa)
    axm, axd = ensembles_panel(fig, gs[0, 2])

    below = dict(frameon=False, fontsize=LEGEND_PT, handletextpad=0.5,
                 labelspacing=0.25, borderpad=0.1, columnspacing=1.2,
                 bbox_transform=fig.transFigure)
    # legends in two columns -- two rows under a, one row under b -- set
    # against the figure's left and right edges, since centred on their
    # panels the two would overlap
    legend_y = 0.16
    axa.legend(loc="upper left", bbox_to_anchor=(0.02, legend_y), ncol=2,
               **below)
    axm.legend(loc="upper right", bbox_to_anchor=(0.985, legend_y), ncol=2,
               **below)

    # panel letters + left-aligned headers; which runs each panel pools
    # is stated in the caption rather than in a subtitle
    fig.canvas.draw()
    heads = [
        (axa, "Agreement of predictions"),
        (axm, "Averaging the 10 methods (left) and\n"
              "other datasets' classifiers (right)"),
    ]
    for letter, (ax, head) in zip("ab", heads):
        box = ax.get_position(original=True)
        head_y = 0.985
        fig.text(box.x0 - 0.016, head_y, f"{letter})", ha="right",
                 va="top", fontsize=fc.pt("title"), fontweight="bold",
                 color=fc.INK)
        fig.text(box.x0, head_y, head, ha="left",
                 va="top", fontsize=fc.pt("label"), color=fc.INK,
                 linespacing=1.0)

    fc.save(fig, "figure3_similarity_and_labels",
            fc.caption_text("figure3_similarity_and_labels"))


def label_type_figure() -> None:
    """Single-source transfer by source and target label type: median
    diff-in-means AUC over the ordered source -> target dataset pairs of
    each pair of label types and the 68% / 88% depths, one matrix per
    model. Blood ranges is excluded as a dead target."""
    pair = pd.read_csv(C.DATA / "supplementary_figure5_label_transfer.csv")
    mid = pair[pair.depth.isin(MID_DEPTHS) & (pair.method == "diff")
               & (pair.target_dataset != DEAD_TARGET)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.13, right=0.86, top=0.76, bottom=0.21,
                        wspace=0.30)
    norm = TwoSlopeNorm(vmin=0.30, vcenter=0.5, vmax=1.0)
    labels = [GROUP_LABEL[g].replace("/", "/\n") for g in GROUP_ORDER]
    for j, (ax, model) in enumerate(zip(axes, ("gemma", "qwen"))):
        g = (mid[mid.model == model]
             .groupby(["source_group", "target_group"]).auc.median()
             .unstack().reindex(index=list(GROUP_ORDER),
                                columns=list(GROUP_ORDER)).to_numpy())
        im = ax.imshow(g, cmap="RdBu_r", norm=norm)
        ax.set_title(C.MODEL_LABEL[model], loc="left",
                     fontsize=fc.pt("label"), pad=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels if j == 0 else [])
        ax.set_xlabel("Target label type")
        if j == 0:
            ax.set_ylabel("Source label type")
        ax.grid(False)
        for y in range(3):
            for x in range(3):
                ax.text(x, y, f"{g[y, x]:.2f}", ha="center", va="center",
                        fontsize=fc.pt("title"),
                        color="white" if g[y, x] >= 0.84 else fc.INK)
    fig.canvas.draw()
    p = axes[1].get_position()
    cax = fig.add_axes([p.x1 + 0.025, p.y0, 0.018, p.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.3, 0.5, 0.7, 0.9])
    cb.ax.tick_params(labelsize=SUBTITLE_PT)
    cb.set_label("AUC (diff-in-means)", fontsize=fc.pt("tick"), color=fc.INK)
    cb.outline.set_edgecolor(fc.MUTED)
    cax.axhline(0.5, color=fc.INK, lw=1.2)
    # no figure-wide title: the caption names the estimator, prompts and
    # depths
    fc.save(fig, "supplementary_figure5_label_types",
            fc.caption_text("supplementary_figure5_label_types"))


def main() -> None:
    fc.setup()
    similarity_figure()
    label_type_figure()


if __name__ == "__main__":
    main()
