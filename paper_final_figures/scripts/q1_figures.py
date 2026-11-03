"""Q1 figure: linear vs nonlinear probes across training regimes.

figure2_methods   two panels sharing the regime axis
                left     median AUC per method at each method's best layer
                right    pairwise AUC differences on the same cells:
                         nonlinear - linear (random slopes counted as a
                         linear method) and best probe - the model's own
                         paired-token readout. A random slopes - best other
                         linear line was cut 2026-09-07: held out, random
                         slopes is scored without its dataset or domain
                         adjustment, so that comparison is not a fair one.

The model's own readout does not depend on the training regime, so it is a
flat reference in the left panel; the right panel carries the paired version
of the same comparison (this is the old fig08, folded in). A third panel --
the same method ranking with the four depths pooled -- was cut 2026-09-03:
it led to the same conclusions as the best-layer panel. Both legends sit
under their panels, which keeps the panels clear at the paper's column
width.
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

F.setup()
RNG = np.random.default_rng(7)

ORDER = ["within_cv", "merged_included", "leave_dataset_out",
         "leave_domain_out", "single_source"]
SHORT = {"within_cv": "Within\ntarget", "merged_included": "Pooled,\ntarget in",
         "leave_dataset_out": "Target\nabsent", "leave_domain_out": "Domain\nabsent",
         "single_source": "One\nsource"}
X = np.arange(len(ORDER))

# the deep dive's existing method colours; random slopes takes fig_common's green.
# colour alone carries method identity now -- no linear/nonlinear dash encoding --
# so each also gets its own marker. LDA is magenta rather than the deep dive's
# teal, which read as random slopes' green next to it (2026-09-07).
PAL = {"diff": F.BLUE, "l2reg": "#7C3AED", "lda": "#C026D3",
       "random_slopes": F.GREEN,
       "mlp": F.ORANGE, "knn": "#C81E1E", "xgboost": "#8a5a00"}
MARK = {"diff": "o", "l2reg": "s", "lda": "h", "random_slopes": "D",
        "mlp": "P", "knn": "X", "xgboost": "*"}
NAMED = ["diff", "l2reg", "lda", "random_slopes", "mlp", "knn", "xgboost"]
OTHER = ["l1reg", "svm_linear", "qda", "svm_rbf"]
# panel a's legend, three columns under the panel: the three
# single-dataset linear methods, the three nonlinear ones, and random
# slopes with the two reference lines (the widest labels share a column
# so the legend stays within the panel's width)
LEGEND_COLUMNS = [["diff", "l2reg", "lda"], ["mlp", "knn", "xgboost"],
                  ["random_slopes"]]
RO = "#9D174D"      # the model's own answer -- a reference, not a method
RO_LINE = F.INK     # its reference line in panel a: black dashed, unlike any method
CELL = ["model", "perturbation", "regime", "dataset"]
LEGEND_PT = 13.5
LEGEND_Y = -0.155   # axes fraction: below the two-line x tick labels


def _method_panel(ax, frame, title, ro_med) -> None:
    """One method-ranking panel: dodged IQR bars + median markers per method.

    `frame` is one row per unit of aggregation; the median and the bars are
    both taken over those rows within (regime, method). The four methods not
    named individually become one visible grey range band.
    """
    stat = frame.groupby(["regime", "method"]).auc.agg(
        med="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
    for m in OTHER:
        s = stat.xs(m, level="method").reindex(ORDER)
        ax.plot(X, s.med, "-", color=F.GRAY, lw=1.0, marker="o", ms=3, zorder=2)
    dodge = dict(zip(NAMED, np.linspace(-0.26, 0.26, len(NAMED))))
    for m in NAMED:
        s = stat.xs(m, level="method").reindex(ORDER)
        xs = X + dodge[m]
        ax.vlines(xs, s.q1, s.q3, color=PAL[m], lw=2.4, alpha=0.30, zorder=3,
                  capstyle="butt")
        ax.plot(xs, s.med, "-", color=PAL[m], lw=2.2, marker=MARK[m], ms=6.5,
                zorder=5, label=C.METHOD_LABEL[m])
    ax.axhline(ro_med, color=RO_LINE, lw=2.0, ls=(0, (6, 3)), zorder=6)
    ax.axhline(0.5, color=F.MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xticks(X)
    ax.set_xticklabels([SHORT[r] for r in ORDER])
    ax.set_xlim(-0.52, len(ORDER) - 0.48)


def _delta_panel(ax, f) -> None:
    """Three paired differences on one AUC-difference axis, same cells throughout.

    Random slopes counts as a linear method here (it is a linear probe with
    dataset adjustments): "best linear" is the best of the five linear
    methods fit without dataset adjustments and random slopes, and "best
    probe" the better of that and the best nonlinear method. Random slopes
    exists only in the three pooled regimes, where it is fit.
    """
    g = f[f.family.isin(["linear", "nonlinear"]) | (f.method == "random_slopes")]
    g = g.assign(family=g.family.where(g.method != "random_slopes", "linear"))
    fam = (g.groupby(CELL + ["family"], as_index=False).auc.max()
           .pivot_table(index=CELL, columns="family", values="auc").reset_index())
    fam["nl_lin"] = fam.nonlinear - fam.linear
    cmp = pd.read_csv(C.DATA / "figure2_probe_vs_logits.csv")
    cmp = cmp[cmp.dataset.isin(C.PAPER_DATASETS)]
    # the readout AUC comes from the stored table; the probe side is
    # recomputed so that "best probe" uses the same linear family as above
    fam = fam.merge(cmp[CELL + ["logit_auc"]], on=CELL, how="left")
    fam["probe_logit"] = fam[["linear", "nonlinear"]].max(axis=1) - fam.logit_auc

    for i, regime in enumerate(ORDER):
        sel = fam[fam.regime == regime].copy()
        sel["x"] = i + RNG.uniform(-0.20, 0.20, len(sel))
        ax.scatter(sel.x, sel.nl_lin, s=20, c="#B9C0CC", zorder=2,
                   edgecolors="none")

    def band(vals, color, label, mk="o"):
        s = vals.groupby("regime").agg(q1=lambda v: v.quantile(.25), med="median",
                                       q3=lambda v: v.quantile(.75)).reindex(ORDER)
        ok = s.med.notna()
        ax.fill_between(X[ok], s.q1[ok], s.q3[ok], color=color, alpha=0.16,
                        lw=0, zorder=3)
        ax.plot(X[ok], s.med[ok], "-", color=color, lw=2.6, marker=mk, ms=7.5,
                zorder=6, label=label)

    band(fam.set_index("regime").nl_lin, F.INK, "Best nonlinear − best linear")
    band(fam.set_index("regime").probe_logit, RO,
         "Best probe − model's output logit", mk="s")

    ax.axhline(0, color=F.MUTED, lw=0.9, zorder=1)
    ax.set_xticks(X)
    ax.set_xticklabels([SHORT[r] for r in ORDER])
    ax.set_xlim(-0.4, len(ORDER) - 0.6)
    ax.set_ylim(-0.135, 0.165)
    ax.set_ylabel("AUC difference")
    series, labels = ax.get_legend_handles_labels()
    # under the panel; no legend title: that the lines are medians over
    # model x framing x dataset cells with interquartile bands is said in
    # the caption
    leg = ax.legend(series, labels, loc="upper center",
                    bbox_to_anchor=(0.5, LEGEND_Y), frameon=False,
                    fontsize=LEGEND_PT, labelspacing=0.35, borderpad=0.2)
    leg._legend_box.align = "left"


def methods() -> None:
    f = C.collapse_single_source(
        pd.read_csv(C.DATA / "figure2_method_results.csv"))
    named = f[f.method.isin(NAMED + OTHER)]
    # left: each method gets its best of the four matched depths within
    # every run x target cell, then the median across cells
    best = named.groupby(["model", "perturbation", "regime", "dataset", "method"],
                         as_index=False).auc.max()
    cmp = pd.read_csv(C.DATA / "figure2_probe_vs_logits.csv")
    cmp = cmp[cmp.dataset.isin(C.PAPER_DATASETS)]
    ro_med = cmp[cmp.regime == "within_cv"].logit_auc.median()

    fig = plt.figure(figsize=(12.8, 6.6))
    # a thin spacer column keeps b's y-axis clear of panel a; the bottom
    # margin holds one three-row legend under each panel
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.16, 1], wspace=0.09,
                          left=0.075, right=0.99, top=0.87, bottom=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 2])
    _method_panel(ax0, best, "At each method's best layer", ro_med)
    ax0.set_ylabel("AUC")
    columns = [[plt.Line2D([], [], color=PAL[m], lw=2.2, marker=MARK[m],
                           ms=6.5, label=C.METHOD_LABEL[m]) for m in members]
               for members in LEGEND_COLUMNS]
    columns[-1] += [
        plt.Line2D([], [], color=F.GRAY, lw=1.0, label="other methods"),
        plt.Line2D([], [], color=RO_LINE, lw=2.0, ls=(0, (6, 3)),
                   label=f"model's output logit ({ro_med:.2f})")]
    handles = [h for c in columns for h in c]   # column-major fill
    leg = ax0.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, LEGEND_Y), ncol=len(columns),
                     frameon=False, fontsize=LEGEND_PT, labelspacing=0.28,
                     handletextpad=0.6, columnspacing=1.0, borderpad=0.2)
    leg._legend_box.align = "left"
    _delta_panel(ax1, f)
    headings = ((ax0, "a)", "At each method's best layer"),
                (ax1, "b)", "Pairwise AUC differences"))
    fig.canvas.draw()
    for ax, tag, title in headings:
        box = ax.get_position()
        heading_y = box.y1 + 0.035 * box.height
        fig.text(box.x0 - 0.016, heading_y, tag, fontsize=F.pt("title"),
                 fontweight="bold", color=F.INK, va="bottom", ha="right")
        fig.text(box.x0, heading_y, title, fontsize=F.pt("title"),
                 color=F.INK, va="bottom", ha="left")

    F.save(fig, "figure2_methods", F.caption_text("figure2_methods"))


if __name__ == "__main__":
    methods()
