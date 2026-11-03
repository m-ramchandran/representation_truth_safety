"""Q17 figure: the two permanently held-out targets, all ways of reaching them.

WJB-adversarial and VDS are excluded from every pooled fit, so they are the
only datasets where "transfer" is the whole story. One panel per target; one
row per way of scoring it, ordered by how much relevant training data the
scorer saw. The two panels share the rows (and the y axis); a row a target
has no scorer for stays blank. Classifier rows come from
shared_classifier_results.csv and all use the SAME classifier -- L2-penalized
logistic regression -- in every training setting (median over the four
matched depths). Fixing the classifier in advance matters here: picking the best of
the ten methods per cell, as the q11 ladder does, needs the held-out target's
own labels to make the pick, so the resulting AUC is not something a user
without those labels could have obtained. A single linear method throughout
also keeps the three classifier rows comparable with each other and with the
random-slopes rows below them, which are linear as well. Random-slopes rows
come from figure5_heldout_targets.csv (median over the ten leave-one-out
pooled fits and the four depths).

Cost of fixing it: diff-in-means transfers better than L2 on both held-out
rows (target-absent 0.870 vs 0.800 on VDS and 0.778 vs 0.759 on
WJB-adversarial, averaged over the four collections; domain-absent 0.702 vs
0.590 on WJB-adversarial), while L2 is the stronger within-target ceiling
(0.932 vs 0.902 and 0.980 vs 0.826). The transfer rows here are therefore the
conservative reading, not the most favourable one.

WJB-adversarial caveat: every pool that contains WJB-vanilla shares source
prompts with the target, so the jailbreak-domain-out classifier row is the
clean transfer number; the other pooled rows (and all RS rows) include it.
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

METHOD = "l2reg"     # one linear classifier everywhere; see the module docstring

WJB, VDS = "wildjailbreak_adversarial", "claude_vds"
TARGETS = (WJB, VDS)
TARGET_TITLE = {WJB: "WJB adversarial", VDS: "VDS"}

# rows shared by both panels, top to bottom: (label, {target: (source, key)}).
# row labels use the training-regime vocabulary of the other figures --
# Within target / Target absent / Domain absent -- and name the random-slopes
# parts (shared direction, deviation) as the pooled-training figure does.
# VDS is alone in its domain, so its target-absent classifier is also its
# domain-absent one and its domain-absent row is left blank. Its borrowed
# deviation is the similarity-weighted average of the pool's deviations;
# WJB-adversarial borrows the jailbreak-domain deviation.
ROWS = [
    ("Within target\n(L2 probe)",
     {WJB: ("clf", "within_cv"), VDS: ("clf", "within_cv")}),
    ("Target absent\n(L2 probe)",
     {WJB: ("clf", "leave_dataset_out"), VDS: ("clf", "leave_dataset_out")}),
    ("Domain absent\n(L2 probe)",
     {WJB: ("clf", "leave_domain_out")}),
    ("Random slopes:\nshared + domain or\nsimilar deviation",
     {WJB: ("rs", "task_mean"), VDS: ("rs", "sim_soft")}),
    ("Random slopes:\nshared direction only",
     {WJB: ("rs", "shared"), VDS: ("rs", "shared")}),
]


def classifier_values(long: pd.DataFrame) -> pd.Series:
    """One classifier's AUC per (target, training setting, run): the
    median over the four matched depths, so no single lucky layer carries
    a cell. The method is fixed to METHOD everywhere rather than chosen
    per cell -- choosing it would take the held-out target's labels."""
    sub = long[long.test_dataset.isin(TARGETS)
               & (long.method == METHOD)
               & long.regime.isin(["within_cv", "leave_dataset_out",
                                   "leave_domain_out"])]
    return (sub.groupby(["test_dataset", "regime", "model",
                         "perturbation"]).auc.median())


def main() -> None:
    F.setup()
    long = pd.read_csv(C.DATA / "shared_classifier_results.csv")
    clf = classifier_values(long)
    rs = pd.read_csv(C.DATA / "figure5_heldout_targets.csv").set_index(
        ["dataset", "estimator", "model", "perturbation"]).auc_median

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), sharey=True)
    ys = np.arange(len(ROWS))[::-1]
    drawn = []
    for ax, target in zip(axes, TARGETS):
        for (label, scorers), y in zip(ROWS, ys):
            if target not in scorers:
                continue
            kind, key = scorers[target]
            for model, pert in C.COLLECTIONS:
                if kind == "clf":
                    value = clf.loc[(target, key, model, pert)]
                else:
                    value = rs.loc[(target, key, model, pert)]
                drawn.append(f"{TARGET_TITLE[target]:<16} {key:<18} "
                             f"{model}-{pert:<11} {value:.3f}")
                ax.scatter(value, y, s=95, zorder=3,
                           marker=F.COLLECTION_MARKER[(model, pert)],
                           facecolor=F.MODEL_COLOR[model]
                           if pert == "true_false" else "white",
                           edgecolor=F.MODEL_COLOR[model], linewidth=1.8)
        ax.axvline(0.5, color=F.GRAY, lw=1, ls="--")
        ax.set_yticks(ys)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_xlim(0.44, 1.01)
        ax.set_xlabel("AUC on the held-out target")
        ax.grid(axis="x", color=F.GRID, linewidth=0.7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticklabels([label for label, _ in ROWS],
                            fontsize=F.pt("tick"))
    handles = [plt.Line2D([], [], ls="", ms=9.5,
                          marker=F.COLLECTION_MARKER[(m, p)],
                          markerfacecolor=F.MODEL_COLOR[m]
                          if p == "true_false" else "white",
                          markeredgecolor=F.MODEL_COLOR[m],
                          markeredgewidth=1.8,
                          label=F.COLLECTION_LABEL[(m, p)])
               for m, p in C.COLLECTIONS]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=4, frameon=False,
               fontsize=F.pt("legend"), columnspacing=1.2,
               handletextpad=0.55, borderpad=0.1)
    # the shared row labels take the left margin once; the gutter only has
    # to separate the two panels' tick marks
    fig.subplots_adjust(left=0.215, right=0.985, top=0.84, bottom=0.22,
                        wspace=0.14)

    # Match the figure-wide panel-heading convention while retaining the
    # existing axes and inter-panel spacing.
    fig.canvas.draw()
    for letter, ax, target in zip("ab", axes, TARGETS):
        box = ax.get_position()
        heading_y = box.y1 + 0.012
        fig.text(box.x0 - 0.016, heading_y, f"{letter})", ha="right",
                 va="bottom", fontsize=F.pt("title"), fontweight="bold",
                 color=F.INK)
        fig.text(box.x0, heading_y, TARGET_TITLE[target], ha="left",
                 va="bottom",
                 fontsize=F.pt("title"), color=F.INK)

    print(f"\nclassifier rows all use {METHOD}; values drawn:")
    for line in drawn:
        print(" ", line)

    F.save(fig, "figure5_heldout_targets", F.caption_text("figure5_heldout_targets"))


if __name__ == "__main__":
    main()
