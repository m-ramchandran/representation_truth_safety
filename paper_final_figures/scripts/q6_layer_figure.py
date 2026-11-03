"""Depth figure: how much does probe depth matter, and does it
replicate across models and framings?"""
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


def main() -> None:
    f = C.collapse_single_source(
        pd.read_csv(C.DATA / "shared_classifier_results.csv"))
    f = f[f.family.isin(["linear", "nonlinear"])]
    best = (f.groupby(["model", "perturbation", "regime", "dataset", "depth"])
            .auc.max().reset_index())
    panels = [("within_cv", "Within target"),
              ("leave_domain_out", "Domain absent"),
              ("single_source", "One source")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.35), sharey=True)
    for ax, (regime, _) in zip(axes, panels):
        for (m, p), g in best[best.regime == regime].groupby(
                ["model", "perturbation"]):
            med = g.groupby("depth").auc.median()
            ax.plot(med.index, med.values, lw=2.4,
                    color=F.MODEL_COLOR[m],
                    ls="-" if p == "bare" else "--",
                    marker=F.COLLECTION_MARKER[(m, p)], ms=7,
                    label=F.COLLECTION_LABEL[(m, p)])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([C.DEPTH_LABEL[d] for d in [1, 2, 3, 4]])
        ax.set_xlabel("Layer depth")
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("Median best AUC across targets")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False,
               fontsize=F.pt("legend"), columnspacing=1.4,
               handletextpad=0.55, borderpad=0.1)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.815, bottom=0.215,
                        wspace=0.20)
    fig.canvas.draw()
    for letter, ax, (_, title) in zip("abc", axes, panels):
        box = ax.get_position()
        heading_y = box.y1 + 0.018
        fig.text(box.x0 - 0.016, heading_y, f"{letter})", ha="right",
                 va="bottom", fontsize=F.pt("title"), fontweight="bold",
                 color=F.INK)
        fig.text(box.x0, heading_y, title, ha="left", va="bottom",
                 fontsize=F.pt("label"), color=F.INK)
    F.save(fig, "supplementary_figure2_depth", F.caption_text("supplementary_figure2_depth"))


if __name__ == "__main__":
    main()
