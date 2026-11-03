"""Q5 figure: what happens when the probe reads something other than the
raw last-token activation vector.

fig_supplementary_figure3_features   left   binned activations vs raw activations, per dataset
                  right  every alternative feature space against raw, in the
                         two regimes where all of them were run
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
SHORT = {"got": "GoT", "common_claim": "CommonClaim", "counterfact": "CounterFact",
         "pubmedqa_artificial": "PubMed-art", "pubmedqa_labeled": "PubMed-lab",
         "claude_anatomy": "Anatomy", "harmful_harmless": "Harmful/harmless",
         "wildjailbreak_vanilla": "WJB-van", "wildjailbreak_adversarial": "WJB-adv",
         "xstest": "XSTest", "social_iqa": "SocialIQa", "claude_vds": "VDS",
         "claude_blood_test_ranges": "Blood"}
FEATURE_COLOR = {"pca": F.TEAL, "pca_dict": "#5ea8a0", "sae": F.PURPLE,
                 "transcoder": "#a78bda", "binned": F.GRAY}
FEATURE_LABEL = {"pca": "PCA (95% var)", "pca_dict": "PCA + dictionary",
                 "sae": "SAE", "transcoder": "Transcoder",
                 "binned": "Binned activations"}
FEATURE_ORDER = ["pca", "pca_dict", "sae", "transcoder", "binned"]
REGIMES = ["within_cv", "single_source"]
RSHORT = {"within_cv": "Within target", "single_source": "One source"}


def _binned_panel(ax) -> None:
    """Per dataset: how much of the raw AUC the value histogram keeps.

    Each bar runs from chance to the raw AUC and is split at the binned AUC,
    so the pale segment is exactly what neuron identity is worth.
    """
    j = pd.read_csv(C.DATA / "supplementary_figure3_feature_sets.csv")
    w = j[(j.regime == "within_cv") & j.feature.isin(["raw", "binned"])]
    # best over methods and layers within a run, then median over the 4 runs
    best = w.groupby(["model", "perturbation", "dataset", "feature"],
                     as_index=False).auc.max()
    piv = best.pivot_table(index=["model", "perturbation", "dataset"],
                           columns="feature", values="auc").reset_index()
    piv["gap"] = piv.raw - piv.binned
    med = piv.groupby("dataset")[["binned", "raw", "gap"]].median()
    med = med.loc[[d for d in med.index if d in C.PAPER_DATASETS]]
    # closest first: smallest raw-minus-binned gap ends up at the top
    med = med.sort_values("gap", ascending=False)

    y = np.arange(len(med))
    ax.barh(y, med.binned - 0.5, left=0.5, height=0.64, color=F.TEAL,
            zorder=3, label="Binned activations")
    ax.barh(y, med.gap, left=med.binned, height=0.64, color=F.GRID,
            zorder=3, label="Additional AUC from raw activations")
    for yi, g in zip(y, med.gap):
        ax.text(1.012, yi, f"−{g:.3f}", va="center", fontsize=F.pt("tick"),
                color=F.MUTED)
    # one-line column header, with enough headroom that it clears the
    # top row's number
    ax.text(1.012, len(med) - 1 + 0.95, "raw − binned",
            fontsize=F.pt("tick"), color=F.INK, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[d] for d in med.index])
    ax.set_ylim(-0.7, len(med) - 1 + 1.45)
    # room past 1.0 for the number column and its header, so neither
    # runs into panel b's y label
    ax.set_xlim(0.5, 1.16)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUC per-dataset")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2,
              frameon=False, fontsize=F.pt("legend"), columnspacing=1.2,
              handletextpad=0.5)


def _delta_panel(ax) -> None:
    j = pd.read_csv(C.DATA / "supplementary_figure3_feature_deltas.csv")
    j = j[j.dataset.isin(C.PAPER_DATASETS) & j.regime.isin(REGIMES)]
    j = j[j.depth > 1]  # depth 1 penalizes every feature space equally
    width = 0.16
    for k, feat in enumerate(FEATURE_ORDER):
        sel = j[j.feature == feat]
        med = sel.groupby("regime").delta.median().reindex(REGIMES)
        q1 = sel.groupby("regime").delta.quantile(.25).reindex(REGIMES)
        q3 = sel.groupby("regime").delta.quantile(.75).reindex(REGIMES)
        x = np.arange(len(REGIMES)) + (k - 2) * width
        ax.bar(x, med, width=width * 0.9, color=FEATURE_COLOR[feat],
               label=FEATURE_LABEL[feat], zorder=3)
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none",
                    ecolor=F.INK, elinewidth=1, capsize=2, zorder=4)
    ax.axhline(0, color=F.INK, lw=1)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([RSHORT[r] for r in REGIMES])
    ax.set_xlim(-0.62, len(REGIMES) - 0.38)
    ax.set_ylabel("[Alt feature - raw activation] AUC")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=F.pt("legend"))


def features() -> None:
    # a narrower canvas than before (16.2 in) with the text at the deep
    # dive's standard sizes, so it reads at the paper's column width; the
    # right margin holds panel b's side legend
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0),
                             gridspec_kw={"width_ratios": [1.5, 1],
                                          "wspace": 0.42})
    fig.subplots_adjust(left=0.135, right=0.815, top=0.77, bottom=0.13)
    _binned_panel(axes[0])
    _delta_panel(axes[1])
    titles = ("Binned activation vs. raw activation\nprobe performance",
              "Difference in performance of alternative\n"
              "features vs. raw features")
    for ax, letter, title in zip(axes, "ab", titles):
        box = ax.get_position()
        fig.text(box.x0 - 0.016, 0.955, f"{letter})", ha="right", va="top",
                 fontsize=F.pt("title"), fontweight="bold", color=F.INK)
        fig.text(box.x0, 0.955, title, ha="left", va="top",
                 fontsize=F.pt("label"), color=F.INK, linespacing=1.0)
    F.save(fig, "supplementary_figure3_features", F.caption_text("supplementary_figure3_features"))


if __name__ == "__main__":
    features()
