"""Follow-up figure: what label-free geometry can and cannot do.

One figure, two panels (absorbs the former q7b_proximity_orthogonality
figure).
Left: how well three quantities rank-predict single-source target AUC —
the supervised covariance-metric probe cosine vs two label-free
activation-geometry measures — split by whether the source and target
share a domain.
Right: |direction cosine| between dataset pairs (L2 probes,
pooled-covariance metric) by activation-centroid-distance tercile and
domain relation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import fig_common as F

F.setup()


GROUPS = [("all", "All pairs", F.INK),
          ("same", "Same domain", F.BLUE),
          ("cross", "Different domain", F.GRAY)]

METRICS = [("probe_cos", 1, "Probe-direction cosine\n(needs target labels)"),
           ("subspace_affinity", 1, "Covariance top-20\nsubspace affinity"),
           ("centroid_dist", -1, "Centroid proximity\n(pooled-whitened)")]


def _predictor_rhos() -> pd.DataFrame:
    """Spearman(metric, transfer AUC) per run x depth, split by whether the
    source and target share a domain. One row per metric x split x group."""
    j = pd.read_csv(C.DATA / "supplementary_figure4_geometry_transfer.csv")
    j = j[j.method == "l2reg"]
    j = j[j.source_dataset.isin(C.PAPER_DATASETS)
          & j.dataset.isin(C.PAPER_DATASETS)]
    ps = pd.read_csv(C.DATA / "supplementary_figure4_direction_similarity.csv")
    ps = ps[(ps.method == "l2reg")
            & (ps.measure == "mahalanobis_reg_cov_test")
            & ps.dataset_a.isin(C.PAPER_DATASETS)
            & ps.dataset_b.isin(C.PAPER_DATASETS)]
    # the test-geometry metric is directional: a = source, b = target, and
    # the target supplies the covariance defining the metric
    j = j.merge(ps.rename(columns={"dataset_a": "source_dataset",
                                   "dataset_b": "dataset",
                                   "value": "probe_cos"})
                [["source_dataset", "dataset", "model", "perturbation",
                  "depth", "probe_cos"]],
                on=["source_dataset", "dataset", "model", "perturbation",
                    "depth"], how="inner")
    keys = ["model", "perturbation", "depth"]
    rows = []
    for feat, sign, label in METRICS:
        for key, _, _ in GROUPS:
            sel = j if key == "all" else j[j.same_domain == (key == "same")]
            for grp, g in sel.groupby(keys):
                rows.append(dict(zip(keys, grp), metric=label, split=key,
                                 n_pairs=len(g),
                                 rho=sign * spearmanr(g[feat], g.auc)[0]))
    return pd.DataFrame(rows)


def predictors_panel(ax) -> None:
    r = _predictor_rhos()
    labels = [m[2] for m in METRICS]
    width = 0.26
    for k, (key, glabel, color) in enumerate(GROUPS):
        s = r[r.split == key].groupby("metric").rho
        med = s.median().reindex(labels)
        q1, q3 = s.quantile(.25).reindex(labels), s.quantile(.75).reindex(labels)
        x = np.arange(len(labels)) + (k - 1) * width
        ax.bar(x, med, width=width * 0.9, color=color, zorder=3, label=glabel)
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none",
                    ecolor=F.INK, elinewidth=1.1, capsize=3, zorder=4)
        for xi, v, top in zip(x, med, q3):
            ax.text(xi, max(v, top) + 0.035, f"{v:.2f}", ha="center",
                    fontsize=13, color=F.INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Spearman ρ with target AUC")
    # both panels share a zero baseline so they read as one aligned pair
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Similarity measure between the two datasets",
                  labelpad=12, fontsize=15)
    ax.legend(loc="upper right", frameon=True, edgecolor=F.GRID,
              framealpha=0.95, fontsize=13.0, title="Source–target pairs",
              title_fontsize=13.5, borderaxespad=0.4)


def proximity_panel(ax) -> None:
    j = pd.read_csv(C.DATA / "supplementary_figure4_proximity_alignment.csv")
    j = j[j.dataset_a.isin(C.PAPER_DATASETS)
          & j.dataset_b.isin(C.PAPER_DATASETS)]
    bins = ["near", "mid", "far"]
    for k, (same, label, color) in enumerate(
            [(True, "Same domain", F.BLUE),
             (False, "Different domain", F.GRAY)]):
        sel = j[j.same_domain == same]
        med = sel.groupby("dist_bin", observed=True).abscos.median().reindex(bins)
        q1 = sel.groupby("dist_bin", observed=True).abscos.quantile(.25).reindex(bins)
        q3 = sel.groupby("dist_bin", observed=True).abscos.quantile(.75).reindex(bins)
        x = np.arange(3) + (k - 0.5) * 0.34
        ax.bar(x, med, width=0.3, color=color, zorder=3, label=label)
        ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="none",
                    ecolor=F.INK, elinewidth=1.1, capsize=3, zorder=4)
        # clear the whisker cap, not just the bar top
        for xi, v, top in zip(x, med, q3):
            ax.text(xi, max(v, top) + 0.022, f"{v:.2f}", ha="center",
                    fontsize=13)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Nearest third", "Middle third", "Farthest third"])
    ax.set_xlabel("Activation-centroid distance\nbetween the two datasets",
                  labelpad=14, fontsize=15)
    ax.set_ylabel("|cosine| between probe directions")
    ax.legend(loc="upper right", frameon=True, edgecolor=F.GRID,
              framealpha=0.95, fontsize=13.0, title="Dataset pairs",
              title_fontsize=13.5, borderaxespad=0.4)
    ax.set_ylim(0, 0.85)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6),
                             width_ratios=[1.18, 1.0])
    fig.subplots_adjust(wspace=0.26, bottom=0.18)
    predictors_panel(axes[0])
    proximity_panel(axes[1])
    titles = ("Performance prediction from unsupervised\n"
              "and supervised geometry",
              "Probe-direction alignment by\n"
              "activation-centroid distance")
    fig.canvas.draw()
    for ax, tag, title in zip(axes, ("a)", "b)"), titles):
        box = ax.get_position()
        heading_y = box.y0 + 1.16 * box.height
        fig.text(box.x0 - 0.016, heading_y, tag,
                 fontsize=F.pt("label"), fontweight="bold", color=F.INK,
                 ha="right", va="top")
        fig.text(box.x0, heading_y, title, fontsize=F.pt("label"),
                 color=F.INK, ha="left", va="top", linespacing=1.05)
    F.save(fig, "supplementary_figure4_geometry", F.caption_text("supplementary_figure4_geometry"))


if __name__ == "__main__":
    main()
