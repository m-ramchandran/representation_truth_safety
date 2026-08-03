"""Recreate the original 4x3 PCA-boundary figure with every data point."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .boundary_common import (ARTIFACTS, FIGURES, SCOPES, SCOPE_LABELS,
                             draw_surface, point_style, production_auc)

def main():
    auc = production_auc(("l2reg", "svm_rbf"))
    fig, axes = plt.subplots(4, 3, figsize=(13.3, 12.2),
                             constrained_layout=False)
    for row, scope in enumerate(SCOPES):
        with np.load(ARTIFACTS / f"{scope}.npz", allow_pickle=True) as d:
            X2, y = np.asarray(d["X2"]), np.asarray(d["y"])
            xx, yy = np.asarray(d["xx"]), np.asarray(d["yy"])
            evr = np.asarray(d["pca_variance_ratio"])
            surfaces = [
                (d["score_l2reg"], float(d["threshold_l2reg"]),
                 f"{SCOPE_LABELS[scope]} — l2, all dimensions\n"
                 f"all-dimensional CV AUC {auc[(scope, 'l2reg')]:.3f}"),
                (d["score_svm_rbf"], float(d["threshold_svm_rbf"]),
                 f"{SCOPE_LABELS[scope]} — svm_rbf, all dimensions\n"
                 f"all-dimensional CV AUC {auc[(scope, 'svm_rbf')]:.3f}"),
            ]
            with np.load(ARTIFACTS / f"{scope}_two_pc_rbf.npz",
                         allow_pickle=True) as pc:
                surfaces.append(
                    (pc["score"], 0.0,
                     f"{SCOPE_LABELS[scope]} — svm_rbf, PC1 and PC2 only\n"
                     f"two-PC CV AUC {float(pc['cv_auc']):.3f}")
                )

            size, alpha = point_style(len(y))
            for col, (score, threshold, title) in enumerate(surfaces):
                ax = axes[row, col]
                draw_surface(ax, xx, yy, score, threshold)
                # Draw both classes separately, but never subsample: every row
                # in the compact all-data artifact is sent to the renderer.
                for value, color in ((0, "#2563EB"), (1, "#EF4444")):
                    take = y == value
                    ax.scatter(X2[take, 0], X2[take, 1], s=size, c=color,
                               alpha=alpha, edgecolors="none",
                               rasterized=True, zorder=3)
                ax.set_title(title, fontsize=9.0, fontweight="semibold", pad=5)
                ax.set_xlabel(f"PC1 ({100*evr[0]:.1f}%)", fontsize=7.5)
                ax.set_ylabel(f"PC2 ({100*evr[1]:.1f}%)", fontsize=7.5)
                ax.tick_params(labelsize=6.5, length=2)
                ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], marker="o", linestyle="", color="#2563EB",
               markersize=4, label="label 0"),
        Line2D([], [], marker="o", linestyle="", color="#EF4444",
               markersize=4, label="label 1"),
        Line2D([], [], color="#111827", lw=1.3, label="decision boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .025), fontsize=8)
    fig.text(.5, .008,
             "All available points are shown. First two columns: full-space "
             "classifier evaluated on the PCA mean-plane; third column: "
             "classifier fit only to PC1 and PC2.",
             ha="center", va="bottom", fontsize=7.5, color="#475467")
    fig.subplots_adjust(left=.065, right=.99, top=.975, bottom=.07,
                        wspace=.14, hspace=.45)
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "decision-boundaries-original-layout-all-points.png"
    fig.savefig(target, dpi=260, bbox_inches="tight", facecolor="white")
    print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
