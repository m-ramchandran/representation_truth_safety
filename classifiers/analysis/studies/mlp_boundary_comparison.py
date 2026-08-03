"""Plot full-D MLP mean-plane slices beside MLPs trained on PC1 and PC2."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .boundary_common import (ARTIFACTS, FIGURES, SCOPES, SCOPE_LABELS,
                             TABLES, draw_surface, point_style,
                             production_auc)

def main() -> None:
    auc_full = production_auc(("mlp",))
    rows = []
    fig, axes = plt.subplots(4, 2, figsize=(9.4, 12.2),
                             constrained_layout=False)
    for row_index, scope in enumerate(SCOPES):
        with np.load(ARTIFACTS / f"{scope}.npz", allow_pickle=True) as full, \
                np.load(ARTIFACTS / f"{scope}_two_pc_mlp.npz",
                        allow_pickle=True) as two_pc:
            X2, y = np.asarray(full["X2"]), np.asarray(full["y"])
            xx, yy = np.asarray(full["xx"]), np.asarray(full["yy"])
            evr = np.asarray(full["pca_variance_ratio"])
            panels = (
                (np.asarray(full["score_mlp"]),
                 float(full["threshold_mlp"]),
                 "MLP, all dimensions (mean-plane slice)",
                 auc_full[scope, "mlp"],
                 "full"),
                (np.asarray(two_pc["score"]), 0.,
                 "MLP, PC1 and PC2 only", float(two_pc["cv_auc"]),
                 "two_pc"),
            )
            size, alpha = point_style(len(y))
            for col, (score, threshold, name, auc, fit_space) in enumerate(panels):
                ax = axes[row_index, col]
                draw_surface(ax, xx, yy, score, threshold)
                for value, color in ((0, "#2563EB"), (1, "#EF4444")):
                    take = y == value
                    ax.scatter(X2[take, 0], X2[take, 1], s=size, c=color,
                               alpha=alpha, edgecolors="none",
                               rasterized=True, zorder=3)
                auc_label = ("all-dimensional CV AUC" if fit_space == "full"
                             else "two-PC CV AUC")
                ax.set_title(f"{SCOPE_LABELS[scope]} — {name}\n"
                             f"{auc_label} {auc:.3f}",
                             fontsize=9.0, fontweight="semibold", pad=5)
                ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}%)", fontsize=7.5)
                ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}%)", fontsize=7.5)
                ax.tick_params(labelsize=6.5, length=2)
                ax.spines[["top", "right"]].set_visible(False)
                rows.append({
                    "scope": scope,
                    "fit_space": fit_space,
                    "layer": 17,
                    "n": len(y),
                    "pca_variance_2d": float(evr.sum()),
                    "cv_auc": auc,
                    "positive_share_on_plane": float(
                        np.mean(score >= threshold)),
                })

    handles = [
        Line2D([], [], marker="o", linestyle="", color="#2563EB",
               markersize=4, label="label 0"),
        Line2D([], [], marker="o", linestyle="", color="#EF4444",
               markersize=4, label="label 1"),
        Line2D([], [], color="#111827", lw=1.3,
               label="decision boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .025), fontsize=8)
    fig.suptitle("MLP boundaries: full-dimensional slice vs two-PC fit",
                 fontsize=15, color="#172033", fontweight="bold", y=.992)
    fig.text(
        .5, .008,
        "All available points are shown. Left: the full-D MLP evaluated on "
        "the PCA mean-plane. Right: an MLP trained using only PC1 and PC2.",
        ha="center", va="bottom", fontsize=7.5, color="#475467")
    fig.subplots_adjust(left=.085, right=.99, top=.935, bottom=.07,
                        wspace=.17, hspace=.45)

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "mlp-boundaries-fullspace-vs-two-pc.png"
    fig.savefig(target, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        TABLES / "mlp_boundaries_fullspace_vs_two_pc.csv", index=False)
    print(target)


if __name__ == "__main__":
    main()
