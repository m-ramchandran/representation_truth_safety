"""Render filled PCA mean-plane slices from the completed full-data refits.

    python -m classifiers.analysis.studies.boundary_slices
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .boundary_common import ARTIFACTS, FIGURES, SCOPES

METHODS = [
    ("diff", "diff in means"),
    ("l1reg", "L1 logistic"),
    ("l2reg", "L2 logistic"),
    ("svm_linear", "linear SVM"),
    ("lda", "LDA"),
    ("qda", "QDA"),
    ("knn", "kNN"),
    ("svm_rbf", "RBF SVM"),
    ("mlp", "MLP"),
    ("xgboost", "XGBoost"),
]


def probability_surface(score, threshold):
    if threshold == .5:  # kNN already stores positive-neighbour fraction
        return score
    return 1 / (1 + np.exp(-np.clip(score, -8, 8)))


def draw_panel(ax, d, method, display, show_labels=True):
    X2, y, xx, yy = d["X2"], d["y"], d["xx"], d["yy"]
    score = d[f"score_{method}"]
    threshold = float(d[f"threshold_{method}"])
    surface = probability_surface(score, threshold).reshape(xx.shape)
    ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 17),
                cmap="RdBu_r", vmin=0, vmax=1, alpha=.66)
    if score.min() <= threshold <= score.max():
        ax.contour(xx, yy, score.reshape(xx.shape), levels=[threshold],
                   colors="#172033", linewidths=1.15)
    size = .28 if len(y) > 100_000 else (.65 if len(y) > 20_000 else 3)
    alpha = .08 if len(y) > 100_000 else (.12 if len(y) > 20_000 else .25)
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm",
               vmin=0, vmax=1, s=size, alpha=alpha, edgecolor="none",
               rasterized=True)
    ax.set_title(display, fontsize=10.5, fontweight="bold")
    if show_labels:
        evr = d["pca_variance_ratio"]
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    FIGURES.mkdir(exist_ok=True)
    loaded = {
        scope: np.load(ARTIFACTS / f"{scope}.npz", allow_pickle=True)
        for scope in SCOPES
    }

    # One readable 2x5 figure per training scope.
    for scope, d in loaded.items():
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.7),
                                 layout="constrained")
        for ax, (method, display) in zip(axes.ravel(), METHODS):
            draw_panel(ax, d, method, display)
        fig.suptitle(
            f"{scope.replace('_', ' ')} — full-data classifier slices "
            f"through the all-data PCA mean plane · all {len(d['y']):,} points",
            fontsize=16, color="#172033", fontweight="bold")
        stem = f"decision-boundary-slices-all-data-{scope}"
        fig.savefig(FIGURES / f"{stem}.png", dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    # Compact overview: scopes are rows, methods are columns.
    fig, axes = plt.subplots(4, 10, figsize=(29, 12.4),
                             layout="constrained")
    for row, scope in enumerate(SCOPES):
        d = loaded[scope]
        for col, (method, display) in enumerate(METHODS):
            draw_panel(axes[row, col], d, method, display,
                       show_labels=(row == len(SCOPES) - 1))
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{scope.replace('_', ' ')}\nPC2", fontsize=9,
                    fontweight="bold")
    fig.suptitle(
        "All ten full-data classifiers — filled PCA mean-plane contour slices",
        fontsize=18, color="#172033", fontweight="bold")
    stem = "decision-boundary-slices-all-data-overview"
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    for d in loaded.values():
        d.close()


if __name__ == "__main__":
    main()
