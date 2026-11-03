"""Goodness-of-fit metrics for a clustering.

Three kinds:
  - model selection (GMM/MFA only): AIC, BIC — read off the fitted model.
  - geometry (any method): silhouette, Calinski-Harabasz, Davies-Bouldin,
    computed in the space the clustering was done in. Noise points (HDBSCAN's
    -1) are excluded.
  - against labels/topics (evaluation only): ARI, NMI, and per-cluster
    composition + purity for the composition heatmap.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
)

NAN = float("nan")


def model_selection(result, X) -> dict:
    """AIC/BIC if the fitted model is probabilistic (GMM, MFA), else empty."""
    m = result.model
    if m is not None and hasattr(m, "aic") and hasattr(m, "bic"):
        return {"aic": float(m.aic(X)), "bic": float(m.bic(X))}
    return {}


def rms_silhouette(Z, labels) -> float:
    """Silhouette with RMS (root-mean-square) distances, exact in O(N·k·D).

    Same s(i) = (b − a) / max(a, b) as the textbook score, but a(i)/b(i) are
    RMS distances to the point's own / nearest other cluster instead of mean
    distances. Exact for any cluster shape via the decomposition
    mean_j ||x − x_j||² = ||x − mu_c||² + Var_c — no N×N distances, no
    subsampling. The own-cluster mean excludes the point itself; singleton
    clusters score 0. Since sqrt is monotone, each point's own-vs-neighbor
    verdict matches the squared-distance silhouette exactly.
    """
    Z = np.asarray(Z, dtype=np.float64)  # cancellation-prone expansion below
    uniq, own, n = np.unique(labels, return_inverse=True, return_counts=True)
    mu = np.stack([Z[labels == c].mean(0) for c in uniq])           # (k, D)
    var = np.array([((Z[labels == c] - mu[j]) ** 2).sum(1).mean()
                    for j, c in enumerate(uniq)])                   # (k,)
    # ||x_i - mu_c||^2 for all i, c; + Var_c = mean squared distance to c.
    d2 = (Z ** 2).sum(1)[:, None] - 2 * Z @ mu.T + (mu ** 2).sum(1)
    m2 = np.maximum(d2 + var, 0.0)                                  # (N, k)

    idx = np.arange(len(Z))
    n_own = n[own]
    a2 = m2[idx, own] * n_own / np.maximum(n_own - 1, 1)            # exclude self
    m2[idx, own] = np.inf
    b2 = m2.min(1)
    a, b = np.sqrt(a2), np.sqrt(b2)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-30)
    s[n_own == 1] = 0.0
    return float(s.mean())


def geometry_metrics(Z, labels) -> dict:
    """Internal validity indices on the clustered space; excludes noise.

    `silhouette` is the exact RMS-distance variant (see rms_silhouette) — the
    textbook mean-distance score needs all N² pairwise distances, which is
    prohibitive on the ~50k-point merged runs.
    """
    labels = np.asarray(labels)
    mask = labels >= 0
    Zc, lc = Z[mask], labels[mask]
    if len(np.unique(lc)) < 2:
        return {"silhouette": NAN, "calinski_harabasz": NAN, "davies_bouldin": NAN}
    return {
        "silhouette": rms_silhouette(Zc, lc),
        "calinski_harabasz": float(calinski_harabasz_score(Zc, lc)),
        "davies_bouldin": float(davies_bouldin_score(Zc, lc)),
    }


def info_metrics(labels, reference) -> dict:
    """ARI and NMI of a clustering against a reference partition (e.g. labels)."""
    return {
        "ari": float(adjusted_rand_score(reference, labels)),
        "nmi": float(normalized_mutual_info_score(reference, labels)),
    }


def composition(labels, category):
    """Cluster x category counts for the composition heatmap.

    Returns clusters (sorted, noise excluded), the category values, the counts
    matrix (n_clusters x n_categories), its row-normalized fractions, and each
    cluster's size.
    """
    labels, category = np.asarray(labels), np.asarray(category)
    clusters = np.unique(labels[labels >= 0])
    cats = np.unique(category)
    counts = np.zeros((len(clusters), len(cats)), dtype=int)
    for i, c in enumerate(clusters):
        vals, n = np.unique(category[labels == c], return_counts=True)
        counts[i, np.searchsorted(cats, vals)] = n
    sizes = counts.sum(axis=1)
    fractions = counts / sizes[:, None]
    return clusters, cats, counts, fractions, sizes


def purity(labels, category) -> dict:
    """Per-cluster dominant-category fraction and entropy (label/topic homogeneity)."""
    clusters, _, _, fractions, _ = composition(labels, category)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(fractions > 0, fractions * np.log(fractions), 0.0), axis=1)
    return {int(c): {"top_fraction": float(fractions[i].max()), "entropy": float(ent[i])}
            for i, c in enumerate(clusters)}


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from .methods import fit_gmm

    Z, y = make_blobs(n_samples=600, centers=3, random_state=0)
    res = fit_gmm(Z, 3, "full", 0)
    print("model_selection:", {k: round(v, 1) for k, v in model_selection(res, Z).items()})
    print("geometry:", {k: round(v, 3) for k, v in geometry_metrics(Z, res.labels).items()})
    print("vs truth:", {k: round(v, 3) for k, v in info_metrics(res.labels, y).items()})
    clusters, cats, counts, frac, sizes = composition(res.labels, y)
    print("sizes:", sizes.tolist(), " purity:", {k: round(v["top_fraction"], 2)
                                                  for k, v in purity(res.labels, y).items()})
