"""NESS stability for the stochastic embeddings (t-SNE / UMAP).

t-SNE and UMAP rearrange under different random seeds. Following Ma et al. 2025
(arXiv:2506.22228), we treat the seed as a perturbation: run an ensemble of
embeddings, then measure how consistent each point's local neighborhood is
across the ensemble.

  - local stability : per point, the paper's score (Section 4.3). Build the
    neighbor co-occurrence count matrix K, where K[j, l] counts how many of the
    N embeddings place l among j's k nearest neighbors; symmetrize by
    K <- max(K, K^T); then S_j is the lambda-th quantile (lambda=0.75) of the
    normalized counts K[j, l] / N over the neighbors l that ever co-occur.
    High S_j means j keeps the same neighbors across seeds. In [0, 1].
  - global stability : mean local stability — one quality number per setting.
  - stability curve  : global stability vs the graph-connectivity parameter
    (perplexity / n_neighbors); pick the smallest value that plateaus.
  - unstable points  : the lowest-stability fraction, to grey out on plots.

This measures reproducibility of the picture, not its faithfulness for
clustering — clustering stays on PCA/raw.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _knn_indices(coords, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    return nn.kneighbors(return_distance=False)[:, 1:]  # drop self


def local_stability(embeddings, k=15, lam=0.75) -> np.ndarray:
    """Per-point NESS local stability score across an ensemble of embeddings."""
    N = len(embeddings[0])
    K = sparse.csr_matrix((N, N), dtype=np.float64)
    for e in embeddings:
        knn = _knn_indices(e, k)
        rows = np.repeat(np.arange(N), k)
        K += sparse.csr_matrix((np.ones(N * k), (rows, knn.ravel())), shape=(N, N))
    K = K.maximum(K.T).tocsr()          # symmetric neighbor co-occurrence counts

    counts = K.data / len(embeddings)   # normalized counts, only where K > 0
    return np.array([np.quantile(counts[K.indptr[j]:K.indptr[j + 1]], lam)
                     for j in range(N)])


def global_stability(embeddings, k=15, lam=0.75) -> float:
    return float(local_stability(embeddings, k, lam).mean())


def stability_curve(X, embed_fn, param_name, param_grid, seeds=range(5), k=15) -> dict:
    """Global stability at each connectivity value. `embed_fn(X, seed, **{param_name: v})`."""
    return {
        v: global_stability([embed_fn(X, seed=s, **{param_name: v}) for s in seeds], k)
        for v in param_grid
    }


def unstable_points(local_stab, frac=0.1) -> np.ndarray:
    """Indices of the least-stable `frac` of points (to flag on plots)."""
    return np.where(local_stab <= np.quantile(local_stab, frac))[0]


if __name__ == "__main__":
    # Two ensembles: one of near-identical embeddings (stable), one of random
    # layouts (unstable). Global stability should separate them clearly.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((300, 2))
    stable = [base + 0.01 * rng.standard_normal((300, 2)) for _ in range(5)]
    noisy = [rng.standard_normal((300, 2)) for _ in range(5)]
    print(f"stable ensemble  global stability = {global_stability(stable):.3f}")
    print(f"random ensemble  global stability = {global_stability(noisy):.3f}")
    ls = local_stability(stable)
    print(f"flagged unstable (bottom 10%): {len(unstable_points(ls))} of {len(ls)} points")
