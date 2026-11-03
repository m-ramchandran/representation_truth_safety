"""Low-dimensional embeddings for *viewing* structure (not for clustering in).

  - t-SNE (openTSNE) and UMAP (umap-learn): local structure. Stochastic, so they
    feed the NESS stability layer (ness.py).
  - MDS (sklearn): global geometry. Its objective preserves pairwise distances,
    so it is the tool for "how far apart are the clusters".

`centroid_distance_map` is the headline global-geometry output: cluster centroids
MDS-embedded to 2D (few points, so exact and cheap) plus the raw centroid
distance matrix as numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import MDS


def embed_tsne(X, seed=0, perplexity=30, n_components=2, n_jobs=-1) -> np.ndarray:
    try:
        from openTSNE import TSNE
    except ImportError as e:
        raise ImportError("t-SNE needs `pip install openTSNE`.") from e
    emb = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed,
               n_jobs=n_jobs).fit(np.asarray(X))
    return np.asarray(emb)


def embed_umap(X, seed=0, n_neighbors=15, min_dist=0.1, n_components=2) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise ImportError("UMAP needs `pip install umap-learn`.") from e
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=n_components, random_state=seed,
    )
    return reducer.fit_transform(np.asarray(X))


def embed_mds(X, metric=True, n_components=2, seed=0, subsample=None):
    """Metric (or non-metric) MDS of points. Returns (coords, subsample_index).

    O(N^2); use `subsample` for more than a few thousand points.
    """
    X = np.asarray(X)
    idx = np.arange(len(X))
    if subsample is not None and subsample < len(X):
        idx = np.random.default_rng(seed).choice(len(X), subsample, replace=False)
    mds = MDS(n_components=n_components, metric_mds=metric, metric="euclidean",
              n_init=4, init="random", random_state=seed)
    return mds.fit_transform(X[idx]), idx


def centroid_distance_map(X, labels, metric=True, seed=0):
    """Faithful map of inter-cluster distances.

    Returns cluster ids, their centroids, the pairwise centroid distance matrix
    (the numbers), and a 2D MDS layout of the centroids (exact — few points).
    """
    X, labels = np.asarray(X), np.asarray(labels)
    clusters = np.unique(labels[labels >= 0])
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in clusters])
    dist = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    coords = MDS(n_components=2, metric_mds=metric, metric="precomputed",
                 n_init=4, init="random", random_state=seed).fit_transform(dist)
    return clusters, centroids, dist, coords


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=0)
    clusters, centroids, dist, coords = centroid_distance_map(X, y)
    print(f"clusters: {clusters.tolist()}")
    print("centroid distance matrix:\n", dist.round(2))
    print(f"MDS centroid layout shape: {coords.shape}")
    mds_pts, idx = embed_mds(X, subsample=200)
    print(f"subsampled point MDS: {mds_pts.shape} from {len(idx)} points")
