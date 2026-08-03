"""Clustering methods, each returning a uniform `ClusterResult`.

GMM, k-means, HDBSCAN come from scikit-learn; Leiden uses leidenalg + igraph on
a kNN graph. MFA lives in `mfa.py` and returns the same `ClusterResult`.

A result carries the cluster labels *and* the fitted model, so held-out scoring
(GMM/MFA log-likelihood) and re-plotting read the saved fit instead of redoing
it. Probabilistic methods (GMM, MFA) expose `log_likelihood(X)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterResult:
    method: str
    params: dict                       # config that produced this fit
    labels: np.ndarray                 # (N,), -1 = noise (HDBSCAN only)
    model: object = None               # fitted estimator, kept for scoring/reuse
    scores: dict = field(default_factory=dict)  # optional per-point arrays

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Per-point log-density; only for probabilistic models (GMM, MFA)."""
        if not hasattr(self.model, "score_samples"):
            raise TypeError(f"{self.method} has no likelihood")
        return self.model.score_samples(X)

    def save(self, out_dir) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "assignments.npz", labels=self.labels, **self.scores)
        (out_dir / "config.json").write_text(
            json.dumps({"method": self.method, **self.params}, indent=2)
        )
        if self.model is not None:
            joblib.dump(self.model, out_dir / "model.joblib")


def fit_gmm(Z, k, covariance_type, seed) -> ClusterResult:
    """GMM on PC scores. covariance_type is 'diag' or 'full'."""
    gm = GaussianMixture(
        n_components=k, covariance_type=covariance_type, random_state=seed
    ).fit(Z)
    return ClusterResult(
        "gmm",
        {"k": k, "covariance_type": covariance_type, "seed": seed},
        gm.predict(Z),
        model=gm,
        scores={
            "proba": gm.predict_proba(Z).astype(np.float32),
            "log_likelihood": gm.score_samples(Z),
        },
    )


def fit_kmeans(Z, k, seed) -> ClusterResult:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    return ClusterResult("kmeans", {"k": k, "seed": seed}, km.labels_, model=km)


def fit_hdbscan(Z, min_cluster_size, min_samples=None) -> ClusterResult:
    """Density-based; picks its own cluster count, labels sparse points -1."""
    hd = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True
    ).fit(Z)
    return ClusterResult(
        "hdbscan",
        {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        hd.labels_,
        model=hd,
        scores={"proba": hd.probabilities_},  # membership strength, 0 for noise
    )


def leiden_graph(Z, n_neighbors):
    """Symmetric kNN graph of the points as an igraph Graph. Building it is the
    expensive part of a Leiden fit and depends only on (data, n_neighbors), so
    callers can build once and share it across resolutions and seeds."""
    try:
        import igraph as ig
    except ImportError as e:
        raise ImportError(
            "Leiden needs `pip install leidenalg igraph`."
        ) from e
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(Z, n_neighbors, mode="connectivity", include_self=False)
    A = A.maximum(A.T).tocoo()  # symmetrize
    edges = np.vstack([A.row, A.col]).T[A.row < A.col]
    return ig.Graph(n=len(Z), edges=edges.tolist(), directed=False)


def fit_leiden(Z, resolution=1.0, n_neighbors=15, seed=0, graph=None) -> ClusterResult:
    """Leiden community detection on a symmetric kNN graph of the points.
    Pass `graph` (from leiden_graph) to reuse one graph across fits."""
    try:
        import leidenalg
    except ImportError as e:
        raise ImportError(
            "Leiden needs `pip install leidenalg igraph`."
        ) from e
    if graph is None:
        graph = leiden_graph(Z, n_neighbors)
    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return ClusterResult(
        "leiden",
        {"resolution": resolution, "n_neighbors": n_neighbors, "seed": seed},
        np.asarray(part.membership),
    )


if __name__ == "__main__":
    # Three well-separated blobs; each method should recover them.
    from sklearn.datasets import make_blobs

    Z, y = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=0)
    for name, res in [
        ("gmm-full", fit_gmm(Z, 3, "full", 0)),
        ("gmm-diag", fit_gmm(Z, 3, "diag", 0)),
        ("kmeans", fit_kmeans(Z, 3, 0)),
        ("hdbscan", fit_hdbscan(Z, 20)),
    ]:
        print(f"{name:10s} n_clusters={res.n_clusters}")
    try:
        print(f"leiden     n_clusters={fit_leiden(Z).n_clusters}")
    except ImportError as e:
        print(f"leiden     skipped ({e})")
