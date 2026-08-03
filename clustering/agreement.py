"""Agreement between clusterings — the sensitivity read-out.

Given several label vectors over the *same* points (different seeds, PCA
bootstraps, #PCs, or fit-sets), measure how much the partitions agree. This is
how the seed / bootstrap / PC-count arms are summarized and how the method
comparison matrix is built.

Works on label arrays so it reads saved runs uniformly; `load_labels` pulls the
labels out of a saved run directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def ari(a, b) -> float:
    return float(adjusted_rand_score(a, b))


def nmi(a, b) -> float:
    return float(normalized_mutual_info_score(a, b))


def pairwise_ari(label_sets) -> np.ndarray:
    """Symmetric (F, F) ARI matrix over a list of label vectors."""
    F = len(label_sets)
    M = np.ones((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            M[i, j] = M[j, i] = ari(label_sets[i], label_sets[j])
    return M


def mean_pairwise_ari(label_sets) -> float:
    """Mean off-diagonal ARI — one stability number for a set of runs."""
    if len(label_sets) < 2:
        return float("nan")
    M = pairwise_ari(label_sets)
    iu = np.triu_indices(len(label_sets), k=1)
    return float(M[iu].mean())


def load_labels(run_dir) -> np.ndarray:
    return np.load(Path(run_dir) / "assignments.npz")["labels"]


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from .methods import fit_kmeans

    Z, _ = make_blobs(n_samples=600, centers=4, random_state=0)
    seeds = [fit_kmeans(Z, 4, s).labels for s in range(5)]
    print(f"mean pairwise ARI across 5 seeds: {mean_pairwise_ari(seeds):.3f}")
    print("pairwise ARI matrix:\n", pairwise_ari(seeds).round(3))
