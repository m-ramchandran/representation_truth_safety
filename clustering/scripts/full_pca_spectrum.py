"""Full PCA eigenspectrum of the merged last-token data, per layer.

The saved run artifacts (pca__pc=60.npz) keep at most 60 components, which
reach only ~50% cumulative variance — not enough to answer "how many PCs for
80/90/95/99%?". This computes all 2560 eigenvalues of the covariance of the
standardized merged data (same standardization as pca.fit), checks the top 60
against the saved run, and saves the full ratio vector next to the run's PCA
files as pca__full_spectrum.npz.

    python -m clustering.scripts.full_pca_spectrum
"""

import numpy as np

from . import load_data

DATASETS = ["geometry_truth", "harmful_harmless", "pubmedqa_artificial",
            "pubmedqa_labeled", "social_iqa", "wildjailbreak", "xstest"]
LAYERS = [9, 17, 22, 29]
THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

for L in LAYERS:
    X = load_data.load_merged(DATASETS, "last", L)["X"].astype(np.float64)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    Xs = (X - mean) / scale
    Xs -= Xs.mean(axis=0)

    C = (Xs.T @ Xs) / (len(Xs) - 1)
    ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    ratio = ev / ev.sum()

    saved = np.load(f"clustering/results/merged_last/runs/L{L}/pca__pc=60.npz")["explained_variance_ratio"]
    match = np.allclose(ratio[:60], saved, atol=1e-4)

    np.savez(f"clustering/results/merged_last/runs/L{L}/pca__full_spectrum.npz",
             explained_variance_ratio=ratio)

    cum = np.cumsum(ratio)
    counts = "  ".join(f"{t:.0%}: {int(np.searchsorted(cum, t)) + 1}" for t in THRESHOLDS)
    print(f"L{L}: N={len(X)}  top-60 matches saved run: {match}  {counts}", flush=True)
