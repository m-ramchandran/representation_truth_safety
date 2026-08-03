"""The standardize+PCA projection shared by the clustering runs.

A `PCAComponents` holds everything needed to map raw activations to PC scores:
per-dimension standardization (fit on the same data as the PCA) followed by
PCA loadings (no whitening). It saves to / loads from a plain `.npz` of arrays
so a fit is never repeated — plots, held-out scoring, and reprojection all read
the saved parameters.

Three ways to choose what the PCA is fit on (all transform the *full* data
afterwards):
    fit(X)                     — fit on X itself
    fit_bootstrap(X, seed)     — fit on a bootstrap resample of X's rows
    fit_leave_out(X, ids, d)   — fit on the rows whose dataset_id != d

Also here: the stability checks — do bootstrap / leave-one-dataset-out fits
recover the same components? `assess` compares perturbed fits to the full-data
reference via principal angles and per-component cosines (see its docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def n_components_for_variance(X_std: np.ndarray, frac: float = 0.95) -> int:
    """Smallest #components whose cumulative explained variance reaches `frac`."""
    ev = PCA().fit(X_std).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ev), frac) + 1)


@dataclass
class PCAComponents:
    scaler_mean: np.ndarray      # (D,) standardization center
    scaler_scale: np.ndarray     # (D,) standardization scale
    pca_mean: np.ndarray         # (D,) PCA center (of the standardized fit data)
    components: np.ndarray       # (k, D) PCA loadings, no whitening
    explained_variance_ratio: np.ndarray  # (k,)

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project (N, D) activations onto the fitted PCs -> (N, k) scores."""
        X_std = (X - self.scaler_mean) / self.scaler_scale
        return (X_std - self.pca_mean) @ self.components.T

    def save(self, path) -> None:
        np.savez(
            path,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            pca_mean=self.pca_mean,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
        )

    @classmethod
    def load(cls, path) -> "PCAComponents":
        d = np.load(path)
        return cls(**{k: d[k] for k in d.files})


def fit(X_fit: np.ndarray, n_components) -> PCAComponents:
    """Standardize on X_fit, then PCA on X_fit. `n_components` int or '95%'."""
    mean = X_fit.mean(axis=0)
    scale = X_fit.std(axis=0)
    scale[scale == 0] = 1.0  # leave constant dimensions untouched
    X_std = (X_fit - mean) / scale

    k = n_components_for_variance(X_std) if n_components == "95%" else int(n_components)
    pca = PCA(n_components=k, whiten=False).fit(X_std)
    return PCAComponents(
        scaler_mean=mean,
        scaler_scale=scale,
        pca_mean=pca.mean_,
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def fit_bootstrap(X: np.ndarray, n_components, seed: int) -> PCAComponents:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=len(X))
    return fit(X[idx], n_components)


def fit_leave_out(X: np.ndarray, dataset_id: np.ndarray, held_out, n_components) -> PCAComponents:
    return fit(X[dataset_id != held_out], n_components)


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces given as
    (m, D) matrices of orthonormal rows."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_overlap(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared cosine of principal angles, in [0, 1]. 1 = same subspace."""
    return float(((A @ B.T) ** 2).sum()) / A.shape[0]


def per_component_cos(V_ref: np.ndarray, V_pert: np.ndarray, best_match: bool) -> np.ndarray:
    """|cos| between reference and perturbed components, per reference axis.

    same-index: compares axis j to axis j (a swap looks like instability).
    best_match: matches axes by max |cos| (a swap is recognized, not penalized).
    """
    if not best_match:
        return np.abs((V_ref * V_pert).sum(axis=1))
    C = np.abs(V_ref @ V_pert.T)
    row, col = linear_sum_assignment(-C)
    return C[row, col]


@dataclass
class StabilityResult:
    m: int
    fit_names: list[str]            # 'full', 'boot0'.., 'leaveout_<d>'..
    fit_kind: dict[str, str]        # name -> 'reference'|'bootstrap'|'leaveout'
    overlap_to_ref: dict[str, float]
    largest_angle_to_ref: dict[str, float]   # radians
    cos_bestmatch: dict[str, np.ndarray]     # name -> (m,) best-matched |cos|
    cos_sameidx: dict[str, np.ndarray]       # name -> (m,) same-index |cos|
    pairwise_overlap: np.ndarray             # (F, F) overlap among all fits


def assess(X, m: int, n_boot: int = 5, seed: int = 0, dataset_id=None) -> StabilityResult:
    """Fit reference + bootstrap (+ leave-one-out) PCAs at `m` components and
    measure their agreement. `m` must be an int (resolve '95%' before calling)."""
    fits = {"full": fit(X, m)}
    kind = {"full": "reference"}
    for b in range(n_boot):
        fits[f"boot{b}"] = fit_bootstrap(X, m, seed + b)
        kind[f"boot{b}"] = "bootstrap"
    if dataset_id is not None:
        for d in np.unique(dataset_id):
            fits[f"leaveout_{d}"] = fit_leave_out(X, dataset_id, d, m)
            kind[f"leaveout_{d}"] = "leaveout"
    return compare_fits(fits, kind)


def compare_fits(fits: dict[str, "PCAComponents"], kind: dict[str, str]) -> StabilityResult:
    """Agreement among already-fitted PCAs; 'full' is the reference."""
    names = list(fits)
    V = {n: fits[n].components for n in names}  # each (m, D), orthonormal rows
    ref = V["full"]
    m = ref.shape[0]

    overlap = {n: subspace_overlap(ref, V[n]) for n in names}
    max_angle = {n: float(principal_angles(ref, V[n]).max()) for n in names}
    cos_bm = {n: per_component_cos(ref, V[n], best_match=True) for n in names}
    cos_si = {n: per_component_cos(ref, V[n], best_match=False) for n in names}

    F = len(names)
    pairwise = np.ones((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            pairwise[i, j] = pairwise[j, i] = subspace_overlap(V[names[i]], V[names[j]])

    return StabilityResult(m, names, kind, overlap, max_angle, cos_bm, cos_si, pairwise)


if __name__ == "__main__":
    # Smoke test on synthetic data with known low-rank structure.
    rng = np.random.default_rng(0)
    N, D = 500, 50
    Z = rng.standard_normal((N, 3)) @ rng.standard_normal((3, D))
    X = Z + 0.1 * rng.standard_normal((N, D)) + rng.standard_normal(D) * 5

    comps = fit(X, n_components=10)
    scores = comps.transform(X)
    print(f"scores: {scores.shape}, top-3 var ratio: {comps.explained_variance_ratio[:3].round(3)}")
    print(f"95% variance at k = {n_components_for_variance((X - X.mean(0)) / X.std(0))}")

    # Save/reload round-trip must reproduce the projection exactly.
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "pca.npz")
    comps.save(p)
    assert np.allclose(scores, PCAComponents.load(p).transform(X))
    print("save/load round-trip OK")

    # Stability: two datasets sharing a subspace + dataset-specific noise.
    rng = np.random.default_rng(0)
    shared = rng.standard_normal((5, 40))
    def block(n, extra):
        return rng.standard_normal((n, 5)) @ shared + extra * rng.standard_normal((n, 40))
    X = np.vstack([block(300, 0.3), block(300, 0.3)])
    ids = np.array([0] * 300 + [1] * 300)

    r = assess(X, m=5, n_boot=5, seed=0, dataset_id=ids)
    print(f"m={r.m}")
    for n in r.fit_names:
        print(f"  {n:14s} [{r.fit_kind[n]:9s}] overlap={r.overlap_to_ref[n]:.3f} "
              f"max_angle={np.degrees(r.largest_angle_to_ref[n]):5.1f} deg  "
              f"min|cos|(bm)={r.cos_bestmatch[n].min():.3f}")
    print(f"pairwise overlap range: {r.pairwise_overlap.min():.3f}..{r.pairwise_overlap.max():.3f}")
