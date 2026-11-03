"""All classifier fit functions, one section per method.

Every method follows the same signature (see the package docstring):

    fit_<name>(X_train, y_train, X_test=None, *, seed, **kwargs)
        -> (cv_scores, test_scores, params, fitted_classifier)

`cv_scores` and `test_scores` are dicts of per-sample arrays with two keys:
"proba", a positive-class probability, and "margin", the method's native
decision score, stored untouched. `cv_scores` always holds CV predictions
on X_train over `folds`: tuning already fits every candidate on every fold
and predicts the held-out part, so the winner's CV predictions are kept
from those fits — nothing is refit to recover them. `test_scores` is None
when no test set is passed. Inputs and outputs are numpy / scipy-sparse
throughout — GPU use is internal to each fit function, so everything
downstream (npz outputs, csvs) never sees a GPU array.

The margin, method by method:

  - l1reg / l2reg : the linear score X @ w + b — exactly the log-odds of
        `proba` (proba = sigmoid(margin)). Stored directly because
        recovering it as logit(proba) loses precision once proba saturates.
  - lda / qda : the log posterior odds (sklearn's decision_function),
        likewise the exact log-odds of `proba`.
  - svm_linear / svm_rbf : the raw SVM decision value — 0 on the decision
        boundary, +-1 at the margin edges. For svm_rbf that value lives in
        the Nystrom feature space (see fit_svm).
  - diff : the signed distance along the unit class-mean direction, in raw
        activation units.
  - knn : no margin. Its probability — the fraction of positive neighbors —
        hits exactly 0 and 1, so it has no finite log-odds.

The SVMs and diff, whose margins are not log-odds, get their probabilities
from Platt calibration: a one-variable logistic regression of y on the CV
margins gives (a, b) with proba = sigmoid(a * margin + b). CV margins, not
training-fit margins — training margins are overconfident. (a, b) is
returned in `params`, so `a * margin + b` puts those margins on the
log-odds scale exactly whenever a cross-method comparison needs it.

Methods here:
  - fit_logreg : L1- or L2-regularized logistic regression on the GPU via
                 cuML (the quasi-Newton solver covers both penalties;
                 OWL-QN for l1). fit_l1 / fit_l2 fix the penalty. The C
                 grid runs sequentially on the device — no joblib.
  - fit_svm  : linear SVM on the GPU via cuML. The rbf variant first maps
               the data through a Nystrom approximation of the RBF kernel
               (exact when n_train <= NYSTROM_M) and fits the linear SVM
               on the mapped features. fit_svm_linear / fit_svm_rbf fix
               the kernel.
  - fit_knn  : k-nearest-neighbors on the GPU via cuML. Brute-force GPU
               distances make the 30-50k-sample datasets practical
               (prediction cost is n_train x n_test, impractical on CPU).
  - fit_diff / fit_lda / fit_qda : CPU implementations (numpy / sklearn) —
               cuML has no LDA/QDA, and diff-in-means costs milliseconds
               either way. Their tuning grids parallelize over the node's
               CPU cores via `inner_n_jobs`.

CV selects hyperparameters by mean per-fold AUC, computed on the margin
where the method has one, else the probability (identical rankings — the
two are monotonically linked). Standardization for the GPU methods is fit
on each fold's train part only, once per fold (it does not depend on the
hyperparameter), and is variance-only for sparse input (a sparse matrix
cannot be mean-centered without densifying; scaling alone is the standard
treatment for nonnegative sparse features).
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp
from cuml.linear_model import LogisticRegression as CumlLogisticRegression
from cuml.neighbors import KNeighborsClassifier as CumlKNeighborsClassifier
from cuml.svm import LinearSVC as CumlLinearSVC
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import (
    DEFAULT_INNER_CV, FittedClassifier, Fold, SklearnClassifier,
    ensure_folds,
)


# --- hyperparameter grids ----------------------------------------------------

# L1 grid: C in [1e-3, 10] at ~1.9x log spacing. Below 1e-3 every coef is
# zero on standardized features; by C~10 the sparsity path is fully in and
# larger C barely changes the solution — so all 15 points sit in the range
# where the path actually moves. C is INVERSE regularization (smaller C =
# sparser).
DEFAULT_L1_CS = np.logspace(-3, 1, 15)

# L2 grid: C in [1e-4, 1e3] at ~3.2x log spacing
DEFAULT_L2_CS = np.logspace(-4, 3, 15)

# Linear SVM C grid: same wide log-spaced spirit as the logistic grids;
# linear fits are cheap, so 15 points is affordable.
DEFAULT_SVM_LINEAR_CS = np.logspace(-4, 2, 15)

# RBF SVM C grid: 9 points (~3.2x spacing, the same as the L2 grid) over
# the range where standardized features move from under- to over-fitting.
DEFAULT_SVM_RBF_CS = np.logspace(-2, 2, 9)

# Nystrom landmark count for svm_rbf. The exact n x n RBF kernel is
# impossible at pooled corpus sizes (~128k samples -> ~65 GB); the Nystrom
# map replaces it with the n x m kernel against m landmark training rows
# (~2 GB), and is exact whenever n_train <= m.
NYSTROM_M = 4096

# kNN neighborhood sizes. Datasets here have ~1000+ samples, so the grid
# spans very local (5) to fairly smoothed (100) decision boundaries.
DEFAULT_KNN_KS = (5, 10, 25, 50, 100)

# QDA shrinkage candidates for CV: Ledoit-Wolf ("auto") plus fixed
# values from light to heavy regularization of the per-class covariances.
DEFAULT_QDA_SHRINKAGES = ("auto", 0.05, 0.15, 0.3, 0.6, 0.9)


# --- Platt calibration -------------------------------------------------------

def _fit_platt(margins: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) with proba = sigmoid(a * margin + b): a one-variable logistic
    regression of y on the margins, effectively unregularized. See the
    module docstring for which methods use this and why it is fit on CV
    margins."""
    calib = LogisticRegression(C=1e6).fit(margins.reshape(-1, 1), y)
    return float(calib.coef_[0, 0]), float(calib.intercept_[0])


# --- GPU plumbing -------------------------------------------------------------

def _to_gpu(X):
    """numpy -> cupy, scipy CSR -> cupyx CSR, both as float32."""
    if sp.issparse(X):
        return cusp.csr_matrix(X.tocsr().astype(np.float32))
    return cp.asarray(np.asarray(X, dtype=np.float32))


def _fit_scale(Xg) -> tuple[Optional[cp.ndarray], cp.ndarray]:
    """Per-column standardization parameters, computed on the GPU. Dense
    returns (mean, scale); sparse returns (None, scale) — variance-only,
    since a sparse matrix cannot be mean-centered without densifying.
    Zero-variance columns get scale 1, as in sklearn's StandardScaler."""
    if cusp.issparse(Xg):
        m = cp.asarray(Xg.mean(axis=0)).ravel()
        m2 = cp.asarray(Xg.multiply(Xg).mean(axis=0)).ravel()
        var = cp.maximum(m2 - m * m, 0.0)
        mean = None
    else:
        mean = Xg.mean(axis=0)
        var = Xg.var(axis=0)
    scale = cp.sqrt(var)
    scale[scale == 0] = 1.0
    return mean, scale


def _apply_scale(Xg, mean, scale):
    if mean is None:
        return Xg.multiply(1.0 / scale).tocsr()
    return (Xg - mean) / scale


def _margin(model, Xg) -> np.ndarray:
    """A fitted cuML model's decision score X @ w + b, as numpy. Every cuML
    model used here is linear (logistic regression, or a linear SVM —
    possibly on Nystrom features) and exposes its coefficients."""
    w = cp.asarray(model.coef_).ravel()
    b = float(cp.asarray(model.intercept_).ravel()[0])
    return cp.asnumpy(Xg @ w + b).ravel().astype(np.float64)


class CumlClassifier(FittedClassifier):
    """Wraps a fitted cuML estimator, the standardization fit on its
    training data, and (for the SVMs) the Platt map. `predict` takes
    numpy / scipy-sparse input, moves it to the GPU, scales, and returns
    numpy score arrays — downstream code never sees GPU arrays."""

    def __init__(self, model, mean, scale, model_type: str, platt=None,
                 feature_map=None):
        self.model = model
        self.mean = mean
        self.scale = scale
        self.model_type = model_type
        self.platt = platt              # (a, b), or None if proba is native
        self.feature_map = feature_map  # svm_rbf's Nystrom map, else None

    def predict(self, X) -> dict:
        Xg = _apply_scale(_to_gpu(X), self.mean, self.scale)
        if self.feature_map is not None:
            Xg = self.feature_map(Xg)
        if self.model_type == "knn":    # no margin (see module docstring)
            proba = cp.asnumpy(self.model.predict_proba(Xg)[:, 1])
            return {"proba": proba.astype(np.float64)}
        margin = _margin(self.model, Xg)
        if self.platt is None:          # logreg: the margin is the log-odds
            return {"proba": expit(margin), "margin": margin}
        a, b = self.platt
        return {"proba": expit(a * margin + b), "margin": margin}

    def get_coef(self) -> Optional[np.ndarray]:
        # For svm_rbf these are weights in the Nystrom feature space, not
        # over the input features.
        if hasattr(self.model, "coef_"):
            return cp.asnumpy(self.model.coef_).ravel()
        return None

    def save_arrays(self) -> dict:
        # The standardization is over the input features even for svm_rbf
        # (it sits before the Nystrom map; the map itself — landmarks and
        # whitening — is not persisted, so svm_rbf's coef stays tied to
        # this fitted object).
        out = super().save_arrays()
        out["standardize_scale"] = cp.asnumpy(self.scale)
        if self.mean is not None:            # dense; sparse scales only
            out["standardize_mean"] = cp.asnumpy(self.mean)
        return out


def _tune_cv_gpu(make_estimator, predict, grid, Xg, y, folds,
                 fold_transform=None):
    """CV-tune on the GPU: standardize each fold once (fit on its train
    part), then fit one estimator per grid value and score the held-out
    part. `predict(estimator, X_gpu)` returns the method's numpy score
    arrays ("proba" and/or "margin"). `fold_transform(Xtr, Xte, f)`, if
    given, remaps both parts after scaling — like the scaling it is
    hyperparameter-independent, so it runs once per fold (svm_rbf's
    Nystrom map). The winner is the grid value with the best mean per-fold
    AUC; its CV scores are kept from these tuning fits — nothing is refit
    to recover them. Returns (best_index, cv_scores)."""
    yg = cp.asarray(y)
    cv = [{} for _ in grid]            # per grid value: key -> (n,) array
    aucs = np.empty((len(grid), len(folds)))
    for f, (tr, te) in enumerate(folds):
        tr_g, te_g = cp.asarray(tr), cp.asarray(te)
        mean, scale = _fit_scale(Xg[tr_g])
        Xtr = _apply_scale(Xg[tr_g], mean, scale)
        Xte = _apply_scale(Xg[te_g], mean, scale)
        if fold_transform is not None:
            Xtr, Xte = fold_transform(Xtr, Xte, f)
        ytr = yg[tr_g]
        for i, value in enumerate(grid):
            est = make_estimator(value)
            est.fit(Xtr, ytr)
            scores = predict(est, Xte)
            for key, s in scores.items():
                cv[i].setdefault(key, np.empty(len(y)))[te] = s
            s = scores["margin"] if "margin" in scores else scores["proba"]
            aucs[i, f] = roc_auc_score(y[te], s)
    best = int(aucs.mean(axis=1).argmax())
    return best, cv[best]


# --- regularized logistic regression (GPU) -------------------------------------

def fit_logreg(
    X_train, y_train: np.ndarray,
    X_test=None,
    *, seed: int,
    penalty: str,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    Cs=None,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, CumlClassifier]:
    """L1- or L2-regularized logistic regression on the GPU (cuML's
    quasi-Newton solver; OWL-QN for l1). CV over `folds` picks C from `Cs`
    (default grid depends on the penalty). `stratify_on` sets what
    self-built folds are stratified on (default y_train); ignored when
    `folds` is given.

    Sparse (CSR) input stays sparse on the GPU, with variance-only
    standardization."""
    if penalty not in ("l1", "l2"):
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}")
    if Cs is None:
        Cs = DEFAULT_L1_CS if penalty == "l1" else DEFAULT_L2_CS
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)

    def make_estimator(C):
        return CumlLogisticRegression(C=float(C), penalty=penalty,
                                      max_iter=1000, output_type="cupy")

    def predict(est, Xte):
        margin = _margin(est, Xte)
        return {"proba": expit(margin), "margin": margin}

    Xg = _to_gpu(X_train)
    best, cv = _tune_cv_gpu(make_estimator, predict, list(Cs), Xg, y_train,
                            folds)
    mean, scale = _fit_scale(Xg)
    model = make_estimator(Cs[best])
    model.fit(_apply_scale(Xg, mean, scale), cp.asarray(y_train))
    fitted = CumlClassifier(model, mean, scale, model_type=penalty)
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"C": float(Cs[best])}, fitted


fit_l1 = partial(fit_logreg, penalty="l1")
fit_l2 = partial(fit_logreg, penalty="l2")


# --- support vector machines (GPU) ---------------------------------------------
#
# Probabilities are never fit inside cuML; the margin is the score, ranked
# by AUC during tuning, and the Platt map fit on the winner's CV margins
# supplies `proba`. The stored margin stays the raw decision value, so its
# geometry (boundary at 0, margin edges at +-1) is preserved.
#
# svm_rbf approximates the RBF kernel with the Nystrom method: represent
# each point by its kernel similarity to m landmark training rows, whiten
# by the landmarks' own m x m kernel, and fit a *linear* SVM on those m
# features — dot products there approximate the kernel. The landmarks come
# from the data itself, so the approximation is spent where the data
# lives; with m = n it reproduces the exact kernel.

def _rbf_kernel(A: cp.ndarray, B: cp.ndarray, gamma: float) -> cp.ndarray:
    """exp(-gamma * squared distance) between the rows of two GPU arrays."""
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return cp.exp(-gamma * cp.maximum(d2, 0.0))


def _nystrom_map(Xs: cp.ndarray, seed: int):
    """The Nystrom feature map of a standardized training matrix. Landmarks
    are m = min(n, NYSTROM_M) rows sampled without replacement; whitening is
    the pseudo-inverse square root of their kernel (eigendecomposition,
    eigenvalues below 1e-6 of the largest dropped). gamma follows sklearn's
    "scale": 1 / (n_features * variance of Xs). Returns (transform, gamma,
    m), where transform maps any standardized GPU array into the feature
    space in row chunks (the distance block stays chunk x m)."""
    n, d = Xs.shape
    gamma = 1.0 / (d * float(Xs.var()))
    if n <= NYSTROM_M:
        L = Xs
    else:
        idx = np.random.default_rng(seed).choice(n, NYSTROM_M, replace=False)
        L = Xs[cp.asarray(idx)]
    K = _rbf_kernel(L, L, gamma)
    K = (K + K.T) / 2.0
    lam, U = cp.linalg.eigh(K)                    # ascending eigenvalues
    keep = lam > 1e-6 * float(lam[-1])
    W = U[:, keep] / cp.sqrt(lam[keep])

    def transform(X: cp.ndarray, _chunk: int = 8192) -> cp.ndarray:
        return cp.concatenate([_rbf_kernel(X[i:i + _chunk], L, gamma) @ W
                               for i in range(0, X.shape[0], _chunk)])

    return transform, gamma, int(L.shape[0])


def fit_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *, seed: int,
    kernel: str,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    Cs=None,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, CumlClassifier]:
    """Linear SVM on the GPU (cuML). kernel="rbf" first maps the data
    through the Nystrom approximation of the RBF kernel — built per fold on
    the fold's standardized train part (it is hyperparameter-independent,
    like the scaling) — and fits the linear SVM on the mapped features;
    kernel="linear" fits the input features directly. CV over `folds` picks
    C from `Cs` (default grid depends on the kernel), the winner's CV
    margins fit the Platt map, and one refit on all of X_train (with its
    own Nystrom map for rbf) gives the final model. Dense input only — the
    runner densifies sparse feature sets for the SVMs. `stratify_on` sets
    what self-built folds are stratified on (default y_train); ignored when
    `folds` is given."""
    if kernel not in ("linear", "rbf"):
        raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}")
    if Cs is None:
        Cs = DEFAULT_SVM_LINEAR_CS if kernel == "linear" else DEFAULT_SVM_RBF_CS
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)

    def make_estimator(C):
        return CumlLinearSVC(C=float(C), output_type="cupy")

    def predict(est, Xte):
        return {"margin": _margin(est, Xte)}

    fold_transform = None
    if kernel == "rbf":
        def fold_transform(Xtr, Xte, f):
            transform, _, _ = _nystrom_map(Xtr, seed=seed + f)
            return transform(Xtr), transform(Xte)

    Xg = _to_gpu(X_train)
    best, cv = _tune_cv_gpu(make_estimator, predict, list(Cs), Xg, y_train,
                            folds, fold_transform=fold_transform)
    a, b = _fit_platt(cv["margin"], y_train)
    cv["proba"] = expit(a * cv["margin"] + b)
    mean, scale = _fit_scale(Xg)
    Xs = _apply_scale(Xg, mean, scale)
    params = {"C": float(Cs[best]), "platt_a": a, "platt_b": b}
    feature_map = None
    if kernel == "rbf":
        feature_map, gamma, m = _nystrom_map(Xs, seed=seed)
        Xs = feature_map(Xs)
        params.update({"gamma": gamma, "n_landmarks": m})
    model = make_estimator(Cs[best])
    model.fit(Xs, cp.asarray(y_train))
    fitted = CumlClassifier(model, mean, scale, model_type=f"svm_{kernel}",
                            platt=(a, b), feature_map=feature_map)
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, params, fitted


fit_svm_linear = partial(fit_svm, kernel="linear")
fit_svm_rbf = partial(fit_svm, kernel="rbf")


# --- k-nearest-neighbors (GPU) ---------------------------------------------------

def fit_knn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    ks=DEFAULT_KNN_KS,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, CumlClassifier]:
    """k-nearest-neighbors on the GPU (cuML, uniform weights, brute-force
    distances). CV over `folds` picks k from `ks`. Standardized in front so
    no single high-variance activation dominates the Euclidean distance.
    The predicted probability is the fraction of the k neighbors in the
    positive class, so scores are coarse (k+1 distinct values) — expect
    tied scores when reading its AUCs — and there is no margin. Dense
    input only. `stratify_on` sets what self-built folds are stratified on
    (default y_train); ignored when `folds` is given."""
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)

    def make_estimator(k):
        return CumlKNeighborsClassifier(n_neighbors=int(k), output_type="cupy")

    def predict(est, Xte):
        proba = cp.asnumpy(est.predict_proba(Xte)[:, 1])
        return {"proba": proba.astype(np.float64)}

    # cuML requires k <= the fitted index size, and small datasets
    # (claude_vds, n=126) undercut the default grid inside the CV folds —
    # clamp to the smallest fold's training part (that size as the lone k
    # when even the smallest candidate is too big).
    min_train = min(len(tr) for tr, _ in folds)
    ks = [k for k in ks if k <= min_train] or [min_train]
    Xg = _to_gpu(X_train)
    best, cv = _tune_cv_gpu(make_estimator, predict, ks, Xg, y_train, folds)
    mean, scale = _fit_scale(Xg)
    model = make_estimator(ks[best])
    model.fit(_apply_scale(Xg, mean, scale), cp.asarray(y_train))
    fitted = CumlClassifier(model, mean, scale, model_type="knn")
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"n_neighbors": int(ks[best])}, fitted


# --- CPU methods ----------------------------------------------------------------
#
# diff / lda / qda run on the node's CPU cores: cuML has no LDA/QDA, and
# diff-in-means is a handful of matmuls.

def _fit_predict_fold(estimator, X, y, tr, te):
    estimator.fit(X[tr], y[tr])
    return {"proba": estimator.predict_proba(X[te])[:, 1],
            "margin": estimator.decision_function(X[te])}


def _tune_cv(candidates, X, y, folds, n_jobs):
    """CV-tune over candidate estimators (one per hyperparameter value): fit
    each candidate on each fold's train part and score its held-out part;
    the winner is the candidate with the best mean per-fold AUC. Its CV
    scores come straight from these tuning fits — nothing is refit to
    recover them. A single candidate means no selection, just CV scores.
    Returns (best_index, cv_scores)."""
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(clone(est), X, y, tr, te)
        for est in candidates for (tr, te) in folds
    )
    cv = [{"proba": np.empty(len(y)), "margin": np.empty(len(y))}
          for _ in candidates]
    aucs = np.empty((len(candidates), len(folds)))
    for i in range(len(candidates)):
        for f, (tr, te) in enumerate(folds):
            scores = preds[i * len(folds) + f]
            cv[i]["proba"][te] = scores["proba"]
            cv[i]["margin"][te] = scores["margin"]
            aucs[i, f] = roc_auc_score(y[te], scores["margin"])
    best = int(aucs.mean(axis=1).argmax())
    return best, cv[best]


# --- diff-in-means ------------------------------------------------------------

def _fit_diff_direction(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit-norm class-mean-difference direction and midpoint threshold."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    proj = X @ d
    return d, 0.5 * (proj[y == 1].mean() + proj[y == 0].mean())


class DiffInMeansClassifier(FittedClassifier):
    """Unit-norm class-mean-difference direction, midpoint threshold, and a
    Platt map. The margin `X @ direction - threshold` is a signed distance
    in activation space (kept raw); the Platt map turns it into `proba`."""

    model_type = "diff_in_means"

    def __init__(self, direction: np.ndarray, threshold: float, platt):
        self.direction = np.asarray(direction)
        self.threshold = float(threshold)
        self.platt = platt              # (a, b)

    def predict(self, X: np.ndarray) -> dict:
        margin = X @ self.direction - self.threshold
        a, b = self.platt
        return {"proba": expit(a * margin + b), "margin": margin}

    def get_coef(self) -> np.ndarray:
        # The unit-norm class-mean-difference direction — in raw
        # activation units, unlike the other methods (no standardization
        # sits in front of diff).
        return self.direction


def fit_diff(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *, seed: int = 0,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,                          # unused; nothing to parallelize
) -> tuple[dict, Optional[dict], dict, DiffInMeansClassifier]:
    """Diff-in-means: project onto the class-mean-difference direction and
    subtract the midpoint of the two projected class means. No
    hyperparameters — the CV margins are simply each fold's direction
    scored on its held-out part; they also fit the Platt map that turns
    margins into probabilities. `stratify_on` sets what self-built folds
    are stratified on (default y_train); ignored when `folds` is given."""
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    cv_margin = np.empty(len(y_train))
    for tr, te in folds:
        d, thr = _fit_diff_direction(X_train[tr], y_train[tr])
        cv_margin[te] = X_train[te] @ d - thr
    a, b = _fit_platt(cv_margin, y_train)
    direction, threshold = _fit_diff_direction(X_train, y_train)
    fitted = DiffInMeansClassifier(direction, threshold, platt=(a, b))
    cv = {"proba": expit(a * cv_margin + b), "margin": cv_margin}
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"platt_a": a, "platt_b": b}, fitted


# --- LDA / QDA -----------------------------------------------------------------
#
# Both are Gaussian generative classifiers: `decision_function` is the log
# posterior odds (the margin) and `predict_proba` its sigmoid, so they
# reuse the `SklearnClassifier` wrapper directly.
#
#   - fit_lda : one pooled covariance shared across classes (linear boundary).
#               Defaults to Ledoit-Wolf shrinkage ("auto"), a closed-form choice
#               that keeps the covariance well-conditioned when d is large.
#   - fit_qda : a separate covariance per class (quadratic boundary). Uses the
#               eigen solver with Ledoit-Wolf shrinkage ("auto") — sklearn's
#               default svd solver cannot regularize and errors out when a class
#               has fewer samples than features, the usual case here.
#
# LDA fits once at fixed Ledoit-Wolf: it pools both classes into one
# covariance (comparatively well-fed with samples) and its boundary is fairly
# insensitive to shrinkage, so CV-tuning tends to reproduce "auto" within
# noise. QDA tunes shrinkage by CV by default: each class covariance is
# estimated from only half the data and the quadratic boundary is genuinely
# sensitive to how much it is regularized. Either behavior can be overridden
# via the `shrinkage` / `shrinkage_grid` args.
# StandardScaler sits in front of both: their shrinkage targets a *scaled
# identity*, which assumes the features share a scale, so standardizing first
# makes the regularization behave sensibly.

def _fit_discriminant(
    make_da, model_type: str,
    X_train, y_train, X_test, *,
    seed, folds, stratify_on, shrinkage, shrinkage_grid, inner_cv,
    inner_n_jobs,
) -> tuple[dict, Optional[dict], dict, SklearnClassifier]:
    """Shared body of fit_lda / fit_qda: build one standardize->discriminant
    pipeline per shrinkage candidate (`make_da(shrinkage)` supplies the
    discriminant step), CV-select among them, and refit the winner on all
    of X_train. A grid of one means no selection, just CV scores."""
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    grid = [shrinkage] if shrinkage_grid is None else list(shrinkage_grid)
    candidates = [
        make_pipeline(
            StandardScaler(with_mean=not sp.issparse(X_train)),
            make_da(s),
        )
        for s in grid
    ]
    best, cv = _tune_cv(candidates, X_train, y_train, folds, inner_n_jobs)
    model = clone(candidates[best]).fit(X_train, y_train)
    fitted = SklearnClassifier(model, model_type=model_type)
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"shrinkage": grid[best]}, fitted


def fit_lda(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    shrinkage="auto",
    shrinkage_grid: Optional[Sequence] = None,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,
) -> tuple[dict, Optional[dict], dict, SklearnClassifier]:
    """Linear discriminant analysis. Fits once at `shrinkage` (default "auto",
    Ledoit-Wolf). Pass `shrinkage_grid` to instead pick shrinkage by CV over
    `folds`. `stratify_on` sets what self-built folds are stratified on
    (default y_train); ignored when `folds` is given."""
    return _fit_discriminant(
        lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s),
        "lda", X_train, y_train, X_test, seed=seed, folds=folds,
        stratify_on=stratify_on, shrinkage=shrinkage,
        shrinkage_grid=shrinkage_grid, inner_cv=inner_cv,
        inner_n_jobs=inner_n_jobs)


def fit_qda(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    shrinkage="auto",
    shrinkage_grid: Optional[Sequence] = DEFAULT_QDA_SHRINKAGES,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,
) -> tuple[dict, Optional[dict], dict, SklearnClassifier]:
    """Quadratic discriminant analysis (eigen solver). By default CV over
    `folds` picks shrinkage from `shrinkage_grid` (Ledoit-Wolf "auto" is one
    of the candidates). Pass `shrinkage_grid=None` to fit once at `shrinkage`.
    `stratify_on` sets what self-built folds are stratified on (default
    y_train); ignored when `folds` is given."""
    return _fit_discriminant(
        lambda s: QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=s),
        "qda", X_train, y_train, X_test, seed=seed, folds=folds,
        stratify_on=stratify_on, shrinkage=shrinkage,
        shrinkage_grid=shrinkage_grid, inner_cv=inner_cv,
        inner_n_jobs=inner_n_jobs)
