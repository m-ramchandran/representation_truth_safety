"""Shared base for all classifier methods.

`FittedClassifier` is the uniform interface every method in this package
returns from its `fit_*` function. Anything downstream (experiment runners,
prediction code, coefficient introspection) talks to this interface — no
branching on model type. `SklearnClassifier` lives here too: it wraps a
fitted sklearn estimator and serves the methods built on sklearn (LDA, QDA).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


# --- shared constants -------------------------------------------------------

DEFAULT_INNER_CV = 5

# Feature-firing filter default. A "fire" is any nonzero value — only
# meaningful for sparse inputs (the sae / transcoder variants).
MIN_FIRES = 5


# --- creating inner/outer CV folds ------------------------------------------
#
# Fold indices are lists of (train_idx, test_idx) tuples so the same fold
# list can be shared across every method in a comparison.

Fold = tuple[np.ndarray, np.ndarray]
NestedFold = tuple[np.ndarray, np.ndarray, list[Fold]]


def make_folds(
    X: np.ndarray, y: np.ndarray, *,
    seed: int, n_splits: int = DEFAULT_INNER_CV,
    stratify_on: Optional[np.ndarray] = None,
) -> list[Fold]:
    """Divide a dataset into stratified folds: one k-fold split as a list of
    (train_idx, test_idx) numpy arrays. Folds are stratified on y, or on
    `stratify_on` if given (e.g. a joint (dataset, label) key on a merged
    corpus). Serves any fold need: CV inside `fit_*`, outer/reporting folds
    in experiment runners, or tuning on the reporting folds themselves (pass
    the same list to `fit_*` as `folds=`)."""
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )
    return list(splitter.split(X, y if stratify_on is None else stratify_on))


def ensure_folds(
    X: np.ndarray, y: np.ndarray, folds: Optional[list[Fold]], *,
    seed: int, n_splits: int = DEFAULT_INNER_CV,
    stratify_on: Optional[np.ndarray] = None,
) -> list[Fold]:
    """The folds passed to a `fit_*` function, or self-built ones when the
    caller omitted them. Every fit function opens with this, so runners can
    share one fold list across methods while standalone calls (notebooks,
    smoke tests) need not build folds at all."""
    if folds is not None:
        return folds
    return make_folds(X, y, seed=seed, n_splits=n_splits,
                      stratify_on=stratify_on)


def make_nested_folds(
    X: np.ndarray, y: np.ndarray, *,
    seed: int,
    n_outer_splits: int = 5,
    n_outer_repeats: int = 1,
    n_inner_splits: int = DEFAULT_INNER_CV,
    stratify_on: Optional[np.ndarray] = None,
) -> list[NestedFold]:
    """Nested CV folds. Outer and inner are separate `RepeatedStratifiedKFold`
    instances. Returns a list of (outer_train_idx, outer_test_idx, inner_folds)
    tuples; length is `n_outer_splits * n_outer_repeats`.

    `stratify_on` overrides what both splits are stratified on — pass a
    joint (dataset, label) key on a merged corpus so every fold, outer and
    inner, is balanced across both the source datasets and the labels.
    Without it, both stratify on y."""
    outer = RepeatedStratifiedKFold(
        n_splits=n_outer_splits, n_repeats=n_outer_repeats, random_state=seed,
    )
    key = y if stratify_on is None else stratify_on
    return [
        (outer_tr, outer_te,
         make_folds(X[outer_tr], y[outer_tr],
                    seed=seed, n_splits=n_inner_splits,
                    stratify_on=None if stratify_on is None
                    else stratify_on[outer_tr]))
        for outer_tr, outer_te in outer.split(X, key)
    ]


def filter_sparse_features(
    X: np.ndarray,
    min_fires: int = MIN_FIRES,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop features that fire on fewer than `min_fires` samples. A "fire" is
    any nonzero value — cuts sae/transcoder width down to features actually in
    use on the dataset. Returns (X_filtered, kept_indices)."""
    if sp.issparse(X):
        fire_count = np.asarray((X != 0).sum(axis=0)).ravel()
        kept = fire_count >= min_fires
        return X.tocsc()[:, kept].tocsr(), np.flatnonzero(kept)
    fire_count = (X != 0).sum(axis=0)
    kept = fire_count >= min_fires
    return X[:, kept], np.flatnonzero(kept)


# --- uniform predictor interface --------------------------------------------

class FittedClassifier:
    """Uniform interface for any trained binary classifier in this repo.

    Subclasses set `model_type` (used as the saved-model tag) and override
    `predict`. `get_coef` defaults to None for methods without a linear
    coefficient vector (e.g. qda, knn)."""

    model_type: str = ""

    def predict(self, X: np.ndarray) -> dict:
        """Per-sample scores: {"proba": positive-class probability in
        [0, 1], "margin": the method's native decision score}, each shape
        (n,). "margin" is absent for methods that have none (knn)."""
        raise NotImplementedError

    def get_coef(self) -> Optional[np.ndarray]:
        """1-D coefficient vector if this method has one, else None (qda,
        knn, mlp, moe). For every method but diff the vector lives in
        standardized-feature space (the scaling fit in front of the model);
        diff's is the raw-space class-mean direction, and svm_rbf's lives
        in its Nystrom feature space. Saved into the runners' prediction
        npz files for the coefficient analyses."""
        return None

    def save_arrays(self) -> dict:
        """Every array about the fitted model worth persisting into the
        runners' prediction npz files: "coef" (when get_coef() is not
        None), and per subclass the standardization fit in front of the
        model ("standardize_scale", plus "standardize_mean" unless the
        scaling was variance-only or the method takes raw input),
        random_slopes' per-dataset deviations, and moe's gate and expert
        weights. The base implementation is the coefficient alone (diff,
        which standardizes nothing)."""
        coef = self.get_coef()
        return {} if coef is None else {"coef": coef}


class SklearnClassifier(FittedClassifier):
    """Wraps a fitted sklearn estimator (usually a Pipeline) that has both
    `predict_proba` and `decision_function` — here LDA and QDA, whose
    decision function is the log posterior odds, i.e. the margin.

    `model_type` is set per-method by the `fit_*` caller ("lda", "qda") so
    downstream code can tell which method a saved model came from."""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type

    def predict(self, X: np.ndarray) -> dict:
        return {"proba": self.model.predict_proba(X)[:, 1],
                "margin": self.model.decision_function(X)}

    def get_coef(self) -> Optional[np.ndarray]:
        # The estimator sits at the end of the pipeline. LDA has a linear
        # coef_ (in standardized-feature space, the scaler in front); QDA
        # has none.
        est = self.model[-1] if hasattr(self.model, "steps") else self.model
        coef = getattr(est, "coef_", None)
        return None if coef is None else np.asarray(coef).ravel()

    def save_arrays(self) -> dict:
        out = super().save_arrays()
        if hasattr(self.model, "steps"):     # the StandardScaler in front
            scaler = self.model[0]
            out["standardize_scale"] = np.asarray(scaler.scale_)
            if scaler.with_mean:
                out["standardize_mean"] = np.asarray(scaler.mean_)
        return out
