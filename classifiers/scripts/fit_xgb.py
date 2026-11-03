"""XGBoost: gradient-boosted trees, trained on the GPU.

The one tree method in the package — everything else is linear, kernel,
prototype, or a small NN — so it contributes a genuinely different
inductive bias (axis-aligned splits, feature interactions). No
standardization is fit in front: trees are invariant to monotone
per-feature rescaling. Scipy-sparse input is accepted directly, so the
method is in features.SPARSE_OK.

The margin is the raw boosted score, a native log-odds under the
binary:logistic objective, so proba = sigmoid(margin) and there is no
Platt calibration.

Training mirrors the torch methods' early-stopping scheme: each
candidate boosts up to `n_rounds` rounds, stopping on the validation AUC
of a seeded, stratified VAL_FRAC slice of the data it was given — during
CV tuning that data is the fold's train part, so the fold's held-out
part never touches training, and the final model likewise keeps its
stopping slice out of training.

CV tuning follows the package rule: every candidate is fit on every
fold, the winner is the best mean per-fold AUC on margins, and the
winner's fold predictions are returned as the CV scores — nothing is
refit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .base import DEFAULT_INNER_CV, FittedClassifier, Fold, ensure_folds

# The grid crosses tree depth with the learning rate (and reg_lambda,
# one value by default); the round count is not tuned — early stopping
# picks it per fit. Ranges follow the literature (survey 2026-07-24,
# Planning.md): depth spans the shallow 2-5 range that won for XGBoost
# probes on LLM activations (Kantamneni et al. 2025, arXiv:2502.16681)
# up through the tabular benchmarks' deeper end (Grinsztajn et al. 2022
# tunes 1-11, TabZilla 2-12); the log-spaced learning rates cover
# Kantamneni's 0.001-0.1 upper half and TabZilla's 0.01-0.3 (fixed
# n_estimators + early stopping is Grinsztajn's setup too, 1000 + 20).
# Row/column subsampling is fixed at 0.8 rather than tuned — inside
# every surveyed range (0.5-1 / 0.7-1), and what the one fixed-config
# activation probe used.
DEFAULT_XGB_DEPTHS = (2, 4, 6, 9)
DEFAULT_XGB_LRS = (0.02, 0.08, 0.3)
DEFAULT_XGB_LAMBDAS = (1.0,)   # widen via partial (e.g. (0.1, 1, 10))
XGB_SUBSAMPLE = 0.8            # row subsampling per tree
XGB_COLSAMPLE = 0.8            # feature subsampling per tree
DEFAULT_XGB_ROUNDS = 2000      # ceiling; early stopping decides the rest
EARLY_STOPPING_ROUNDS = 20     # rounds without validation-AUC improvement
VAL_FRAC = 0.1                 # early-stopping slice


def _fit_one(X, y: np.ndarray, depth: int, lr: float, lam: float,
             n_rounds: int, seed: int) -> XGBClassifier:
    """One boosted fit with early stopping on a seeded, stratified
    VAL_FRAC slice of (X, y); the slice never enters training."""
    tr, val = train_test_split(np.arange(len(y)), test_size=VAL_FRAC,
                               random_state=seed, stratify=y)
    model = XGBClassifier(
        n_estimators=n_rounds, max_depth=depth, learning_rate=lr,
        reg_lambda=lam, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        tree_method="hist", device="cuda", eval_metric="auc",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=seed,
        importance_type="gain",   # what save_arrays persists; explicit
        verbosity=0)              # so the npz key means one thing always
    model.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
    return model


def _margin(model: XGBClassifier, X) -> np.ndarray:
    # the raw boosted score (log-odds); predict honors best_iteration
    return model.predict(X, output_margin=True).astype(np.float64)


class XGBoostClassifier(FittedClassifier):
    """Wraps a fitted XGBClassifier. No scaler sits in front, and there
    is no coefficient vector (get_coef stays None). save_arrays carries
    two things instead: the per-feature gain importance — each feature's
    share of the total split-gain, a magnitude with no sign, so NOT a
    direction (nothing in analysis_scripts/directions.py applies, and
    credit is split arbitrarily among correlated activation dimensions)
    — and the serialized booster, so the fit itself is reusable.

    The booster is stored as UBJSON bytes in a uint8 array rather than
    as a decomposed tree table: a boosted forest is a ragged graph of
    per-node (feature, threshold, children, leaf value) records whose
    shape varies per fit, so there is no faithful matrix form, but
    XGBoost's own serialization round-trips exactly and a byte array
    still satisfies the array-per-key contract."""

    model_type = "xgboost"

    def __init__(self, model: XGBClassifier):
        self.model = model

    def predict(self, X) -> dict:
        m = _margin(self.model, X)
        return {"proba": expit(m), "margin": m}

    def save_arrays(self) -> dict:
        # The booster itself rides along so the fit is reusable without a
        # refit (it is the only method whose predictions cannot be
        # rebuilt from a summary vector). UBJSON is XGBoost's own
        # portable format; as a uint8 array it satisfies the
        # array-per-key contract and stays inside the one npz per fit
        # that npz_path / resume / redraw all assume. ~0.4 MB deflated
        # at 300 rounds x depth 6 (measured 2026-07-24), and callers
        # that pass `keys=` to load_prediction never decompress it.
        # Round-trip: load_results.load_booster (bit-identical margins).
        booster = self.model.get_booster()
        best = getattr(self.model, "best_iteration", None)
        if best is not None:
            # Early stopping keeps training PATIENCE rounds past the best
            # one, and XGBClassifier.predict quietly ignores that tail
            # while a bare Booster.predict does not. Slicing the tail off
            # here makes the saved artifact reproduce this fit's margins
            # for any reader, without them having to know an
            # iteration_range. (Caught by the smoke test 2026-07-24: the
            # unsliced booster disagreed with the fit it came from.)
            booster = booster[: best + 1]
        raw = booster.save_raw(raw_format="ubj")
        return {"feature_importance":
                np.asarray(self.model.feature_importances_),
                "booster_ubj": np.frombuffer(bytes(raw), dtype=np.uint8)}


def fit_xgboost(
    X_train, y_train: np.ndarray,
    X_test=None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    depths=DEFAULT_XGB_DEPTHS,
    learning_rates=DEFAULT_XGB_LRS,
    reg_lambdas=DEFAULT_XGB_LAMBDAS,
    n_rounds: int = DEFAULT_XGB_ROUNDS,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, XGBoostClassifier]:
    """Gradient-boosted trees. CV over `folds` picks (depth, learning
    rate, reg_lambda) from the crossed grids; each candidate's round
    count comes from its own early stopping. `stratify_on` sets what
    self-built folds are stratified on (default y_train); ignored when
    `folds` is given."""
    y = np.asarray(y_train)
    folds = ensure_folds(X_train, y, folds, seed=seed, n_splits=inner_cv,
                         stratify_on=stratify_on)
    grid = [(d, lr, lam) for d in depths for lr in learning_rates
            for lam in reg_lambdas]

    cv = [{"proba": np.empty(len(y)), "margin": np.empty(len(y))}
          for _ in grid]
    aucs = np.empty((len(grid), len(folds)))
    for f, (tr, te) in enumerate(folds):
        for i, (depth, lr, lam) in enumerate(grid):
            model = _fit_one(X_train[tr], y[tr], depth, lr, lam, n_rounds,
                             seed + 1009 * f + i)
            m = _margin(model, X_train[te])
            cv[i]["margin"][te] = m
            cv[i]["proba"][te] = expit(m)
            aucs[i, f] = roc_auc_score(y[te], m)
    best = int(aucs.mean(axis=1).argmax())
    depth, lr, lam = grid[best]

    model = _fit_one(X_train, y, depth, lr, lam, n_rounds, seed)
    fitted = XGBoostClassifier(model)
    test = fitted.predict(X_test) if X_test is not None else None
    params = {"max_depth": depth, "learning_rate": lr, "reg_lambda": lam,
              "n_rounds": int(model.best_iteration) + 1}
    return cv[best], test, params, fitted
