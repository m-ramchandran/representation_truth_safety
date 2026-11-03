"""Classifier package. Runs on GPU nodes: l1reg / l2reg / svm_linear /
svm_rbf / knn fit on the GPU via cuML, mlp / random_slopes / moe on the
GPU via torch, xgboost on the GPU via the xgboost package, diff / lda /
qda on the node's CPU cores.

random_slopes, random_slopes, random_slopes_task, and moe are run by the pooled
experiments only (they need a multi-dataset corpus); xgboost is an
ordinary method and runs in all three. The two random-slopes variants
share one fit function and differ only in the required `groups=` array —
each sample's source dataset, or its task for the per-task variant
(fewer, larger groups); moe's expert-count grid is chosen by the runner
from the pool's composition (features.pooled_fit_kwargs supplies all of
these).

Every method is a function with a shared signature::

    fit_<name>(X_train, y_train, X_test=None, *, seed, **kwargs)
        -> (cv_scores, test_scores, params, fitted_classifier)

    cv_scores          : dict of per-sample CV predictions on X_train over
                         `folds` — "proba" (positive-class probability) and
                         "margin" (the method's native decision score, see
                         fit_classifiers; absent for knn, which has none) —
                         kept from the winning hyperparameters' tuning fits
                         (for methods with nothing to tune: the same
                         per-fold fits, just without selection)
    test_scores        : the same dict for X_test, or None if no test set
    params             : dict of chosen hyperparameters, plus the Platt
                         (a, b) for the methods that calibrate (svm, diff)
    fitted_classifier  : a `FittedClassifier` fit on all of X_train — call
                         `.predict(X)` on it later to score new data
                         without branching on type

Hyperparameter tuning is done inside the function via CV on
(X_train, y_train) only. Experiment runners precompute the fold indices
once per training set (via `make_folds` / `make_nested_folds`) and pass
that same list to every method via `folds=` — so when you compare methods
A and B on the same data, they are fit and selected against identical
splits. Standalone calls (notebooks, smoke tests) can omit `folds`; each
method then builds its own via `make_folds(X_train, y_train, seed=seed)`,
stratified on `stratify_on=` if given.

A fit function has no `groups=` of its own — CV grouping (`--group-folds`:
keeping rows expanded from one source item in one fold, see base.py and
data.GROUP_RULES) is a property of the fold list, so the runners build the
folds with `groups=` and pass them in. Note that `groups=` on
`fit_random_slopes` / `fit_moe` is a different thing entirely: those are
the per-sample dataset or task the model fits deviations for.

To override a method's hyperparameter grid for a single experiment without
editing this package, use ``functools.partial`` in the run script::

    from classifiers import fit_l1
    methods = {"l1reg": partial(fit_l1, Cs=(0.0001, 0.001, 0.01, 0.1, 1.0))}

`METHODS` maps the short name to the default fit function. Adding a new
method = a new section in `fit_classifiers.py`, then one entry below.
"""

from __future__ import annotations

from .base import (
    DEFAULT_INNER_CV,
    MIN_FIRES,
    FittedClassifier,
    Fold,
    NestedFold,
    SklearnClassifier,
    filter_sparse_features,
    make_folds,
    make_nested_folds,
)
from .fit_classifiers import (
    DEFAULT_KNN_KS,
    DEFAULT_L1_CS,
    DEFAULT_L2_CS,
    DEFAULT_QDA_SHRINKAGES,
    DEFAULT_SVM_LINEAR_CS,
    DEFAULT_SVM_RBF_CS,
    NYSTROM_M,
    CumlClassifier,
    DiffInMeansClassifier,
    fit_diff,
    fit_knn,
    fit_l1,
    fit_l2,
    fit_lda,
    fit_logreg,
    fit_qda,
    fit_svm,
    fit_svm_linear,
    fit_svm_rbf,
)
from .fit_torch import (
    RandomSlopesClassifier,
    TorchClassifier,
    fit_mlp,
    fit_moe,
    fit_moe_hard,
    fit_random_slopes,
)
from .fit_xgb import (
    XGBoostClassifier,
    fit_xgboost,
)


METHODS: dict = {
    "l1reg": fit_l1,
    "l2reg": fit_l2,
    "svm_linear": fit_svm_linear,
    "svm_rbf": fit_svm_rbf,
    "diff":  fit_diff,
    "lda":   fit_lda,
    "qda":   fit_qda,
    "knn":   fit_knn,
    "mlp":   fit_mlp,
    "xgboost": fit_xgboost,
    "random_slopes": fit_random_slopes,
    # same fit function as random_slopes; the runners pass task names as
    # `groups` (features.pooled_fit_kwargs), so deviations are per task
    "random_slopes_task": fit_random_slopes,
    "moe":   fit_moe,
    # same experts, same gate, same grid; routed to the single arg-max
    # expert instead of the gate-weighted mixture (fit_torch._MoEHard)
    "moe_hard": fit_moe_hard,
}


__all__ = [
    "METHODS",
    "DEFAULT_INNER_CV", "DEFAULT_KNN_KS", "DEFAULT_L1_CS", "DEFAULT_L2_CS",
    "DEFAULT_QDA_SHRINKAGES", "DEFAULT_SVM_LINEAR_CS", "DEFAULT_SVM_RBF_CS",
    "MIN_FIRES", "NYSTROM_M",
    "CumlClassifier", "DiffInMeansClassifier", "FittedClassifier",
    "RandomSlopesClassifier", "SklearnClassifier", "TorchClassifier",
    "XGBoostClassifier",
    "Fold", "NestedFold",
    "filter_sparse_features", "make_folds", "make_nested_folds",
    "fit_diff", "fit_knn", "fit_l1", "fit_l2", "fit_lda", "fit_logreg",
    "fit_mlp", "fit_moe", "fit_moe_hard", "fit_qda", "fit_random_slopes", "fit_svm",
    "fit_svm_linear", "fit_svm_rbf", "fit_xgboost",
]
