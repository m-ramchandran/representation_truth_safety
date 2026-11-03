"""Run ON compute cluster (CPU node): the expanded-pool merged run without GPUs.

Same experiment as run_merged_expanded_pool.py (pool expanded to
include claude_vds and wildjailbreak_adversarial), executed on CPU:

  - cupy/cuml are stubbed out before `classifiers.scripts` imports (the
    cuML methods are not used);
  - the torch methods (mlp, random_slopes, random_slopes_task) run
    unmodified with fit_torch.DEVICE = "cpu";
  - l2reg is swapped for a contract-identical sklearn implementation
    (StandardScaler -> LogisticRegression(lbfgs), same fold protocol,
    native log-odds margins);
  - xgboost is dropped (2000-round fits are GPU-priced; MLP is the modal
    pooled nonlinear winner in the original runs and stands in for the
    family);
  - tuning grids are pinned to the neighbourhood of the hyperparameters
    the original merged runs chose (hyperparam_stability.csv):
    mlp hidden {64,256} x wd 1e-4; rs shared_l2 {1e-7,1e-6} x dev
    multiplier {10,100}; l2 C {0.01, 0.0316, 0.1}.

    python expanded_pool_cpu.py <base_collection> <layer> [methods_csv]

The optional third argument restricts the run to a comma-separated
method subset (default: diff,l2reg,mlp,random_slopes,random_slopes_task).
Because the runner's resume record is (features, layer, method), several
jobs with disjoint method subsets can safely fill the same result tree
in parallel.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import runpy
import sys
import types

# ---- 1. real heavy imports first, then stub cupy/cuml -------------------
# (torch's own import machinery uses `inspect`, which walks sys.modules;
# stub modules must never answer dunder lookups like __file__.)

import numpy as np
import torch


class _Anything:
    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return self


def _stub(name: str) -> None:
    mod = types.ModuleType(name)

    def module_getattr(attr, _mod=name):
        if attr.startswith("__"):
            raise AttributeError(f"{_mod} stub has no {attr}")
        return _Anything()

    mod.__getattr__ = module_getattr                # PEP 562
    sys.modules[name] = mod


for _n in ("cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.sparse",
           "cuml", "cuml.linear_model", "cuml.neighbors", "cuml.svm"):
    _stub(_n)

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK")
                          or os.environ.get("SLURM_NTASKS_PER_NODE") or 32))

import classifiers.data as data
import classifiers.scripts as S
import classifiers.scripts.fit_torch as ft
import classifiers.scripts.runner_common as rc
from classifiers.scripts.base import DEFAULT_INNER_CV, SklearnClassifier
from classifiers.scripts.fit_classifiers import (StandardScaler, _tune_cv,
                                                 ensure_folds, make_pipeline)

ft.DEVICE = "cpu"
rc.release_gpu_memory = lambda: None

L2_CS = (0.01, 0.03162277660168379, 0.1)


def fit_l2_cpu(X_train, y_train, X_test=None, *, seed: int,
               folds=None, stratify_on=None, Cs=L2_CS,
               inner_cv: int = DEFAULT_INNER_CV, inner_n_jobs: int = 1):
    """sklearn twin of the cuML fit_l2: standardize -> L2 logistic,
    CV-select C over `folds`, refit the winner on all of X_train."""
    from sklearn.linear_model import LogisticRegression

    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    grid = list(Cs)
    candidates = [make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c), penalty="l2", solver="lbfgs",
                           max_iter=1000, random_state=seed))
        for c in grid]
    best, cv = _tune_cv(candidates, X_train, y_train, folds, inner_n_jobs)
    from sklearn.base import clone
    model = clone(candidates[best]).fit(X_train, y_train)
    fitted = SklearnClassifier(model, model_type="l2reg")
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"C": grid[best], "solver": "sklearn_lbfgs"}, fitted


def main() -> None:
    base, layer = sys.argv[1], sys.argv[2]
    methods = (sys.argv[3].split(",") if len(sys.argv) > 3 else
               ["diff", "l2reg", "mlp", "random_slopes",
                "random_slopes_task"])
    name = base + "_expanded_pool"
    if name not in data.COLLECTIONS:
        data.COLLECTIONS[name] = dataclasses.replace(
            data.COLLECTIONS[base],
            pool_excluded=("claude_blood_test_ranges",))

    S.METHODS["l2reg"] = fit_l2_cpu
    S.METHODS["mlp"] = functools.partial(
        ft.fit_mlp, hiddens=(64, 256), weight_decays=(1e-4,))
    S.METHODS["random_slopes"] = functools.partial(
        ft.fit_random_slopes, shared_l2=(1e-7, 1e-6),
        dev_multipliers=(10.0, 100.0))
    S.METHODS["random_slopes_task"] = functools.partial(
        ft.fit_random_slopes, shared_l2=(1e-7, 1e-6),
        dev_multipliers=(10.0, 100.0))

    sys.argv = ["run_merged", "--data", name, "--raw", "last",
                "--features", "raw", "--layers", layer,
                "--methods", *methods, "--n-jobs", "1"]
    runpy.run_module("classifiers.scripts.run_merged", run_name="__main__")


if __name__ == "__main__":
    main()
