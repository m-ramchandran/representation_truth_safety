"""Fit the four all-data two-PC RBF classifiers used by the 4x3 figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .boundary_common import ARTIFACTS, SCOPES


def gpu_scores(X: np.ndarray, y: np.ndarray,
               grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the large-scope fit with cuML, imported only when requested."""
    import cupy as cp
    from cuml.svm import SVC

    def as_numpy(value):
        return value.get() if hasattr(value, "get") else np.asarray(value)

    Xg = cp.asarray(X)
    yg = cp.asarray(y)
    oof = np.empty(len(y), dtype=np.float32)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train, test) in enumerate(folds.split(X, y), 1):
        model = SVC(C=2.0, kernel="rbf", gamma="scale", cache_size=8192,
                    output_type="cupy")
        model.fit(Xg[cp.asarray(train)], yg[cp.asarray(train)])
        oof[test] = as_numpy(
            model.decision_function(Xg[cp.asarray(test)])
        ).reshape(-1)
        print(f"fold {fold}/5 complete", flush=True)
        del model
        cp.get_default_memory_pool().free_all_blocks()

    model = SVC(C=2.0, kernel="rbf", gamma="scale", cache_size=8192,
                output_type="cupy")
    model.fit(Xg, yg)
    score = as_numpy(model.decision_function(cp.asarray(grid)))
    return score.reshape(-1).astype(np.float32), oof, str(model.gamma)


def cpu_scores(X: np.ndarray, y: np.ndarray, grid: np.ndarray,
               n_jobs: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the same fit with scikit-learn for smaller scopes or local use."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.svm import SVC

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model = SVC(C=2.0, kernel="rbf", gamma="scale")
    oof = cross_val_predict(
        model, X, y, cv=folds, method="decision_function", n_jobs=n_jobs,
    ).astype(np.float32)
    model.fit(X, y)
    score = model.decision_function(grid).astype(np.float32)
    return score, oof, str(model._gamma)


def fit(scope: str, backend: str = "gpu", n_jobs: int = 5) -> Path:
    with np.load(ARTIFACTS / f"{scope}.npz") as data:
        X = np.asarray(data["X2"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int32)
        grid = np.asarray(data["grid"], dtype=np.float32)

    scorer = gpu_scores if backend == "gpu" else cpu_scores
    args = (X, y, grid) if backend == "gpu" else (X, y, grid, n_jobs)
    grid_score, oof, gamma = scorer(*args)
    auc = float(roc_auc_score(y, oof))

    target = ARTIFACTS / f"{scope}_two_pc_rbf.npz"
    np.savez_compressed(target, score=grid_score, oof_score=oof, cv_auc=auc,
                        C=np.float32(2.0), gamma=np.asarray(gamma))
    print(f"{scope}: wrote {target} (CV AUC {auc:.6f})", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--jobs", type=int, default=5,
                        help="parallel CV jobs for the CPU backend")
    args = parser.parse_args()
    fit(args.scope, args.backend, args.jobs)
