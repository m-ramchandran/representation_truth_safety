"""Shared plumbing for the three experiment runners (run_per_dataset,
run_leave_out, run_merged): the command-line arguments they have in
common, the metrics-csv append, the guarded call that skips
data-dependent failures, sparse -> dense conversion, the nested-CV outer
loop, packing per-dataset scores into npz entries, and the AUC heatmap
drawing. What the datasets and feature sets *are* lives in features.py;
this module is only about running and reporting.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from ..fig_style import FS, install_rcparams
from ..methods.base import make_nested_folds
from .features import COLLECTIONS, FEATURE_SETS, RAW_VARIANTS

# Two data-dependent failures where the right move is to skip the unit so
# the remaining ones still run (absent csv rows mark it): qda on the
# densified sparse feature sets, whose class covariances can be
# rank-deficient (or break LAPACK outright), and the GPU running out of
# memory on the widest feature sets. Anything else stays uncaught — a
# crash beats a silent gap when the cause is a bug, not the data.
SKIP_ERRORS = (np.linalg.LinAlgError, cp.cuda.memory.OutOfMemoryError)


def release_gpu_memory() -> None:
    """Return CuPy's cached device blocks to the driver.

    cuML allocates through CuPy's default memory pool, which keeps freed
    blocks cached for reuse instead of handing them back — the right
    default inside one library, and invisible for as long as every fit in
    a process goes through it. XGBoost does not: it allocates through its
    own CUDA calls and sees only what the pool has not claimed. On the
    data_unwrap pooled corpus that meant

        Memory allocation error on worker 0: cudaErrorMemoryAllocation
        - Free memory: 2.6059GB   - Requested memory: 4.93503GB

    on a 32 GB V100 — ~29 GB sat in the pool, unused, after the cuML
    methods that ran earlier in the same task. The fits themselves were
    fine; each task simply died four hours in, at its last method.
    XGBoostError is not in SKIP_ERRORS (it is not a cupy OOM and most of
    what it covers is a real bug), so the whole task failed rather than
    skipping the method.

    Called between methods by all three runners. The cost is that the
    next cuML fit re-allocates from the driver instead of reusing a
    cached block; the benefit is that whichever library asks next sees
    the whole card.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# --- command line -------------------------------------------------------------

def add_shared_args(ap, *, method_names, default_features) -> None:
    """The arguments all three runners take. Runner-specific ones
    (--holdout, --datasets) are added by the runner itself."""
    ap.add_argument("--data", default="original", choices=sorted(COLLECTIONS),
                    help="which data collection to run on; results are "
                         "kept apart under results/<data>/")
    ap.add_argument("--raw", required=True, choices=RAW_VARIANTS,
                    help="raw activations the raw-built feature sets "
                         "(raw / pca / pca_dict / binned_activations) use")
    ap.add_argument("--features", nargs="+", default=list(default_features),
                    choices=FEATURE_SETS)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Scope layers (default: the collection's layers)")
    ap.add_argument("--methods", nargs="+", default=list(method_names),
                    choices=method_names)
    ap.add_argument("--skip", nargs="*", default=[], metavar="METHOD:FEATURES",
                    help="method / feature-set pairs to leave out, e.g. "
                         "--skip qda:sae qda:transcoder")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="DATASET",
                    help="datasets to drop from the run entirely — never "
                         "trained on, never evaluated (e.g. --exclude "
                         "claude_blood_test_ranges)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="cores for the CPU-fit methods' tuning grids "
                         "(diff/lda/qda); the GPU methods run theirs "
                         "sequentially on the device")


def parse_exclude(ap, names, collection) -> set:
    """--exclude names -> a set, validated against the collection's roster;
    runners filter the loaded units with it, so an excluded dataset is
    never trained on, never evaluated, and never plotted."""
    if unknown := set(names) - set(collection.datasets):
        ap.error(f"--exclude {sorted(unknown)}: not in the collection "
                 f"(datasets: {list(collection.datasets)})")
    return set(names)


def parse_skip(ap, tokens, method_names) -> set:
    """--skip METHOD:FEATURES tokens -> a set of (method, features) pairs;
    unknown names are an argparse error."""
    skip = set()
    for token in tokens:
        method, _, features = token.partition(":")
        if method not in method_names or features not in FEATURE_SETS:
            ap.error(f"--skip {token}: expected <method>:<feature set> "
                     f"with names from --methods / --features")
        skip.add((method, features))
    return skip


# --- running ------------------------------------------------------------------

def to_dense(X):
    """Sparse -> dense ndarray (identity on dense input). Runners call this
    on the feature matrix for methods not in SPARSE_OK."""
    return np.asarray(X.todense()) if sp.issparse(X) else X


def nested_cv(fit, X, y, *, seed: int, n_jobs: int, stratify_on=None,
              fold_kwargs=None):
    """The nested-CV outer loop shared by the per-dataset diagonal and the
    merged run: on each outer fold, `fit` tunes on the fold's inner folds
    and predicts the outer test part. `fold_kwargs(outer_tr, outer_te)`,
    if given, supplies extra per-fold fit arguments (the pooled methods'
    groups). Returns (cv_scores, cv_fold, fold_aucs, fold_params):
    per-sample score arrays assembled over the outer folds, each sample's
    outer fold index, and the per-fold AUCs and chosen hyperparameters."""
    cv = {}                      # "proba" / "margin" -> (n,) over outer folds
    cv_fold = np.empty(len(y), dtype=int)
    fold_aucs, fold_params = [], []
    for f, (outer_tr, outer_te, inner_folds) in enumerate(
            make_nested_folds(X, y, seed=seed, stratify_on=stratify_on)):
        kwargs = fold_kwargs(outer_tr, outer_te) if fold_kwargs else {}
        _, scores, params, _ = fit(X[outer_tr], y[outer_tr], X[outer_te],
                                   seed=seed, folds=inner_folds,
                                   inner_n_jobs=n_jobs, **kwargs)
        for key, s in scores.items():
            cv.setdefault(key, np.empty(len(y)))[outer_te] = s
        cv_fold[outer_te] = f
        fold_aucs.append(roc_auc_score(y[outer_te], scores["proba"]))
        fold_params.append(params)
    return cv, cv_fold, fold_aucs, fold_params


def scores_npz(parts, prefix: str) -> dict:
    """npz entries for scores on several evaluation datasets. `parts` is a
    list of (dataset_name, unit_dict, scores_dict); the entries are
    {prefix}_dataset / _sample_id / _y plus one {prefix}_<key> per score
    key, each concatenated over the parts."""
    out = {
        f"{prefix}_dataset": np.concatenate(
            [np.full(len(u["y"]), name) for name, u, _ in parts]),
        f"{prefix}_sample_id": np.concatenate(
            [u["sample_id"] for _, u, _ in parts]),
        f"{prefix}_y": np.concatenate([u["y"] for _, u, _ in parts]),
    }
    if any(u.get("topic") is not None for _, u, _ in parts):
        out[f"{prefix}_topic"] = np.concatenate([
            (u["topic"].astype(str) if u.get("topic") is not None
             else np.full(len(u["y"]), ""))
            for _, u, _ in parts
        ])
    for key in parts[0][2]:      # every part has the method's score keys
        out[f"{prefix}_{key}"] = np.concatenate([s[key] for _, _, s in parts])
    return out


def model_npz(fitted, prefix: str = "") -> dict:
    """npz entries describing the fitted model — its coefficient vector,
    the standardization fit in front of it, and any method-specific
    arrays (see FittedClassifier.save_arrays for the full key list and
    which space each vector lives in). `prefix` distinguishes the merged
    run's whole-corpus fit ("full_", giving full_coef etc.)."""
    return {prefix + k: v for k, v in fitted.save_arrays().items()}


def append_metrics(csv_path: Path, columns, rows) -> None:
    """Append metric rows, writing the header first when the file is new.
    (One csv per (tag, layer) run means parallel jobs never share a
    file.)"""
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        w.writerows(rows)


# --- plotting -----------------------------------------------------------------
# The AUC figures live in classifiers.auc_figures (import-safe, so
# analysis.redraw can rebuild them locally from the csvs);
# re-exported here for the runners.

from ..auc_figures import (  # noqa: E402,F401
    dataset_order, draw_auc_matrix, plot_heatmaps, plot_method_by_dataset,
    read_metric_rows,
)
