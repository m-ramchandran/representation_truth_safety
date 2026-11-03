"""Shared plumbing for the three experiment runners (run_per_dataset,
run_leave_out, run_merged): the command-line arguments they have in
common, the metrics-csv append, the guarded call that skips
data-dependent failures, sparse -> dense conversion, the nested-CV outer
loop, packing per-dataset scores into npz entries, and the AUC heatmap
drawing. What the datasets and feature sets *are* lives in features.py;
this module is only about running and reporting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cupy as cp
import matplotlib
import torch
matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from ..fig_style import FS, install_rcparams
from .base import make_nested_folds
from .features import (COLLECTIONS, DEFAULT_COLLECTION, FEATURE_SETS,
                       RAW_VARIANTS)

# Data-dependent failures where the right move is to skip the unit so the
# remaining ones still run (absent csv rows mark it): qda on the densified
# sparse feature sets, whose class covariances can be rank-deficient (or
# break LAPACK outright), and the GPU running out of memory on the widest
# feature sets. Anything else stays uncaught — a crash beats a silent gap
# when the cause is a bug, not the data.
SKIP_ERRORS = (np.linalg.LinAlgError, cp.cuda.memory.OutOfMemoryError)


def is_gpu_oom(exc: BaseException) -> bool:
    """Whether an exception is a GPU out-of-memory, whatever raised it.

    XGBoost reports OOM as a plain XGBoostError with the condition only in
    the message, so the class cannot go in SKIP_ERRORS — most of what it
    covers is a real bug, and swallowing those would turn a crash into a
    silent gap. Match the message instead.
    """
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return ("xgboosterror" in text or "runtimeerror" in text) and any(
        m in text for m in ("out of memory", "bad_alloc",
                            "cudaerrormemoryallocation"))


def release_gpu_memory() -> None:
    """Hand cached device blocks back to the driver, for every allocator
    that caches: CuPy's pool (what cuML fits through) and torch's.

    Neither returns freed blocks on its own, and XGBoost allocates through
    its own CUDA calls, so it sees only what the others have not claimed —
    it has run out of memory on a card that was 95% idle, twice. Called
    before every fit. Costs a re-allocation per fit; buys each library the
    whole card. A new GPU library means a new line here.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# --- command line -------------------------------------------------------------

def add_shared_args(ap, *, method_names, default_features) -> None:
    """The arguments all three runners take. Runner-specific ones
    (--holdout, --datasets) are added by the runner itself."""
    ap.add_argument("--data", default=DEFAULT_COLLECTION, choices=sorted(COLLECTIONS),
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
    ap.add_argument("--group-folds", dest="group_folds",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="keep every row expanded from one source item in "
                         "the same CV fold (data.GROUP_RULES: Social IQa's "
                         "answer choices, geometry_truth's negation pairs, "
                         "PubMedQA's abstracts). On by default. "
                         "--no-group-folds restores the pre-2026-08 split, "
                         "which is what the results already on disk used")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="cores for the CPU-fit methods' tuning grids "
                         "(diff/lda/qda); the GPU methods run theirs "
                         "sequentially on the device")


def warn_fold_policy(results: Path, group_folds: bool) -> None:
    """Say which fold policy is in force, and shout if results already on
    disk were produced under the other one.

    The resume record (`completed_groups`) keys on features/layer/method/
    dataset — it has no notion of the fold policy, so a rerun in a
    directory written before --group-folds existed keeps the old rows and
    appends new ones beside them. Those are not comparable numbers. Move
    or delete the old tables/ and predictions/ first."""
    policy = "group-aware (--group-folds)" if group_folds else "ungrouped"
    print(f"CV folds: {policy}")
    if not group_folds:
        return
    stale = sorted((results / "tables").glob("metrics_*.csv"))
    if stale:
        print(f"  WARNING: {len(stale)} metrics csv(s) already in "
              f"{results / 'tables'} were written with ungrouped folds. "
              f"Resume will keep those rows and add group-aware ones next "
              f"to them. Move the tables/ and predictions/ directories "
              f"aside for a clean group-aware run.", flush=True)


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
              fold_groups=None, fold_kwargs=None):
    """The nested-CV outer loop shared by the per-dataset diagonal and the
    merged run: on each outer fold, `fit` tunes on the fold's inner folds
    and predicts the outer test part. `fold_kwargs(outer_tr, outer_te)`,
    if given, supplies extra per-fold fit arguments (the pooled methods'
    groups). `fold_groups` is the CV fold-group id per sample (data.group_ids
    — rows expanded from one source item), which keeps each group whole in
    both the outer and the inner splits; None splits by label alone, the
    behaviour before --group-folds existed. Returns (cv_scores, cv_fold,
    fold_aucs, fold_params): per-sample score arrays assembled over the
    outer folds, each sample's outer fold index, and the per-fold AUCs and
    chosen hyperparameters."""
    cv = {}                      # "proba" / "margin" -> (n,) over outer folds
    cv_fold = np.empty(len(y), dtype=int)
    fold_aucs, fold_params = [], []
    for f, (outer_tr, outer_te, inner_folds) in enumerate(
            make_nested_folds(X, y, seed=seed, stratify_on=stratify_on,
                              groups=fold_groups)):
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
    {prefix}_dataset / _sample_id / _y / _group plus one {prefix}_<key>
    per score key, each concatenated over the parts.

    An empty `parts` contributes nothing rather than raising: a
    pool_excluded unit is fitted for its own diagonal and evaluated on
    nothing else, so its npz has no transfer_* arrays."""
    if not parts:
        return {}
    out = {
        f"{prefix}_dataset": np.concatenate(
            [np.full(len(u["y"]), name) for name, u, _ in parts]),
        f"{prefix}_sample_id": np.concatenate(
            [u["sample_id"] for _, u, _ in parts]),
        f"{prefix}_y": np.concatenate([u["y"] for _, u, _ in parts]),
    }
    # Saved on evaluation rows, not just training ones: wildjailbreak's
    # two halves are separate units sharing one source_prompt_id per pair,
    # so a vanilla-trained probe scored on adversarial sees rewrites of
    # its own training prompts. This column makes that measurable later.
    if any(u.get("group") is not None for _, u, _ in parts):
        out[f"{prefix}_group"] = np.concatenate([
            (np.asarray(u["group"]).astype(str) if u.get("group") is not None
             else np.full(len(u["y"]), ""))
            for _, u, _ in parts
        ])
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
    (One csv per (tag, layer) run means parallel Slurm tasks never share
    a file.)"""
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        w.writerows(rows)


# --- plotting -----------------------------------------------------------------
# The AUC figures live in classifiers.auc_figures (import-safe, so
# analysis_scripts.redraw can rebuild them locally from the csvs);
# re-exported here for the runners.

from ..auc_figures import (  # noqa: E402,F401
    dataset_order, draw_auc_matrix, plot_heatmaps, plot_method_by_dataset,
    read_metric_rows,
)
