"""Shared layer for the classifier experiment runners: which layers,
methods, and feature sets exist, and turning raw caches into each feature
set's design matrix. Loading (the dataset roster, label conventions,
`load_datasets`) lives in `classifiers/data.py` — import-safe off-GPU,
shared with `analysis/` — and is re-exported here for the runners.
run_per_dataset, run_leave_out, and run_merged all import from here, so a
dataset or feature set added there / here joins every experiment.

Feature sets (each fit on the training data only, label-free; the
returned `transform` maps any other data into the same feature space):

  raw                 the raw activations themselves
  sae / transcoder    per-layer dictionary activations (sparse); features
                      firing on >= MIN_FIRES training samples are kept.
                      Cached at the last prompt token only, so both ignore
                      --raw.
  pca                 PCs of the raw activations explaining 95% of the
                      variance, fit on the training data.
  pca_dict            pca + sae + transcoder side by side, each block
                      preprocessed exactly as when used alone. (The
                      crosscoder is excluded: it is one joint latent over
                      layers 9/17/22/29, not separable per layer.)
  binned_activations  each prompt's raw activation vector as a 100-bin
                      histogram. Bins are equal-width over the training
                      data's 0.5-99.5 percentile range.

Tasks group the datasets by what the model is being asked to judge; the
leave-one-task-out experiment holds out a whole task's datasets at once.
A dataset in no task stands as its own task. The roster, tasks, and file
locations all come from the named data collection (`--data`, see
data.COLLECTIONS); a dataset added to a collection's roster joins every
experiment on that collection automatically.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

from ..data import (COLLECTIONS, load_datasets,  # noqa: F401  (re-exported)
                    task_of)
from ..methods.base import filter_sparse_features

RAW_VARIANTS = ("last", "mean")
FEATURE_SETS = ("raw", "sae", "transcoder", "pca", "pca_dict",
                "binned_activations")
# which cache each feature set is built from ("raw" resolves to --raw)
SOURCES = {
    "raw": ("raw",),
    "sae": ("sae",),
    "transcoder": ("transcoder",),
    "pca": ("raw",),
    "pca_dict": ("raw", "sae", "transcoder"),
    "binned_activations": ("raw",),
}
METHOD_NAMES = ("l1reg", "l2reg", "svm_linear", "svm_rbf", "diff", "lda",
                "qda", "knn", "mlp", "xgboost")

# The pooled experiments (run_leave_out, run_merged) add the methods that
# need a multi-dataset corpus — the two random-slopes variants and moe —
# and for now run only the raw and binned feature sets (more can be
# enabled later — the machinery is the same, this tuple is the only
# gate). xgboost is an ordinary method and sits in METHOD_NAMES above,
# so it runs in all three experiments.
POOLED_METHOD_NAMES = METHOD_NAMES + ("random_slopes",
                                      "random_slopes_task", "moe")
POOLED_FEATURE_SETS = ("raw", "binned_activations")

PCA_VARIANCE = 0.95        # keep the top PCs explaining this much variance
N_BINS = 100               # binned_activations histogram bins
ROBUST_PCT = (0.5, 99.5)   # bin range percentiles

# methods that fit scipy-sparse input directly; the rest get a densified
# (post-cull) matrix
SPARSE_OK = {"l1reg", "l2reg", "xgboost"}


def moe_k_grid(pool_datasets, data: str = "original") -> tuple:
    """MoE expert counts to try for a training pool: the defaults plus the
    pool's number of tasks and number of datasets."""
    pool_datasets = list(pool_datasets)
    n_tasks = len({task_of(d, data) for d in pool_datasets})
    return tuple(sorted({2, 4, n_tasks, len(pool_datasets)}))


def method_groups(method: str, groups: np.ndarray,
                  data: str = "original") -> np.ndarray:
    """The grouping label a pooled method sees for each sample: the
    dataset names as given, mapped to task names for random_slopes_task
    (one deviation per task, fit jointly on every member dataset's
    samples)."""
    if method != "random_slopes_task":
        return groups
    names, inv = np.unique(np.asarray(groups).astype(str),
                           return_inverse=True)
    return np.asarray([task_of(n, data) for n in names])[inv]


def pooled_fit_kwargs(method: str, groups: np.ndarray,
                      data: str = "original") -> dict:
    """Extra fit arguments a method needs on a pooled corpus, given the
    per-sample dataset names. Every other method takes none — this is the
    only place the pooled runners branch on method."""
    if method in ("random_slopes", "random_slopes_task"):
        return {"groups": method_groups(method, groups, data)}
    if method == "moe":
        return {"Ks": moe_k_grid(np.unique(groups), data)}
    return {}


def _histograms(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-row histogram counts, the same bin edges for every row."""
    return np.stack([np.histogram(row, bins=edges)[0]
                     for row in X]).astype(np.float32)


def fit_features(features: str, blocks: dict):
    """One feature set's design matrix for the training data (a role ->
    matrix dict — a single dataset's blocks, or pooled blocks stacked
    across a training pool), a `transform` mapping any other blocks dict
    into the same feature space, and the npz entries describing the fit.
    Everything here is label-free and computed on the training data only."""
    if features == "raw":
        return blocks["raw"], lambda b: b["raw"], {}
    if features in ("sae", "transcoder"):
        X, kept = filter_sparse_features(blocks[features])
        return X, lambda b: b[features][:, kept], {"kept_features": kept}
    if features == "pca":
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
        pca.fit(blocks["raw"])
        return (pca.transform(blocks["raw"]), lambda b: pca.transform(b["raw"]),
                {"n_pcs": pca.n_components_, "pca_mean": pca.mean_,
                 "pca_components": pca.components_})
    if features == "binned_activations":
        lo, hi = np.percentile(blocks["raw"], ROBUST_PCT)
        edges = np.linspace(lo, hi, N_BINS + 1)
        return (_histograms(blocks["raw"], edges),
                lambda b: _histograms(b["raw"], edges), {"bin_edges": edges})
    if features == "pca_dict":
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
        pca.fit(blocks["raw"])
        sae, kept_sae = filter_sparse_features(blocks["sae"])
        tc, kept_tc = filter_sparse_features(blocks["transcoder"])

        def stack(pcs, sae_m, tc_m):
            return sp.hstack([sp.csr_matrix(pcs), sae_m, tc_m], format="csr")

        return (stack(pca.transform(blocks["raw"]), sae, tc),
                lambda b: stack(pca.transform(b["raw"]), b["sae"][:, kept_sae],
                                b["transcoder"][:, kept_tc]),
                {"n_pcs": pca.n_components_, "pca_mean": pca.mean_,
                 "pca_components": pca.components_, "kept_sae": kept_sae,
                 "kept_transcoder": kept_tc})
    raise ValueError(f"unknown feature set {features!r}")


def usable_units(units: dict, features: str) -> list:
    """The loaded units that have every cache block this feature set needs
    (a unit missing one simply sits the feature set out)."""
    return [name for name, u in units.items()
            if all(r in u["blocks"] for r in SOURCES[features])]


def pool_blocks(units: dict, names) -> tuple[dict, np.ndarray, dict]:
    """Stack several units into one training pool. Returns (blocks, groups,
    arrays): the pooled role -> matrix dict (over the roles every named
    unit has), the per-sample dataset name, and the pooled y / sample_id /
    topic arrays (topic "" for the samples of units whose cache has
    none)."""
    names = list(names)
    roles = set.intersection(*(set(units[n]["blocks"]) for n in names))
    blocks = {}
    for role in roles:
        mats = [units[n]["blocks"][role] for n in names]
        blocks[role] = (sp.vstack(mats, format="csr")
                        if sp.issparse(mats[0]) else np.concatenate(mats))
    groups = np.concatenate([np.full(len(units[n]["y"]), n) for n in names])
    arrays = {k: np.concatenate([units[n][k] for n in names])
              for k in ("y", "sample_id")}
    arrays["topic"] = np.concatenate(
        [units[n]["topic"].astype(str) if units[n]["topic"] is not None
         else np.full(len(units[n]["y"]), "") for n in names])
    return blocks, groups, arrays


def completed_groups(csv_path: Path, key_cols) -> set:
    """The resume record: key tuples already present in a metrics csv. A
    group's rows are written all at once, so presence = complete."""
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {tuple(r[c] for c in key_cols) for r in csv.DictReader(f)}
