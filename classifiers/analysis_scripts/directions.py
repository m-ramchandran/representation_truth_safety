"""Saved coefficient vectors as comparable directions in cache units.

Every coefficient module starts here. A fit's `coef` lives in the
standardized feature space of its own preprocessing — each fit has its
own scaler, PCA basis, or kept-feature list — so two fits' vectors
cannot be compared as stored. This module maps each vector back to the
units of the activation cache it was built from, and stacks several such
directions into one aligned matrix.

Cache units per feature set (the space is inferred from the arrays in
the npz — see the README's npz table):

  raw               w = coef / standardize_scale
  pca               w = pca_components.T @ (coef / scale) — exact: the
                    PCA map is linear, so the fit IS a linear model on
                    the raw activations, directly comparable to raw fits
  sae / transcoder  w = coef / scale on the kept dictionary features;
                    `stack_directions` aligns different fits' supports
  diff              stored in raw units already (no scaler), passthrough

Refused, with the reason in the error: binned_activations (each fit's
bin edges come from its own training percentiles, so bin i is a
different feature per fit), pca_dict (one vector spanning three spaces),
svm_rbf (Nystrom space; the map is not persisted), and the methods
without a coefficient (qda / knn / mlp / moe / xgboost). The coef of
both random_slopes variants is their shared weights.

Intercepts — and the standardize/PCA means, which only shift them — are
dropped: directions are compared by angle, and AUC is threshold-free.

Also here, since they rebuild the same saved pipeline: `to_cache_units`
(the row-wise mapping underneath raw_direction, for non-coefficient
vectors like moe gate rows) and `apply_featurization` (the transform
half of scripts' fit_features from the npz arrays, for scoring saved
models on new activations).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data import DEFAULT_COLLECTION, task_of
from .load_results import RESULTS, load_prediction, npz_path

COEF_METHODS = ("l1reg", "l2reg", "svm_linear", "lda", "diff",
                "random_slopes", "random_slopes_task")
# the per-dataset methods with a comparable direction (random_slopes is
# pooled-only)
LINEAR_METHODS = ("l1reg", "l2reg", "svm_linear", "lda", "diff")
# feature sets whose directions live in raw activation space
RAW_SPACE = ("raw", "pca")
# cache roles each feature set is rebuilt from (data.load_datasets roles)
ROLES = {"raw": ("raw",), "sae": ("sae",), "transcoder": ("transcoder",),
         "pca": ("raw",), "binned_activations": ("raw",),
         "pca_dict": ("raw", "sae", "transcoder")}

# every npz key a direction load can need (raw_direction +
# apply_featurization, both prefixes) — load_direction decompresses
# only these instead of the whole archive (the cv/transfer score
# arrays dwarf everything else)
DIRECTION_KEYS = frozenset(
    p + k for p in ("", "full_")
    for k in ("coef", "standardize_scale", "n_pcs", "pca_components",
              "pca_mean", "kept_features", "bin_edges", "kept_sae",
              "kept_transcoder"))


def load_units(layer: int, raw: str, roles=("raw",),
               data: str = DEFAULT_COLLECTION):
    """Activation caches per dataset — THE patchable seam: tests and
    harnesses substitute synthetic units by monkeypatching this one
    name, and modules call it as directions.load_units(...)."""
    from ..data import load_datasets
    return load_datasets(layer, raw, roles, data)


def unit_rows(W: np.ndarray) -> np.ndarray:
    """Rows scaled to unit norm."""
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine of two dense vectors."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pair_cos(a, b) -> float:
    """Cosine of two (w, idx) directions, aligned pairwise."""
    W, _ = stack_directions([a, b])
    return cosine(W[0], W[1])


def leave_out_name(dataset: str, kind: str, data: str = DEFAULT_COLLECTION) -> str:
    """The leave_out npz dataset key for a held-out dataset under a
    holdout kind — the one spelling of the runner's naming convention
    ("<kind>_<holdout>", task kind resolving through task_of)."""
    holdout = dataset if kind == "dataset" else task_of(dataset, data)
    return f"{kind}_{holdout}"


def to_cache_units(W, d: dict,
                   prefix: str = "") -> tuple[np.ndarray, np.ndarray | None]:
    """Rows of W — vectors in the fit's standardized model space
    (coefficients, moe gate rows, ...) — mapped to cache units. Returns
    (mapped rows, feature indices): indices are None for a dense space
    (raw, or pca back-projected to raw) and the kept dictionary indices
    for sae / transcoder. `prefix` is "" or "full_" (merged's
    whole-corpus fit; featurization arrays are stored unprefixed, so
    those fall back). Raises for the spaces with no cache mapping."""
    get = lambda key: d.get(prefix + key, d.get(key))
    if get("bin_edges") is not None:
        raise ValueError("binned_activations directions are not "
                         "comparable: each fit's bins are different "
                         "features")
    if get("kept_sae") is not None:
        raise ValueError("pca_dict spans pca + sae + transcoder; "
                         "no single cache space")
    W = np.atleast_2d(np.asarray(W, dtype=float))
    scale = get("standardize_scale")
    if scale is not None:                        # absent for diff (raw input)
        W = W / np.asarray(scale, dtype=float)
    if get("n_pcs") is not None:                 # pca -> raw space
        return W @ np.asarray(get("pca_components"), dtype=float), None
    kept = get("kept_features")
    return W, (None if kept is None else np.asarray(kept))


def raw_direction(d: dict, method: str,
                  prefix: str = "") -> tuple[np.ndarray, np.ndarray | None]:
    """One loaded prediction npz -> (direction in cache units, feature
    indices), via to_cache_units."""
    if method == "svm_rbf":
        raise ValueError("svm_rbf's coef lives in its Nystrom feature "
                         "space and cannot be mapped to cache units")
    if method not in COEF_METHODS:
        raise ValueError(f"{method} has no coefficient vector")
    if prefix + "coef" not in d:
        raise KeyError(f"no '{prefix}coef' in the npz (runs before "
                       "2026-07-20 lack coefficients)")
    W, idx = to_cache_units(np.asarray(d[prefix + "coef"]).ravel(), d,
                            prefix)
    return W[0], idx


def apply_featurization(d: dict, tag: str, blocks: dict,
                        prefix: str = ""):
    """One unit's cache blocks through the fit's saved featurization —
    the transform half of scripts' fit_features rebuilt from the npz
    arrays — giving the matrix the fit's scaler and model actually saw.
    `blocks` maps role -> matrix as in data.load_datasets; the roles a
    tag needs are ROLES[features]."""
    features = tag.rsplit("_", 1)[0]
    get = lambda key: d.get(prefix + key, d.get(key))
    if features == "raw":
        return blocks["raw"]
    if features in ("sae", "transcoder"):
        return blocks[features][:, get("kept_features")]
    if features == "pca":
        return ((np.asarray(blocks["raw"]) - get("pca_mean"))
                @ np.asarray(get("pca_components")).T)
    if features == "binned_activations":
        edges = np.asarray(get("bin_edges"))
        return np.stack([np.histogram(row, bins=edges)[0]
                         for row in np.asarray(blocks["raw"])]
                        ).astype(np.float32)
    if features == "pca_dict":
        import scipy.sparse as sp
        pcs = ((np.asarray(blocks["raw"]) - get("pca_mean"))
               @ np.asarray(get("pca_components")).T)
        return sp.hstack([sp.csr_matrix(pcs),
                          blocks["sae"][:, get("kept_sae")],
                          blocks["transcoder"][:, get("kept_transcoder")]],
                         format="csr")
    raise ValueError(f"unknown feature set in tag {tag!r}")


def load_direction(experiment: str, tag: str, layer: int, method: str,
                   dataset: str = None,
                   root: Path = RESULTS) -> tuple[np.ndarray,
                                                  np.ndarray | None]:
    """The direction of one saved fit, by run coordinates. per_dataset:
    `dataset` is the training dataset (the full training-dataset fit);
    leave_out: "<kind>_<holdout>" as in the npz filenames (the pool
    fit); merged: no dataset (the whole-corpus `full_` fit)."""
    d = load_prediction(npz_path(experiment, tag, layer, method, dataset,
                                 root), keys=DIRECTION_KEYS)
    return raw_direction(d, method,
                         prefix="full_" if experiment == "merged" else "")


def stack_directions(directions) -> tuple[np.ndarray, np.ndarray | None]:
    """A sequence of (w, idx) pairs from the same (feature set, layer) ->
    an (n_fits, d) matrix in one shared column space, plus that space's
    dictionary indices (None when dense). Dense directions must agree on
    length; sparse ones are scattered onto the union of their supports
    (a feature one fit never saw gets coefficient 0 there, which is what
    absence means for the kept-feature filter)."""
    ws = [np.asarray(w, dtype=float) for w, _ in directions]
    idxs = [idx for _, idx in directions]
    if all(idx is None for idx in idxs):
        dims = {len(w) for w in ws}
        if len(dims) > 1:
            raise ValueError(f"dense directions of different lengths: {dims}")
        return np.vstack(ws), None
    if any(idx is None for idx in idxs):
        raise ValueError("cannot mix dense and dictionary-indexed "
                         "directions")
    union = np.unique(np.concatenate(idxs))
    W = np.zeros((len(ws), len(union)))
    for row, (w, idx) in enumerate(zip(ws, idxs)):
        W[row, np.searchsorted(union, idx)] = w
    return W, union
