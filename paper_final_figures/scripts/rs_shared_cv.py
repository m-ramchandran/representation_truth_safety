"""Run ON compute cluster (GPU node): cross-validated AUC of the pooled random-slopes
SHARED direction on its own, for the expanded pool.

The expanded-pool merged run kept each outer fold's out-of-fold predictions
and its chosen hyperparameters (npz keys cv_fold, params) but not the fold
models, so the shared-only score has to be re-fit. For each of the five
outer folds the random-slopes model is trained again on that fold's
training rows at the fold's stored (shared_l2, dev_l2) -- no inner tuning
-- and the held-out rows are scored two ways: with the shared weights only,
and with shared + the row's own dataset deviation. The second is the stored
prediction re-made, and is written next to the stored out-of-fold margin as
a check that the refit reproduces the run. Nothing under
results/<base>_expanded_pool/ is written to.

    python paper_final_figures/scripts/rs_shared_cv.py <base> <layer>

from the repository root (or with REPO_ROOT set). Writes
classifiers/results/_rs_shared_cv/<base>_expanded_pool_layer<L>.csv: one row
per (fold, dataset) and, with fold = "all", per dataset over the pooled
out-of-fold scores; columns n, auc_shared, auc_full, auc_stored.
"""
from __future__ import annotations

import ast
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("REPO_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(ROOT))
import classifiers.data as data                                  # noqa: E402
from classifiers.scripts import fit_torch as ft                  # noqa: E402
from classifiers.scripts.features import (                       # noqa: E402
    SOURCES, fit_features, load_datasets, pool_blocks, usable_units)
from classifiers.scripts.runner_common import to_dense           # noqa: E402

RESULTS = ROOT / "classifiers" / "results"
OUT = RESULTS / "_rs_shared_cv"
SEED = 0        # run_merged's --seed default, used by the expanded-pool runs


def register(base: str) -> str:
    """The derived collection run_merged_expanded_pool.py fitted: the base
    collection with only Blood ranges kept out of the pool."""
    name = base + "_expanded_pool"
    if name not in data.COLLECTIONS:
        data.COLLECTIONS[name] = dataclasses.replace(
            data.COLLECTIONS[base],
            pool_excluded=("claude_blood_test_ranges",))
    return name


def load_pool(name: str, layer: int):
    """The pooled corpus exactly as run_merged.run_pooled builds it for the
    'raw' feature set: (X dense, y, dataset per row, sample id per row)."""
    roles = sorted(SOURCES["raw"])
    units = load_datasets(layer, "last", roles, name, groups=True)
    c = data.COLLECTIONS[name]
    usable = usable_units(units, "raw")
    pool = [n for n in usable if n in c.pool_datasets]
    blocks, groups, arrays = pool_blocks(units, pool)
    X, _, _ = fit_features("raw", blocks)
    return (to_dense(X), np.asarray(arrays["y"]).astype(int),
            np.asarray(groups).astype(str),
            np.asarray(arrays["sample_id"]).astype(str))


def score_fold(X, y, groups, tr, te, value, seed):
    """Refit one outer fold as fit_torch.fit_random_slopes would for the
    winning candidate, then score the held-out rows shared-only and with
    deviations. Returns (shared margins, full margins)."""
    group_names, codes = np.unique(groups[tr], return_inverse=True)
    g = torch.as_tensor(codes.astype(np.int64), device=ft.DEVICE)
    Xt = ft._as_tensor(X[tr])

    def fit_candidate(value, Xtr, ytr, gtr, cand_seed):
        lam_shared, lam_dev = value
        torch.manual_seed(cand_seed)
        model = ft._RandomSlopes(Xtr.shape[1], len(group_names), lam_shared,
                                 lam_dev).to(ft.DEVICE)
        return ft._train(model, Xtr, ytr, gtr, seed=cand_seed)

    model, mean, scale = ft._fit_final(fit_candidate, value, Xt, y[tr], g, seed)
    fitted = ft.RandomSlopesClassifier(model, mean, scale, group_names)
    shared = fitted.predict(X[te])["margin"]
    full = fitted.predict_with_groups(X[te], groups[te])["margin"]
    del Xt, model, fitted
    torch.cuda.empty_cache()
    return shared, full


def auc_rows(y, groups, idx, shared, full, stored, tag, extra):
    rows = []
    for d in np.unique(groups[idx]):
        m = groups[idx] == d
        yy = y[idx][m]
        if len(np.unique(yy)) < 2:
            continue
        rows.append({**extra, "fold": tag, "dataset": d, "n": int(m.sum()),
                     "auc_shared": roc_auc_score(yy, shared[m]),
                     "auc_full": roc_auc_score(yy, full[m]),
                     "auc_stored": roc_auc_score(yy, stored[m])})
    return rows


def main() -> None:
    base, layer = sys.argv[1], int(sys.argv[2])
    name = register(base)
    npz_path = (RESULTS / name / "merged" / "predictions"
                / f"raw_last_layer{layer}_random_slopes.npz")
    npz = np.load(npz_path, allow_pickle=True)
    t0 = time.time()
    X, y, groups, sid = load_pool(name, layer)
    print(f"{name} layer {layer}: pool {X.shape}, {len(np.unique(groups))} "
          f"datasets, loaded in {time.time() - t0:.0f}s", flush=True)

    # line the corpus up with the stored run, row for row
    stored_id = np.asarray(npz["cv_sample_id"]).astype(str)
    if not np.array_equal(stored_id, sid):
        pos = {s: i for i, s in enumerate(sid)}
        idx = np.array([pos[s] for s in stored_id])
        X, y, groups, sid = X[idx], y[idx], groups[idx], sid[idx]
        print("  rows re-ordered to the stored sample order", flush=True)
    assert np.array_equal(np.asarray(npz["cv_dataset"]).astype(str), groups)
    assert np.array_equal(np.asarray(npz["cv_y"]).astype(int), y)
    fold = np.asarray(npz["cv_fold"]).astype(int)
    params = ast.literal_eval(str(npz["params"]))
    stored = np.asarray(npz["cv_margin"], dtype=float)
    extra = {"collection": name, "layer": layer}

    shared_oof = np.full(len(y), np.nan)
    full_oof = np.full(len(y), np.nan)
    rows = []
    for k in range(int(fold.max()) + 1):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        value = (float(params[k]["shared_l2"]), float(params[k]["dev_l2"]))
        t0 = time.time()
        shared, full = score_fold(X, y, groups, tr, te, value, SEED)
        shared_oof[te], full_oof[te] = shared, full
        rows += auc_rows(y, groups, te, shared, full, stored[te], k,
                         {**extra, "shared_l2": value[0], "dev_l2": value[1]})
        print(f"  fold {k}: n_train {len(tr)}, params {value}, "
              f"AUC shared {roc_auc_score(y[te], shared):.3f} / full "
              f"{roc_auc_score(y[te], full):.3f} / stored "
              f"{roc_auc_score(y[te], stored[te]):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    assert not np.isnan(shared_oof).any()
    every = np.arange(len(y))
    rows += auc_rows(y, groups, every, shared_oof, full_oof, stored, "all",
                     {**extra, "shared_l2": np.nan, "dev_l2": np.nan})

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}_layer{layer}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    pooled = frame[frame.fold == "all"].set_index("dataset")
    print(f"\n{name} layer {layer}, out-of-fold AUC per dataset:")
    print(pooled[["n", "auc_shared", "auc_full", "auc_stored"]]
          .round(3).to_string())
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
