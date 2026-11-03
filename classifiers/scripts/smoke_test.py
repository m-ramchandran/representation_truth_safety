"""GPU smoke test: everything the Mac cannot verify, on synthetic data.

Run on a GPU node from the repo root (~10 minutes, no HF downloads, all
output under a temp directory that is printed and left for inspection):

    python -m classifiers.scripts.smoke_test

What it checks:
  1. environment — cupy / cuml / torch import, torch sees the GPU;
  2. every method through the fit contract on synthetic dense data
     (score-dict keys, margin <-> probability consistency, CV and test
     AUCs, fitted.predict reproducing the returned test scores, the
     get_coef / save_arrays persistence contract), plus l1reg / l2reg on
     sparse CSR input;
  3. the Nystrom map: reconstructs the exact RBF kernel when m = n, and
     svm_rbf still works with NYSTROM_M shrunk so the landmark-sampling
     path runs;
  4. the three runners end to end with load_datasets replaced by nine
     small synthetic datasets named like the real ones (so the
     collection's task map and unit roles apply): csvs / npz / figures
     appear, AUCs parse, and a re-run skips completed work. Then the
     pool_excluded contract, which is what makes the three held-out sets
     held out — per_dataset trains one for its own diagonal but writes NO
     transfer row, while it stays a test column for the others; no pooled
     run ever puts one in cv_dataset; leave_out gives it no dataset-mode
     holdout of its own (that pool would equal the merged one) yet scores
     it against EVERY holdout in both modes; merged scores it with the
     whole-corpus fit as kind eval_only. Plus: --exclude drops a dataset
     from pools, holdouts and evaluation everywhere, the pooled npz carry
     cv_topic, and every npz carries cv_group (the fold-group id, which
     doubles as the wildjailbreak vanilla/adversarial pairing key).

Grids are shrunk for speed everywhere — the full default grids are
exercised by the real experiments, not here.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)


# --- 1. environment ----------------------------------------------------------

import cupy as cp
import cuml
import torch

from . import METHODS
from . import fit_classifiers as fc

print(f"cupy {cp.__version__}, cuml {cuml.__version__}, "
      f"torch {torch.__version__}")
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(f"torch device: {torch.cuda.get_device_name(0)}")


# --- 2. every method through the fit contract --------------------------------

def make_dense(n: int, d: int, seed: int):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)).astype(np.float32)
    w = np.zeros(d)
    w[: d // 4] = 2.0
    y = (X @ w + r.logistic(size=n) > 0).astype(int)
    return X, y


X, y = make_dense(2500, 40, seed=1)
X_test, y_test = make_dense(600, 40, seed=2)
groups = np.array(["alpha", "beta"])[np.arange(len(y)) % 2]

# small grids for speed; same code paths as the defaults
FIT_KWARGS = {
    "l1reg": dict(Cs=np.logspace(-2, 1, 4)),
    "l2reg": dict(Cs=np.logspace(-2, 2, 5)),
    "svm_linear": dict(Cs=(0.01, 0.1, 1.0)),
    "svm_rbf": dict(Cs=(0.1, 1.0, 10.0)),
    "knn": dict(ks=(5, 25)),
    "diff": {}, "lda": {}, "qda": {},
    "mlp": dict(hiddens=(16,), weight_decays=(1e-3,)),
    "xgboost": dict(depths=(3,), learning_rates=(0.3,), n_rounds=200),
    "random_slopes": dict(groups=groups, shared_l2=(1e-5, 1e-3),
                          dev_multipliers=(10.0,)),
    # same fit function as random_slopes; here the contract just runs it
    # under its registered name (the task mapping is a runner concern,
    # exercised in section 4)
    "random_slopes_task": dict(groups=groups, shared_l2=(1e-5,),
                               dev_multipliers=(10.0,)),
    "moe": dict(Ks=(2,)),
    "moe_hard": dict(Ks=(2,)),
}
NATIVE_LOGODDS = {"l1reg", "l2reg", "lda", "qda", "mlp", "xgboost",
                  "random_slopes", "random_slopes_task", "moe",
                  "moe_hard"}
PLATT = {"svm_linear", "svm_rbf", "diff"}
# get_coef contract: a (d,) vector in the input-feature space for the
# linear methods, None for the rest. (svm_rbf is neither — its vector
# lives in the Nystrom feature space — so it is exempt from both checks.)
HAS_COEF = {"l1reg", "l2reg", "svm_linear", "diff", "lda", "random_slopes",
            "random_slopes_task"}
NO_COEF = {"qda", "knn", "mlp", "xgboost", "moe", "moe_hard"}

print("\nper-method contract checks (n=2500, d=40):")
for name, fit in METHODS.items():
    cv, test, params, fitted = fit(X, y, X_test, seed=0, **FIT_KWARGS[name])
    for label, scores, n_exp in (("cv", cv, len(y)),
                                 ("test", test, len(y_test))):
        expected = {"proba"} if name == "knn" else {"proba", "margin"}
        assert set(scores) == expected, (name, label, set(scores))
        assert all(len(s) == n_exp for s in scores.values())
        if name in NATIVE_LOGODDS:
            assert np.allclose(scores["proba"], expit(scores["margin"]),
                               atol=1e-4), f"{name} {label}: proba != " \
                                           "sigmoid(margin)"
        if name in PLATT:
            a, b = params["platt_a"], params["platt_b"]
            assert np.allclose(scores["proba"],
                               expit(a * scores["margin"] + b),
                               atol=1e-4), f"{name} {label}: Platt mismatch"
    auc_cv = roc_auc_score(y, cv["proba"])
    auc_te = roc_auc_score(y_test, test["proba"])
    assert auc_cv > 0.75 and auc_te > 0.75, (name, auc_cv, auc_te)
    again = fitted.predict(X_test)
    assert np.allclose(again["proba"], test["proba"], atol=1e-6), \
        f"{name}: fitted.predict disagrees with returned test scores"
    coef = fitted.get_coef()
    if name in HAS_COEF:
        assert coef is not None and coef.shape == (X.shape[1],), \
            f"{name}: get_coef() should be ({X.shape[1]},), got " \
            f"{None if coef is None else coef.shape}"
    elif name in NO_COEF:
        assert coef is None, f"{name}: get_coef() should be None"
    # save_arrays contract: coef iff the method has one (svm_rbf's is in
    # Nystrom space but still saved); the standardization for every
    # method but diff and xgboost (raw-space input, nothing to save),
    # with the mean present on dense input; the method-specific extras
    # for random_slopes / moe / xgboost.
    arrays = fitted.save_arrays()
    assert ("coef" in arrays) == (name in HAS_COEF or name == "svm_rbf"), \
        (name, sorted(arrays))
    if name == "diff":
        assert set(arrays) == {"coef"}, (name, sorted(arrays))
    elif name == "xgboost":
        assert set(arrays) == {"feature_importance", "booster_ubj"}, \
            (name, sorted(arrays))
        assert arrays["feature_importance"].shape == (X.shape[1],)
        # the saved booster must reproduce the fit's own margins exactly,
        # so a later analysis can score new data without a refit
        import xgboost as _xgb
        _b = _xgb.Booster()
        _b.load_model(bytearray(arrays["booster_ubj"].tobytes()))
        _m = _b.predict(_xgb.DMatrix(X_test), output_margin=True)
        assert np.allclose(_m, test["margin"], atol=1e-6), \
            "xgboost: reloaded booster disagrees with the fit's margins"
    else:
        assert arrays["standardize_scale"].shape == (X.shape[1],)
        assert arrays["standardize_mean"].shape == (X.shape[1],)
    if name in ("random_slopes", "random_slopes_task"):
        assert arrays["dev_w"].shape == (2, X.shape[1])
        assert arrays["dev_b"].shape == (2,)
        assert list(arrays["group_labels"]) == ["alpha", "beta"]
    if name in ("moe", "moe_hard"):
        k = params["K"]
        assert arrays["gate_w"].shape == (k, X.shape[1])
        assert arrays["expert_w"].shape == (k, X.shape[1])
        assert arrays["gate_b"].shape == (k,)
        assert arrays["expert_b"].shape == (k,)
    print(f"  {name:14s} cv AUC {auc_cv:.3f}  test AUC {auc_te:.3f}  "
          f"params {params}")

# sparse CSR through the GPU logistic regressions (variance-only scaling)
Z = rng.exponential(size=(2000, 300)).astype(np.float32) \
    * (rng.random(size=(2000, 300)) < 0.1)
y_sp = rng.integers(0, 2, 2000)
Z[:, :8] += 2.0 * y_sp[:, None]                  # signal, still nonnegative
Z_sp = sp.csr_matrix(Z)
for name in ("l1reg", "l2reg"):
    cv, _, params, fitted = METHODS[name](Z_sp, y_sp, seed=0,
                                          **FIT_KWARGS[name])
    auc = roc_auc_score(y_sp, cv["proba"])
    assert auc > 0.75, (name, "sparse", auc)
    assert fitted.get_coef().shape == (300,)
    assert "standardize_mean" not in fitted.save_arrays()  # variance-only
    print(f"  {name} on sparse CSR: cv AUC {auc:.3f}")

# xgboost is in SPARSE_OK too, so the pooled runners hand it CSR on the
# sae / transcoder feature sets: it must fit, score sparse test input,
# and still carry no coefficient or scaler
cv, test_sp, _, fitted = METHODS["xgboost"](Z_sp, y_sp, Z_sp[:50], seed=0,
                                            **FIT_KWARGS["xgboost"])
auc = roc_auc_score(y_sp, cv["proba"])
assert auc > 0.75, ("xgboost", "sparse", auc)
assert len(test_sp["proba"]) == 50
arrays = fitted.save_arrays()
assert arrays["feature_importance"].shape == (300,)
assert "coef" not in arrays and "standardize_scale" not in arrays
print(f"  xgboost on sparse CSR: cv AUC {auc:.3f}")

# knn on a tiny dataset: the default k grid (up to 100) must clamp to the
# folds' training size instead of crashing cuML (claude_vds, n=126, took
# down the first tf_wrapped_a array this way)
X_t, y_t = make_dense(60, 10, seed=3)
_, _, params, fitted = METHODS["knn"](X_t, y_t, seed=0)
assert params["n_neighbors"] <= 48, params
assert len(fitted.predict(X_t)["proba"]) == 60
print(f"  knn on n=60: k grid clamped, chose k={params['n_neighbors']}")


# --- 3. Nystrom --------------------------------------------------------------

# exact regime (m = n): the map must reproduce the kernel
Xs = cp.asarray(rng.normal(size=(400, 20)).astype(np.float32))
transform, gamma, m = fc._nystrom_map(Xs, seed=0)
err = float(cp.abs(transform(Xs) @ transform(Xs).T
                   - fc._rbf_kernel(Xs, Xs, gamma)).max())
assert m == 400 and err < 1e-2, f"exact Nystrom reconstruction error {err}"
print(f"\nNystrom exact regime: max |Phi Phi^T - K| = {err:.2e}")

# landmark-sampling regime: shrink NYSTROM_M so n > m
saved = fc.NYSTROM_M
fc.NYSTROM_M = 256
try:
    cv, test, params, _ = METHODS["svm_rbf"](X, y, X_test, seed=0,
                                             Cs=(1.0, 10.0))
    auc = roc_auc_score(y_test, test["proba"])
    assert params["n_landmarks"] == 256 and auc > 0.75, (params, auc)
    print(f"Nystrom with 256 landmarks on n=2500: test AUC {auc:.3f}")
finally:
    fc.NYSTROM_M = saved


# --- 4. the runners end to end -----------------------------------------------

from ..data import COLLECTIONS, DEFAULT_COLLECTION
from . import (fit_l2, fit_mlp, fit_moe, fit_moe_hard, fit_random_slopes,
               fit_svm_rbf, fit_xgboost)
from . import run_leave_out, run_merged, run_per_dataset

# Small datasets named like the real ones so the collection's task map and
# role assignments apply unchanged. Six poolable units across the three
# tasks (tf: got + counterfact, medical: pubmedqa_labeled +
# claude_anatomy, jailbreak: wildjailbreak_vanilla + xstest) and all three
# pool_excluded ones, chosen to cover the three cases that behave
# differently:
#   wildjailbreak_adversarial  its task (jailbreak) keeps poolable members
#   claude_blood_test_ranges   likewise (medical), and it is the unit the
#                              random_slopes_task deviation check uses
#   claude_vds                 a singleton task that is ENTIRELY
#                              pool_excluded, so it is never a holdout
POOL_EXCLUDED = ("wildjailbreak_adversarial", "claude_blood_test_ranges",
                 "claude_vds")
SIZES = {"got": 300, "counterfact": 300, "pubmedqa_labeled": 200,
         "claude_anatomy": 150, "wildjailbreak_vanilla": 300, "xstest": 150,
         "wildjailbreak_adversarial": 200, "claude_blood_test_ranges": 120,
         "claude_vds": 100}
D = 30


def fake_load_datasets(layer, raw, roles, data=DEFAULT_COLLECTION,
                       groups=False):
    """`groups=True` gives every unit a two-rows-per-group structure, so the
    runners' --group-folds path is exercised rather than degenerating to one
    group per row."""
    sizes = SIZES
    w = np.zeros(D)
    w[:6] = 2.0                                   # shared signal
    units = {}
    for i, (name, n) in enumerate(sizes.items()):
        r = np.random.default_rng(100 + i)
        Xd = r.normal(size=(n, D)).astype(np.float32)
        dev = np.zeros(D)
        dev[6 + i] = 1.5                          # a dataset-specific feature
        yd = ((Xd @ (w + dev) + r.logistic(size=n)) > 0).astype(int)
        sample_id = np.array([f"{name}:{j}" for j in range(n)])
        units[name] = {
            "y": yd,
            "sample_id": sample_id,
            "topic": np.array(["p1", "p2"])[np.arange(n) % 2],
            "group": (np.array([f"{name}:g{j // 2}" for j in range(n)])
                      if groups else sample_id),
            "blocks": {"raw": Xd},
        }
    return units


FAST_METHODS = {
    "l2reg": partial(fit_l2, Cs=np.logspace(-2, 2, 5)),
    "svm_rbf": partial(fit_svm_rbf, Cs=(0.1, 1.0, 10.0)),
    "mlp": partial(fit_mlp, hiddens=(16,), weight_decays=(1e-3,)),
    "xgboost": partial(fit_xgboost, depths=(3,), learning_rates=(0.3,),
                       n_rounds=200),
    "random_slopes": partial(fit_random_slopes, shared_l2=(1e-5,),
                             dev_multipliers=(10.0,)),
    "random_slopes_task": partial(fit_random_slopes, shared_l2=(1e-5,),
                                  dev_multipliers=(10.0,)),
    "moe": partial(fit_moe, Ks=(2,)),
    "moe_hard": partial(fit_moe_hard, Ks=(2,)),
}

tmp = Path(tempfile.mkdtemp(prefix="classifiers_smoke_"))
print(f"\nrunner outputs -> {tmp}")


def run_runner(module, results_sub, argv, data=DEFAULT_COLLECTION):
    module.load_datasets = fake_load_datasets
    module.METHODS = {**METHODS, **FAST_METHODS}
    module.RESULTS = tmp             # runners write under <RESULTS>/<data>/
    sys.argv = argv
    module.main()
    out = tmp / data / results_sub
    tables = list((out / "tables").glob("*.csv"))
    preds = list((out / "predictions").glob("*.npz"))
    figs = list((out / "figures").glob("*.png"))
    assert tables and preds and figs, (results_sub, tables, preds, figs)
    n_rows = 0
    for t in tables:
        with open(t) as f:
            for row in csv.DictReader(f):
                assert 0.0 <= float(row["auc"]) <= 1.0
                n_rows += 1
    print(f"  {results_sub}: {len(tables)} csv(s), {n_rows} rows, "
          f"{len(preds)} npz, {len(figs)} figure(s)")
    return n_rows


MERGED_ARGV = ["smoke", "--raw", "last", "--layers", "9",
               "--features", "raw",
               "--methods", "l2reg", "svm_rbf", "mlp", "xgboost",
               "random_slopes", "random_slopes_task", "moe", "moe_hard",
               "--n-jobs", "4"]

print("run_per_dataset (2 training datasets, raw + binned + pca):")
run_runner(run_per_dataset, "per_dataset",
           ["smoke", "--raw", "last", "--layers", "9",
            "--features", "raw", "binned_activations", "pca",
            "--methods", "l2reg", "knn", "svm_rbf", "mlp", "xgboost",
            "--datasets", "got", "counterfact", "--n-jobs", "4"])

# the model arrays land in the npz: coef + scaler for a linear method,
# and the PCA basis for the pca feature set
pd_npz = np.load(tmp / DEFAULT_COLLECTION / "per_dataset" / "predictions"
                 / "pca_last_layer9_l2reg_got.npz")
assert {"coef", "standardize_mean", "standardize_scale", "cv_topic",
        "n_pcs", "pca_mean", "pca_components"} <= set(pd_npz.files), \
    sorted(pd_npz.files)
assert pd_npz["pca_components"].shape == (int(pd_npz["n_pcs"]), D)
assert pd_npz["pca_mean"].shape == (D,)

# xgboost in per_dataset (it joined METHOD_NAMES 2026-07-25): model
# arrays are unprefixed here, and it still has no coef and no scaler
xgb_pd = np.load(tmp / DEFAULT_COLLECTION / "per_dataset" / "predictions"
                 / "raw_last_layer9_xgboost_got.npz")
assert {"feature_importance", "booster_ubj", "transfer_proba",
        "cv_proba"} <= set(xgb_pd.files), sorted(xgb_pd.files)
assert "coef" not in xgb_pd.files and "standardize_scale" not in xgb_pd.files
assert xgb_pd["feature_importance"].shape == (D,)

# every prediction npz carries the fold-group id — the pairing key the
# wildjailbreak vanilla/adversarial contamination check needs after the fact
assert "cv_group" in pd_npz.files and len(pd_npz["cv_group"]) == SIZES["got"]
assert "transfer_group" in xgb_pd.files, sorted(xgb_pd.files)

# --- the pool_excluded contract in run_per_dataset ---------------------
# A held-out set IS trained here, for its own diagonal only: one cv row,
# no transfer rows out of it, and it still appears as a test column for
# the datasets that do train normally.
print("run_per_dataset (pool_excluded units: diagonal only, no transfer):")
run_runner(run_per_dataset, "per_dataset",
           ["smoke", "--raw", "last", "--layers", "9", "--features", "raw",
            "--methods", "l2reg", "--datasets", "got", *POOL_EXCLUDED,
            "--n-jobs", "4"])
with open(tmp / DEFAULT_COLLECTION / "per_dataset" / "tables"
          / "metrics_raw_last_layer9.csv") as f:
    pe_rows = [r for r in csv.DictReader(f) if r["method"] == "l2reg"]
for unit in POOL_EXCLUDED:
    mine = [r for r in pe_rows if r["train_dataset"] == unit]
    assert [r["kind"] for r in mine] == ["cv"], \
        f"{unit}: expected exactly one cv row, got {[r['kind'] for r in mine]}"
    assert mine[0]["test_dataset"] == unit
    assert any(r["train_dataset"] == "got" and r["test_dataset"] == unit
               and r["kind"] == "transfer" for r in pe_rows), \
        f"{unit} should still be a test column for an ordinary learner"
    # and the npz genuinely has no transfer arrays — scores_npz returns
    # {} for an empty parts list rather than raising on the concatenate
    pe_npz = np.load(tmp / DEFAULT_COLLECTION / "per_dataset" / "predictions"
                     / f"raw_last_layer9_l2reg_{unit}.npz")
    assert not [k for k in pe_npz.files if k.startswith("transfer_")], \
        f"{unit}: unexpected transfer arrays {sorted(pe_npz.files)}"
    assert {"cv_proba", "cv_y", "cv_group", "coef"} <= set(pe_npz.files)

for mode in ("dataset", "task"):
    print(f"run_leave_out --holdout {mode}:")
    run_runner(run_leave_out, "leave_out",
               ["smoke", "--holdout", mode, "--raw", "last",
                "--layers", "9", "--features", "raw",
                "--methods", "l2reg", "svm_rbf", "mlp", "xgboost",
                "random_slopes", "random_slopes_task", "moe", "moe_hard",
                "--n-jobs", "4"])

# random_slopes_task runs in task mode only:
# no dataset-mode rows or npz, and in task mode its deviations are per
# task (group_labels are the pool's tasks, not its datasets — holding
# out tf leaves medical + jailbreak)
with open(tmp / DEFAULT_COLLECTION / "leave_out" / "tables"
          / "metrics_raw_last_layer9_dataset.csv") as f:
    assert all(r["method"] != "random_slopes_task"
               for r in csv.DictReader(f)), \
        "random_slopes_task should not run in dataset mode"
assert not list((tmp / DEFAULT_COLLECTION / "leave_out" / "predictions")
                .glob("*random_slopes_task_dataset*"))
rst_t = np.load(tmp / DEFAULT_COLLECTION / "leave_out" / "predictions"
                / "raw_last_layer9_random_slopes_task_task_tf.npz")
assert sorted(rst_t["group_labels"].astype(str)) \
    == ["jailbreak", "medical"], sorted(rst_t["group_labels"])
assert "holdout_proba" in rst_t.files

# --- the pool_excluded contract in run_leave_out -----------------------
# None of the three is ever in a pool; none gets a holdout of its own in
# dataset mode (that pool would be the merged pool); and every one is
# scored against EVERY holdout, in both modes, because no pool ever saw
# it. claude_vds additionally must not appear as a task-mode holdout: it
# is a singleton task whose only member is pool_excluded.
_C = COLLECTIONS[DEFAULT_COLLECTION]
assert set(_C.pool_excluded) == set(POOL_EXCLUDED), \
    "the smoke roster no longer matches the collection's held-out sets"
for mode in ("dataset", "task"):
    with open(tmp / DEFAULT_COLLECTION / "leave_out" / "tables"
              / f"metrics_raw_last_layer9_{mode}.csv") as f:
        lo = [r for r in csv.DictReader(f) if r["method"] == "l2reg"]
    holdouts = {r["holdout"] for r in lo}
    for unit in POOL_EXCLUDED:
        assert unit not in holdouts, \
            f"{unit} got its own {mode}-mode holdout; its pool is the " \
            f"merged pool, which already scores it"
        scored = {r["holdout"] for r in lo if r["test_dataset"] == unit}
        assert scored == holdouts, \
            f"{unit} scored on {sorted(scored)}, expected every holdout " \
            f"{sorted(holdouts)}"
    if mode == "task":
        assert "claude_vds" not in holdouts, \
            "a task whose only member is pool_excluded must not be held out"
    # and the pool itself never contains one
    npz = np.load(tmp / DEFAULT_COLLECTION / "leave_out" / "predictions"
                  / f"raw_last_layer9_l2reg_{mode}_{sorted(holdouts)[0]}.npz")
    pooled = set(np.unique(npz["cv_dataset"]).astype(str))
    assert not pooled & set(POOL_EXCLUDED), sorted(pooled)
    assert "cv_group" in npz.files and "holdout_group" in npz.files

print("run_merged:")
n_rows = run_runner(run_merged, "merged", MERGED_ARGV)

# the whole-corpus fit's arrays carry the full_ prefix, including the
# method-specific extras (random_slopes deviations, moe gate/experts)
rs_npz = np.load(tmp / DEFAULT_COLLECTION / "merged" / "predictions"
                 / "raw_last_layer9_random_slopes.npz")
assert {"full_coef", "full_standardize_scale", "full_dev_w", "full_dev_b",
        "full_group_labels"} <= set(rs_npz.files), sorted(rs_npz.files)
# one deviation row per POOLED dataset — the pool_excluded units are not
# in the corpus, so they get no deviation. This is the assertion that
# distinguishes "they were left out" from "they were left out of the
# metrics table but still trained on".
N_POOLED = len(SIZES) - len(POOL_EXCLUDED)
assert rs_npz["full_dev_w"].shape == (N_POOLED, D), \
    f"expected {N_POOLED} deviation rows, got {rs_npz['full_dev_w'].shape}"
assert not (set(rs_npz["full_group_labels"].astype(str)) & set(POOL_EXCLUDED)), \
    "a pool_excluded unit got a random_slopes deviation row"
moe_npz = np.load(tmp / DEFAULT_COLLECTION / "merged" / "predictions"
                  / "raw_last_layer9_moe.npz")
assert {"full_gate_w", "full_gate_b", "full_expert_w",
        "full_expert_b"} <= set(moe_npz.files), sorted(moe_npz.files)
assert "full_coef" not in moe_npz.files      # moe has no coefficient vector

# random_slopes_task on the merged pool: one deviation row per task,
# never per dataset
rst_npz = np.load(tmp / DEFAULT_COLLECTION / "merged" / "predictions"
                  / "raw_last_layer9_random_slopes_task.npz")
assert sorted(rst_npz["full_group_labels"].astype(str)) \
    == ["jailbreak", "medical", "tf"], sorted(rst_npz["full_group_labels"])
assert rst_npz["full_dev_w"].shape == (3, D)

# xgboost: no coefficient and no scaler; gain importances and the
# serialized booster are saved, and the booster reloads and scores
xgb_npz = np.load(tmp / DEFAULT_COLLECTION / "merged" / "predictions"
                  / "raw_last_layer9_xgboost.npz")
assert xgb_npz["full_feature_importance"].shape == (D,)
assert "full_coef" not in xgb_npz.files \
    and "full_standardize_scale" not in xgb_npz.files
import xgboost as _xgb
_b = _xgb.Booster()
_b.load_model(bytearray(xgb_npz["full_booster_ubj"].tobytes()))
assert len(_b.predict(_xgb.DMatrix(np.zeros((3, D), dtype=np.float32)),
                      output_margin=True)) == 3, \
    "the merged whole-corpus booster should reload and score"

print("run_merged again (resume record should skip everything):")
n_rows_again = run_runner(run_merged, "merged", MERGED_ARGV)
assert n_rows_again == n_rows, "re-run changed the csv"

# --- the pool_excluded contract in run_merged --------------------------
# Never pooled, but scored by the whole-corpus fit as kind eval_only —
# and because that fit's pool is every poolable dataset, those rows ARE
# the held-out-dataset numbers. --exclude still removes a dataset from
# everything, and the pooled npz carry cv_topic and cv_group.
# A SECOND collection, so this run gets its own results tree: the merged
# resume key is (features, layer, method), so re-running l2reg under the
# default collection would be skipped and these assertions would silently
# be reading the earlier run's rows — the ones that still have xstest.
OTHER = "gemma_original"
assert OTHER != DEFAULT_COLLECTION
print(f"run_merged --data {OTHER} --exclude xstest "
      "(pool_excluded + --exclude):")
run_runner(run_merged, "merged",
           ["smoke", "--data", OTHER, "--raw", "last", "--layers", "9",
            "--features", "raw", "--methods", "l2reg",
            "random_slopes_task", "--exclude", "xstest", "--n-jobs", "4"],
           data=OTHER)
with open(tmp / OTHER / "merged" / "tables"
          / "metrics_raw_last_layer9.csv") as f:
    mg_rows = [r for r in csv.DictReader(f) if r["method"] == "l2reg"]
assert all(r["test_dataset"] != "xstest" for r in mg_rows), \
    "--exclude leaked into merged"
for unit in POOL_EXCLUDED:
    rows = [r for r in mg_rows if r["test_dataset"] == unit]
    assert rows and all(r["kind"] == "eval_only" for r in rows), \
        f"{unit} should be scored by the whole-corpus fit as eval_only"
    assert not any(r["kind"] == "per_dataset" and r["test_dataset"] == unit
                   for r in mg_rows), f"{unit} entered the pooled CV"
mg_npz = np.load(tmp / OTHER / "merged" / "predictions"
                 / "raw_last_layer9_l2reg.npz")
pooled = set(np.unique(mg_npz["cv_dataset"]).astype(str))
assert not pooled & set(POOL_EXCLUDED) and "xstest" not in pooled, \
    sorted(pooled)
assert {"eval_only_proba", "eval_only_y", "eval_only_dataset", "cv_topic",
        "cv_group"} <= set(mg_npz.files), sorted(mg_npz.files)
assert set(np.unique(mg_npz["eval_only_dataset"]).astype(str)) \
    == set(POOL_EXCLUDED)

# A held-out unit whose TASK is still in the pool gets that task's
# deviation from random_slopes_task, with the shared-only scores stored
# alongside; claude_vds, whose task is only itself, cannot and falls back
# to shared weights per sample.
rst_eo = np.load(tmp / OTHER / "merged" / "predictions"
                 / "raw_last_layer9_random_slopes_task.npz")
assert {"eval_only_shared_proba", "eval_only_shared_margin"} \
    <= set(rst_eo.files), sorted(rst_eo.files)
_names = rst_eo["eval_only_dataset"].astype(str)
_known = np.isin(_names, ["wildjailbreak_adversarial",
                          "claude_blood_test_ranges"])
assert not np.allclose(rst_eo["eval_only_margin"][_known],
                       rst_eo["eval_only_shared_margin"][_known]), \
    "a held-out unit of a pooled task should get that task's deviation"
_vds = _names == "claude_vds"
assert np.allclose(rst_eo["eval_only_margin"][_vds],
                   rst_eo["eval_only_shared_margin"][_vds]), \
    "claude_vds has no pooled task, so it must score shared-weights only"

print(f"\nall smoke tests passed (outputs kept at {tmp})")
