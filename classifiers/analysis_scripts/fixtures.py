"""A small synthetic results tree for developing the analysis modules on
the Mac before real results exist.

Three fake datasets share a planted linear signal and each has a private
direction on top; four stand-in "methods" score with fixed linear rules
(no fitting), chosen to cover the corners of the npz contract: l2reg
(dense coef, log-odds margin), l1reg (sparse coef), diff (raw-space unit
direction, Platt-calibrated), knn (probability only — no margin, no
coef). The merged tree adds random_slopes (per-dataset deviations),
random_slopes_task (per-task deviations), moe (gate/expert arrays), and
xgboost (feature_importance only — no coef, no scaler). Layout, keys,
and csv columns match the runners' outputs exactly. "got" gets two-part
sample ids ("p1:3") so the per-part analyses have something to slice.

    python -m classifiers.analysis_scripts.fixtures OUTDIR
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

TAG, LAYER, D, FOLDS = "raw_last", 9, 20, 5
# got + common_claim are both in the original collection's "tf" task, so
# one task has two members: task-mode pools genuinely differ from
# dataset-mode ones, and a held-out dataset can have a same-task sibling
# left in the pool (what rs_holdout's task-deviation averaging needs).
# beta / gamma are in no task and stand as their own.
SIZES = {"got": 400, "common_claim": 250, "beta": 400, "gamma": 300,
         "delta": 200}
# delta plays the pool_excluded role: never in a pool, so every
# leave-out fit scores it on top of the holdout's own members. That is
# what makes a leave-out npz hold SEVERAL evaluated units, and modules
# that assumed one (rs_holdout did) break on it.
POOL_EXCLUDED = ("delta",)
METHODS = ("l2reg", "l1reg", "diff", "knn")
POOLED_METHODS = METHODS + ("random_slopes", "random_slopes_task", "moe",
                            "moe_hard", "xgboost")


def make_datasets(rng) -> dict:
    """name -> {X, y, sample_id}: labels follow a shared direction (dims
    0-4) plus one private dimension per dataset, so transfer is real but
    imperfect."""
    shared = np.zeros(D)
    shared[:5] = 1.5
    out = {}
    for i, (name, n) in enumerate(SIZES.items()):
        w = shared.copy()
        w[5 + i] = 2.0
        X = rng.normal(size=(n, D))
        y = (X @ w + rng.logistic(size=n) > 0).astype(int)
        part = np.array(["p1", "p2"])[np.arange(n) % 2] \
            if name == "got" else np.full(n, name)
        sample_id = np.array([f"{p}:{j}" for j, p in enumerate(part)])
        out[name] = {"X": X, "y": y, "sample_id": sample_id, "w": w}
    return out


def score(method: str, X, w, rng) -> tuple[dict, dict, dict]:
    """One method's (scores, params, model_arrays) on X, given the true
    direction w. Per-sample noise makes the methods err on different
    samples; per-method quirks mirror the real contract."""
    mean, scale = X.mean(0), X.std(0)
    z = ((X - mean) / scale) @ (w * scale)      # standardized-space score
    arrays = {"standardize_mean": mean, "standardize_scale": scale}
    if method == "knn":                          # neighbor fractions: proba
        proba = np.round(25 * expit(z + rng.normal(0, 2, len(z)))) / 25
        return {"proba": proba}, {"k": 25}, arrays
    if method == "diff":                         # raw space, Platt map
        direction = w / np.linalg.norm(w)
        margin = X @ direction + rng.normal(0, 1, len(z))
        return ({"proba": expit(2.0 * margin), "margin": margin},
                {"platt_a": 2.0, "platt_b": 0.0}, {"coef": direction})
    coef = w * scale                             # standardized-space coef
    if method == "l1reg":
        coef = np.where(np.abs(coef) >= np.sort(np.abs(coef))[-6], coef, 0)
    margin = ((X - mean) / scale) @ coef + rng.normal(0, 1.5, len(z))
    arrays["coef"] = coef
    return {"proba": expit(margin), "margin": margin}, {"C": 1.0}, arrays


def pooled_arrays(method: str, arrays: dict, groups, rng) -> dict:
    """A pooled fit's model arrays with the method-specific extras: the
    random-slopes variants get one deviation row per group (`groups` is
    the pool's group names — datasets, or tasks for the _task variant),
    moe gets gate/expert weights instead of a coefficient, and xgboost
    gets gain importances and nothing else (no coef, no scaler)."""
    if method == "xgboost":
        # a real fit also saves booster_ubj (UBJSON bytes); the fixture
        # plants a stand-in blob so callers can check for the key, but it
        # is not a loadable model — tests that score a booster need a
        # real fit
        return {"feature_importance": rng.random(D),
                "booster_ubj": np.zeros(8, dtype=np.uint8)}
    out = dict(arrays)
    if method in ("random_slopes", "random_slopes_task"):
        labels = sorted(set(groups))
        out.update(dev_w=rng.normal(0, .1, (len(labels), D)),
                   dev_b=rng.normal(0, .1, len(labels)),
                   group_labels=np.array(labels))
    if method in ("moe", "moe_hard"):
        del out["coef"]                    # neither moe has a coefficient
        out.update(gate_w=rng.normal(size=(2, D)), gate_b=np.zeros(2),
                   expert_w=rng.normal(size=(2, D)), expert_b=np.zeros(2))
    return out


def write_npz(root: Path, experiment: str, name: str, npz: dict) -> None:
    d = root / experiment / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}.npz", **npz)


def write_csv(root: Path, experiment: str, columns, rows) -> None:
    d = root / experiment / "tables"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"metrics_{TAG}_layer{LAYER}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def eval_parts(parts, prefix: str) -> tuple[dict, list]:
    """npz score entries and (dataset, n, auc) rows for scored datasets;
    `parts` is a list of (name, unit, scores)."""
    npz = {
        f"{prefix}_dataset": np.concatenate(
            [np.full(len(u["y"]), n) for n, u, _ in parts]),
        f"{prefix}_sample_id": np.concatenate(
            [u["sample_id"] for _, u, _ in parts]),
        f"{prefix}_y": np.concatenate([u["y"] for _, u, _ in parts]),
    }
    for key in parts[0][2]:
        npz[f"{prefix}_{key}"] = np.concatenate([s[key] for _, _, s in parts])
    rows = [(n, len(u["y"]), f"{roc_auc_score(u['y'], s['proba']):.3f}")
            for n, u, s in parts]
    return npz, rows


def main(outdir: Path) -> None:
    rng = np.random.default_rng(0)
    data = make_datasets(rng)

    # per_dataset: nested CV on each dataset + transfer to the others
    rows = []
    for method in METHODS:
        for train, tr in data.items():
            cv, params, arrays = score(method, tr["X"], tr["w"], rng)
            transfer = [(name, u, score(method, u["X"], tr["w"], rng)[0])
                        for name, u in data.items() if name != train]
            t_npz, t_rows = eval_parts(transfer, "transfer")
            npz = {"cv_sample_id": tr["sample_id"], "cv_y": tr["y"],
                   "cv_fold": np.arange(len(tr["y"])) % FOLDS,
                   "params": repr(params),
                   **{f"cv_{k}": v for k, v in cv.items()},
                   **t_npz, **arrays}
            write_npz(outdir, "per_dataset",
                      f"{TAG}_layer{LAYER}_{method}_{train}", npz)
            auc = f"{roc_auc_score(tr['y'], cv['proba']):.3f}"
            rows.append((TAG, LAYER, method, train, train, "cv",
                         len(tr["y"]), auc))
            rows += [(TAG, LAYER, method, train, t, "transfer", n, a)
                     for t, n, a in t_rows]
    write_csv(outdir, "per_dataset",
              ("features", "layer", "method", "train_dataset",
               "test_dataset", "kind", "n_test", "auc"), rows)

    # leave_out: dataset mode holds out one dataset, task mode holds out
    # a whole task's members at once (got + common_claim are both tf), so
    # the two modes' pools genuinely differ — as in the runner.
    # random_slopes_task is task mode only, mirroring run_leave_out.
    from ..data import task_of
    tasks = {}
    for ds in data:
        tasks.setdefault(task_of(ds), []).append(ds)
    poolable = [d for d in data if d not in POOL_EXCLUDED]
    holdouts = [("dataset", ds, [ds]) for ds in poolable] \
        + [("task", t, [m for m in members if m in poolable])
           for t, members in tasks.items()
           if any(m in poolable for m in members)]
    rows = []
    for method in POOLED_METHODS:
        base = "l2reg" if method in ("random_slopes", "random_slopes_task",
                                     "moe", "moe_hard",
                                     "xgboost") else method
        for kind, holdout, members in holdouts:
            if method == "random_slopes_task" and kind == "dataset":
                continue                          # runner runs task mode only
            pool = {n: u for n, u in data.items()
                    if n not in members and n not in POOL_EXCLUDED}
            w = np.mean([u["w"] for u in pool.values()], axis=0)
            X = np.concatenate([u["X"] for u in pool.values()])
            cv, params, arrays = score(base, X, w, rng)
            groups = [task_of(n) for n in pool] \
                if method == "random_slopes_task" else list(pool)
            arrays = pooled_arrays(method, arrays, groups, rng)
            # members + every pool_excluded unit, as run_leave_out does
            scored = list(members) + [n for n in POOL_EXCLUDED
                                      if n not in members]
            h_npz, h_rows = eval_parts(
                [(m, data[m], score(base, data[m]["X"], w, rng)[0])
                 for m in scored], "holdout")
            npz = {"cv_sample_id": np.concatenate(
                       [u["sample_id"] for u in pool.values()]),
                   "cv_y": np.concatenate([u["y"] for u in pool.values()]),
                   "cv_dataset": np.concatenate(
                       [np.full(len(u["y"]), n) for n, u in pool.items()]),
                   "params": repr(params),
                   **{f"cv_{k}": v for k, v in cv.items()},
                   **h_npz, **arrays}
            write_npz(outdir, "leave_out",
                      f"{TAG}_layer{LAYER}_{method}_{kind}_{holdout}", npz)
            rows += [(TAG, LAYER, method, kind, holdout, t, n, a)
                     for t, n, a in h_rows]
    write_csv(outdir, "leave_out",
              ("features", "layer", "method", "holdout_kind", "holdout",
               "test_dataset", "n_test", "auc"), rows)

    # merged: everything pooled, plus the two pooled-only methods
    X = np.concatenate([u["X"] for u in data.values()])
    y = np.concatenate([u["y"] for u in data.values()])
    dataset = np.concatenate([np.full(len(u["y"]), n)
                              for n, u in data.items()])
    w = np.mean([u["w"] for u in data.values()], axis=0)
    rows = []
    for method in POOLED_METHODS:
        base = "l2reg" if method in ("random_slopes", "random_slopes_task",
                                     "moe", "moe_hard",
                                     "xgboost") else method
        cv, params, arrays = score(base, X, w, rng)
        groups = [task_of(n) for n in data] \
            if method == "random_slopes_task" else list(data)
        arrays = pooled_arrays(method, arrays, groups, rng)
        npz = {"cv_sample_id": np.concatenate(
                   [u["sample_id"] for u in data.values()]),
               "cv_y": y, "cv_dataset": dataset,
               "cv_fold": np.arange(len(y)) % FOLDS,
               "params": repr([params] * FOLDS),
               "full_params": repr(params),
               **{f"cv_{k}": v for k, v in cv.items()},
               **{f"full_{k}": v for k, v in arrays.items()}}
        write_npz(outdir, "merged", f"{TAG}_layer{LAYER}_{method}", npz)
        rows.append((TAG, LAYER, method, "overall", "all", len(y),
                     f"{roc_auc_score(y, cv['proba']):.3f}"))
        for name in data:
            m = dataset == name
            rows.append((TAG, LAYER, method, "per_dataset", name,
                         int(m.sum()),
                         f"{roc_auc_score(y[m], cv['proba'][m]):.3f}"))
    write_csv(outdir, "merged",
              ("features", "layer", "method", "kind", "test_dataset",
               "n_test", "auc"), rows)
    print(f"fixture results tree written to {outdir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
