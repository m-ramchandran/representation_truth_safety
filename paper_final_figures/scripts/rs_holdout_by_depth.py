"""Run ON compute cluster (CPU): figure 5's random-slopes rows, kept per depth.

figure5_heldout_targets.csv stores only the median over the ten
leave-one-dataset-out fits AND the four depths, so it cannot answer
questions of the form "does the random-slopes estimator beat the L2
target-absent probe at the layers we actually trust?". This rebuilds the
per-(collection, layer, fit) numbers behind those medians -- the
rs_similarity_holdout.csv part of followups.py, nothing else -- for the
two permanently held-out targets.

Estimators, all label-free (identical construction to psc_followups.rs_rows,
so the medians of this file reproduce figure5_heldout_targets.csv):

  shared     the leave-one-out fit's shared weights alone
  task_mean  shared + the mean deviation of the target's own domain
             (WJB-adversarial borrows the jailbreak domain; VDS is alone in
             its domain and has none, so it gets no task_mean row)
  sim_top1   shared + the deviation of the nearest group by whitened
             centroid distance
  sim_soft   shared + the softmax-of-distance weighted mean of all
             deviations -- the row figure 5 draws for VDS

    python paper_final_figures/scripts/rs_holdout_by_depth.py \\
        <results_root> <out_csv>

Reads only. Writes one row per (collection, layer, holdout fit, target,
estimator).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("REPO_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(ROOT))
from classifiers.data import COLLECTIONS                            # noqa: E402
from classifiers.analysis_scripts.directions import (                # noqa: E402
    ROLES, apply_featurization, load_units)

TAG = "raw_last"
COLLS = ("gemma_original", "gemma_tf_a", "qwen_original", "qwen_tf_a")
TARGETS = ("claude_vds", "wildjailbreak_adversarial")


def auc(y, scores):
    """Mann-Whitney AUC; the scores are continuous so ties are measure
    zero."""
    order = np.argsort(scores, kind="stable")
    yy = np.asarray(y, dtype=bool)[order]
    n1 = yy.sum()
    n0 = len(yy) - n1
    if n0 == 0 or n1 == 0:
        return np.nan
    ranks = np.arange(1, len(yy) + 1, dtype=np.float64)
    return float((ranks[yy].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def whitened_distances(units, covs):
    names = [n for n in units if f"cov_{n}" in covs.files]
    chol = np.linalg.cholesky(np.asarray(covs["cov_pool"], dtype=np.float64))
    mu = {n: np.linalg.solve(
        chol, np.asarray(units[n]["blocks"]["raw"].mean(0), dtype=np.float64))
        for n in names}
    return {(a, b): float(np.linalg.norm(mu[a] - mu[b]))
            for i, a in enumerate(names) for b in names[i + 1:]}


def rows_for(coll, root, layer, units, dist, task_of):
    files = sorted(glob.glob(os.path.join(
        root, coll, "leave_out", "predictions",
        f"{TAG}_layer{layer}_random_slopes_dataset_*.npz")))
    med = float(np.median(list(dist.values()))) if dist else 1.0

    def d_of(a, b):
        return 0.0 if a == b else dist.get((a, b), dist.get((b, a), np.nan))

    out = []
    for f in files:
        holdout = re.search(r"_dataset_(.+)\.npz$", f).group(1)
        d = dict(np.load(f, allow_pickle=True))
        labels = np.asarray(d["group_labels"]).astype(str)
        dev_w = np.atleast_2d(np.asarray(d["dev_w"], dtype=float))
        scale = np.asarray(d["standardize_scale"], dtype=float)
        shared = np.asarray(d["coef"], dtype=float).ravel()
        hd = np.asarray(d["holdout_dataset"]).astype(str)
        for target in TARGETS:
            mask = hd == target
            if not mask.any() or target not in units:
                continue
            stored = np.asarray(d["holdout_sample_id"]).astype(str)[mask]
            unit = units[target]
            if not np.array_equal(
                    np.asarray(unit["sample_id"]).astype(str), stored):
                continue
            y = np.asarray(d["holdout_y"]).astype(int)[mask]
            if len(np.unique(y)) < 2:
                continue
            X = apply_featurization(d, TAG, unit["blocks"])

            dists = np.array([d_of(target, g) for g in labels])
            ok = ~np.isnan(dists)
            est = {"shared": shared}
            same = [i for i, g in enumerate(labels)
                    if task_of(g) == task_of(target) and g != target]
            if same:
                est["task_mean"] = shared + dev_w[same].mean(0)
            if ok.any():
                near = int(np.where(ok)[0][np.argmin(dists[ok])])
                est["sim_top1"] = shared + dev_w[near]
                w = np.exp(-(dists[ok] / med) ** 2)
                w /= w.sum()
                est["sim_soft"] = shared + (dev_w[ok] * w[:, None]).sum(0)
            for name, v in est.items():
                out.append({"collection": coll, "layer": layer,
                            "holdout_fit": holdout, "target": target,
                            "estimator": name,
                            "auc": auc(y, X @ (v / scale)),
                            "n": int(mask.sum())})
        print(f"  {coll} L{layer} {holdout}: done", flush=True)
    return out


def main() -> None:
    root, out_csv = sys.argv[1], sys.argv[2]
    rows = []
    for coll in COLLS:
        for layer in COLLECTIONS[coll].layers:
            print(f"== {coll} layer {layer}", flush=True)
            units = load_units(layer, "last", ROLES["raw"], coll)
            with np.load(os.path.join(root, coll, "covariances",
                                      f"{TAG}_layer{layer}.npz"),
                         allow_pickle=True) as covs:
                dist = whitened_distances(units, covs)
            rows += rows_for(coll, root, layer, units, dist,
                             COLLECTIONS[coll].task_of)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    print(f"\nwrote {out_csv}  ({len(df)} rows)")
    print(df.pivot_table(index=["target", "collection"], columns="estimator",
                         values="auc", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
