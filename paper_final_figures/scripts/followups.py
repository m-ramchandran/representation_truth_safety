"""Run ON compute cluster (CPU): three follow-ups needing activations/prediction npz.

1. Label-free geometry per (collection, layer): dataset centroids
   whitened by the pooled Ledoit-Wolf covariance (pairwise distances),
   covariance Frobenius cosines, and top-20 covariance-eigenspace
   affinities. -> geometry_pairs.csv
2. claude_vds holdout sample ids for every leave_out / merged fit, so
   the local parquet rows can be split into book vs claude halves.
   -> vds_ids.csv
3. rs_holdout with unsupervised deviation borrowing: score each saved
   leave-one-dataset-out random_slopes fit on its held-out unit (and on
   the permanently excluded units) with
     shared, shared+task_mean, shared+dev[nearest group],
     shared+softmax-weighted mean of all deviations,
   where nearest/weights come from the whitened centroid distances of
   part 1 (no labels used). -> rs_similarity_holdout.csv

    python followups.py <results_root> <out_dir>
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classifiers.data import COLLECTIONS
from classifiers.analysis_scripts.directions import (ROLES, apply_featurization,
                                                     load_units)

TAG = "raw_last"
TOPK = 20
COLLS = ("gemma_original", "gemma_tf_a", "qwen_original", "qwen_tf_a")
EXCLUDED = ("claude_blood_test_ranges", "claude_vds",
            "wildjailbreak_adversarial")


def geometry_rows(coll, layer, units, covs):
    names = [n for n in units if f"cov_{n}" in covs.files]
    chol = np.linalg.cholesky(np.asarray(covs["cov_pool"], dtype=np.float64))
    mu = {n: np.linalg.solve(
        chol, np.asarray(units[n]["blocks"]["raw"].mean(0), dtype=np.float64))
        for n in names}
    eig = {}
    for n in names:
        c = np.asarray(covs[f"cov_{n}"], dtype=np.float64)
        w, v = np.linalg.eigh(c)
        eig[n] = (v[:, -TOPK:], c)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ua, ca = eig[a]
            ub, cb = eig[b]
            rows.append({
                "collection": coll, "layer": layer, "dataset_a": a,
                "dataset_b": b,
                "centroid_dist": float(np.linalg.norm(mu[a] - mu[b])),
                "cov_frob_cos": float(np.sum(ca * cb) /
                                      (np.linalg.norm(ca) * np.linalg.norm(cb))),
                "subspace_affinity": float(np.sum((ua.T @ ub) ** 2) / TOPK)})
    dist = {(r["dataset_a"], r["dataset_b"]): r["centroid_dist"] for r in rows}
    return rows, dist


def vds_id_rows(coll, root, layer):
    out = []
    for kind in ("dataset", "task"):
        for f in glob.glob(os.path.join(
                root, coll, "leave_out", "predictions",
                f"{TAG}_layer{layer}_l2reg_{kind}_*.npz")):
            holdout = re.search(f"_{kind}_(.+)\\.npz$", f).group(1)
            with np.load(f, allow_pickle=True) as d:
                ds = d["holdout_dataset"].astype(str)
                pos = np.nonzero(ds == "claude_vds")[0]
                if not len(pos):
                    continue
                ids = d["holdout_sample_id"].astype(str)[pos]
            out.append(pd.DataFrame({
                "collection": coll, "layer": layer, "kind": f"leave_{kind}",
                "holdout": holdout, "idx": pos,
                "row": [int(s.split(":")[1]) for s in ids]}))
    f = os.path.join(root, coll, "merged", "predictions",
                     f"{TAG}_layer{layer}_l2reg.npz")
    if os.path.exists(f):
        with np.load(f, allow_pickle=True) as d:
            if "eval_only_sample_id" in d.files:
                ds = d["eval_only_dataset"].astype(str)
                pos = np.nonzero(ds == "claude_vds")[0]
                ids = d["eval_only_sample_id"].astype(str)[pos]
                out.append(pd.DataFrame({
                    "collection": coll, "layer": layer, "kind": "merged_eval",
                    "holdout": "pool", "idx": pos,
                    "row": [int(s.split(":")[1]) for s in ids]}))
    return out


def rs_rows(coll, root, layer, units, dist, task_of):
    files = sorted(glob.glob(os.path.join(
        root, coll, "leave_out", "predictions",
        f"{TAG}_layer{layer}_random_slopes_dataset_*.npz")))
    med = np.median(list(dist.values())) if dist else 1.0

    def d_of(a, b):
        if a == b:
            return 0.0
        return dist.get((a, b), dist.get((b, a), np.nan))

    rows = []
    for f in files:
        holdout = re.search(r"_dataset_(.+)\.npz$", f).group(1)
        d = dict(np.load(f, allow_pickle=True))
        labels = np.asarray(d["group_labels"]).astype(str)
        dev_w = np.atleast_2d(np.asarray(d["dev_w"], dtype=float))
        scale = np.asarray(d["standardize_scale"], dtype=float)
        shared = np.asarray(d["coef"], dtype=float).ravel()
        hd = np.asarray(d["holdout_dataset"]).astype(str)
        for dataset in [holdout] + [e for e in EXCLUDED if e != holdout]:
            mask = hd == dataset
            if not mask.any() or dataset not in units:
                continue
            stored = np.asarray(d["holdout_sample_id"]).astype(str)[mask]
            unit = units[dataset]
            if not np.array_equal(
                    np.asarray(unit["sample_id"]).astype(str), stored):
                continue
            y = np.asarray(d["holdout_y"]).astype(int)[mask]
            if len(np.unique(y)) < 2:
                continue
            X = apply_featurization(d, TAG, unit["blocks"])

            def auc(v):
                return float(roc_auc_score(y, X @ (v / scale)))

            dists = np.array([d_of(dataset, g) for g in labels])
            ok = ~np.isnan(dists)
            est = {"shared": shared}
            same = [i for i, g in enumerate(labels)
                    if task_of(g) == task_of(dataset) and g != dataset]
            if same:
                est["task_mean"] = shared + dev_w[same].mean(0)
            if ok.any():
                near = int(np.where(ok)[0][np.argmin(dists[ok])])
                est["sim_top1"] = shared + dev_w[near]
                w = np.exp(-(dists[ok] / med) ** 2)
                w /= w.sum()
                est["sim_soft"] = shared + (dev_w[ok] * w[:, None]).sum(0)
                nearest = labels[near]
            else:
                nearest = ""
            for name, v in est.items():
                rows.append({"collection": coll, "layer": layer,
                             "holdout": holdout, "dataset": dataset,
                             "estimator": name, "auc": auc(v),
                             "nearest_group": nearest,
                             "stored_auc": float(roc_auc_score(
                                 y, np.asarray(d["holdout_proba"])[mask]))})
    return rows


def main() -> None:
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    geo, ids, rs = [], [], []
    for coll in COLLS:
        c = COLLECTIONS[coll]
        for layer in c.layers:
            print(f"== {coll} layer {layer}", flush=True)
            covp = os.path.join(root, coll, "covariances",
                                f"{TAG}_layer{layer}.npz")
            units = load_units(layer, "last", ROLES["raw"], coll)
            with np.load(covp, allow_pickle=True) as covs:
                g, dist = geometry_rows(coll, layer, units, covs)
            geo += g
            ids += vds_id_rows(coll, root, layer)
            rs += rs_rows(coll, root, layer, units, dist, c.task_of)
            pd.DataFrame(geo).to_csv(
                os.path.join(out_dir, "geometry_pairs.csv"), index=False)
            pd.concat(ids, ignore_index=True).to_csv(
                os.path.join(out_dir, "vds_ids.csv"), index=False)
            pd.DataFrame(rs).to_csv(
                os.path.join(out_dir, "rs_similarity_holdout.csv"), index=False)
    print("done", len(geo), len(rs), flush=True)


if __name__ == "__main__":
    main()
