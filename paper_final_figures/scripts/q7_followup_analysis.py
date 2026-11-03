"""Consume the compute cluster follow-up outputs:

1. Label-free geometry -> transfer: correlate pooled-whitened centroid
   distance / covariance Frobenius cosine / top-20 subspace affinity
   (computed from activations only) with single-source transfer AUC, and
   with domain membership.
2. Similarity-weighted deviation borrowing for held-out datasets
   (rs_similarity_holdout.csv): shared vs task_mean vs sim_top1/sim_soft.
3. VDS book-vs-claude split of the probe predictions (vds_ids.csv +
   data/heldout parquets), alongside the logits split.

Expects the followup csvs in data/followups/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C

FU = C.DATA / "followups"
COLL_KEY = {v: k for k, v in C.COLLECTIONS.items()}


def geometry_transfer() -> None:
    g = pd.read_csv(FU / "geometry_pairs.csv")
    g = g[g.dataset_a.isin(C.PAPER_DATASETS)
          & g.dataset_b.isin(C.PAPER_DATASETS)].copy()   # paper roster
    g[["model", "perturbation"]] = [COLL_KEY[c] for c in g.collection]
    g["depth"] = [C.DEPTH_OF[(m, l)] for m, l in zip(g.model, g.layer)]
    g["same_domain"] = (g.dataset_a.map(C.DOMAIN_OF)
                        == g.dataset_b.map(C.DOMAIN_OF))
    g.to_csv(C.DATA / "q7_geometry_pairs.csv", index=False)

    tr = C.long_results()
    tr = tr[tr.regime == "single_source"]
    tr = (tr[tr.method.isin(["l2reg", "diff"])]
          .groupby(["model", "perturbation", "depth", "method",
                    "source_dataset", "dataset"], as_index=False).auc.median())
    # symmetric geometry joins both directions of each ordered pair
    both = pd.concat([
        g.rename(columns={"dataset_a": "source_dataset", "dataset_b": "dataset"}),
        g.rename(columns={"dataset_b": "source_dataset", "dataset_a": "dataset"}),
    ], ignore_index=True)
    j = both.merge(tr, on=["model", "perturbation", "depth",
                           "source_dataset", "dataset"])
    j.to_csv(C.DATA / "supplementary_figure4_geometry_transfer.csv",
             index=False)

    print("== label-free geometry vs transfer AUC (Spearman, per run+depth) ==")
    for method in ("l2reg", "diff"):
        for feat, sign in [("centroid_dist", -1), ("cov_frob_cos", 1),
                           ("subspace_affinity", 1)]:
            rhos = [sign * spearmanr(gg[feat], gg.auc)[0]
                    for _, gg in j[j.method == method].groupby(
                        ["model", "perturbation", "depth"]) if len(gg) > 30]
            print(f"  {method:6s} {feat:18s} median rho {np.median(rhos):+.2f} "
                  f"(range {min(rhos):+.2f}..{max(rhos):+.2f}, n={len(rhos)})")
    print("\n== does unsupervised geometry separate domains? ==")
    for feat in ("centroid_dist", "cov_frob_cos", "subspace_affinity"):
        w = g.groupby("same_domain")[feat].median()
        print(f"  {feat:18s} within {w.get(True):.3f}  between {w.get(False):.3f}")


def rs_similarity() -> None:
    r = pd.read_csv(FU / "rs_similarity_holdout.csv")
    r[["model", "perturbation"]] = [COLL_KEY[c] for c in r.collection]
    r.to_csv(C.DATA / "q7_rs_similarity.csv", index=False)
    own = r[r.holdout == r.dataset]
    piv = (own.groupby(["model", "perturbation", "estimator"]).auc.median()
           .unstack("estimator"))
    print("\n== held-out dataset, deviation borrowing (median AUC) ==")
    cols = [c for c in ("shared", "task_mean", "sim_top1", "sim_soft")
            if c in piv]
    print(piv[cols].round(3))
    print("\nper dataset (median over runs/layers):")
    t = (own.groupby(["dataset", "estimator"]).auc.median()
         .unstack("estimator"))
    print(t[cols].round(3).to_string())
    exc = r[r.holdout != r.dataset]
    print("\n== permanently excluded targets under every holdout fit ==")
    t = exc.groupby(["dataset", "estimator"]).auc.median().unstack("estimator")
    print(t[[c for c in cols if c in t]].round(3).to_string())


def vds_split() -> None:
    ids = pd.read_csv(FU / "vds_ids.csv")
    ids[["model", "perturbation"]] = [COLL_KEY[c] for c in ids.collection]
    from huggingface_hub import hf_hub_download
    from classifiers.data import COLLECTIONS as CO, HF_REPO
    src = {}
    for coll in ids.collection.unique():
        hf_dir, stem = CO[coll].hf_files["claude_vds"][0]
        p = hf_hub_download(HF_REPO, f"{hf_dir}/{stem}_tf_logits.npz",
                            repo_type="dataset")
        src[coll] = np.load(p, allow_pickle=True)["source"].astype(str)
    rows = []
    for coll, g in ids.groupby("collection"):
        pq = C.DATA / "heldout" / f"{coll}_heldout.parquet"
        df = pd.read_parquet(pq, filters=[("dataset", "==", "claude_vds")])
        df = df[df.method.isin(C.METHODS)]
        key = g.set_index(["layer", "kind", "holdout", "idx"]).row
        for (layer, kind, holdout, method), gg in df.groupby(
                ["layer", "kind", "holdout", "method"], observed=True):
            try:
                row = key.loc[layer, str(kind), str(holdout)].reindex(gg.idx)
            except KeyError:
                continue
            half = pd.Series(src[coll][row.to_numpy()], index=gg.index)
            for h in ("book", "claude"):
                m = half == h
                if m.sum() < 20 or gg[m].y.nunique() < 2:
                    continue
                rows.append({
                    "collection": coll, "layer": layer, "kind": str(kind),
                    "holdout": str(holdout), "method": method, "half": h,
                    "n": int(m.sum()),
                    "auc": roc_auc_score(gg[m].y, gg[m].proba)})
    out = pd.DataFrame(rows)
    out[["model", "perturbation"]] = [COLL_KEY[c] for c in out.collection]
    out.to_csv(C.DATA / "q7_vds_split_probes.csv", index=False)
    best = (out.groupby(["model", "perturbation", "kind", "holdout",
                         "layer", "half"]).auc.max().reset_index())
    print("\n== VDS split: best-method probe AUC per half "
          "(median over fits/layers) ==")
    print(best.groupby(["half", "model", "perturbation"]).auc.median()
          .unstack(["model", "perturbation"]).round(3))
    lg = pd.read_csv(C.DATA / "q_vds_split_logits.csv")
    print("\nmodel-readout comparison (p_true):")
    print(lg.pivot_table(index="half", columns=["model", "pert"],
                         values="logit_auc").round(3))


def proximity_alignment() -> None:
    """Supplementary Figure 4 proximity data: every unordered dataset pair with its
    centroid distance (pooled-whitened) and the |cosine| between the two
    datasets' L2 directions under the pooled-covariance metric, binned into
    per-(model, framing, depth) distance terciles. Paper roster only, so
    the terciles are recomputed without claude_blood_test_ranges (this
    table previously had no writer; the rule here reproduces the stored
    13-dataset bins exactly when run with the full roster)."""
    g = pd.read_csv(C.DATA / "q7_geometry_pairs.csv")
    g = g[g.dataset_a.isin(C.PAPER_DATASETS)
          & g.dataset_b.isin(C.PAPER_DATASETS)].copy()
    ps = pd.read_csv(C.DATA / "supplementary_figure4_direction_similarity.csv")
    ps = ps[(ps.method == "l2reg")
            & (ps.measure == "mahalanobis_reg_cov_pool")].copy()
    ps["a"] = ps[["dataset_a", "dataset_b"]].min(axis=1)
    ps["b"] = ps[["dataset_a", "dataset_b"]].max(axis=1)
    ps = (ps.groupby(["model", "perturbation", "depth", "a", "b"],
                     as_index=False).value.first())
    ps["abscos"] = ps.value.abs()
    g["a"] = g[["dataset_a", "dataset_b"]].min(axis=1)
    g["b"] = g[["dataset_a", "dataset_b"]].max(axis=1)
    j = g.merge(ps, on=["model", "perturbation", "depth", "a", "b"])
    j = j[["collection", "layer", "dataset_a", "dataset_b", "centroid_dist",
           "cov_frob_cos", "subspace_affinity", "model", "perturbation",
           "depth", "same_domain", "a", "b", "abscos", "value"]]

    def bins(s: pd.Series) -> pd.Series:
        q1, q2 = s.quantile([1 / 3, 2 / 3]).values
        return pd.Series(np.where(s <= q1, "near",
                                  np.where(s <= q2, "mid", "far")),
                         index=s.index)

    j["dist_bin"] = (j.groupby(["model", "perturbation", "depth"])
                     .centroid_dist.transform(bins))
    j.to_csv(C.DATA / "supplementary_figure4_proximity_alignment.csv",
             index=False)
    print("\n== |cos| between L2 directions by centroid-distance tercile ==")
    print(j.groupby(["same_domain", "dist_bin"]).abscos.median()
          .unstack("dist_bin")[["near", "mid", "far"]].round(2))


if __name__ == "__main__":
    geometry_transfer()
    proximity_alignment()
    rs_similarity()
    vds_split()
