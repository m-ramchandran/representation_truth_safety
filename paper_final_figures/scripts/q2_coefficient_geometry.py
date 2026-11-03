"""Q2/Q9: coefficient geometry across collections and layers.

Aggregates the coefficient analysis tables of all four collections:
  similarity_{method}    pairwise direction similarity (3 measures)
  similarity_summary     within/cross-task means + spearman vs transfer
  mls                    merged / leave-out / single triangle
  pca_merged             merged direction vs the singles' span
  pca_heldout            each direction rebuilt from the others' span
  pca_spectrum           spectrum of the 13 unit directions
  coef_norms             cos(merged, single) etc.
  rs_vs_single           random-slopes decomposition vs single fits
  rs_holdout             should a held-out set borrow its task deviation
Writes tidy csvs to data/ and prints a digest.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def _iter(coll_pattern: str):
    for (model, pert), coll in C.COLLECTIONS.items():
        for layer in C.LAYERS[model]:
            path = C.RESULTS / coll / coll_pattern.format(L=layer)
            if path.exists():
                yield model, pert, layer, C.DEPTH_OF[(model, layer)], path


def pair_similarity() -> pd.DataFrame:
    rows = []
    for method in ("l2reg", "diff", "svm_linear", "lda"):
        for model, pert, layer, depth, f in _iter(
                "coefficients/tables/similarity_" + method +
                "_raw_last_layer{L}.csv"):
            s = pd.read_csv(f)
            s = s[s.dataset_a != s.dataset_b]
            s["model"], s["perturbation"] = model, pert
            s["layer"], s["depth"], s["method"] = layer, depth, method
            rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["domain_a"] = out.dataset_a.map(C.DOMAIN_OF)
    out["domain_b"] = out.dataset_b.map(C.DOMAIN_OF)
    out["same_domain"] = out.domain_a == out.domain_b
    return out


def summaries() -> pd.DataFrame:
    rows = []
    for model, pert, layer, depth, f in _iter(
            "coefficients/tables/similarity_summary_raw_last_layer{L}.csv"):
        s = pd.read_csv(f)
        s["model"], s["perturbation"] = model, pert
        s["layer"], s["depth"] = layer, depth
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def simple_concat(pattern: str) -> pd.DataFrame:
    rows = []
    for model, pert, layer, depth, f in _iter(pattern):
        s = pd.read_csv(f)
        s["model"], s["perturbation"] = model, pert
        s["layer"], s["depth"] = layer, depth
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rs_holdout() -> pd.DataFrame:
    rows = []
    for (model, pert), coll in C.COLLECTIONS.items():
        f = C.RESULTS / coll / "leave_out/tables/rs_holdout_raw_last.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s["model"], s["perturbation"] = model, pert
        s["depth"] = [C.DEPTH_OF[(model, l)] for l in s.layer]
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    pd.set_option("display.width", 250)

    pairs = pair_similarity()
    pairs.to_csv(C.DATA / "supplementary_figure4_direction_similarity.csv",
                 index=False)
    summ = summaries()
    summ.to_csv(C.DATA / "q2_similarity_summary.csv", index=False)
    mls = simple_concat("coefficients/tables/mls_raw_last_layer{L}_dataset.csv")
    mls.to_csv(C.DATA / "q2_mls.csv", index=False)
    pm = simple_concat("coefficients/tables/pca_merged_l2reg_raw_last_layer{L}_dataset.csv")
    pm.to_csv(C.DATA / "q2_pca_merged.csv", index=False)
    ph = simple_concat("coefficients/tables/pca_heldout_l2reg_raw_last_layer{L}.csv")
    ph.to_csv(C.DATA / "figure3_heldout_direction_ensemble.csv", index=False)
    spec = simple_concat("coefficients/tables/pca_spectrum_l2reg_raw_last_layer{L}.csv")
    spec.to_csv(C.DATA / "q2_spectrum.csv", index=False)
    rsv = simple_concat("coefficients/tables/rs_vs_single_raw_last_layer{L}.csv")
    rsv.to_csv(C.DATA / "q2_rs_vs_single.csv", index=False)
    rsh = rs_holdout()
    rsh.to_csv(C.DATA / "q2_rs_holdout.csv", index=False)

    print("== within vs between domain similarity (median over pairs), "
          "l2reg, mahalanobis_reg_cov_pool ==")
    sel = pairs[(pairs.method == "l2reg")
                & (pairs.measure == "mahalanobis_reg_cov_pool")]
    t = (sel.groupby(["model", "perturbation", "depth", "same_domain"])
         .value.median().unstack("same_domain")
         .rename(columns={True: "within", False: "between"}))
    print(t.round(3))

    print("\n== same, diff-in-means, plain cosine ==")
    sel = pairs[(pairs.method == "diff") & (pairs.measure == "plain")]
    t = (sel.groupby(["model", "perturbation", "depth", "same_domain"])
         .value.median().unstack("same_domain")
         .rename(columns={True: "within", False: "between"}))
    print(t.round(3))

    print("\n== spearman(similarity, transfer auc) by measure (l2reg + diff; "
          "median over layers/collections) ==")
    print(summ[summ.method.isin(["l2reg", "diff"])]
          .groupby(["method", "measure"]).spearman_vs_transfer_auc
          .median().round(3))

    print("\n== merged direction vs singles' span (basis=single) ==")
    scalars = pm[pm.dataset.isna() & (pm.basis == "single")]
    t = scalars.pivot_table(index=["model", "perturbation", "depth"],
                            columns="quantity", values="value")
    print(t[["cos_merged_mean_dir", "energy_in_span", "energy_along_mean",
             "energy_outside_span"]].round(3))

    print("\n== held-out direction from the others' span: k0 (their mean), "
          "best k, own fit ==")
    kcols = [c for c in ph.columns if c.startswith("k")]
    ph["k_best"] = ph[kcols].max(axis=1)
    ph["private_gap"] = ph.original - ph.k_best
    print(ph.groupby("dataset")[["k0", "k_best", "original", "private_gap"]]
          .median().round(3).sort_values("private_gap"))

    print("\n== spectrum: PC1 share of the centered single directions ==")
    pc1 = spec[spec.pc == "PC1"]
    print(pc1.pivot_table(index=["model", "perturbation"], columns="depth",
                          values="explained_variance_ratio").round(3))

    print("\n== random slopes vs single directions (median over datasets, "
          "l2reg comparison) ==")
    r = rsv[(rsv.method == "l2reg") & (rsv.variant == "random_slopes")]
    print(r.groupby(["model", "perturbation", "comparison"])
          .value.median().unstack("comparison")
          [["shared", "dev", "shared_plus_dev"]].round(3))

    if not rsh.empty:
        print("\n== rs_holdout: borrow the task deviation? (median AUC by "
              "estimator) ==")
        print(rsh.groupby(["model", "perturbation", "estimator"])
              .auc.median().unstack("estimator").round(3))
        print("\nper dataset (gemma bare, median over layers):")
        g = rsh[(rsh.model == "gemma") & (rsh.perturbation == "bare")]
        print(g.pivot_table(index="dataset", columns="estimator",
                            values="auc").round(3))


if __name__ == "__main__":
    main()
