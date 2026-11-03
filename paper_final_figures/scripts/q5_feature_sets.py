"""Q5: raw activations vs PCA / PCA+dict / SAE / transcoder / binned.

Best-over-methods AUC per (run, regime, dataset, feature, depth), the
delta to raw at the matched cell, and the binned control read-out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def main() -> None:
    all_ = pd.read_csv(C.DATA / "supplementary_figure3_feature_sets.csv")
    keys = ["model", "perturbation", "regime", "dataset", "domain",
            "feature", "depth"]
    # collapse single-source over sources (per method), then best over methods
    single = all_[all_.regime == "single_source"]
    other = all_[all_.regime != "single_source"]
    single = (single.groupby(keys + ["method"], as_index=False).auc.median())
    both = pd.concat([other, single], ignore_index=True)
    best = both.groupby(keys, as_index=False).auc.max()

    raw = best[best.feature == "raw"].rename(columns={"auc": "auc_raw"})
    j = best[best.feature != "raw"].merge(
        raw.drop(columns=["feature"]),
        on=[k for k in keys if k != "feature"])
    j["delta"] = j.auc - j.auc_raw
    j.to_csv(C.DATA / "supplementary_figure3_feature_deltas.csv", index=False)

    pd.set_option("display.width", 250)
    print("== best-over-methods AUC minus raw, median (IQR) over datasets ==")
    for regime in ["within_cv", "single_source", "merged_included",
                   "leave_dataset_out", "leave_domain_out"]:
        sel = j[j.regime == regime]
        if sel.empty:
            continue
        t = sel.groupby(["feature", "depth"]).delta.median().unstack("depth")
        print(f"\n-- {regime} --")
        print(t.round(3))

    print("\n== cells where a non-raw feature beats raw by > 0.01 ==")
    win = j[j.delta > 0.01]
    print(win.groupby(["regime", "feature"]).size().unstack(fill_value=0))
    print("\n  detail (median delta per dataset with >=2 supporting runs):")
    d = (win.groupby(["regime", "feature", "dataset"])
         .agg(n_cells=("delta", "size"), med=("delta", "median")).reset_index())
    print(d[d.n_cells >= 3].round(3).to_string(index=False))

    # ---- binned control ------------------------------------------------
    print("\n== binned control: best method AUC on the value-histogram ==")
    b = best[best.feature == "binned"]
    t = (b[b.regime == "within_cv"]
         .groupby(["dataset"]).auc.median().sort_values(ascending=False))
    raw_w = raw[raw.regime == "within_cv"].groupby("dataset").auc_raw.median()
    out = pd.DataFrame({"binned_best": t, "raw_best": raw_w})
    out["share_of_raw"] = (out.binned_best - 0.5) / (out.raw_best - 0.5)
    print(out.round(3).sort_values("share_of_raw", ascending=False))

    # which methods do well on binned?
    bb = both[(both.feature == "binned") & (both.regime == "within_cv")]
    idx = bb.groupby(["model", "perturbation", "dataset"]).auc.idxmax()
    winners = bb.loc[idx]
    print("\nbinned within-CV winning method counts:")
    print(winners.method.value_counts())
    print("\nbinned within-CV: median AUC by method family:")
    fam = bb.groupby(bb.method.map(C.FAMILY_OF)).auc.median()
    print(fam.round(3))


if __name__ == "__main__":
    main()
