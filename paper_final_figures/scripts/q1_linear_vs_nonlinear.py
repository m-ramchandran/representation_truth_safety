"""Q1: where (if anywhere) does nonlinearity beat linearity?

Computes, for raw_last only:
  - per (model, perturbation, regime, dataset): best linear vs best
    nonlinear AUC over methods x layers ("oracle" comparison), and the
    typical (median over methods, fixed best layer) comparison;
  - per-depth version to look for layer patterns;
  - replication counts across the four collections;
  - the wildjailbreak_adversarial / claude_vds / blood_ranges detail.

Writes tables to paper_final_figures/data/ and prints a digest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def main() -> None:
    # shared_classifier_results.csv is the uncollapsed frame used by more
    # than one final figure; written here so it follows the
    # loader's roster (paper roster, no claude_blood_test_ranges).
    raw = C.long_results()
    raw.to_csv(C.DATA / "shared_classifier_results.csv", index=False)
    # Figure 2 additionally draws random slopes beside the ten standard
    # methods. Cache that broader frame separately so the final figure can
    # be regenerated without keeping classifiers/results on this machine.
    C.long_results(keep_all_methods=True).to_csv(
        C.DATA / "figure2_method_results.csv", index=False)
    f = C.collapse_single_source(raw)
    f = f[f.family.isin(["linear", "nonlinear"])]

    # ---- oracle best per family, over methods AND layers ----------------
    cell = ["model", "perturbation", "regime", "dataset", "domain"]
    best = (f.groupby(cell + ["family"])
            .auc.max().unstack("family").reset_index())
    best["gap"] = best.nonlinear - best.linear

    # winner identity: which method/layer is the overall best per cell
    idx = f.groupby(cell).auc.idxmax()
    winners = f.loc[idx, cell + ["method", "family", "layer", "depth", "auc"]]

    # ---- typical: median over methods within family at each depth -------
    per_depth = (f.groupby(cell + ["family", "depth"])
                 .auc.agg(best="max", median="median").reset_index())

    # ---- digest ---------------------------------------------------------
    pd.set_option("display.width", 250)

    print("== Oracle gap (nonlinear best - linear best), all datasets ==")
    g = best.groupby(["regime"]).gap.describe()[["count", "mean", "50%", "min", "max"]]
    print(g.round(3))

    print("\n== Gap by regime x collection (mean over datasets) ==")
    piv = best.pivot_table(index="regime", columns=["model", "perturbation"],
                           values="gap", aggfunc="mean")
    print(piv.round(3).reindex(C.REGIMES))

    print("\n== Datasets where |gap| > 0.01 in ANY collection ==")
    flagged = best[best.gap.abs() > 0.01]
    print(flagged.groupby(["regime", "dataset"]).gap.agg(["count", "mean", "min", "max"]).round(3))

    print("\n== Replication: sign of gap (>+0.01 / <-0.01 / tie) counts over 4 collections ==")
    def sign(x):
        return "nonlinear" if x > 0.01 else ("linear" if x < -0.01 else "tie")
    best["winner"] = best.gap.map(sign)
    rep = (best.groupby(["regime", "dataset"]).winner
           .value_counts().unstack(fill_value=0))
    rep = rep[[c for c in ("nonlinear", "tie", "linear") if c in rep]]
    print(rep[rep.get("nonlinear", 0) > 0])

    print("\n== Winning method frequency (over all cells) ==")
    print(winners.groupby(["regime", "family"]).size().unstack(fill_value=0))
    print()
    print(winners.method.value_counts())

    print("\n== Stress targets detail (oracle best per family) ==")
    stress = best[best.dataset.isin(C.POOL_EXCLUDED)]
    print(stress.sort_values(["dataset", "regime"]).round(3).to_string(index=False))

    print("\n== Depth pattern: mean gap (best NL - best L per depth) across datasets ==")
    pdep = per_depth.pivot_table(index=["regime", "depth"],
                                 columns="family", values="best", aggfunc="mean")
    pdep["gap"] = pdep.nonlinear - pdep.linear
    print(pdep.round(3))


if __name__ == "__main__":
    main()
