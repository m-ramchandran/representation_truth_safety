"""Q3: how similar are the classifiers' predictions, and does
agreement track separability / transferability? Do they ensemble?

Uses the analysis package's precomputed tables:
  prediction_correlation_*  Spearman correlation between per-sample
                            predictions of every method pair
  ensemble_auc_*            per dataset: each method's AUC + the
                            mean-probability ensemble (CV settings)
Writes the two Figure 3 input tables and prints a digest.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

# random slopes is a linear estimator (a penalized linear model with
# per-dataset offsets), so it belongs in the agreement analysis on the linear
# side. It only exists in the pooled regimes, so it simply has no rows in
# within_cv; the other pooled-only methods stay out (moe is a gate over
# experts, not a single linear rule).
AGREEMENT_METHODS = C.METHODS + ("random_slopes", "random_slopes_task")
AGREEMENT_FAMILY = {**C.FAMILY_OF, "random_slopes": "linear",
                    "random_slopes_task": "linear"}

PAIR_KIND = {}
for a in AGREEMENT_METHODS:
    for b in AGREEMENT_METHODS:
        fa, fb = AGREEMENT_FAMILY[a], AGREEMENT_FAMILY[b]
        PAIR_KIND[(a, b)] = ("within_linear" if fa == fb == "linear" else
                             "within_nonlinear" if fa == fb == "nonlinear"
                             else "cross_family")


def load_correlations() -> pd.DataFrame:
    rows = []
    for (model, pert), coll in C.COLLECTIONS.items():
        root = C.RESULTS / coll
        layers = C.LAYERS[model]
        specs = [("per_dataset", "prediction_correlation_raw_last_layer{L}.csv",
                  "within_cv"),
                 ("merged", "prediction_correlation_raw_last_layer{L}.csv",
                  "merged_included"),
                 ("leave_out", "prediction_correlation_raw_last_layer{L}_dataset.csv",
                  "leave_dataset_out"),
                 ("leave_out", "prediction_correlation_raw_last_layer{L}_task.csv",
                  "leave_domain_out")]
        for exp, pat, regime in specs:
            for layer in layers:
                f = root / exp / "tables" / pat.format(L=layer)
                if not f.exists():
                    continue
                c = pd.read_csv(f)
                if regime == "leave_dataset_out":
                    # keep the matched holdout, plus permanently excluded
                    # targets (never equal to any holdout): for those every
                    # holdout is target-absent; keep all their rows.
                    c = c[(c.holdout == c.dataset)
                          | c.dataset.isin(C.POOL_EXCLUDED)]
                elif regime == "leave_domain_out":
                    td = c.dataset.map(C.DOMAIN_OF)
                    hd = c.holdout.map(C.DOMAIN_RENAME)
                    c = c[td == hd]
                methods = [m for m in c.columns if m in AGREEMENT_METHODS]
                melted = c.melt(
                    id_vars=[x for x in ("method", "dataset") if x in c.columns],
                    value_vars=methods, var_name="method_b",
                    value_name="corr").rename(columns={"method": "method_a"})
                melted = melted[melted.method_a.isin(AGREEMENT_METHODS)
                                & (melted.method_a != melted.method_b)]
                melted["corr"] = pd.to_numeric(melted["corr"], errors="coerce")
                melted = melted.dropna(subset=["corr"])
                if "dataset" not in melted:
                    melted["dataset"] = "all"
                melted["pair_kind"] = [PAIR_KIND[(a, b)] for a, b in
                                       zip(melted.method_a, melted.method_b)]
                melted["model"], melted["perturbation"] = model, pert
                melted["regime"], melted["layer"] = regime, layer
                melted["depth"] = C.DEPTH_OF[(model, layer)]
                rows.append(melted)
    return pd.concat(rows, ignore_index=True)


def load_ensembles() -> pd.DataFrame:
    rows = []
    for (model, pert), coll in C.COLLECTIONS.items():
        for exp, regime in (("per_dataset", "within_cv"),
                            ("merged", "merged_included")):
            for f in glob.glob(str(C.RESULTS / coll / exp / "tables" /
                                   "ensemble_auc_raw_last_layer*.csv")):
                layer = int(f.rsplit("layer", 1)[1].split(".")[0])
                t = pd.read_csv(f)
                # NOT AGREEMENT_METHODS: we cannot verify which methods the
                # stored mean_proba column averaged over, so the singles it is
                # compared against stay on the standard 10-method roster.
                # Otherwise F13 would pit an ensemble of 10 against the best
                # of 12 and the gap would be an artifact of the rosters.
                methods = [m for m in t.columns if m in C.METHODS]
                t["best_single"] = t[methods].max(axis=1)
                t["median_single"] = t[methods].median(axis=1)
                rows.append(pd.DataFrame({
                    "model": model, "perturbation": pert, "regime": regime,
                    "layer": layer, "depth": C.DEPTH_OF[(model, layer)],
                    "dataset": t.dataset, "ensemble": t.mean_proba,
                    "best_single": t.best_single,
                    "median_single": t.median_single}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    corr = load_correlations()
    corr.to_csv(C.DATA / "figure3_prediction_similarity.csv", index=False)
    agg = (corr.groupby(["model", "perturbation", "regime", "layer", "depth",
                         "dataset", "pair_kind"], observed=True)["corr"]
           .median().unstack("pair_kind").reset_index())

    pd.set_option("display.width", 250)
    print("== median Spearman between predictions, by pair kind and regime ==")
    print(corr.groupby(["regime", "pair_kind"], observed=True)["corr"]
          .median().unstack().round(3))

    print("\n== cross-family agreement by depth (median over runs/datasets) ==")
    print(agg.groupby(["regime", "depth"]).cross_family.median().unstack().round(3))

    # does agreement track the target's difficulty?
    methods = pd.read_csv(C.DATA / "figure2_method_results.csv")
    collapsed = C.collapse_single_source(methods)
    best = (collapsed[collapsed.family.isin(["linear", "nonlinear"])]
            .groupby(["model", "perturbation", "regime", "dataset",
                      "family"]).auc.max().unstack("family").reset_index())
    best["probe_best"] = best[["linear", "nonlinear"]].max(axis=1)
    j = agg.merge(best[["model", "perturbation", "regime", "dataset",
                        "probe_best"]],
                  on=["model", "perturbation", "regime", "dataset"])
    from scipy.stats import spearmanr
    print("\n== Spearman(cross-family agreement, best AUC on that target) ==")
    for regime, g in j.groupby("regime"):
        # a few cells have no cross-family pair at all (a method missing for
        # that holdout); without nan_policy those propagate and the whole
        # regime silently prints as nan
        r, p = spearmanr(g.cross_family, g.probe_best, nan_policy="omit")
        n = int(g.cross_family.notna().sum())
        print(f"  {regime:18s} r={r:+.2f} (p={p:.1e}, n={n}, "
              f"{len(g) - n} cells dropped)")

    ens = load_ensembles()
    ens["ens_minus_best"] = ens.ensemble - ens.best_single
    ens["ens_minus_median"] = ens.ensemble - ens.median_single
    ens.to_csv(C.DATA / "figure3_method_ensemble.csv", index=False)
    print("\n== mean-probability ensemble vs singles (CV settings) ==")
    print(ens.groupby("regime")[["ens_minus_best", "ens_minus_median"]]
          .describe().round(3).loc[:, (slice(None), ["mean", "50%", "min", "max"])])
    print("\nper dataset (median over runs/layers), merged regime:")
    print(ens[ens.regime == "merged_included"]
          .groupby("dataset")[["ens_minus_best", "ens_minus_median"]]
          .median().round(3).sort_values("ens_minus_best"))


if __name__ == "__main__":
    main()
