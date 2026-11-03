"""Q12 — the full pooled-regime method comparison behind F2.

F2's "random slopes beats the best nonlinear method" was supported by
Anatomy alone; this writes the complete merged-regime picture: every
pool dataset x run, best-over-layers AUC per method family (5 plain
linear, 5 nonlinear) next to the pooled-only estimators
(random_slopes, random_slopes_task, moe, moe_hard), plus the
corpus-wide AUC (run_merged's "overall" row: the mean over the five
outer folds of the pooled test-fold AUC — not one score over the whole
pooled test set).

Headline shape of the result (see digest): the only cells where the
best nonlinear beats the best plain linear by more than 0.02 are the
four Anatomy cells, and exactly there random slopes overtakes the
nonlinear family; 24/40 cells are three-way ties within ±0.01 (30/40
for the rs-vs-best-nonlinear pair alone); the one replicated
random-slopes loss is PubMedQA-lab (small dataset, dominant
same-domain sibling), which the domain-level variant recovers.

Input: classifiers/results/*/merged/tables (pooled-only methods are
not in long_results_raw).
Output: data/q12_pooled_methods.csv, data/q12_pooled_overall.csv,
data/q12_digest.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

LINEAR = ["l1reg", "l2reg", "svm_linear", "diff", "lda"]
NONLINEAR = ["svm_rbf", "knn", "mlp", "xgboost", "qda"]
POOLED_ONLY = ["random_slopes", "random_slopes_task", "moe", "moe_hard"]

_LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _LINES.append(text)


def load_merged() -> pd.DataFrame:
    frames = []
    for (model, pert), coll in C.COLLECTIONS.items():
        for f in sorted((C.RESULTS / coll / "merged" / "tables")
                        .glob("metrics_raw_last_layer*.csv")):
            frames.append(pd.read_csv(f).assign(model=model,
                                                perturbation=pert))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    mg = load_merged()

    # ---- corpus-wide single AUC over the pooled test set ---------------
    ov = mg[mg.kind == "overall"].groupby(
        ["model", "perturbation", "method"]).auc
    overall = pd.DataFrame({"auc_best_layer": ov.max(),
                            "auc_mean_layer": ov.mean()}).reset_index()
    overall.to_csv(C.DATA / "q12_pooled_overall.csv", index=False)
    best = overall.pivot_table(index=["model", "perturbation"],
                               columns="method", values="auc_best_layer")
    say("== corpus-wide pooled AUC (best over layers) ==")
    say(pd.DataFrame({
        "best_linear": best[LINEAR].max(axis=1),
        "best_nonlinear": best[NONLINEAR].max(axis=1),
        "random_slopes": best.random_slopes,
        "rs_task": best.random_slopes_task,
        "moe": best.moe}).round(3).to_string())

    # ---- per-dataset table, both layer aggregations --------------------
    pdset = mg[mg.kind == "per_dataset"]
    frames = []
    for agg in ("best_layer", "mean_layer"):
        g = pdset.groupby(["model", "perturbation", "test_dataset",
                           "method"]).auc
        wide = (g.max() if agg == "best_layer" else g.mean()).unstack()
        t = pd.DataFrame({
            "best_linear": wide[LINEAR].max(axis=1),
            "best_nonlinear": wide[NONLINEAR].max(axis=1),
            "mlp": wide.mlp,
            "random_slopes": wide.random_slopes,
            "random_slopes_task": wide.random_slopes_task,
            "moe": wide.moe, "moe_hard": wide.moe_hard,
        })
        t["rs_vs_nonlinear"] = t.random_slopes - t.best_nonlinear
        t["rs_vs_linear"] = t.random_slopes - t.best_linear
        t["nonlinear_vs_linear"] = t.best_nonlinear - t.best_linear
        frames.append(t.assign(aggregation=agg).reset_index())
    table = pd.concat(frames, ignore_index=True).rename(
        columns={"test_dataset": "dataset"})
    table["domain"] = table.dataset.map(C.DOMAIN_OF)
    table.to_csv(C.DATA / "q12_pooled_methods.csv", index=False)

    tb = table[table.aggregation == "best_layer"].set_index(
        ["model", "perturbation", "dataset"])
    say("\n== per-dataset summary over the 40 (dataset x run) cells "
        "(best over layers) ==")
    for col, lab in [("rs_vs_nonlinear", "rs - best nonlinear"),
                     ("rs_vs_linear", "rs - best plain linear"),
                     ("nonlinear_vs_linear", "best nonlinear - best linear")]:
        # round to the runner csvs' 3-dp grain first, so the counts
        # reproduce from the written CSV (0.92-0.91 is 0.010000000000000009
        # in float, which a bare > 0.01 counts as a win)
        v = tb[col].round(3)
        say(f"  {lab:30s}: median {v.median():+.3f}  "
            f"win/tie/loss (±0.01): {(v > 0.01).sum()}/"
            f"{((v >= -0.01) & (v <= 0.01)).sum()}/{(v < -0.01).sum()}")

    say("\n== interference cells (best nonlinear - best linear > 0.02) ==")
    say(tb[tb.nonlinear_vs_linear > 0.02]
        [["best_linear", "best_nonlinear", "random_slopes",
          "rs_vs_nonlinear"]].round(3).to_string())

    say("\n== replicated random-slopes losses (rs - best nonlinear < -0.01) ==")
    say(tb[tb.rs_vs_nonlinear < -0.01]
        [["best_linear", "best_nonlinear", "random_slopes",
          "random_slopes_task", "moe"]].round(3).to_string())

    say("\n== full per-dataset table (best over layers) ==")
    for (m, p), g in tb.groupby(level=[0, 1]):
        say(f"\n  {m} {p}:")
        say(g.droplevel([0, 1])
            [["best_linear", "best_nonlinear", "random_slopes",
              "random_slopes_task", "moe"]].round(3).to_string())

    (C.DATA / "q12_digest.txt").write_text("\n".join(_LINES) + "\n")
    say("\nwrote data/q12_pooled_methods.csv, q12_pooled_overall.csv, "
        "q12_digest.txt")


if __name__ == "__main__":
    main()
