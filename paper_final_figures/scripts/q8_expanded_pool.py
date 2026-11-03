"""Expanded-pool comparison: with VDS and WJB-adversarial INSIDE the
pooled corpus, does a single pooled linear rule underserve them, and
does partial pooling (random slopes) recover them?

Reads classifiers/results/<coll>_expanded_pool/merged/tables at the
85%-depth layer (gemma 29 / qwen 24) plus the original-run context
(within-CV and target-absent numbers). Writes figure4_expanded_pool_auc.csv,
which the
combined pooled-training figure (q12_figures.py) draws its lower block from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

LAYER = {"gemma": 29, "qwen": 24}
SHORT = {"claude_anatomy": "Anatomy", "claude_vds": "VDS",
         "wildjailbreak_adversarial": "WJB-adv"}


def load() -> pd.DataFrame:
    rows = []
    orig = C.collapse_single_source(C.long_results())
    for (model, pert), coll in C.COLLECTIONS.items():
        path = (C.RESULTS / f"{coll}_expanded_pool" / "merged" / "tables" /
                f"metrics_raw_last_layer{LAYER[model]}.csv")
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t = t[t.kind == "per_dataset"]
        g = orig[(orig.model == model) & (orig.perturbation == pert)
                 & (orig.depth == 3) & (orig.method == "l2reg")]
        wc = g[g.regime == "within_cv"].set_index("dataset").auc
        for r in t.itertuples():
            rows.append({"model": model, "perturbation": pert,
                         "dataset": r.test_dataset, "method": r.method,
                         "auc": r.auc,
                         "within_cv_l2": wc.get(r.test_dataset, np.nan)})
    return pd.DataFrame(rows)


def main() -> None:
    t = load()
    t.to_csv(C.DATA / "figure4_expanded_pool_auc.csv", index=False)
    pd.set_option("display.width", 220)
    have_l2 = t[t.method == "l2reg"].groupby(["model", "perturbation"]).size()
    print("l2reg present in:", dict(have_l2))

    piv = t.pivot_table(index=["model", "perturbation", "dataset"],
                        columns="method", values="auc").reset_index()
    piv["rs_minus_l2"] = piv.random_slopes - piv.get("l2reg", np.nan)
    piv["l2_minus_within"] = piv.get("l2reg", np.nan) - piv.merge(
        t[t.method == "l2reg"][["model", "perturbation", "dataset",
                                "within_cv_l2"]].drop_duplicates(),
        on=["model", "perturbation", "dataset"], how="left").within_cv_l2
    print("\n== stress targets ==")
    sel = piv[piv.dataset.isin(SHORT)]
    print(sel.round(3).to_string(index=False))
    print("\n== all 12 pool datasets, rs - pooled L2 ==")
    for (m, p), g in piv.groupby(["model", "perturbation"]):
        w = int((g.rs_minus_l2 > 0.01).sum())
        l = int((g.rs_minus_l2 < -0.01).sum())
        print(f"  {m:6s} {p:11s} n={len(g):2d}  win {w}  tie {len(g)-w-l}  "
              f"loss {l}  (median {g.rs_minus_l2.median():+.3f})")



if __name__ == "__main__":
    main()
