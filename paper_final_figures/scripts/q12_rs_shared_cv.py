"""Consolidate the cross-validated shared-direction-only AUCs.

rs_shared_cv.py (run locally through local_rs_shared_cv.py on
2026-09-03, compute cluster being down) writes one csv per expanded-pool collection
under classifiers/results/_rs_shared_cv/. This gathers them into
data/figure4_shared_direction_cv.csv, one row per
(model, perturbation, dataset, fold)
with the out-of-fold AUC of the shared direction alone (auc_shared), of the
full random-slopes model refit (auc_full) and of the stored run
(auc_stored); fold "all" pools the five folds' out-of-fold scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

SRC = C.REPO / "classifiers" / "results" / "_rs_shared_cv"


def main() -> None:
    coll = {v + "_expanded_pool": k for k, v in C.COLLECTIONS.items()}
    frames = []
    for f in sorted(SRC.glob("*_expanded_pool_layer*.csv")):
        d = pd.read_csv(f)
        d["model"] = d.collection.map(lambda c: coll[c][0])
        d["perturbation"] = d.collection.map(lambda c: coll[c][1])
        d["depth"] = [C.DEPTH_OF[(m, l)] for m, l in zip(d.model, d.layer)]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(C.DATA / "figure4_shared_direction_cv.csv", index=False)
    pooled = out[out.fold == "all"]
    print(f"{len(frames)} collections -> data/figure4_shared_direction_cv.csv; "
          f"max |auc_full - auc_stored| = "
          f"{(pooled.auc_full - pooled.auc_stored).abs().max():.4f}")
    print(pooled.pivot_table(index="dataset", columns=["model", "perturbation"],
                             values="auc_shared").round(3).to_string())


if __name__ == "__main__":
    main()
