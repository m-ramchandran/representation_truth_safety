"""Figure 2 probe-vs-logit data: best linear / nonlinear probe AUC per (model,
framing, training setting, dataset) beside the model's own answer AUC.

Joins data/q1_best_family_gap.csv (written by q1_linear_vs_nonlinear.py,
paper roster) with the canonical readout rows of data/q4_logit_auc.csv
(written by q4_model_logits.py). This table previously had no writer; the
join reproduces the stored file column for column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def main() -> None:
    gap = pd.read_csv(C.DATA / "q1_best_family_gap.csv")
    gap = gap[gap.dataset.isin(C.PAPER_DATASETS)]
    lg = pd.read_csv(C.DATA / "q4_logit_auc.csv")
    lg = (lg[lg.canonical][["model", "perturbation", "dataset", "auc"]]
          .rename(columns={"auc": "logit_auc"}))
    out = gap.merge(lg, on=["model", "perturbation", "dataset"], how="inner")
    out["probe_best"] = out[["linear", "nonlinear"]].max(axis=1)
    out["probe_minus_logit"] = out.probe_best - out.logit_auc
    out.to_csv(C.DATA / "figure2_probe_vs_logits.csv", index=False)
    print(f"wrote figure2_probe_vs_logits.csv: {len(out)} rows, "
          f"{out.dataset.nunique()} datasets")
    print(out.groupby("regime").probe_minus_logit.median().round(3))


if __name__ == "__main__":
    main()
