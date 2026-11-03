"""Distill the projected-coefficient steering runs into the figure's data.

Reads, for Gemma-3-4B layer 29 and Qwen2.5-7B layer 24, the analyzer's
summary and curve tables of the runs tagged gemmaproj / qwenproj under
steering/out/ (pulled from the cluster), the direction manifest, and the
builder's geometry table, and writes

  data/figure6_projected_effects.csv   one row per vector and dataset:
      effect_per_1000     answer shift per 1000 units of push (True/False)
      per_step            the same per steering step of size 1, i.e. per
                          one True-minus-False class separation of that
                          dataset (effect_per_1000 x dose base / 1000)
      *_safe              the same read under the safe/unsafe word pair
      source, k           which coefficient vector, how many components kept
      kept_share          share of the vector's squared length inside them
      auc_target/train    how well the projected vector separates the
                          classes on the target rows / its training rows
                          (in-sample for the own-dataset fit)
      cos_own_diff, cos_famout_diff
  data/figure6_projected_curves.csv    per-dose means for both word pairs

    python paper_final_figures/scripts/distill_projected_steering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

STEER = C.REPO / "steering" / "out"
RUNS = [("gemma", "google/gemma-3-4b-it", "gemmaproj", 29),
        ("qwen", "Qwen/Qwen2.5-7B-Instruct", "qwenproj", 24)]


def source_of(kind: str) -> tuple[str, int]:
    if "_pc" in kind:
        stem, k = kind.rsplit("_pc", 1)
        return stem, int(k)
    if kind in ("taskout_diff", "datasetout_diff"):
        return "famout_diff", 0
    return kind, 0


def main() -> None:
    effects, curves = [], []
    for model, model_id, tag, L in RUNS:
        rd = STEER / f"runs_{tag}_layer{L}"
        man = pd.read_csv(STEER / f"manifest_{tag}_layer{L}.csv")
        geo = pd.read_csv(STEER / f"projected_geometry_{tag}_layer{L}.csv")
        geo = geo[geo.in_sweep].rename(columns={"name": "direction"})
        s_true = pd.read_csv(rd / "summary.csv")
        s_safe = pd.read_csv(rd / "summary_safe.csv")
        c_true = pd.read_csv(rd / "curves.csv")
        c_safe = pd.read_csv(rd / "curves_safe.csv")
        base = c_true[c_true.frac == 1.0].groupby("target").alpha.first()
        df = s_true[["target", "direction", "kind", "uses_target_data", "flipped",
                     "effect_per_1000", "effect_per_1pct", "n_usable",
                     "moved_right_way", "baseline_auc", "dose_base"]].copy()
        if not np.allclose(df.dose_base, df.target.map(base), rtol=1e-3):
            raise SystemExit(f"{tag}: dose base in run_meta and alpha at size 1 disagree")
        safe = s_safe[["direction", "effect_per_1000", "moved_right_way", "baseline_auc"]] \
            .rename(columns={"effect_per_1000": "effect_safe_per_1000",
                             "moved_right_way": "moved_right_way_safe",
                             "baseline_auc": "baseline_auc_safe"})
        df = df.merge(safe, on="direction", how="left")
        df = df.merge(man[["name", "training_set", "k", "kept_share"]]
                      .rename(columns={"name": "direction"}), on="direction", how="left")
        df = df.merge(geo[["direction", "auc_target", "auc_train",
                           "cos_own_diff", "cos_famout_diff"]], on="direction", how="left")
        src = df.kind.map(source_of)
        df["source"] = [s for s, _ in src]
        df["per_step"] = df.effect_per_1000 * df.dose_base / 1000.0
        df["per_step_safe"] = df.effect_safe_per_1000 * df.dose_base / 1000.0
        df.insert(0, "layer", L)
        df.insert(0, "model_id", model_id)
        df.insert(0, "model", model)
        effects.append(df)
        for pair, c in (("true", c_true), ("safe", c_safe)):
            c = c.copy()
            c["pair"] = pair
            c.insert(0, "layer", L)
            c.insert(0, "model", model)
            curves.append(c.merge(man[["name", "kind"]].rename(columns={"name": "direction"}),
                                  on="direction", how="left"))
        print(f"{tag}: {len(df)} vectors over {df.target.nunique()} datasets; "
              f"baseline AUC {df.groupby('target').baseline_auc.first().round(3).to_dict()}")
    eff = pd.concat(effects, ignore_index=True)
    cols = ["model", "model_id", "layer", "target", "direction", "kind", "source", "k",
            "uses_target_data", "training_set", "kept_share", "flipped", "baseline_auc",
            "baseline_auc_safe", "dose_base", "effect_per_1000", "per_step",
            "effect_per_1pct", "effect_safe_per_1000", "per_step_safe", "n_usable",
            "moved_right_way", "moved_right_way_safe", "auc_target", "auc_train",
            "cos_own_diff", "cos_famout_diff"]
    eff[cols].to_csv(C.DATA / "figure6_projected_effects.csv", index=False,
                     float_format="%.6g")
    pd.concat(curves, ignore_index=True).to_csv(
        C.DATA / "figure6_projected_curves.csv", index=False, float_format="%.6g")
    print("wrote data/figure6_projected_effects.csv and figure6_projected_curves.csv")


if __name__ == "__main__":
    main()
