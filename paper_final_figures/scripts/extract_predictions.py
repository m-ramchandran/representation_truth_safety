"""Run ON compute cluster: reduce prediction npz trees to compact parquet.

For each collection, writes paper_final_figures_cache/<collection>_heldout.parquet
with one row per (layer, kind, holdout, method, sample): the held-out
probabilities of every leave_out fit, plus the merged run's eval_only
scores (kind='merged_eval') and the merged run's pooled-CV scores
(kind='merged_cv', holdout='pool').

Columns: layer int16, kind, holdout, method, dataset, idx int32,
y int8, proba float32. `idx` is the position inside that fit's stored
arrays, so methods align within one (layer, kind, holdout) group —
verified locally by comparing y sequences.

    python extract_predictions.py <results_root> <out_dir>
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

COLLECTIONS = ("gemma_original", "gemma_tf_a", "qwen_original", "qwen_tf_a")


def leave_out_rows(root: str) -> list[pd.DataFrame]:
    rows = []
    pattern = os.path.join(root, "leave_out", "predictions",
                           "raw_last_layer*.npz")
    rx = re.compile(r"raw_last_layer(\d+)_([a-z0-9_]+?)_(dataset|task)_(.+)\.npz$")
    for path in sorted(glob.glob(pattern)):
        m = rx.search(os.path.basename(path))
        if not m:
            continue
        layer, method, kind, holdout = m.groups()
        try:
            with np.load(path, allow_pickle=True) as d:
                if "holdout_proba" not in d.files:
                    continue
                ds = d["holdout_dataset"].astype(str)
                y = d["holdout_y"].astype(np.int8)
                p = d["holdout_proba"].astype(np.float32)
        except Exception as exc:                        # noqa: BLE001
            print(f"  skip {os.path.basename(path)}: {exc}")
            continue
        rows.append(pd.DataFrame({
            "layer": np.int16(layer), "kind": f"leave_{kind}",
            "holdout": holdout, "method": method, "dataset": ds,
            "idx": np.arange(len(y), dtype=np.int32), "y": y, "proba": p}))
    return rows


def merged_rows(root: str) -> list[pd.DataFrame]:
    rows = []
    pattern = os.path.join(root, "merged", "predictions", "raw_last_layer*.npz")
    rx = re.compile(r"raw_last_layer(\d+)_([a-z0-9_]+)\.npz$")
    for path in sorted(glob.glob(pattern)):
        m = rx.search(os.path.basename(path))
        if not m:
            continue
        layer, method = m.groups()
        try:
            with np.load(path, allow_pickle=True) as d:
                if "cv_proba" in d.files:
                    y = d["cv_y"].astype(np.int8)
                    rows.append(pd.DataFrame({
                        "layer": np.int16(layer), "kind": "merged_cv",
                        "holdout": "pool", "method": method,
                        "dataset": d["cv_dataset"].astype(str),
                        "idx": np.arange(len(y), dtype=np.int32),
                        "y": y, "proba": d["cv_proba"].astype(np.float32)}))
                if "eval_only_proba" in d.files:
                    y = d["eval_only_y"].astype(np.int8)
                    rows.append(pd.DataFrame({
                        "layer": np.int16(layer), "kind": "merged_eval",
                        "holdout": "pool", "method": method,
                        "dataset": d["eval_only_dataset"].astype(str),
                        "idx": np.arange(len(y), dtype=np.int32),
                        "y": y,
                        "proba": d["eval_only_proba"].astype(np.float32)}))
        except Exception as exc:                        # noqa: BLE001
            print(f"  skip {os.path.basename(path)}: {exc}")
    return rows


def main() -> None:
    results_root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for coll in COLLECTIONS:
        root = os.path.join(results_root, coll)
        if not os.path.isdir(os.path.join(root, "leave_out", "predictions")):
            print(f"{coll}: no predictions, skipped")
            continue
        frames = leave_out_rows(root) + merged_rows(root)
        if not frames:
            print(f"{coll}: nothing extracted")
            continue
        out = pd.concat(frames, ignore_index=True)
        for c in ("kind", "holdout", "method", "dataset"):
            out[c] = out[c].astype("category")
        path = os.path.join(out_dir, f"{coll}_heldout.parquet")
        out.to_parquet(path, compression="zstd")
        print(f"{coll}: {len(out):,} rows -> {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
