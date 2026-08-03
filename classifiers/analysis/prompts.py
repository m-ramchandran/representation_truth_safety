"""Candidate mislabels: samples the in-domain classifiers get
confidently wrong, reported with their prompt text.

Uses the per_dataset CV predictions (each dataset judged by its own
fit — the high-AUC, in-domain view, which is where a consistent
confident error most plausibly means the label is wrong rather than the
model). A sample counts as confidently wrong for a method when the
method misclassifies it with |log-odds| > threshold (default 2, i.e.
probability past ~0.88); the report keeps samples that are confidently
wrong for at least --min-frac of the methods that have a margin (knn
never counts). Prompt text is joined through the activation caches via
classifiers.data — sample_id is "<source>:<row>", stable across layers.

    python -m classifiers.analysis.prompts --tag raw_last --layer 9 \
        [--threshold 2.0] [--min-frac 1.0] [--root ...]

Output: <root>/per_dataset/analysis/confidently_wrong_<tag>_layer<L>.csv
(dataset, sample_id, y, methods scored / wrong / confidently wrong,
mean log-odds, prompt), sorted worst-first. Treat pre-production output
as placeholder — the single-dataset runs are being redone.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import COLLECTIONS, load_prompts
from .load_results import (add_data_args, analysis_dirs,
                           eval_predictions, is_wrong, resolve_root)


def load_prompt_map(layer: int, data: str = "original") -> dict:
    """sample_id -> prompt text, from every HF source of the collection
    that stores prompts (patchable for tests). Reads only the prompts
    arrays (data.load_prompts) — no activation decompression."""
    out = {}
    for source in COLLECTIONS[data].hf_files:
        try:
            found = load_prompts(source, layer, data)
        except Exception as e:                    # missing file on HF
            print(f"  {source}: no cache loaded ({e})")
            continue
        if not found:
            print(f"  {source}: cache has no prompts key")
            continue
        out.update(found)
    return out


def confidently_wrong(cv: pd.DataFrame, threshold: float,
                      min_frac: float) -> pd.DataFrame:
    """Per sample of each dataset's own CV predictions: how many
    margin-methods got it wrong, and confidently so; keep samples
    confidently wrong for >= min_frac of them."""
    cv = cv[cv["fit"] == cv["dataset"]].copy()    # in-domain CV only
    cv = cv[cv["logodds"].notna()]
    cv["wrong"] = is_wrong(cv["proba"], cv["y"])
    cv["conf_wrong"] = cv["wrong"] & (cv["logodds"].abs() > threshold)
    g = cv.groupby(["dataset", "sample_id", "y"]).agg(
        n_scored=("method", "nunique"),
        n_wrong=("wrong", "sum"),
        n_conf_wrong=("conf_wrong", "sum"),
        mean_logodds=("logodds", "mean")).reset_index()
    g = g[g["n_conf_wrong"] >= np.ceil(min_frac * g["n_scored"])]
    return g.sort_values(["n_conf_wrong", "mean_logodds"],
                         key=lambda c: -c.abs() if c.name ==
                         "mean_logodds" else -c).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="|log-odds| above this counts as confident")
    ap.add_argument("--min-frac", type=float, default=1.0,
                    help="fraction of margin-methods that must be "
                         "confidently wrong")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    cv, _ = eval_predictions("per_dataset", args.tag, args.layer,
                             args.root, with_held=False)
    report = confidently_wrong(cv, args.threshold, args.min_frac)

    prompts = load_prompt_map(args.layer, args.data)
    report["prompt"] = [prompts.get(s, "") for s in report["sample_id"]]
    missing = int((report["prompt"] == "").sum())
    if missing:
        print(f"  {missing}/{len(report)} samples had no prompt text")

    tables, _ = analysis_dirs(args.root, "per_dataset")
    path = tables / f"confidently_wrong_{args.tag}_layer{args.layer}.csv"
    report.round({"mean_logodds": 3}).to_csv(path, index=False)
    per_ds = report.groupby("dataset").size()
    print(f"prompts: {len(report)} candidate mislabels "
          f"({dict(per_ds)}) -> {path}")


if __name__ == "__main__":
    main()
