"""Performance below the dataset level.

Two views of one (tag, layer)'s CV predictions:

- got datasets: got is seven curated datasets merged; each part's
  AUC comes
  from the saved cv_topic where the npz has one, else the sample_id
  prefix ("cities:12" — the original collection's per-part files). Works
  on per_dataset (got's own fit) and merged (the pooled fit).
- merged slices: per (method, dataset), the AUC next to accuracy, the
  base rate, and the predicted-positive rate — a dataset can have fine
  AUC yet sit wholesale on one side of the pooled model's threshold.

    python -m classifiers.analysis.per_slice --experiment merged \
        --tag raw_last [--layers 17 22 29 33] [--root ...]

Csvs are per layer; the got-by-dataset figure holds all requested
layers as panels (got_datasets_<tag>.png). Outputs join the
experiment's tables/ and figures/.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import COLLECTIONS
from .load_results import (add_data_args, analysis_dirs, auc_or_nan,
                           cv_predictions, resolve_root)
from .plots import layer_panels, save


def got_datasets(preds: pd.DataFrame) -> pd.DataFrame:
    """method x part AUC inside got."""
    got = preds[preds["dataset"] == "got"].copy()
    got["part"] = got["topic"].where(got["topic"] != "",
                                     got["sample_id"].str.split(":").str[0])
    return (got.groupby(["method", "part"])
            .apply(lambda d: auc_or_nan(d["y"], d["proba"]))
            .unstack("part").round(3))


def dataset_slices(preds: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset): AUC, accuracy at 0.5, the dataset's base
    rate, and the predicted-positive rate."""
    def stats(d):
        pred = d["proba"] >= 0.5
        return pd.Series({"auc": auc_or_nan(d["y"], d["proba"]),
                          "accuracy": (pred == (d["y"] == 1)).mean(),
                          "base_rate": (d["y"] == 1).mean(),
                          "positive_rate": pred.mean()})
    return preds.groupby(["method", "dataset"]).apply(stats).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", default="merged",
                    choices=("per_dataset", "merged"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="default: the collection's layers")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    layers = args.layers or list(COLLECTIONS[args.data].layers)

    tables, figures = analysis_dirs(args.root, args.experiment)
    parts_by_layer, written = {}, []
    for layer in layers:
        try:
            preds = cv_predictions(args.experiment, args.tag, layer,
                                   args.root)
        except FileNotFoundError as e:
            print(f"  layer {layer}: skipped ({e})")
            continue
        stem = f"{args.tag}_layer{layer}"
        if (preds["dataset"] == "got").any():
            parts_by_layer[layer] = got_datasets(preds)
            parts_by_layer[layer].to_csv(tables / f"got_datasets_{stem}.csv")
            written += [f"got_datasets L{layer}"]
        if args.experiment == "merged":
            dataset_slices(preds).to_csv(tables / f"dataset_slices_{stem}.csv")
            written += [f"dataset_slices L{layer}"]

    # one figure across the layers, a panel each: fewer files, and
    # panel counts that stay legible on a slide
    if parts_by_layer:
        save(layer_panels(parts_by_layer,
                          f"AUC by got dataset ({args.experiment})",
                          args.tag, vmin=0.5, vmax=1.0),
             figures / f"got_datasets_{args.tag}.png")
        written += ["got_datasets figure"]

    print(f"per_slice: wrote {', '.join(written)} under {tables.parent}")


if __name__ == "__main__":
    main()
