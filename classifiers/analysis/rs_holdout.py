"""Would a held-out dataset be better served by its task's deviation?

In leave-one-dataset-out, `random_slopes` scores the held-out dataset
with the SHARED weights only: that dataset has no deviation row of its
own, and in a mixed model the prediction for an unseen group is the
fixed effect (the deviations carry a Gaussian prior centered at 0, so an
unobserved group's best guess is 0). But the groups here are not
exchangeable — dataset sits inside task — so for a held-out dataset
whose task is still represented in the pool (hold out counterfact and
got + common_claim remain), that task's own deviation is a better guess
than zero. This module re-scores the saved fits and compares held-out
AUC under three estimators — settled post hoc on the existing results,
before any runner commits to one:

  shared              w                    what the runner's csv reports
  task_mean           w + mean(dev_g)      g = the same-task pool members
  task_mean_weighted  w + weighted mean    weighted by each g's pool size

The weighted variant exists because each dev_g is shrunk roughly in
proportion to its own sample size, so the big datasets' deviations are
the better-determined ones (wildjailbreak's 50k vs xstest's 450).

Nothing is refit: `dev_w` / `group_labels` are already in every
leave_out npz, so this needs only the held-out activations and the fit's
saved featurization (directions.apply_featurization). AUC is rank-based,
which removes every constant from the comparison — the shared intercept
(never saved), dev_b, and the standardizer's mean all shift a dataset's
margins uniformly — so only the per-feature scale matters, and it is
folded into the vectors as `v / standardize_scale`. That also means this
works for EVERY feature set, binned_activations included: nothing is
compared across fits, so there is no shared-space requirement.

A held-out dataset whose task has no other member in the pool
(social_iqa, claude_vds, and every dataset in the fixture tree that
stands as its own task) has no task deviation to borrow; its two
task_mean rows are left empty.

    python -m classifiers.analysis.rs_holdout --tag raw_last \
        [--layers 9 17 ...] [--data ...] [--root ...]

Outputs under <root>/leave_out/: rs_holdout_<tag>.csv (long form:
layer, dataset, task, n_same_task, estimator, auc) and
rs_holdout_<tag>.png (a panel per layer, rows = estimator, columns =
held-out dataset). The csv also carries the `shared_stored` estimator —
the AUC of the scores the runner itself saved — as a check on this
module's rebuild of the featurization; on real runs it should match
`shared` to the third decimal, and any gap is printed. (On the fixture
tree it differs by design: fixtures add per-sample noise to margins.)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..data import COLLECTIONS, task_of
from . import directions
from .directions import DIRECTION_KEYS, ROLES, apply_featurization
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction, npz_path, resolve_root)
from .plots import layer_panels, save

RS_METHOD = "random_slopes"
HOLDOUT_KIND = "dataset"       # task mode holds out the whole task: no sibling
ESTIMATORS = ("shared", "task_mean", "task_mean_weighted")
RS_KEYS = DIRECTION_KEYS | {"dev_w", "group_labels", "cv_dataset",
                            "holdout_y", "holdout_sample_id",
                            "holdout_proba"}


def task_deviations(d: dict, dataset: str, data: str) -> dict:
    """{estimator -> deviation vector} for a held-out dataset, in the
    fit's standardized feature space. The same-task pool members are the
    deviation rows whose group shares the held-out dataset's task; with
    none, only the shared estimator is available (empty dict beyond it).
    Weights are each member's pool sample count (from cv_dataset)."""
    labels = np.asarray(d["group_labels"]).astype(str)
    dev_w = np.atleast_2d(np.asarray(d["dev_w"], dtype=float))
    task = task_of(dataset, data)
    same = [i for i, g in enumerate(labels) if task_of(g, data) == task]
    if not same:
        return {}
    counts = pd.Series(np.asarray(d["cv_dataset"]).astype(str)).value_counts()
    n = np.array([float(counts.get(labels[i], 0.0)) for i in same])
    rows = dev_w[same]
    weighted = (rows * n[:, None]).sum(0) / n.sum() if n.sum() > 0 \
        else rows.mean(0)
    return {"task_mean": rows.mean(0), "task_mean_weighted": weighted}


def holdout_rows(d: dict, tag: str, unit: dict, dataset: str,
                 data: str) -> list[dict]:
    """One held-out dataset's AUC under each estimator. `unit` is the
    dataset's loaded cache entry; its rows must be the ones the fit
    scored (checked against holdout_sample_id)."""
    stored_ids = np.asarray(d["holdout_sample_id"]).astype(str)
    if not np.array_equal(np.asarray(unit["sample_id"]).astype(str),
                          stored_ids):
        raise ValueError(f"{dataset}: cache rows differ from the fit's "
                         "holdout_sample_id (wrong layer or --raw?)")
    y = np.asarray(d["holdout_y"]).astype(int)
    X = apply_featurization(d, tag, unit["blocks"])
    scale = np.asarray(d["standardize_scale"], dtype=float)
    shared = np.asarray(d["coef"], dtype=float).ravel()

    def auc(v) -> float:
        # constants drop out of a rank metric, so only 1/scale matters;
        # X may be sparse (sae / transcoder) and @ handles that
        return float(roc_auc_score(y, X @ (v / scale)))

    labels = np.asarray(d["group_labels"]).astype(str)
    n_same = sum(task_of(g, data) == task_of(dataset, data) for g in labels)
    devs = task_deviations(d, dataset, data)
    values = {"shared": auc(shared),
              "shared_stored": float(roc_auc_score(y, d["holdout_proba"])),
              **{name: auc(shared + dev) for name, dev in devs.items()}}
    return [{"dataset": dataset, "task": task_of(dataset, data),
             "n_same_task": n_same, "estimator": name, "auc": round(v, 4)}
            for name, v in values.items()]


def rs_figure(df: pd.DataFrame, tag: str):
    """A panel per layer, rows = estimator, columns = held-out dataset.
    Also called by redraw.py on the saved csv."""
    by_layer = {
        int(layer): g[g["estimator"].isin(ESTIMATORS)]
        .pivot(index="estimator", columns="dataset", values="auc")
        .reindex(list(ESTIMATORS))
        for layer, g in df.groupby("layer")}
    return layer_panels(
        by_layer, "Held-out AUC: shared weights vs the task's deviation",
        f"{tag} — random_slopes, leave one dataset out; task_mean rows "
        "are blank where no same-task dataset stayed in the pool",
        vmin=0.5, vmax=1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="default: the collection's layers")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    features, raw = args.tag.rsplit("_", 1)
    layers = args.layers or list(COLLECTIONS[args.data].layers)
    tables, figures = analysis_dirs(args.root, "leave_out")

    metrics = load_metrics("leave_out", args.root)
    metrics = metrics[(metrics["features"] == args.tag)
                      & (metrics["method"] == RS_METHOD)
                      & (metrics["holdout_kind"] == HOLDOUT_KIND)]
    if metrics.empty:
        raise FileNotFoundError(f"no {RS_METHOD} {HOLDOUT_KIND}-mode "
                                f"leave_out runs for tag={args.tag!r}")

    rows = []
    for layer in layers:
        held = sorted(metrics[metrics["layer"] == layer]["holdout"].unique())
        if not held:
            print(f"  layer {layer}: no runs, skipped")
            continue
        units = directions.load_units(layer, raw, ROLES[features], args.data)
        for dataset in held:
            try:
                d = load_prediction(
                    npz_path("leave_out", args.tag, layer, RS_METHOD,
                             f"{HOLDOUT_KIND}_{dataset}", args.root),
                    keys=RS_KEYS)
                new = holdout_rows(d, args.tag, units[dataset], dataset,
                                   args.data)
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  layer {layer} {dataset}: skipped ({e})")
                continue
            rows += [{"features": args.tag, "layer": layer, **r} for r in new]
    if not rows:
        raise FileNotFoundError("no fits could be re-scored")

    df = pd.DataFrame(rows)
    df.to_csv(tables / f"rs_holdout_{args.tag}.csv", index=False)
    save(rs_figure(df, args.tag), figures / f"rs_holdout_{args.tag}.png")

    wide = df.pivot_table(index=["layer", "dataset"], columns="estimator",
                          values="auc")
    gap = (wide["shared"] - wide["shared_stored"]).abs().max()
    print(f"  rebuild check: largest |shared - shared_stored| = {gap:.4f}"
          + ("  (expected on the fixture tree: fixtures add margin noise)"
             if gap > 0.005 else ""))
    for est in ("task_mean", "task_mean_weighted"):
        if est in wide:
            delta = (wide[est] - wide["shared"]).dropna()
            if len(delta):
                print(f"  {est} - shared: mean {delta.mean():+.4f}, "
                      f"best {delta.max():+.4f}, worst {delta.min():+.4f} "
                      f"over {len(delta)} (layer, dataset) pairs")
    print(f"rs_holdout: {len(df)} rows -> {tables.parent}")


if __name__ == "__main__":
    main()
