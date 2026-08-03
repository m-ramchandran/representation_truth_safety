"""Leave-one-out: train pooled on the other datasets, evaluate on held-out data.

Two modes, one loop. --holdout dataset holds out each dataset D in turn:
train pooled on every other dataset, evaluate on D, which the training
pool never saw. --holdout task is the strict version: a held-out dataset
in dataset mode still has same-task siblings in the pool (counterfact is
held out but got remains), so hold out a whole task's datasets at once —
tasks are declared per collection (data.COLLECTIONS; original: tf = got,
common_claim, counterfact / medical = pubmedqa_artificial,
pubmedqa_labeled / jailbreak = harmful_harmless, wildjailbreak, xstest)
— and report an AUC per member dataset. Datasets added to a collection's
roster join every pool automatically (its eval-only units never do —
the pool is every *trainable* dataset outside the held-out unit);
membership counts come from what is actually loaded, never a hardcoded
roster. A holdout that removes nothing from the pool is skipped: an
eval-only unit alone (or a task whose loaded members are all eval-only)
would leave the pool identical to run_merged's whole-corpus fit, which
already scores the eval-only units — holding it out here would just
refit that same model. Eval-only units are still evaluated as members
of a held-out task alongside their trainable siblings, where the pool
genuinely differs. --exclude drops a dataset from the run entirely
(never pooled, never held out, never evaluated).

For every (feature set, layer, method) and held-out unit:
  - featurization (binning etc.) is fit on the pooled training data;
  - the method is tuned by CV on the pool, folds stratified on the joint
    (dataset, label) key, and refit on the whole pool;
  - the fit is applied to each held-out member dataset; AUC against that
    dataset's own labels.
The pool's CV predictions come free from the tuned fit and are stored too.

random_slopes scores held-out data with its shared weights only (the
held-out dataset's deviations were never fit). random_slopes_task fits
one deviation per task instead of per dataset and runs in task mode
only, because a dataset-mode row would mix semantics: most held-out
datasets' tasks are still in the pool but singleton tasks' never are.
A held-out task has no deviation row,
so its scoring is likewise shared-weights only. moe's gate reads the
activations, so the fitted mixture applies to unseen datasets unchanged;
its expert-count grid adds the pool's number of tasks and of datasets to
the defaults. svm_rbf runs through its Nystrom map (the exact kernel is
impossible at pool sizes); xgboost boosts trees on the GPU with early
stopping. Methods on pooled corpora default to the raw and
binned_activations feature sets (features.POOLED_FEATURE_SETS).

Outputs, under classifiers/results/<data>/leave_out/ — <tag> is
<features>_<raw>:
  tables/metrics_<tag>_layer<L>_<dataset|task>.csv
                        one row per (method, held-out unit, member test
                        dataset), AUC to 3 decimals; the resume record — a
                        (method, holdout) group present in the csv is
                        skipped on re-run.
  predictions/          one .npz per (tag, layer, method, kind, holdout)
                        with the pool's per-sample CV scores and the
                        held-out per-sample scores (probabilities, and
                        margins for every method but knn), the chosen
                        hyperparameters, the pool fit's coefficient
                        vector where the method has one (`coef`), and
                        what the featurization kept.
  figures/              one figure per (feature set, holdout kind)
                        (auc_<tag>_<kind>.png): a methods x held-out
                        test datasets panel per layer, two per row.

Run from the repo root:

    python -m classifiers.experiments.run_leave_out --holdout dataset --raw mean \
        [--data original] \
        [--features raw binned_activations] [--layers 9 17 22 29] \
        [--methods ... mlp random_slopes moe] [--skip qda:raw ...] \
        [--exclude claude_blood_test_ranges ...] [--seed 0] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..methods import METHODS, make_folds
from .features import (
    COLLECTIONS, POOLED_FEATURE_SETS, POOLED_METHOD_NAMES, SOURCES, SPARSE_OK,
    completed_groups, fit_features, load_datasets, pool_blocks,
    pooled_fit_kwargs, usable_units,
)
from .runner_common import (
    SKIP_ERRORS, add_shared_args, append_metrics, model_npz, parse_exclude,
    parse_skip, plot_method_by_dataset, release_gpu_memory, scores_npz,
    to_dense,
)

RESULTS = Path(__file__).parents[1] / "results"    # / <data> / leave_out
CSV_COLUMNS = ("features", "layer", "method", "holdout_kind", "holdout",
               "test_dataset", "n_test", "auc")
RESUME_KEY = ("features", "layer", "method", "holdout")


def holdout_units(data: dict, features: str, kind: str, c) -> dict:
    """holdout name -> its member datasets, over what is actually loaded
    and usable for this feature set. Dataset mode: every unit stands
    alone. Task mode: each of the collection's tasks with at least one
    loaded member, and a unit in no task stands as its own task. Either
    way, a holdout with no trainable member is dropped — it would leave
    the full trainable pool intact, i.e. refit the merged whole-corpus
    model (which already scores the eval-only units)."""
    usable = usable_units(data, features)
    if kind == "dataset":
        out = {name: [name] for name in usable}
    else:
        out = {}
        for task, datasets in c.tasks.items():
            members = [d for d in datasets if d in usable]
            if members:
                out[task] = members
        in_task = {d for members in c.tasks.values() for d in members}
        out.update({name: [name] for name in usable if name not in in_task})
    return {name: members for name, members in out.items()
            if any(m in c.train_datasets for m in members)}


def run_holdout(method: str, features: str, members: list, data: dict,
                densify: bool, seed: int, n_jobs: int,
                collection: str) -> tuple[list, dict]:
    """One (feature set, method, held-out unit): featurize and tune on the
    pool of every other usable trainable dataset, refit on the whole pool,
    and score each held-out member. Returns the metric rows (test_dataset,
    n_test, auc) and the npz contents."""
    c = COLLECTIONS[collection]
    pool = [n for n in usable_units(data, features)
            if n not in members and n in c.train_datasets]
    blocks, groups, arrays = pool_blocks(data, pool)
    X, transform, extras = fit_features(features, blocks)
    y = arrays["y"]
    if densify:
        X = to_dense(X)
    key = np.char.add(groups, y.astype(str))
    folds = make_folds(X, y, seed=seed, stratify_on=key)
    cv, _, params, fitted = METHODS[method](
        X, y, seed=seed, folds=folds, inner_n_jobs=n_jobs,
        **pooled_fit_kwargs(method, groups, collection))

    rows = []
    held_out = []                # (member dataset, its unit, its scores)
    for member in members:
        te = data[member]
        X_h = transform(te["blocks"])
        if densify:
            X_h = to_dense(X_h)
        scores = fitted.predict(X_h)
        rows.append((member, len(te["y"]),
                     f"{roc_auc_score(te['y'], scores['proba']):.3f}"))
        held_out.append((member, te, scores))

    npz = {
        "cv_sample_id": arrays["sample_id"], "cv_y": y, "cv_dataset": groups,
        "params": repr(params),
    }
    if (arrays["topic"] != "").any():   # per-sample topic (e.g. got's parts)
        npz["cv_topic"] = arrays["topic"]
    npz.update({f"cv_{k}": v for k, v in cv.items()})
    npz.update(scores_npz(held_out, "holdout"))
    npz.update(model_npz(fitted))      # the pool fit's coef, scaler, extras
    npz.update(extras)
    return rows, npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout", required=True, choices=("dataset", "task"))
    add_shared_args(ap, method_names=POOLED_METHOD_NAMES,
                    default_features=POOLED_FEATURE_SETS)
    args = ap.parse_args()
    skip = parse_skip(ap, args.skip, POOLED_METHOD_NAMES)
    c = COLLECTIONS[args.data]
    exclude = parse_exclude(ap, args.exclude, c)
    layers = args.layers or list(c.layers)
    results = RESULTS / args.data / "leave_out"

    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "predictions").mkdir(exist_ok=True)

    for layer in layers:
        roles = sorted({r for f in args.features for r in SOURCES[f]})
        print(f"layer{layer}: loading {roles} caches")
        data = load_datasets(layer, args.raw, roles, args.data)
        data = {n: u for n, u in data.items() if n not in exclude}

        for features in args.features:
            tag = f"{features}_{args.raw}"
            csv_path = (results / "tables" /
                        f"metrics_{tag}_layer{layer}_{args.holdout}.csv")
            done = completed_groups(csv_path, RESUME_KEY)
            units = holdout_units(data, features, args.holdout, c)
            for method in args.methods:
                if (method, features) in skip:
                    continue
                if method == "random_slopes_task" \
                        and args.holdout == "dataset":
                    # task-mode only: a dataset-mode row would mix
                    # semantics — a held-out
                    # dataset's task is usually still in the pool, a
                    # singleton task's never is
                    continue
                for holdout, members in units.items():
                    if (tag, str(layer), method, holdout) in done:
                        continue
                    print(f"  {tag} {method}, hold out {holdout} "
                          f"({sum(len(data[m]['y']) for m in members)} "
                          "held-out samples)", flush=True)
                    release_gpu_memory()   # see runner_common
                    try:
                        rows, npz = run_holdout(
                            method, features, members, data,
                            densify=method not in SPARSE_OK,
                            seed=args.seed, n_jobs=args.n_jobs,
                            collection=args.data)
                    except SKIP_ERRORS as e:  # see runner_common.SKIP_ERRORS
                        print(f"    skipped, {type(e).__name__}: {e}",
                              flush=True)
                        continue
                    np.savez_compressed(
                        results / "predictions" /
                        f"{tag}_layer{layer}_{method}_{args.holdout}_"
                        f"{holdout}.npz", **npz)
                    append_metrics(
                        csv_path, CSV_COLUMNS,
                        [(tag, layer, method, args.holdout, holdout) + r
                         for r in rows])
            # one figure per feature set, a panel per layer — drawn
            # from every layer csv present, so parallel single-layer
            # tasks converge on the complete figure
            plot_method_by_dataset(
                results / "tables",
                lambda L: f"metrics_{tag}_layer{L}_{args.holdout}.csv",
                tag, sorted(set(c.layers) | set(layers)),
                args.methods, c.datasets,
                xlabel="held-out test dataset",
                title=f"held-out AUC, {tag}, leave one "
                      f"{args.holdout} out",
                out_path=(results / "figures" /
                          f"auc_{tag}_{args.holdout}.png"))
    print("done")


if __name__ == "__main__":
    main()
