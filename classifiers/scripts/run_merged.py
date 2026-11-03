"""Merged: nested CV on all datasets pooled into one corpus.

Does one classifier fit all the data at once, and which datasets does it
serve well or badly? Every loaded trainable dataset (the --data
collection's roster minus its eval-only and pool_excluded units) is
pooled and evaluated by
nested CV — the same protocol as the per-dataset diagonal, on the pooled
corpus: 5 outer folds; on each outer train part the method tunes its
hyperparameters on inner folds and then predicts the outer test fold.
Outer and inner folds are stratified on the joint (dataset, label) key,
and group-aware by default (--group-folds): rows expanded from one source
item — Social IQa's answer choices, got's negation pairs — stay in the
same fold at both levels, so a fold is never scored on a near duplicate of
something it trained on. Groups are namespaced by unit, so two datasets
cannot collide on one key. See data.GROUP_RULES; --no-group-folds restores
the plain stratified split the results on disk were produced with.
Featurization (binning etc.) is fit on the pooled corpus, label-free.

Reported per (feature set, layer, method), all from the same CV
predictions (no extra fits): the overall AUC (mean over the outer folds)
and a per-dataset AUC (that dataset's samples' CV predictions, pooled
across outer folds — per-fold-per-dataset would leave the small datasets
with too few samples).

random_slopes fits per-dataset deviations and uses them for every
prediction (unlike leave-one-out, every dataset here is seen in
training); random_slopes_task is the same model with one deviation per
task instead of per dataset (the eval-only units get their task's
deviation, with the shared-only scores stored alongside); moe's
expert-count grid adds the pool's number of tasks and of datasets to the
defaults; svm_rbf runs through its Nystrom map; xgboost boosts trees on
the GPU with early stopping. Pooled runs default to the raw and
binned_activations feature sets (features.POOLED_FEATURE_SETS). Datasets added to a collection's roster
join the pool automatically; --exclude drops a dataset from the run
entirely (never pooled, never evaluated).

The collection's never-pooled units — eval_only and pool_excluded alike
— are scored here by the whole-corpus fit (kind "eval_only" — inference
only, no extra training): the pool behind that fit is exactly the pool a
leave-one-dataset-out holdout of such a unit would train on, so this row
IS that experiment's answer, without the redundant refit. For the three
held-out sets this is their headline held-out-dataset number.

Outputs, under classifiers/results/<data>/merged/ — <tag> is
<features>_<raw>:
  tables/metrics_<tag>_layer<L>.csv
                        one overall row + one row per dataset for each
                        method (plus one "eval_only" row per eval-only
                        unit, scored by the whole-corpus fit), AUC to 3
                        decimals; the resume record — a method whose
                        rows are present is skipped on re-run.
  predictions/          one .npz per (tag, layer, method) with per-sample
                        CV scores over the outer folds (probabilities,
                        and margins for every method but knn), the outer
                        fold index, the per-outer-fold chosen
                        hyperparameters, the pooled samples' topic where
                        any cache has one (`cv_topic`), and what the
                        featurization kept. Also one tuned fit on the
                        WHOLE pooled corpus (not part of the nested-CV
                        AUCs): its chosen hyperparameters
                        (`full_params`), coefficient vector where the
                        method has one (`full_coef`) — the pooled model
                        the coefficient analyses read — and its
                        eval-only scores (`eval_only_*`).
  figures/              one figure per feature set (auc_<tag>.png): a
                        methods x (all + per-dataset + eval-only) AUC
                        panel per layer, two per row.

Run from the repo root:

    python -m classifiers.scripts.run_merged --raw mean \
        [--data gemma_tf_a] \
        [--features raw binned_activations] [--layers 17 22 29 33] \
        [--methods ... mlp random_slopes moe] [--skip qda:raw ...] \
        [--exclude claude_blood_test_ranges ...] [--no-group-folds] \
        [--seed 0] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import METHODS, make_folds
from .features import (
    COLLECTIONS, POOLED_FEATURE_SETS, POOLED_METHOD_NAMES, SOURCES, SPARSE_OK,
    completed_groups, fit_features, load_datasets, method_groups, pool_blocks,
    pooled_fit_kwargs, usable_units,
)
from .runner_common import (
    SKIP_ERRORS, add_shared_args, is_gpu_oom, append_metrics, dataset_order, model_npz,
    nested_cv, parse_exclude, parse_skip, plot_method_by_dataset,
    release_gpu_memory, scores_npz, to_dense, warn_fold_policy
)

RESULTS = Path(__file__).parents[1] / "results"    # / <data> / merged
CSV_COLUMNS = ("features", "layer", "method", "kind", "test_dataset",
               "n_test", "auc")
RESUME_KEY = ("features", "layer", "method")


def run_pooled(method: str, features: str, data: dict, densify: bool,
               seed: int, n_jobs: int, collection: str,
               group_folds: bool = True) -> tuple[list, dict]:
    """Nested CV for one (feature set, method) on the pool of every usable
    trainable dataset. Returns the metric rows (kind, test_dataset, n_test,
    auc) and the npz contents."""
    c = COLLECTIONS[collection]
    usable = usable_units(data, features)
    pool = [n for n in usable if n in c.pool_datasets]
    blocks, groups, arrays = pool_blocks(data, pool)
    X, transform, extras = fit_features(features, blocks)
    y = arrays["y"]
    if densify:
        X = to_dense(X)
    key = np.char.add(groups, y.astype(str))
    # `groups` above is the source dataset (what random_slopes partitions
    # on); `fold_groups` is the CV group — unit-namespaced by pool_blocks
    fold_groups = arrays["group"] if group_folds else None

    def fold_kwargs(outer_tr, outer_te):
        kwargs = pooled_fit_kwargs(method, groups[outer_tr], collection)
        if "groups" in kwargs:   # random slopes: outer-fold groups are known
            kwargs["groups_test"] = method_groups(method, groups[outer_te],
                                                  collection)
        return kwargs

    cv, cv_fold, fold_aucs, fold_params = nested_cv(
        METHODS[method], X, y, seed=seed, n_jobs=n_jobs, stratify_on=key,
        fold_groups=fold_groups, fold_kwargs=fold_kwargs)

    rows = [("overall", "all", len(y), f"{np.mean(fold_aucs):.3f}")]
    for d in dataset_order(np.unique(groups), c.datasets):
        m = groups == d
        rows.append(("per_dataset", d, int(m.sum()),
                     f"{roc_auc_score(y[m], cv['proba'][m]):.3f}"))

    # One tuned fit on the whole pooled corpus. It plays no part in the
    # AUCs above (those stay nested-CV, each fold's model discarded); it
    # exists to be *the* pooled model — its coefficient vector and chosen
    # hyperparameters are what the coefficient analyses read.
    full_folds = make_folds(X, y, seed=seed, stratify_on=key,
                            groups=fold_groups)
    _, _, full_params, fitted = METHODS[method](
        X, y, seed=seed, folds=full_folds, inner_n_jobs=n_jobs,
        **pooled_fit_kwargs(method, groups, collection))

    # The units that were kept out of the pool — eval_only and
    # pool_excluded alike — scored by the whole-corpus fit (inference
    # only). Its pool is every poolable dataset, which is exactly what a
    # leave-one-dataset-out holdout of one of these units would train on,
    # so this row IS their held-out-dataset number and replaces that
    # redundant refit. Reported under kind "eval_only", the established
    # name for "scored, never pooled".
    eval_only = []               # (unit, its data, its scores)
    held_out = set(c.eval_only) | set(c.pool_excluded)
    for name in (n for n in usable if n in held_out):
        te = data[name]
        X_e = transform(te["blocks"])
        if densify:
            X_e = to_dense(X_e)
        if method == "random_slopes_task":
            # the unit's task was seen in training, so its task deviation
            # applies (an own-task unit has no
            # deviation row and falls back to shared weights per sample);
            # the shared-only scores ride along in the npz for comparison
            shared = fitted.predict(X_e)
            scores = fitted.predict_with_groups(
                X_e, method_groups(method, np.full(len(te["y"]), name),
                                   collection))
            scores["shared_proba"] = shared["proba"]
            scores["shared_margin"] = shared["margin"]
        else:
            scores = fitted.predict(X_e)
        rows.append(("eval_only", name, len(te["y"]),
                     f"{roc_auc_score(te['y'], scores['proba']):.3f}"))
        eval_only.append((name, te, scores))

    npz = {
        "cv_sample_id": arrays["sample_id"], "cv_y": y, "cv_dataset": groups,
        "cv_fold": cv_fold, "cv_group": np.asarray(arrays["group"]).astype(str),
        "params": repr(fold_params),
        "full_params": repr(full_params),
    }
    if (arrays["topic"] != "").any():   # per-sample topic (e.g. got's parts)
        npz["cv_topic"] = arrays["topic"]
    npz.update({f"cv_{k}": v for k, v in cv.items()})
    if eval_only:
        npz.update(scores_npz(eval_only, "eval_only"))
    npz.update(model_npz(fitted, "full_"))   # full_coef, full_standardize_*, ...
    npz.update(extras)
    return rows, npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_shared_args(ap, method_names=POOLED_METHOD_NAMES,
                    default_features=POOLED_FEATURE_SETS)
    args = ap.parse_args()
    skip = parse_skip(ap, args.skip, POOLED_METHOD_NAMES)
    c = COLLECTIONS[args.data]
    exclude = parse_exclude(ap, args.exclude, c)
    layers = args.layers or list(c.layers)
    results = RESULTS / args.data / "merged"

    warn_fold_policy(results, args.group_folds)
    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "predictions").mkdir(exist_ok=True)

    for layer in layers:
        roles = sorted({r for f in args.features for r in SOURCES[f]})
        print(f"layer{layer}: loading {roles} caches")
        data = load_datasets(layer, args.raw, roles, args.data,
                             groups=args.group_folds)
        data = {n: u for n, u in data.items() if n not in exclude}

        for features in args.features:
            tag = f"{features}_{args.raw}"
            csv_path = results / "tables" / f"metrics_{tag}_layer{layer}.csv"
            done = completed_groups(csv_path, RESUME_KEY)
            for method in args.methods:
                if (method, features) in skip \
                        or (tag, str(layer), method) in done:
                    continue
                print(f"  {tag} {method} on the pooled corpus", flush=True)
                release_gpu_memory()   # see runner_common
                try:
                    rows, npz = run_pooled(method, features, data,
                                           densify=method not in SPARSE_OK,
                                           seed=args.seed, n_jobs=args.n_jobs,
                                           collection=args.data,
                                           group_folds=args.group_folds)
                except Exception as e:     # see runner_common.SKIP_ERRORS
                    if not (isinstance(e, SKIP_ERRORS) or is_gpu_oom(e)):
                        raise
                    print(f"    skipped, {type(e).__name__}: {e}", flush=True)
                    continue
                np.savez_compressed(
                    results / "predictions" /
                    f"{tag}_layer{layer}_{method}.npz", **npz)
                append_metrics(csv_path, CSV_COLUMNS,
                               [(tag, layer, method) + r for r in rows])
            # one figure per feature set, a panel per layer — drawn
            # from every layer csv present, so parallel single-layer
            # tasks converge on the complete figure
            plot_method_by_dataset(
                results / "tables",
                lambda L: f"metrics_{tag}_layer{L}.csv",
                tag, sorted(set(c.layers) | set(layers)),
                args.methods, c.datasets, xlabel="evaluated on",
                title=f"pooled nested-CV AUC, {tag}",
                out_path=results / "figures" / f"auc_{tag}.png")
    print("done")


if __name__ == "__main__":
    main()
