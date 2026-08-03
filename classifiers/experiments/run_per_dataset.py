"""Per-dataset performance: within-dataset CV and cross-dataset transfer.

Runs on a GPU node: l1reg / l2reg / svm_linear / svm_rbf / knn fit on the
GPU (cuML), mlp on the GPU (torch), xgboost on the GPU (hist boosting),
diff / lda / qda on the node's CPU cores. The GPU is what makes knn
(brute-force n_train x n_test distances) and the Nystrom-mapped RBF SVM
practical on the 30-50k-sample datasets.

The binary units come from the --data collection (data.COLLECTIONS;
default "original": got — the 7 curated geometry-of-truth parts merged —
common_claim, counterfact, harmful_harmless, pubmedqa_artificial,
pubmedqa_labeled, wildjailbreak, xstest). A collection's eval-only units
(e.g. tf_wrapped_a's claude_diagnosis_severity, n=40) appear as test
datasets but are never trained on. Labels follow the convention set in
classifiers/data.py: 1 = true / yes / safe / harmless.

Each run names the raw activations to build on (--raw last|mean) and the
feature sets to run (--features). Feature sets:

  raw                 the raw activations themselves
  sae / transcoder    per-layer dictionary activations (sparse); features
                      firing on >= MIN_FIRES samples of the training
                      dataset are kept (label-free), and the same kept
                      features index the test datasets. The caches hold
                      these at the last prompt token only, so both ignore
                      --raw; run them under a single --raw value or you
                      compute identical results twice under two tags.
  pca                 PCs of the raw activations explaining 95% of the
                      variance, fit on the training dataset (label-free,
                      like the firing filter); test datasets are projected
                      with the training dataset's PCs.
  pca_dict            pca + sae + transcoder side by side, each block
                      preprocessed exactly as when used alone. (The
                      crosscoder is excluded: it is one joint latent over
                      layers 9/17/22/29, not separable per layer.)
  binned_activations  each prompt's raw activation vector as a 100-bin
                      histogram (counts per bin). Bins are equal-width over
                      the training dataset's 0.5-99.5 percentile range (so
                      a few outliers don't stretch them); the same edges
                      bin the test datasets.

For every (feature set, layer, method) and every training dataset D:
  - within-dataset (diagonal): nested CV on D. 5 outer folds; on each outer
    train part the method tunes its hyperparameters on inner folds and then
    predicts the outer test fold. Reported AUC = mean over the outer folds.
  - transfer (off-diagonal): one fit on all of D (hyperparameters tuned by
    CV), applied to every other loaded dataset; AUC against that dataset's
    own labels. Cross-family transfer (truth vs safety) is computed like any
    other pair.

Sparse feature sets (sae / transcoder / pca_dict) are fit sparse by
l1reg/l2reg (CSR stays sparse on the GPU, scaling is variance-only —
sparse matrices cannot be mean-centered); the other methods get the culled
slice densified. got's folds are stratified on the joint (part, label) key.
Missing HF files are skipped with a note.

Outputs, under classifiers/results/<data>/per_dataset/ — <tag> is always
<features>_<raw> (raw_mean, pca_mean, sae_last, ...), so no two runs can
ever write the same file:
  tables/metrics_<tag>_layer<L>.csv
                        one row per (method, train, test), AUC to 3
                        decimals; one csv per (tag, layer) so parallel
                        jobs never share a file. Also the resume
                        record: a (method, train) group whose rows are
                        present is skipped on re-run. Concatenate the csvs
                        for the full table (columns include features/layer).
  predictions/          one .npz per (tag, layer, method, train) with
                        per-sample CV and transfer scores (probabilities,
                        and margins for every method but knn), the
                        training samples' topic where the cache has one
                        (`cv_topic` — e.g. got's parts), the chosen
                        hyperparameters, the full fit's coefficient
                        vector where the method has one (`coef`, see
                        FittedClassifier.get_coef), and what the
                        featurization kept (kept features / number of
                        PCs / bin edges).
  figures/              one heatmap per (tag, layer), a panel per method:
                        the train x test AUC matrix, diagonal = CV.

Run from the repo root:

    python -m classifiers.experiments.run_per_dataset --raw mean \
        [--data original] \
        [--features raw sae transcoder pca pca_dict binned_activations] \
        [--layers 9 17 22 29] \
        [--methods l1reg l2reg svm_linear svm_rbf diff lda qda knn mlp] \
        [--skip qda:sae qda:transcoder ...] \
        [--datasets ...] [--seed 0] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..methods import METHODS
from .features import (
    COLLECTIONS, METHOD_NAMES, FEATURE_SETS, SOURCES, SPARSE_OK,
    completed_groups, fit_features, load_datasets,
)
from .runner_common import (
    SKIP_ERRORS, add_shared_args, append_metrics, model_npz, nested_cv,
    parse_exclude, parse_skip, plot_heatmaps, release_gpu_memory, scores_npz,
    to_dense,
)

RESULTS = Path(__file__).parents[1] / "results"    # / <data> / per_dataset
CSV_COLUMNS = ("features", "layer", "method", "train_dataset", "test_dataset",
               "kind", "n_test", "auc")
RESUME_KEY = ("features", "layer", "method", "train_dataset")


def run_unit(fit, features: str, train_name: str, data: dict, densify: bool,
             seed: int, n_jobs: int) -> tuple[list[tuple], dict]:
    """All fits for one (feature set, method, training dataset): nested CV on
    the training dataset plus one full fit applied to every other dataset
    that has the blocks this feature set needs. Returns the metric rows
    (without features/layer/method) and the npz contents."""
    tr = data[train_name]
    X, transform, extras = fit_features(features, tr["blocks"])
    y = tr["y"]
    if densify:
        X = to_dense(X)
    stratify_on = None
    if train_name == "got":  # merged corpus: balance folds across (part, label)
        stratify_on = np.char.add(tr["topic"].astype(str), y.astype(str))

    cv, cv_fold, fold_aucs, _ = nested_cv(fit, X, y, seed=seed, n_jobs=n_jobs,
                                          stratify_on=stratify_on)
    _, _, params, fitted = fit(X, y, seed=seed, stratify_on=stratify_on,
                               inner_n_jobs=n_jobs)

    rows = [(train_name, train_name, "cv", len(y), f"{np.mean(fold_aucs):.3f}")]
    transfer = []                # (test dataset, its unit, its scores)
    for test_name, te in data.items():
        if test_name == train_name \
                or any(r not in te["blocks"] for r in SOURCES[features]):
            continue
        X_test = transform(te["blocks"])
        if densify:
            X_test = to_dense(X_test)
        scores = fitted.predict(X_test)
        rows.append((train_name, test_name, "transfer", len(te["y"]),
                     f"{roc_auc_score(te['y'], scores['proba']):.3f}"))
        transfer.append((test_name, te, scores))

    npz = {
        "cv_sample_id": tr["sample_id"], "cv_y": y, "cv_fold": cv_fold,
        "params": repr(params),
    }
    if tr["topic"] is not None:    # per-sample topic (e.g. got's parts)
        npz["cv_topic"] = tr["topic"]
    npz.update({f"cv_{key}": v for key, v in cv.items()})
    npz.update(scores_npz(transfer, "transfer"))
    npz.update(model_npz(fitted))      # the full fit's coef, scaler, extras
    npz.update(extras)
    return rows, npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_shared_args(ap, method_names=METHOD_NAMES,
                    default_features=FEATURE_SETS)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="training datasets to run (default: every "
                         "trainable dataset in the collection)")
    args = ap.parse_args()
    skip = parse_skip(ap, args.skip, METHOD_NAMES)
    c = COLLECTIONS[args.data]
    exclude = parse_exclude(ap, args.exclude, c)
    layers = args.layers or list(c.layers)
    train_datasets = args.datasets or [d for d in c.train_datasets
                                       if d not in exclude]
    if unknown := set(train_datasets) - set(c.train_datasets):
        ap.error(f"--datasets {sorted(unknown)}: not trainable in "
                 f"--data {args.data} (trainable: {list(c.train_datasets)})")
    if both := set(train_datasets) & exclude:
        ap.error(f"--datasets {sorted(both)}: also in --exclude")
    results = RESULTS / args.data / "per_dataset"
    train_roster = [d for d in c.train_datasets if d not in exclude]
    test_roster = [d for d in c.datasets if d not in exclude]

    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "predictions").mkdir(exist_ok=True)

    for layer in layers:
        todo = {}      # feature set -> [(method, train_dataset), ...] left to run
        for features in args.features:
            tag = f"{features}_{args.raw}"
            csv_path = results / "tables" / f"metrics_{tag}_layer{layer}.csv"
            done = completed_groups(csv_path, RESUME_KEY)
            left = [(m, d) for m in args.methods if (m, features) not in skip
                    for d in train_datasets
                    if (tag, str(layer), m, d) not in done]
            if left:
                todo[features] = left
            else:
                print(f"{tag} layer{layer}: all done, skipped")

        if todo:
            roles = sorted({r for f in todo for r in SOURCES[f]})
            print(f"layer{layer}: loading {roles} caches")
            data = load_datasets(layer, args.raw, roles, args.data)
            data = {n: u for n, u in data.items() if n not in exclude}

        for features, left in todo.items():
            tag = f"{features}_{args.raw}"
            csv_path = results / "tables" / f"metrics_{tag}_layer{layer}.csv"
            for method, train_name in left:
                if train_name not in data or any(
                        r not in data[train_name]["blocks"]
                        for r in SOURCES[features]):
                    print(f"  {tag} {method} on {train_name}: missing cache, "
                          "skipped")
                    continue
                print(f"  {tag} {method} on {train_name} "
                      f"(N={len(data[train_name]['y'])})", flush=True)
                release_gpu_memory()   # see runner_common
                try:
                    rows, npz = run_unit(METHODS[method], features, train_name,
                                         data, densify=method not in SPARSE_OK,
                                         seed=args.seed, n_jobs=args.n_jobs)
                except SKIP_ERRORS as e:   # see runner_common.SKIP_ERRORS
                    print(f"    skipped, {type(e).__name__}: {e}", flush=True)
                    continue
                np.savez_compressed(
                    results / "predictions" /
                    f"{tag}_layer{layer}_{method}_{train_name}.npz", **npz)
                append_metrics(csv_path, CSV_COLUMNS,
                               [(tag, layer, method) + r for r in rows])

        # figures are redrawn from the csvs even when every fit was
        # already done — a no-op resubmit refreshes stale styling
        for features in args.features:
            tag = f"{features}_{args.raw}"
            plot_heatmaps(results / "tables" /
                          f"metrics_{tag}_layer{layer}.csv",
                          tag, layer, args.methods, train_roster,
                          test_roster, results)
    print("done")


if __name__ == "__main__":
    main()
