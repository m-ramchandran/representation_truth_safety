# Post-hoc analyses

Analyses of the classifier experiments' outputs — everything the runners
left under `classifiers/results/<collection>/`. Every CLI takes the
`--data <collection>` / `--root` pair (`load_results.add_data_args`):
`--root` defaults to `results/<data>`, and `--data` also picks the
task groups and cache files for the modules that use them (task_of,
covariance estimation, prompt joins). The rule that makes this folder
work:

**Read `results/` (csvs + npz) and the HF activation caches. Never
import `classifiers.scripts`.** Activation loading goes through
`classifiers.data` (package root, import-safe anywhere) — the same module
`scripts/` uses, so the label conventions cannot drift between the two.

`scripts/` imports cuML at load time and only works on a GPU node; the
outputs are plain numpy/csv and the activation caches load with plain
numpy, so everything here runs on the Mac. The caches are for analyses
that need data, not fits — covariance estimates for Mahalanobis cosines,
`topic` / prompt-text joins, scoring saved directions.

## Complete collection runs and paper figures

The two collections remain independent commands and cluster jobs:

```bash
python -m classifiers.analysis_scripts.run_all --data data_unwrap
python -m classifiers.analysis_scripts.run_all --data tf_wrapped_a
```

`run_all` owns the shared module manifest, while each command reads and
writes only its collection's result tree. The final stage inventories the
canonical paper inputs in `results/<data>/paper_inputs/manifest.json`; it
does not duplicate tables or prediction archives.

After both collection trees are available, build the cross-collection paper
outputs once:

```bash
python -m classifiers.analysis_scripts.paper
```

This writes `results/paper/tables/` and `results/paper/figures/`. A fast
table-only redraw is available through
`python -m classifiers.analysis_scripts.paper --redraw`. Full-data boundary
refits remain optional GPU jobs; after their compact artifacts are present,
render their paper figures with `--stages boundaries`. On compute cluster, the
cross-collection core build can be submitted as
`sbatch/classifiers_paper.sbatch` after both collection analysis jobs finish.

## Modules

Pipeline entry points:

- **`run_all.py`** — the shared ordered manifest for one complete collection
  analysis; both collection-specific Slurm jobs call it.
- **`paper/`** — validates the two collection manifests, computes curated
  cross-collection tables/figures, and redraws them independently of the
  per-layer diagnostic figure tree.

Shared infrastructure (no CLI):

- **`load_results.py`** — metrics tables → pandas, prediction npz →
  dicts with `params` parsed back; the per-sample prediction frames
  (`cv_predictions`, `eval_predictions` with the log-odds column).
- **`directions.py`** — saved vectors → comparable directions:
  `load_direction` / `raw_direction` map any coef to cache units,
  `to_cache_units` does the same for non-coefficient rows (moe gates),
  `apply_featurization` rebuilds a fit's feature transform from its npz
  arrays, `stack_directions` aligns sparse supports.
- **`plots.py`** — annotated heatmaps / per-method panels /
  layers-as-panels figures / save (sizes from classifiers.fig_style).
- **`cross_methods.py`** — one row per dataset, one column per
  training regime (in_domain_fit / within_cv / merged_cv /
  leave_one_out / leave_task_out): the generalization gradient in
  a single heatmap per method, plus a `regimes_best_` panel holding
  each cell's winning AUC and the method that won it. Writes its own
  section,
  `results/<data>/cross_methods/`.
- **`redraw.py`** — regenerate every figure from the saved csvs (no
  npz, no activations): the seconds-long local loop for styling and
  formatting changes; `--only <family ...>` narrows it.
- **`fixtures.py`** — writes a synthetic results tree with a planted,
  known signal; how every module is tested without GPU results (it is
  seeded, so tests regenerate the exact activations its fits saw).
  Its `got` + `common_claim` share the `tf` task, so task-mode pools
  differ from dataset-mode ones and a held-out dataset can have a
  same-task sibling left in the pool. Its fits are stand-in linear
  rules with per-sample noise added to the margins, so a module that
  recomputes scores from saved arrays will not reproduce the stored
  ones exactly on this tree (real runs do).

One module per analysis question, run as
`python -m classifiers.analysis_scripts.<module>` (each docstring has the exact
usage). Prediction-based:

- **`errors.py`** — do the methods miss the same samples? Rank
  correlation between their predictions (a figure), error overlap,
  samples everyone misses (a figure: per dataset, the best single
  method's error rate split into the shared misses inside and outside
  a mean probability of 0.2-0.8, and the errors another method
  avoids), mean-probability ensemble. per_dataset and
  merged correlate CV predictions per dataset; leave_out correlates
  held-out predictions per (held-out unit, held-out dataset) and writes
  only that figure — overlapping pools leave it no single CV prediction
  per sample.
- **`per_slice.py`** — AUC by got sub-dataset; per-dataset slices of the
  merged CV predictions.
- **`margins.py`** — is transfer failure uncertain or confidently
  wrong? `margin_summary_*.csv` carries CV vs held-out log-odds
  quantiles and the confidently-wrong share per learner (a method x
  fit) on its own CV and on each dataset it transfers to. The one
  figure is the score-distribution grid those rates come from: rows =
  evaluated unit, columns = method, panels annotated with the AUC
  there (every method that stores a margin; `--hist` narrows it;
  leave_out draws one grid per holdout kind). A per-learner dot plot
  of the rates was dropped 2026-07-25 as redundant with the grid. Its bin
  counts go to `margin_hist_*.csv`, which is what lets `redraw.py`
  rebuild the grid off-cluster — the per-sample npz never leaves it.
  Runs for all three experiments — merged's "transfer" is the
  per-dataset slices of its pooled nested CV.
- **`prompts.py`** — candidate mislabels (confidently wrong in-domain
  for every margin-method) with prompt text.

Coefficient-based (all through `directions.py`):

- **`coef_sparsity.py`** — what l1 keeps: support sizes, features
  recurring across datasets (with signs).
- **`coef_similarity.py`** — dataset x dataset direction similarity
  (plain cosine + shrunk Mahalanobis in two geometries), vs transfer
  AUC, and the Merged / Leave-out / Single triangle. Its Ledoit-Wolf
  covariances are cached under `results/<data>/covariances/` (one npz
  per tag/layer, float64 so they stay invertible, self-invalidating
  on roster or dtype change; big — never rsync it to the laptop).
- **`coef_pca.py`** — the shared subspace of the single directions:
  spectrum, reconstruction and held-out AUC curves, the merged
  direction against the singles' span (least-squares weighted sum).
- **`coef_norms.py`** — norm accounting: direction norms vs chosen C,
  cos(merged, single), movement per dropped dataset.
- **`rs_directions.py`** — the merged random-slopes fits' decomposition
  vs the singles: per dataset, cos of the shared weights, of
  shared + the dataset's group deviation (its own row for
  random_slopes, its task's for random_slopes_task), and of the
  deviation alone, each against that dataset's own per_dataset
  direction.
- **`rs_holdout.py`** — should a held-out dataset borrow its task's
  deviation? Re-scores the saved leave-one-dataset-out random_slopes
  fits with the shared weights (what the runner reports) against
  shared + the mean of the same-task pool members' deviations, plain
  and pool-size-weighted, and compares held-out AUC. No refit: the
  deviations are already in the npz. Writes under `leave_out/`.

Model internals:

- **`gates.py`** — what the moe gate learned: principal angles against
  dataset/task/label structure, recomputed per-sample routing, gate and
  expert vectors vs the per-dataset l1 directions.
- **`hyperparams.py`** — per-outer-fold chosen hyperparameters in the
  merged run: fold disagreement flags a flat tuning landscape.
- **`complexity.py`** — model size at each run's chosen
  hyperparameters, two counts per run: trained parameters (gradient /
  solver / closed-form estimates; 0 for knn) and the total the fitted
  rule carries (adds knn's memorized training set and svm_rbf's
  Nystrom map): a csv and one log-scale bar chart per feature set,
  a row of bars per layer.

Outputs: cross-experiment coefficient modules write under
`results/coefficients/`; everything else joins the runner's own
`tables/` and `figures/` under `results/<experiment>/` of the
experiment it reads (no nested analysis/ level since 2026-07-23;
runner files are the `metrics_*` csvs and `auc_*` figures, analysis
files everything else), via `load_results.analysis_dirs`.

## Which experiments feed the coefficients figures

"S" = the per_dataset single fits, "M" = the merged whole-corpus fit,
"L" = the leave-out pool fits (`<kind>` = dataset | task):

| file | module | uses |
|---|---|---|
| `similarity_<measure>_<tag>_layer<L>.png` | coef_similarity | S only (mahalanobis measures also load activation covariances) |
| `mls_<measure>_<tag>_layer<L>_<kind>.png` | coef_similarity | S + M + L — the Merged/Leave-out/Single cosine triangle |
| `pca_reconstruction_<method>_...` | coef_pca | S only |
| `pca_individual_<method>_...` | coef_pca | S only |
| `pca_heldout_<method>_...` | coef_pca | S only — a held-out *direction* among the singles, NOT the leave_out experiment |
| `pca_merged_<method>_..._<kind>.png` | coef_pca | S + M + L — merged direction over two bases: the singles' span and the leave-out directions' span |
| `pca_leave_out_<method>_..._<kind>.png` | coef_pca | S + L — each leave-out pool direction vs the singles it trained beside |
| `support_sizes.png` | coef_sparsity | S only |
| `rs_vs_single_<tag>_layer<L>_<variant>.png` | rs_directions | S + M — the merged random-slopes fit's shared / effective / deviation directions vs each dataset's single fit |

`rs_holdout_<tag>.png` (under `leave_out/figures/`, not
`coefficients/`) is L only: it re-scores each leave-one-dataset-out
random_slopes fit on its held-out dataset, so it reports AUC rather
than direction geometry.

`coef_norms_<tag>_layer<L>_<kind>.csv` is the one table mixing all
three: single norms and C, cos(merged, single), and delta_leave_out.

## What is in a prediction npz

All experiments:

| key | contents |
|---|---|
| `cv_sample_id`, `cv_y` | sample id ("`<source>:<row>`", joinable to prompts/topics) and label per training sample |
| `cv_topic` | per-training-sample topic where the cache stores one (e.g. got's parts) — per_dataset runs from 2026-07-21 on, pooled runs (leave_out / merged, samples of topic-less units get "") from 2026-07-22 on; `cv_predictions` exposes it as the `topic` column ("" when absent) |
| `cv_proba`, `cv_margin` | per-sample CV predictions (`cv_margin` absent for knn, which has no margin) |
| `params` | chosen hyperparameters (repr; parsed by `load_prediction`) |
| `standardize_mean`, `standardize_scale` | the standardization fit in front of the model (mean absent when scaling was variance-only — sparse input — and both absent for diff, which takes raw input); `w_raw = coef / standardize_scale` maps a coefficient vector back to raw feature units |
| method extras | `dev_w` / `dev_b` / `group_labels` for random_slopes (per-dataset deviations, rows in `group_labels` order) and random_slopes_task (the same arrays with one row per *task*); `gate_w` / `gate_b` / `expert_w` / `expert_b` for moe (gate and experts are linear on the standardized features); `feature_importance` + `booster_ubj` for xgboost (see below) |
| featurization extras | what the featurization fit: `kept_features` / (`n_pcs`, `pca_mean`, `pca_components`) / `bin_edges` / (`kept_sae`, `kept_transcoder`) depending on the feature set |

Per experiment:

- **per_dataset**: `cv_fold` (outer fold index; CV here is nested),
  `transfer_{dataset,sample_id,y,proba,margin}` (the full fit applied to
  every other dataset), plus the full fit's model arrays (`coef`,
  `standardize_*`, method extras).
- **leave_out**: `cv_dataset` (each pool sample's source dataset),
  `holdout_{dataset,sample_id,y,proba,margin}`, plus the pool fit's
  model arrays.
- **merged**: `cv_fold`, `cv_dataset`; `params` is a per-outer-fold
  *list*. Plus one tuned fit on the whole pooled corpus, not part of the
  nested-CV AUCs: its model arrays carry a `full_` prefix (`full_params`,
  `full_coef`, `full_standardize_scale`, `full_gate_w`, ...), and where
  the collection has eval-only units, that fit's scores on them land in
  `eval_only_{dataset,sample_id,y,proba,margin}` (runs from 2026-07-22
  on; the csv rows carry kind `eval_only`). random_slopes_task scores
  the eval-only units with their task's deviation and stores the
  shared-weights-only scores alongside
  (`eval_only_shared_{proba,margin}`).

**xgboost's two keys.** `feature_importance` is each feature's share of
the total split gain — a magnitude with no sign, so it is *not* a
direction (`directions.py` refuses it, and correlated activation
dimensions split the credit between them); its nonzero count is a
support size comparable to `coef_sparsity`'s. `booster_ubj` is the
fitted ensemble itself, UBJSON bytes in a uint8 array (~0.4 MB
deflated), since xgboost is the one method whose predictions cannot be
rebuilt from a summary vector: `load_results.load_booster(d, prefix)`
returns an `xgboost.Booster` that reproduces the fit's margins exactly,
for scoring new data without a refit. It takes raw featurized values —
no standardization in front — so put new activations through
`directions.apply_featurization` only. Pass `keys=` to
`load_prediction` and it is never decompressed.

`coef` / `full_coef` is absent for methods without a coefficient vector
(qda, knn, mlp, moe, xgboost; svm_rbf's is in its Nystrom feature space, and the
Nystrom map itself — landmarks and whitening — is not persisted, so
svm_rbf's vector cannot be taken back to input space). Runs before
2026-07-21 lack the standardization, PCA basis, and method extras; runs
before 2026-07-20 lack `coef` entirely.

## Conventions to respect (set in `scripts/`, mirrored here)

- **Margins.** For l1reg / l2reg / lda / qda / mlp / random_slopes / moe
  the margin is the exact log-odds of `proba`. For svm_linear / svm_rbf /
  diff it is the raw decision value; `proba = sigmoid(a * margin + b)`
  with the Platt `(a, b)` stored in `params` — apply that map whenever a
  cross-method comparison needs those margins on the log-odds scale.
- **Coefficient spaces.** Every method's `coef` lives in
  standardized-feature space (the scaling fit in front of the model) —
  except diff, whose vector is the unit-norm class-mean direction in raw
  activation units. `coef / standardize_scale` maps a standardized-space
  vector to raw feature units (each fit's own scaler — two fits on
  different training data have different scalers, so map both before
  comparing, and always before comparing anything to diff).
  The random_slopes variants' `coef` is their shared weights; the
  per-group deviations (datasets, or tasks for random_slopes_task) are
  in `dev_w` / `dev_b` (same standardized space).
  `directions.py` implements all of this — `load_direction` /
  `raw_direction` map any saved coef to cache units (pca fits
  back-projected to raw space), `stack_directions` aligns sparse
  supports; use it rather than re-deriving the mapping.
- **Feature indexing.** For sparse feature sets, coefficients index the
  *kept* features; map back to dictionary indices via `kept_features`
  (or `kept_sae` / `kept_transcoder` for pca_dict, after the `n_pcs`
  PCA block).
- **Labels.** 1 = true / yes / safe / harmless everywhere (set at load
  time in `classifiers/data.py`, per collection).
