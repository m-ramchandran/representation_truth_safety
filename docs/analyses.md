# Analyses

`classifiers/analysis/` reads what the runners left under
`results/<collection>/` and nothing else: the metrics csvs, the prediction npzs,
and where an analysis needs data rather than fits, the cached activations
through `classifiers.data`. It never imports `classifiers.methods`, which pulls
in cuML at load time. Keep it that way and the whole tree runs on a laptop.

Every CLI takes the `--data <collection>` / `--root` pair from
`load_results.add_data_args`. `--root` defaults to `results/<data>`, and
`--data` also selects the task groups and cache files for the modules that need
them. Each module's docstring has its exact usage.

## Running everything

One collection at a time:

```bash
python -m classifiers.analysis.run_all --data data_unwrap
```

`run_all` owns the shared ordered manifest. Each invocation touches only its own
collection's tree, and the last stage inventories the paper inputs in
`results/<data>/paper_inputs/manifest.json`.

Once both collection trees exist, the cross-collection outputs build once:

```bash
python -m classifiers.analysis.paper
```

That writes `results/paper/tables/` and `results/paper/figures/`. `--redraw`
does a fast table-only pass. `--stages boundaries` renders the boundary figures,
and needs their refit artifacts (see below) to already be there.

## Shared infrastructure

No CLI of their own:

- `load_results.py` turns metrics tables into pandas and prediction npzs into
  dicts with `params` parsed back, and builds the per-sample prediction frames
  `cv_predictions` and `eval_predictions` with their log-odds column.
- `directions.py` turns saved vectors into comparable directions.
  `load_direction` and `raw_direction` map any coef to cache units,
  `to_cache_units` does the same for non-coefficient rows like moe gates,
  `apply_featurization` rebuilds a fit's feature transform from its npz arrays,
  and `stack_directions` aligns sparse supports.
- `plots.py` has the annotated heatmaps, per-method panels, layers-as-panels
  figures, and saving, with sizes from `classifiers.fig_style`.
- `fixtures.py` writes a synthetic results tree with a planted, known signal.
  This is how every module gets tested without GPU results. It is seeded, so
  tests regenerate the exact activations its fits saw. Its `got` and
  `common_claim` share the `tf` task, which makes task-mode pools differ from
  dataset-mode ones and lets a held-out dataset keep a same-task sibling in the
  pool. One caveat: its fits are stand-in linear rules with per-sample noise
  added to the margins, so a module that recomputes scores from saved arrays
  will not reproduce the stored ones exactly on this tree. Real runs do.

## One module per question

Run as `python -m classifiers.analysis.<module>`.

Prediction-based:

- `errors.py` — do the methods miss the same samples? Rank correlation between
  their predictions, error-set overlap, the samples every method gets wrong, and
  a mean-probability ensemble. per_dataset and merged correlate CV predictions
  per dataset. leave_out correlates held-out predictions per (held-out unit,
  held-out dataset), because overlapping pools leave it no single CV prediction
  per sample.
- `margins.py` — is transfer failure uncertain, or confidently wrong?
  `margin_summary_*.csv` carries CV against held-out log-odds quantiles and the
  confidently-wrong share per learner, on its own CV and on each dataset it
  transfers to. The figure is the score-distribution grid those rates come from,
  rows = evaluated unit, columns = method, panels annotated with the AUC there.
  Its bin counts go to `margin_hist_*.csv`, so `redraw.py` can rebuild the grid
  without the per-sample npz.
- `per_slice.py` — AUC by geometry-of-truth sub-dataset, and per-dataset slices
  of the merged CV predictions.
- `prompts.py` — candidate mislabels, meaning samples that are confidently wrong
  in domain for every margin-carrying method, with their prompt text.
- `cross_methods.py` — one row per dataset, one column per training regime
  (in_domain_fit, within_cv, merged_cv, leave_one_out, leave_task_out). The
  generalization gradient in a single heatmap per method, plus a `regimes_best_`
  panel holding each cell's winning AUC and which method won it.

Coefficient-based, all through `directions.py`:

- `coef_sparsity.py` — what l1 keeps: support sizes, and features recurring
  across datasets with their signs.
- `coef_similarity.py` — dataset × dataset direction similarity (plain cosine
  plus shrunk Mahalanobis in two geometries) against transfer AUC, and the
  Merged / Leave-out / Single triangle. Its Ledoit-Wolf covariances are cached
  under `results/<data>/covariances/`, float64 so they stay invertible,
  self-invalidating on a roster or dtype change.
- `coef_pca.py` — the shared subspace of the single directions: spectrum,
  reconstruction and held-out AUC curves, and the merged direction against the
  singles' span.
- `coef_norms.py` — norm accounting. Direction norms against the chosen C,
  cos(merged, single), and movement per dropped dataset.
- `rs_directions.py` — the merged random-slopes fits decomposed against the
  singles. Per dataset: the cosine of the shared weights, of shared plus that
  dataset's group deviation, and of the deviation alone, each against the
  dataset's own per_dataset direction.
- `rs_holdout.py` — should a held-out dataset borrow its task's deviation? Takes
  the saved leave-one-dataset-out random_slopes fits and re-scores them with the
  shared weights, which is what the runner reports, against shared plus the mean
  of the same-task pool members' deviations, plain and pool-size-weighted. No
  refit needed; the deviations are already in the npz.

Model internals:

- `gates.py` — what the moe gate learned. Principal angles against dataset, task
  and label structure, recomputed per-sample routing, and gate and expert
  vectors against the per-dataset l1 directions.
- `hyperparams.py` — per-outer-fold chosen hyperparameters in the merged run.
  Fold disagreement flags a flat tuning landscape.
- `complexity.py` — model size at each run's chosen hyperparameters. Two counts
  per run: trained parameters, and the total the fitted rule carries, which adds
  knn's memorized training set and svm_rbf's Nyström map.

## Studies

`classifiers/analysis/studies/` holds the cross-experiment work. The modules
above each answer one question about one experiment; these cut across
experiments and collections. Run any of them directly, e.g.
`python -m classifiers.analysis.studies.linear_hypothesis`.

| module | what it answers |
|---|---|
| `metrics_frame.py` | every `metrics_*.csv` from both collections in one frame; also caches `metrics.parquet` |
| `common.py` | the shared roster, task map, method families, known coverage holes |
| `coverage.py` | which (feature set × layer × method × dataset) fits actually exist |
| `coverage_gaps.py` | exactly which fits are missing, so gaps don't get read as results |
| `methods.py` | method × regime, paired win-rates against l2, linear against nonlinear by family |
| `datasets.py` | the generalization gradient per dataset, transfer matrix, within- against between-task, asymmetry |
| `features_and_layers.py` | raw against pca / sae / transcoder / binned, paired per cell; layer trends and best-layer agreement |
| `prompt_wrapping.py` | wrapped minus unwrapped, paired on identical cells; the social_iqa noise floor; the jailbreak inversion |
| `linear_hypothesis.py` | `diff` against discriminative fits, per-source transfer spread, pooling gain, nonlinear gain against distance from domain |
| `synthesis.py` | median and oracle performance summaries, cross-layer stability, error profiles, transfer predictors after accounting for diff-in-means alignment, coefficient geometry, boundary slices |
| `covariance_transfer.py` | label-free covariance-shape, variance-profile and main-variation-direction similarity; dataset-identity shuffle tests; held-dataset-out prediction |
| `label_free_pcs.py` | target-label-free PC selection: ordinary against within-dataset PCA, a learned unlabeled PC-shape ranking, random and oracle controls, held-dataset-out AUC, and a full-dimension centering control |
| `theory_checks.py` | numerical verification of the closed forms the transfer analysis rests on, asserted against simulation |

The boundary figures have a staged workflow, because their full-data fits need a
GPU and leave reusable intermediates behind:

| module | role |
|---|---|
| `boundary_refit.py` | fit full-dimensional classifiers for one scope, save model and grid artifacts, or combine completed scopes |
| `boundary_slices.py` | per-scope and overview mean-plane slices from those artifacts |
| `two_pc_rbf.py`, `two_pc_rbf_cpu.py` | the two-PC RBF comparison, GPU and CPU backends |
| `two_pc_mlp.py` | the two-PC MLP comparison |
| `boundary_grid.py` | full-dimensional linear and RBF slices beside the two-PC RBF fit |
| `mlp_boundary_comparison.py` | full-dimensional and two-PC MLP boundaries side by side |

`boundary_common.py` owns the paths and drawing helpers those six share.

## Redrawing

Every figure under `results/` is a pure function of a csv in the same `tables/`
directory. The npz-crunching and activation loading exist only to produce those
tables, so a styling change never needs a rerun:

```bash
python -m classifiers.analysis.redraw --data data_unwrap
```

`--only <family …>` narrows it. Families whose csvs are missing get skipped with
a note.
