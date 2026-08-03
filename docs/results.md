# What a run writes

Each runner writes one metrics csv and one prediction npz per (feature set,
layer) cell, under `results/<collection>/<experiment>/`:

    tables/       metrics_<tag>_layer<L>.csv, one row per fit, AUC to
                  three decimals. Doubles as the resume record: a group
                  whose rows are present gets skipped on re-run.
    predictions/  one .npz per (tag, layer, method, fit), with per-sample
                  scores and the fitted model's arrays
    figures/      auc_<tag>*.png, drawn from the csv

`<tag>` is always `<features>_<raw>`: `raw_last`, `pca_mean`, `sae_last`, and so
on. No two runs can write the same file. Analysis output joins the same
`tables/` and `figures/` directories and is told apart by name, since
`metrics_*` and `auc_*` belong to the runner and everything else does not. The
cross-experiment coefficient modules write to
`results/<collection>/coefficients/` instead.

## Inside a prediction npz

Present for every experiment:

| key | contents |
|---|---|
| `cv_sample_id`, `cv_y` | sample id (`<source>:<row>`, joinable to prompts and topics) and label, per training sample |
| `cv_topic` | per-training-sample topic where the cache stores one. `cv_predictions` exposes it as the `topic` column, `""` when absent |
| `cv_proba`, `cv_margin` | per-sample CV predictions. `cv_margin` is absent for knn, which has no margin |
| `params` | chosen hyperparameters as a repr, parsed back by `load_prediction` |
| `standardize_mean`, `standardize_scale` | the standardization fit in front of the model. The mean is absent when scaling was variance-only (sparse input); both are absent for `diff`, which takes raw input |
| method extras | `dev_w` / `dev_b` / `group_labels` for random_slopes (per-dataset deviations, rows in `group_labels` order) and random_slopes_task (same arrays, one row per task). `gate_w` / `gate_b` / `expert_w` / `expert_b` for moe. `feature_importance` + `booster_ubj` for xgboost |
| featurization extras | whatever the featurization fit: `kept_features`, or (`n_pcs`, `pca_mean`, `pca_components`), or `bin_edges`, or (`kept_sae`, `kept_transcoder`) |

On top of that, per experiment:

**per_dataset** adds `cv_fold` (outer fold index; CV here is nested) and
`transfer_{dataset,sample_id,y,proba,margin}`, the full fit applied to every
other dataset, plus that full fit's own model arrays.

**leave_out** adds `cv_dataset` (each pool sample's source dataset) and
`holdout_{dataset,sample_id,y,proba,margin}`, plus the pool fit's model arrays.

**merged** adds `cv_fold` and `cv_dataset`, and its `params` is a per-outer-fold
*list*. It also carries one tuned fit on the whole pooled corpus, which is not
part of the nested-CV AUCs: its arrays take a `full_` prefix (`full_params`,
`full_coef`, `full_standardize_scale`, `full_gate_w`, …). Where the collection
has eval-only units, that fit's scores on them land in
`eval_only_{dataset,sample_id,y,proba,margin}` and the csv rows carry kind
`eval_only`. random_slopes_task scores eval-only units with their task's
deviation and stores the shared-weights-only scores next to them
(`eval_only_shared_{proba,margin}`).

`coef` and `full_coef` are absent for methods that have no coefficient vector:
qda, knn, mlp, moe, xgboost. svm_rbf's lives in its Nyström feature space, and
the Nyström map (landmarks and whitening) is not persisted, so that vector
cannot be taken back to input space either.

### xgboost's two keys

`feature_importance` is each feature's share of the total split gain. It is a
magnitude with no sign, so it is not a direction. `directions.py` refuses it,
and correlated activation dimensions split the credit between them anyway. Its
nonzero count is a support size comparable to `coef_sparsity`'s.

`booster_ubj` is the fitted ensemble itself, UBJSON bytes in a uint8 array,
about 0.4 MB deflated. xgboost is the one method whose predictions cannot be
rebuilt from a summary vector, so `load_results.load_booster(d, prefix)` hands
back an `xgboost.Booster` that reproduces the fit's margins exactly and scores
new data without a refit. It wants raw featurized values with no standardization
in front, so put new activations through `directions.apply_featurization` and
stop there. Pass `keys=` to `load_prediction` and the booster is never
decompressed.

## Conventions

Set in `classifiers/experiments/`, mirrored in `classifiers/analysis/`. Break
one and you get plausible numbers, not an error.

*Margins.* For l1reg, l2reg, lda, qda, mlp, random_slopes and moe the margin is
the exact log-odds of `proba`. For svm_linear, svm_rbf and diff it is the raw
decision value, with `proba = sigmoid(a * margin + b)` and the Platt `(a, b)` in
`params`. Apply that map before any cross-method comparison that needs those
margins on the log-odds scale.

*Coefficient spaces.* Every method's `coef` lives in standardized-feature space,
the space the scaler in front of the model produces. The exception is `diff`,
whose vector is the unit-norm class-mean direction in raw activation units.
`coef / standardize_scale` maps a standardized-space vector to raw feature
units, using each fit's own scaler: two fits on different training data have
different scalers, so map both before comparing them, and always before
comparing anything to `diff`. The random-slopes variants' `coef` is their shared
weights; the per-group deviations sit in `dev_w` / `dev_b`, same space.

`directions.py` implements all of this. `load_direction` and `raw_direction` map
any saved coef to cache units, back-projecting pca fits to raw space, and
`stack_directions` aligns sparse supports. Use it rather than re-deriving the
mapping.

*Feature indexing.* For sparse feature sets, coefficients index the *kept*
features. Map back to dictionary indices with `kept_features`, or with
`kept_sae` / `kept_transcoder` for `pca_dict`, after the `n_pcs` PCA block.

*Labels.* 1 = true / yes / safe / harmless, set at load time in
`classifiers/data.py`, per collection.
