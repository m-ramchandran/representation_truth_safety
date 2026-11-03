# Classifier experiments

This package trains and evaluates binary probes on Gemma 3 4B and Qwen2.5 7B
activations. All commands are run from the repository root.

## Data

Activation files are read from the Hugging Face dataset
`anqiwang02/Activations`; set `ACTIVATIONS_REPO` to use a different one.
`classifiers.data.COLLECTIONS` defines the four evaluated collections:

| collection | model | prompt format |
|---|---|---|
| `gemma_original` | Gemma 3 4B IT | original |
| `gemma_tf_a` | Gemma 3 4B IT | True/False |
| `qwen_original` | Qwen2.5 7B Instruct | original |
| `qwen_tf_a` | Qwen2.5 7B Instruct | True/False |

Labels are normalized to `1 = true / yes / safe / harmless`. Source-derived
rows share a fold group so related prompts never cross a train/test boundary.

## Experiments

The three entry points correspond to the reported training regimes:

```bash
# Within-target training and cross-dataset transfer.
python -m classifiers.scripts.run_per_dataset --data gemma_tf_a --raw last

# Target-absent or domain-absent training.
python -m classifiers.scripts.run_leave_out --data gemma_tf_a --holdout task --raw last

# Pooled training.
python -m classifiers.scripts.run_merged --data gemma_tf_a --raw last
```

Use `--help` on any command for layer, feature, method, seed, and parallelism
options. The evaluated methods include logistic regression, linear and RBF
SVMs, difference in means, LDA/QDA, k-nearest neighbors, MLP, XGBoost,
random-slopes probes, and soft/hard mixtures of experts. Features include raw
activations, SAE/transcoder activations, PCA, combined dictionaries, and
activation histograms when available.

Outputs are written below `classifiers/results/<collection>/` and are excluded
from git. `classifiers/analysis_scripts/` converts those raw runs into compact
tables; the final plot inputs are checked in under `paper_final_figures/data/`.

## Validation

```bash
python -m pytest classifiers/tests
python -m classifiers.scripts.smoke_test
```

The smoke test covers every estimator contract and all three runners. GPU-backed
methods require a CUDA environment.
