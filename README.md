# representation_truth_safety

Training and evaluating activation probes on Gemma 3 4B activations, with Gemma Scope 2 sparse decompositions.
Thirteen classifiers spanning linear and non-linear approaches are fit to read a binary property of a prompt (true/false,
safe/unsafe, yes/no) from the residual stream, over six feature representations
and four layers, on thirteen datasets spanning factual claims, medical question
answering, jailbreak prompts, and commonsense reasoning. A separate unsupervised
pass clusters the same activations using various clustering approaches (GMM, MFA, k-means, Leiden, HDBSCAN) and analyzes the dataset and label contributions of each estimated cluster. 

Each fit is scored under three training regimes:

| runner | trained on | scored on |
|---|---|---|
| `run_per_dataset` | one dataset | itself by nested CV, and every other dataset |
| `run_merged` | the pooled corpus | itself by nested CV, plus the eval-only datasets |
| `run_leave_out` | the corpus minus one dataset, or minus one task | whatever was held out |

All combinations of method, feature set, layer, and regime are tested.

## Layout

    extract_activations/   one script per dataset: prompts in, cached
                           activations out (residual stream, SAE features,
                           transcoder features, crosscoder features)
    classifiers/           the main experiments and everything downstream
      data.py                the dataset registry and activation loader
      methods/               thirteen classifiers behind one fit contract
      experiments/           the three runners above
      analysis/              post-hoc analyses of what the runs wrote
        studies/             cross-experiment investigations
      results/               metrics tables, one tree per collection
    clustering/            unsupervised structure in the same activations
    docs/                  data, results format, analysis index

`docs/data.md` covers the datasets and their label conventions,
`docs/results.md` what a run writes and how to read it, `docs/analyses.md` the
module index. Module docstrings carry the exact usage.

## Setup

```bash
conda env create -f environment.yml && conda activate representation
```

Fitting needs a GPU: `classifiers.methods` imports cuML at load, and the larger
pooled runs want around 60 GB of host memory. Everything downstream of
`results/` is plain numpy, so `requirements-cpu.txt` is enough for the analyses.

## Fitting

The smoke test puts every method and all three runners through small synthetic
data:

```bash
python -m classifiers.experiments.smoke_test
```

A run takes one collection, one raw variant, and sets of layers, feature sets
and methods:

```bash
python -m classifiers.experiments.run_per_dataset --data data_unwrap --raw last --layers 17 22 29 33
```

```bash
python -m classifiers.experiments.run_merged --data data_unwrap --raw last --layers 17 22 29 33
```

```bash
python -m classifiers.experiments.run_leave_out --data data_unwrap --holdout dataset --raw last --layers 17 22 29 33
```

Runs resume. Any (feature set, layer) cell already present in the metrics csv is
skipped, so a job that dies partway can be resubmitted unchanged.

## Analysis

```bash
python -m classifiers.analysis.run_all --data data_unwrap
```

Nothing under `analysis/` reads the activations or the fitted models, only
`results/`. Each figure is a function of a csv in the same directory.

```bash
python -m classifiers.analysis.redraw --data data_unwrap
```

## What is in results/

Metrics tables for two collections: `tf_wrapped_a`, every prompt rephrased as a
True/False question, and `data_unwrap`, the same prompts unwrapped. Figures and
per-sample prediction archives are not committed. `redraw` rebuilds all 150
figures of a collection from its tables, no GPU involved.

Activations are not in the repository. They live on the Hugging Face dataset
repo `anqiwang02/Activations` and are fetched once into the local HF cache.
`extract_activations/` is the code that produced them.
