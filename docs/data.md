# Data

Activations live in named **collections**: independent snapshots that differ in
prompt wording, dataset roster, and how labels are stored. Every loader and
runner takes `--data <collection>` and writes to `results/<collection>/`. The
registry is `classifiers.data.COLLECTIONS`; adding a collection is one entry
there.

After loading, 1 always means true / yes / safe / harmless. Sources stored the
other way round get flipped at load time.

## original

Bare prompts. Scope layers 9, 17, 22, 29. Last prompt token (`--raw last`) or
the mean over prompt tokens (`--raw mean`).

| dataset | n | task |
|---|---|---|
| got (curated geometry-of-truth statements) | 8,860 | tf |
| common_claim | 4,450 | tf |
| counterfact | 31,964 | tf |
| pubmedqa_artificial | 30,250 | medical |
| pubmedqa_labeled | 890 | medical |
| harmful_harmless | 1,024 | jailbreak |
| wildjailbreak | 49,997 | jailbreak |
| xstest | 450 | jailbreak |

## tf_wrapped_a

Every prompt rewritten as a True/False question. Safety prompts become
"Determine whether the following is True or False. It is safe for an AI
assistant to answer …". All 34 layers cached; runs default to 17, 22, 29, 33.

The convention here is stored in the files themselves (True = factually correct
or safe to answer), so nothing gets flipped. The eight original datasets come
back with the same task groups, wildjailbreak grown to 50,000, and five more
appear:

| dataset | n | task | notes |
|---|---|---|---|
| social_iqa | 50,000 | (own) | binary under the T/F wrapping |
| claude_anatomy | 1,000 | medical | model-generated anatomy statements |
| claude_blood_test_ranges | 450 | medical | model-generated, has a `perturbation` field |
| claude_diagnosis_severity | 40 | medical | eval-only |
| claude_vds | 126 | (own) | model-generated |

## data_unwrap

The same prompts as tf_wrapped_a with the wrapper stripped: the bare statement
or request, re-extracted in the same layout, all 34 layers. It runs at
tf_wrapped_a's layers rather than original's, so the wrapped/unwrapped
comparison sits at matched depths.

Labels are original's, stored per source:

| source | stored as | flipped at load |
|---|---|---|
| geometry_truth (got / common_claim / counterfact) | 1 = true statement | no |
| harmful_harmless | 1 = harmless | no |
| pubmedqa_artificial / pubmedqa_labeled | `'yes'` / `'no'` strings | no (positive = `'yes'`) |
| wildjailbreak | `True = harmful` | **yes** |
| xstest | `True = unsafe` | **yes** |
| claude_* | 1 = true statement | no |

The claude_\* files carry no `meta_label_convention`, so I read their polarity
off the prompt text. `claude_blood_test_ranges` also checks out structurally:
its labels match its `perturbation` field (`none` ↔ 1) on all 450 rows.

### social_iqa comes from the wrapped tree

Even under data_unwrap. There is no unwrapped equivalent to load, because the
two extractions have different rows and not just different wording.

The unwrapped extraction gives one activation per question (n = 35,364). Its
prompt lists choices A/B/C, asserts nothing, and is labelled with the correct
answer index. That is not a binary property of the prompt, which is also why
`original` has no social_iqa unit. The wrapped extraction instead expands each
question into one row per choice ("The correct answer for the question … is …"),
sampled and balanced to n = 50,000, so every activation belongs to a single
assertion that is either true or false.

Nothing computable from the unwrapped vectors recovers those, since the model
never saw a prompt asserting one answer. Keeping the wrapped files costs one
thing: this dataset carries the wrapper line that the other eleven don't, and a
pooled model can see it.

## Datasets with special roles

`claude_diagnosis_severity` is eval-only. It is scored, never trained on.

`claude_blood_test_ranges` is `--exclude`d from the pooled runs, so it gets a
single-dataset learner in `run_per_dataset` and nothing else.

For tf_wrapped_a and data_unwrap alike, that leaves:

- 12 training datasets, 13 test datasets, per-dataset
- a pooled corpus of 11 datasets, 179,011 samples
- 11 leave-one-dataset-out holdouts
- 5 leave-one-task-out holdouts (tf, medical, jailbreak, social_iqa, claude_vds)

## Feature sets

`--features` picks any of:

| name | what it is |
|---|---|
| `raw` | the residual stream activation itself, d = 2,560 |
| `pca` | PCA of the raw activations, fit on training data only |
| `binned_activations` | raw activations discretized into bins |
| `sae` | Gemma Scope 2 SAE features, sparse |
| `transcoder` | Gemma Scope 2 single-layer transcoder features, sparse |
| `pca_dict` | PCA of the raw activations concatenated with kept dictionary features |

Sparse sets keep only features that fire on at least `MIN_FIRES` training
prompts.

## Layer numbering

Layers are Scope layers everywhere: code, configs, figure labels. Cached files
are named by Hugging Face `hidden_states` index, and the two are off by one.

    Scope layer s  ->  file layer{s+1}  ->  hidden_states[s+1]

So Scope SAE `layer_12` reads `hidden_states[13]`. Worth being careful with,
because getting it wrong is silent: the activations load fine, they are just
from the wrong depth. The conversion happens once, in `classifiers.data`, and
nothing downstream repeats it.

## Files

`anqiwang02/Activations` on Hugging Face holds one `.npz` per dataset × variant
(`last` / `mean` / `sae` / `transcoder`) × layer. Downloads go through
`hf_hub_download`, so a file is fetched once and read from
`~/.cache/huggingface` after that.

| key | contents |
|---|---|
| `X` / `activations` | (N, D) activations. Dense, or sparse COO (`row`/`col`/`val`/`shape`) for sae and transcoder, which load as scipy CSR |
| `labels` | (N,) evaluation labels |
| `dataset_id` | (N,) source dataset per sample, where one file holds several |
| `topic` / `types` / `data_types` | (N,) topic or subtype per sample |
| `prompts` | (N,) prompt text |

Every sample also gets a `sample_id` of the form `<source>:<row>`, e.g.
`cities:42`. Row order is identical across layer files, so a sample keeps its id
at every layer. Cross-layer joins key on that.

`extract_activations/` has the code that produced all of this, one directory per
dataset. `sample_prompts/` there holds a few dozen rows of each, enough to see
the prompt formats without downloading anything.
