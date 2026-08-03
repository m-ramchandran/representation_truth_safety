# Activation extraction

One directory per dataset. Each holds an `extract_activations.py` that takes the
dataset's prompts, runs them through Gemma 3 4B, and writes one self-contained
`.npz` per layer per feature set. Datasets that share a pipeline import a common
module (`geometry_truth/got_common.py`, `PubMedQA/pubmedqa_common.py`) and keep
only their loader and defaults locally.

This step needs a GPU and is the only part of the repository that touches the
model. Nothing downstream reruns it, since the outputs are cached on Hugging
Face. The code is here because the extraction settings are part of the method.

## What one script writes

For each Scope layer, at `activations/<dataset>_<variant>_layer<L>.npz`:

| variant | contents |
|---|---|
| `last` | residual stream at the last prompt token |
| `mean` | residual stream averaged over all prompt tokens |
| `sae` | Gemma Scope 2 SAE features, sparse COO |
| `transcoder` | Gemma Scope 2 single-layer transcoder features, sparse COO |
| `crosscoder` | Gemma Scope 2 crosscoder features over Scope layers 9/17/22/29, one file |

Each file also carries labels, prompt text, per-sample topic or subtype where the
dataset has one, and up to ten greedily decoded tokens per prompt. Layers are
Scope layers; see `docs/data.md` for the off-by-one against Hugging Face's
`hidden_states` index.

## Running one

From the dataset's own directory:

```bash
python extract_activations.py --layers all --batch-size 4 --max-len 0 --gen-tokens 10 --decomps transcoder
```

`--max-len 0` means no truncation, so each batch pads only to its own longest
prompt. Decompositions need `sae-lens`; `--no-decomps` skips them and is much
faster. Crosscoder weights are read from `$GEMMA_SCOPE_DIR/crosscoder`, falling
back to the local Hugging Face cache and then the network.

Those flags produced the cached files for every dataset. On top of them:

| dataset | extra arguments |
|---|---|
| geometry_truth (each of the nine parts) | — |
| harmful_harmless, XSTest, social_IQA | `--split all` |
| wildjailbreak | `--config train --n-per-category 12500` |
| PubMedQA/pqa_artificial | `--n 30250 --balance` |
| PubMedQA/pqa_labeled | `--n 0 --no-balance` |

`google/gemma-3-4b-it` is gated, so authenticate once with `hf auth login`
before the first run.

## sample_prompts/

A few dozen rows of each dataset, as csv. Enough to see the prompt formats and
label conventions without downloading any activations.
