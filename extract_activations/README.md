# Activation extraction

Scripts and datasets for saving a model's internal activations.

First, install and activate the Conda environment from the `extract_activations/` directory:

```bash
conda env create -n rare_event -f environment.yaml
conda activate rare_event
```

Use an NVIDIA GPU with CUDA support and enough memory for the model. To use Gemma, request access to `google/gemma-3-4b-it` on Hugging Face, then use a saved Hugging Face login or set `HF_TOKEN` to your access token. Leave enough disk space for model downloads and results. If using the Slurm job scripts, update the partition name and Conda setup path for your cluster.

For a small first run, run this from `extract_activations/` on a machine with a GPU:

```bash
python gemma_3_4b/unwrapped/extract_activations.py \
  --model google/gemma-3-4b-it \
  --dataset claude_anatomy --n 16 \
  --layers all --pool last --batch-size 1 \
  --no-decomps --outdir ./outputs/gemma_anatomy
```

This processes 16 examples from the local anatomy dataset, one at a time. It saves values from every model layer at the last input position. `--no-decomps` skips the optional Gemma Scope analysis. Run the script with `--help` for other options.

Results are saved in the folder given by `--outdir`:

- `{dataset}_last_layer{L}.npz`: One NumPy file per layer, containing an `activations` array (one row per example), input text, labels, model responses, and run settings.
- `{dataset}_tf_logits.npz`: Scores and probabilities for possible next answers, such as True/False and Yes/No.

Without `--outdir`, set `SCRATCH_ROOT` to a writable folder for large files. Each model and prompt format uses a separate subfolder there.


| Folder                           | Purpose                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `gemma_3_4b/`                    | Scripts for Gemma 3 4B IT, including optional Gemma Scope analysis.                                            |
| `qwen_7b/`                       | Scripts for Qwen2.5-7B-Instruct, using the same data and input text as the Gemma scripts.                      |
| `generated_data/`                | Generated true/false CSV datasets covering anatomy, blood-test ranges, and VDS.                                |
| `shared_core/`                   | Code shared by both models for saving activations, selecting examples, and scoring answers.                    |
| `shared_core/shared_sample_ids/` | Saved row numbers and checks that let the scripts use the same examples and detect changes to the source text. |


Both model folders contain:

- `unwrapped/`: Uses the original dataset text as input.
- `wrap_TF/`: Rewrites dataset examples as statements to judge as True or False. 

