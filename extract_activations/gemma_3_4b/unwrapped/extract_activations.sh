#!/bin/bash
#SBATCH --job-name=extract_unwrap_rm_bench
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --output=results/extract-%j.out
#SBATCH --error=results/extract-%j.err

# Extract Gemma-3-4B activations for one unwrapped dataset.
#
# Env: DATASET (required), OUTDIR, SCRATCH_UNWRAP_ROOT, EXTRA_ARGS, HF_HOME

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"

DATASET="${DATASET:?Set DATASET=geometry_truth|harmful_harmless|xstest|social_iqa|social_iqa_binary|wildjailbreak|pubmedqa_artificial|pubmedqa_labeled|claude_anatomy|claude_blood_test_ranges|claude_vds|claude_vds_300|rm_bench}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Large .npz outputs go to scratch by default (see extract_activations.py).
SCRATCH_UNWRAP_ROOT="${SCRATCH_UNWRAP_ROOT:-${SCRATCH_ROOT}/dataset_unwrap}"
OUTDIR="${OUTDIR:-${SCRATCH_UNWRAP_ROOT}/${DATASET}_activations}"

UNWRAP_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${UNWRAP_ROOT}/results" "${OUTDIR}"

cd "${UNWRAP_ROOT}"

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate rare_event

# Put large HF downloads on scratch, but keep auth from the login token.
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/huggingface-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    for token_file in \
        "${XDG_CACHE_HOME:-}/huggingface/token" \
        "${SCRATCH_ROOT}/.xdg-cache/huggingface/token" \
        "${HOME}/.cache/huggingface/token" \
        "${HOME}/.huggingface/token"
    do
        if [[ -n "${token_file}" && -f "${token_file}" ]]; then
            export HF_TOKEN="$(<"${token_file}")"
            echo "HF auth: loaded token from ${token_file}"
            break
        fi
    done
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARNING: no HF token found; gated model google/gemma-3-4b-it will 401" >&2
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname) | dataset=${DATASET}"
echo "Activations outdir: ${OUTDIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python extract_activations.py \
    --dataset "${DATASET}" \
    --outdir "${OUTDIR}" \
    --layers all \
    --batch-size 4 \
    --max-len 0 \
    --gen-tokens 10 \
    --decomps sae crosscoder transcoder \
    ${EXTRA_ARGS}

echo "Job finished with exit code $?"
