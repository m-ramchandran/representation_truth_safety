#!/bin/bash
#SBATCH --job-name=score_logits
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=results/score-logits-%j.out
#SBATCH --error=results/score-logits-%j.err

# Score / refresh {dataset}_tf_logits.npz for one wrapped activation outdir.
#
# Env: DATASET (required), VARIANT=A|B, BATCH_SIZE, OUTDIR, SCRATCH_WRAP_ROOT

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"

DATASET="${DATASET:?Set DATASET=geometry_truth|harmful_harmless|xstest|...}"
VARIANT="${VARIANT:-A}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SCRATCH_WRAP_ROOT="${SCRATCH_WRAP_ROOT:-${SCRATCH_ROOT}/dataset_wrap_TF}"
OUTDIR="${OUTDIR:-${SCRATCH_WRAP_ROOT}/${DATASET}_activations/${VARIANT}}"

WRAP_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${WRAP_ROOT}/results"

cd "${WRAP_ROOT}"

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate rare_event

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

if [[ ! -d "${OUTDIR}" ]]; then
    echo "ERROR: outdir not found: ${OUTDIR}" >&2
    exit 1
fi

echo "Node: $(hostname) | dataset=${DATASET} variant=${VARIANT}"
echo "Outdir: ${OUTDIR} | batch_size=${BATCH_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available — submit with sbatch on a GPU partition." >&2
    exit 1
fi

python "${WRAP_ROOT}/score_tf_logits.py" \
  --outdir "${OUTDIR}" \
  --force \
  --require-gpu \
  --batch-size "${BATCH_SIZE}"

echo "Job finished with exit code 0"
