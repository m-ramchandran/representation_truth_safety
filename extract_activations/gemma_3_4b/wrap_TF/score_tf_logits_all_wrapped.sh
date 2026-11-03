#!/bin/bash
#SBATCH --job-name=score_logits
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=results/score-logits-%j.out
#SBATCH --error=results/score-logits-%j.err

# Backfill {dataset}_tf_logits.npz across every wrapped activation dir on
# scratch, in one job. Forward-only, no re-extraction. Missing dirs are skipped.
#
# Env: DATASETS (default: all), VARIANT=A|B, BATCH_SIZE, SCRATCH_WRAP_ROOT

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"

VARIANT="${VARIANT:-A}"
SCRATCH_WRAP_ROOT="${SCRATCH_WRAP_ROOT:-${SCRATCH_ROOT}/dataset_wrap_TF}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATASETS="${DATASETS:-geometry_truth harmful_harmless xstest pubmedqa_artificial pubmedqa_labeled social_iqa social_iqa_binary wildjailbreak claude_anatomy claude_blood_test_ranges claude_vds claude_vds_300}"

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

echo "Node: $(hostname) | variant=${VARIANT} | batch_size=${BATCH_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: CUDA not available — submit with sbatch on a GPU partition." >&2
    exit 1
fi

for dataset in ${DATASETS}; do
  outdir="${SCRATCH_WRAP_ROOT}/${dataset}_activations/${VARIANT}"
  if [[ ! -d "${outdir}" ]]; then
    echo "Skip (no dir): ${outdir}"
    continue
  fi
  echo "=== ${dataset} variant=${VARIANT} ==="
  python "${WRAP_ROOT}/score_tf_logits.py" \
    --outdir "${outdir}" \
    --force \
    --require-gpu \
    --batch-size "${BATCH_SIZE}"
done

echo "Job finished with exit code 0"
