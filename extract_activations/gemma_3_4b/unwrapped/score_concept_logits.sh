#!/bin/bash
#SBATCH --job-name=score_concept_logits
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/score-concept-logits-%j.out
#SBATCH --error=results/score-concept-logits-%j.err

# Backfill {dataset}_tf_logits.npz for one unwrapped activation outdir.
#
# Env: OUTDIR (required), BATCH_SIZE, MAX_LEN, HF_HOME

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"

UNWRAPPED_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTDIR="${OUTDIR:?Set OUTDIR relative to unwrapped/ e.g. harmful_harmless/activations}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LEN="${MAX_LEN:-0}"

if [[ "${OUTDIR}" != /* ]]; then
  OUTDIR="${UNWRAPPED_ROOT}/${OUTDIR}"
fi

mkdir -p "${UNWRAPPED_ROOT}/results"
cd "${UNWRAPPED_ROOT}"

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate rare_event

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/huggingface-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"

if [[ ! -d "${OUTDIR}" ]]; then
  echo "ERROR: outdir not found: ${OUTDIR}" >&2
  exit 1
fi

echo "Node: $(hostname) | outdir=${OUTDIR} | batch_size=${BATCH_SIZE} max_len=${MAX_LEN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "${UNWRAPPED_ROOT}/score_concept_logits.py" \
  --outdir "${OUTDIR}" \
  --force \
  --require-gpu \
  --batch-size "${BATCH_SIZE}" \
  --max-len "${MAX_LEN}"

echo "Job finished with exit code 0"
