#!/bin/bash
#SBATCH --job-name=extract_wrapped_rm_bench_B
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --output=results/extract-%j.out
#SBATCH --error=results/extract-%j.err

# Extract Gemma-3-4B activations for one T/F-wrapped dataset.
#
# Env: DATASET (required), VARIANT=A|B, BATCH_SIZE, OUTDIR,
#      SCRATCH_WRAP_ROOT, EXTRA_ARGS, HF_HOME

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:?Set SCRATCH_ROOT to a writable scratch directory}"

DATASET="${DATASET:?Set DATASET=geometry_truth|harmful_harmless|xstest|pubmedqa_artificial|pubmedqa_labeled|social_iqa|social_iqa_binary|wildjailbreak|claude_anatomy|claude_blood_test_ranges|claude_vds|claude_vds_300|rm_bench}"
VARIANT="${VARIANT:-A}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# rm_bench's prompt+response text runs far longer/more variable than this
# pipeline's usual short statements (up to 7136 chars), and padding is
# dynamic per-batch (see shared_extract_engine.py's extract_batch) --
# batch_size=4 OOM'd on an A100-40GB partway through a real run (job
# 3528727, batch 628/1991, two ~5000-char rows landed in the same batch).
# Override with BATCH_SIZE=1 if this still OOMs.
if [[ -z "${BATCH_SIZE:-}" ]]; then
    if [[ "${DATASET}" == "rm_bench" ]]; then
        BATCH_SIZE=2
    else
        BATCH_SIZE=4
    fi
fi

# Large .npz outputs go to scratch by default (see extract_activations.py).
SCRATCH_WRAP_ROOT="${SCRATCH_WRAP_ROOT:-${SCRATCH_ROOT}/dataset_wrap_TF}"
OUTDIR="${OUTDIR:-${SCRATCH_WRAP_ROOT}/${DATASET}_activations/${VARIANT}}"

WRAP_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${WRAP_ROOT}/results" "${OUTDIR}"

cd "${WRAP_ROOT}"

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate rare_event

# Put large HF downloads on scratch, but keep auth from the login token.
# Login with XDG_CACHE_HOME=/scratch/.../.xdg-cache stores the token there, not
# under ~/.cache/huggingface — and HF_HOME=scratch alone would miss it → 401.
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

echo "Node: $(hostname) | dataset=${DATASET} variant=${VARIANT}"
echo "Activations outdir: ${OUTDIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python extract_activations.py \
    --dataset "${DATASET}" \
    --variant "${VARIANT}" \
    --outdir "${OUTDIR}" \
    --layers all \
    --batch-size "${BATCH_SIZE}" \
    --max-len 0 \
    --gen-tokens 10 \
    --decomps sae crosscoder transcoder \
    ${EXTRA_ARGS}

echo "Job finished with exit code $?"
