#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --job-name=gemma_steer
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#
# The Gemma sweep, one dataset per array task. Splitting it this way lets a
# single failed dataset rerun on its own and avoids six tasks writing one
# file.
#
# Results are checkpointed every ten batches, so interrupted jobs can resume.
# bf16 requires a compatible GPU.
#
# Order matters — the array reads the file the build writes:
#   sbatch steering/sbatch/gemma_preflight.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/gemma_directions.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/gemma_steer.sbatch

TARGETS=(common_claim claude_vds wildjailbreak_vanilla xstest \
         pubmedqa_labeled wildjailbreak_adversarial)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "array task $SLURM_ARRAY_TASK_ID -> $TARGET"
python -u steering/steer.py --collection gemma_tf_a --layer 29 \
    --directions-tag gemma --targets "$TARGET" "$@"
