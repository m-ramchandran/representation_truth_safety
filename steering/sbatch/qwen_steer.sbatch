#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --job-name=qwen_steer
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out

# The Qwen sweep, one dataset per array task. Splitting it this way lets a
# single failed dataset rerun on its own and avoids six tasks writing one
# file.
#
# Results are checkpointed every ten batches, so interrupted jobs can resume.
# Qwen2.5-7B is ~15 GB in bf16 and requires a compatible GPU.
#
# Order matters — the array reads the directions file the build writes:
#   sbatch steering/sbatch/qwen_preflight.sbatch
#   sbatch steering/sbatch/qwen_directions.sbatch          # note the job id
#   sbatch --dependency=afterok:<id> steering/sbatch/qwen_steer.sbatch
#
# Smoke first (three directions, one dose, one target) before the full array:
#   sbatch --array=0 steering/sbatch/qwen_steer.sbatch --smoke

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
python -u steering/steer.py --collection qwen_tf_a --layer 24 \
    --directions-tag qwen --targets "$TARGET" "$@"
