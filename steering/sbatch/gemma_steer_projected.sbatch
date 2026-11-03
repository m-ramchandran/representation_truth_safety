#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --job-name=gemma_steer_proj
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out

# Projected-coefficient steering run. Each target
# is split over two array tasks (interleaved halves of its 97 vectors); only
# the six push sizes the slope uses (0.25, 0.5, 1 either side), no removal or
# mean matching. Submit with --dependency=afterok:<projected_dirs job>.
TARGETS=(common_claim claude_vds wildjailbreak_vanilla xstest \
         pubmedqa_labeled wildjailbreak_adversarial)
TARGET=${TARGETS[$((SLURM_ARRAY_TASK_ID / 2))]}
PART=$((SLURM_ARRAY_TASK_ID % 2 + 1))

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "array task $SLURM_ARRAY_TASK_ID -> $TARGET part $PART/2"
python -u steering/steer.py --collection gemma_tf_a --layer 29 \
    --directions-tag gemmaproj --targets "$TARGET" \
    --max-frac 1 --adds-only --part ${PART}/2 "$@"
