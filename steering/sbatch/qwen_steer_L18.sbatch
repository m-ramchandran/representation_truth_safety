#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --job-name=qwen_steer_L18
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#
# The Qwen sweep at the SECOND extracted layer (L18, 68% depth) instead of
# L24 (89%). Same design as qwen_steer.sbatch; only the site moves. See
# gemma_steer_L22.sbatch for what this tests.
#
# Five hours, not three: each edit reruns every block above the steering
# layer, and L18 has 10 of Qwen's 28 above it where L24 had 4 — about 2.5x
# the per-edit compute. The longest L24 task ran 1h18. Snapshots every ten
# batches make a walltime kill resumable.
#
# Order:
#   sbatch steering/sbatch/qwen_preflight_L18.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/qwen_directions_L18.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/qwen_steer_L18.sbatch

# Four targets, not six: the expanded-pool fits that serve claude_vds
# and wildjailbreak_adversarial were never made at this layer, and
# nothing is ever refit for steering. The directions build for this
# run uses the CLASS-CENTERED (within-class) covariance, and the
# RIDGE dial ((1-t)Shat + tI)^(1/2) replaces the multiplier rungs.
# t=1 is the coefficients as
# fitted; t=0 is half of one multiplication; nothing goes beyond.
TARGETS=(common_claim wildjailbreak_vanilla xstest pubmedqa_labeled)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "array task $SLURM_ARRAY_TASK_ID -> $TARGET"
python -u steering/steer.py --collection qwen_tf_a --layer 18 \
    --directions-tag qwen --targets "$TARGET" "$@"
