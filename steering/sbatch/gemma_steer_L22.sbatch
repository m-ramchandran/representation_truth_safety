#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --job-name=gemma_steer_L22
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#
# The Gemma sweep at the SECOND extracted layer (L22, 68% depth) instead of
# L29 (88%). Same six datasets, same 64 vectors each, same doses — only the
# probing site moves. The question this answers: does the mid-depth site's
# flatter covariance leave the fitted coefficient vectors with a larger
# component along the class-mean difference, so that they steer partially
# (the ITI / Marks-Tegmark regime) instead of not at all (the L29 result)?
#
# Six hours, not three: each edit reruns every block ABOVE the steering
# layer, and L22 has 12 of Gemma's 34 above it where L29 had 5 — about
# 2.4x the per-edit compute. The longest L29 task ran 2h12; snapshots every
# ten batches still make a walltime kill resumable.
#
# Order:
#   sbatch steering/sbatch/gemma_preflight_L22.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/gemma_directions_L22.sbatch
#   sbatch --dependency=afterok:<id> steering/sbatch/gemma_steer_L22.sbatch

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
python -u steering/steer.py --collection gemma_tf_a --layer 22 \
    --directions-tag gemma --targets "$TARGET" "$@"
