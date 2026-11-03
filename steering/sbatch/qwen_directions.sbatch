#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=1:00:00
#SBATCH --job-name=qwen_dirs
#SBATCH --output=logs/%x_%A.out

# Build the Qwen layer-24 directions from SAVED fits (nothing is refit).
# Must finish before the steering array starts — all six array tasks read
# the one directions_qwen_layer24.npz this writes.
#
# More memory and CPUs than the Gemma build: Qwen's states are 3584-wide,
# so each covariance is 3584x3584, and the build now takes four powers of
# it through one eigendecomposition per target.
#
#   sbatch steering/sbatch/qwen_preflight.sbatch     # first, always
#   sbatch steering/sbatch/qwen_directions.sbatch

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/directions.py --collection qwen_tf_a --layer 24 --tag qwen
