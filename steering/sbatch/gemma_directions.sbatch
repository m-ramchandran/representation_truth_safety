#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=1:00:00
#SBATCH --job-name=gemma_dirs
#SBATCH --output=logs/%x_%A.out

# Build the Gemma layer-29 vectors from SAVED fits (nothing is refitted).
#
# The tag keeps this model-specific artifact separate from untagged outputs.
# The largest covariance here is wildjailbreak's ~25k rows x 2560.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/directions.py --collection gemma_tf_a --layer 29 \
    --tag gemma
