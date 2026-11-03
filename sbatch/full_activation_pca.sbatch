#!/bin/bash
#SBATCH --job-name=full_activation_pca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# The fig12 PCA, refit on every sample instead of 400 per dataset, plus
# the supervised readouts of the resulting map.
#
# CPU only - charges the Regular Memory pool. 32 cores buys ~62 GB, which
# is what the peak needs: the Qwen Original matrix is 200k x 3584 float32
# (2.9 GB), its full-rank projection another 2.9 GB, and a training fold
# of the all-PCs logistic regression ~2.3 GB on top.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

python paper_final_figures/scripts/full_activation_pca.py
