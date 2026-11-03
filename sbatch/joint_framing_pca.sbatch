#!/bin/bash
#SBATCH --job-name=joint_framing_pca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# The supplementary_figure1_framing basis: one PCA per model fit on the
# Original and True/False samples pooled, both framings projected onto it.
#
# CPU only - charges the Regular Memory pool. 16 cores buys ~31 GB; the
# peak holds both Qwen matrices at once (200k x 3584 and 231k x 3584
# float32, 2.9 + 3.3 GB) plus the 103 MB covariance. Only the top four
# components are projected out, so there is no full-rank score matrix and
# no supervised readout here - those stay with full_activation_pca.sbatch.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

python paper_final_figures/scripts/joint_framing_pca.py
