#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate "${CONDA_ENV:-representation}"

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

scripts=(
    q0_activation_geometry.py
    q1_figures.py
    q16_figures.py
    q12_figures.py
    q17_heldout_targets.py
    fig_steering_projected.py
    q6_layer_figure.py
    q5_figures.py
    q7_figures.py
)

for script in "${scripts[@]}"; do
    python "paper_final_figures/scripts/$script"
done
