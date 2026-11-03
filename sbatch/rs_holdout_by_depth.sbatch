#!/bin/bash
#SBATCH --job-name=rs_holdout_by_depth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A.out

# Figure 5's random-slopes rows kept per depth, for both permanently
# held-out targets, so they can be compared with the L2 target-absent probe
# layer by layer. Reproduces psc_followups.rs_rows; reads saved fits only.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
set -eo pipefail
mkdir -p logs classifiers/results/_rs_by_depth

python paper_final_figures/scripts/rs_holdout_by_depth.py \
    classifiers/results \
    classifiers/results/_rs_by_depth/rs_holdout_by_depth.csv
echo done
