#!/bin/bash
#SBATCH --job-name=classifiers_paper
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out

# Submit only after both collection-specific analysis jobs have completed.
# This reads both result trees and writes the curated cross-collection output
# under classifiers/results/paper/. GPU boundary artifacts are assembled by
# a separate optional invocation with --stages boundaries.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

python -m classifiers.analysis_scripts.paper
