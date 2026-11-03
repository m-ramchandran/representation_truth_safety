#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A.out

# Post-hoc analyses + figures for one collection, on CPU. Layers come
# from the collection itself, so Qwen's are picked up automatically.
#
# DO NOT run while that collection's arrays are still going: every module
# writes n_samples=0 PLACEHOLDER rows for any npz not yet on disk and
# exits 0, so a premature run looks successful and silently loses
# methods. submit_all.sh chains this with afterany for that reason.
#
# 24 cores (~47 GB) is for coef_similarity's Ledoit-Wolf covariances,
# which peak around 14 GB; 8 cores OOM-killed it silently once.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

: "${COLL:?set COLL via --export (use sbatch/submit_all.sh)}"

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m classifiers.analysis_scripts.run_all --data $COLL --keep-going
