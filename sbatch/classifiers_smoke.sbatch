#!/bin/bash
#SBATCH --job-name=clf_smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Classifier smoke test: every method through the fit contract (incl.
# the coef/model-array checks), the sparse path, Nystrom exactness, and
# all three runners end to end with resume, on synthetic data. ~9 min
# on a V100; run after any change to scripts/ before production arrays.

# Initialize the environment-module function when the batch shell omits it.
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m classifiers.scripts.smoke_test
