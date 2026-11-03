#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a_%A.out

# Pooled nested CV, one collection. Submit through
# sbatch/submit_all.sh; sbatch/classifiers_grid.sh maps the index to a
# cell. One task = one (feature set, layer); the csvs are the resume
# record.
#
# The heaviest of the three: 5 outer folds over a pool that reaches ~205k
# rows under perturbation A. The pool_excluded units are scored here by
# the whole-corpus fit (kind "eval_only"), and since that fit's pool is
# every poolable dataset, those rows ARE their held-out-dataset numbers.

# Some compute nodes hand the batch script a shell with no `module`
# function; define it from lmod's initializer when that happens.
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

: "${COLL:?set COLL via --export (use sbatch/submit_all.sh)}"

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source sbatch/classifiers_grid.sh

cell_of "$COLL" pooled "$SLURM_ARRAY_TASK_ID"
echo "collection=$COLL features=$FEATURES layer=$LAYER"

python -m classifiers.scripts.run_merged \
    --data $COLL --raw last --features $FEATURES --layers $LAYER --n-jobs 5
