#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a_%A.out

# Within-dataset nested CV + cross-dataset transfer, one collection.
# Submit through sbatch/submit_all.sh, which sizes the array and sets the
# job name; sbatch/classifiers_grid.sh maps the index to a cell.
#
# One task = one (feature set, layer). That is the finest split without
# two tasks appending to the same metrics csv, which is also the resume
# record — a timed-out task is resubmitted unchanged and continues.

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

cell_of "$COLL" per_dataset "$SLURM_ARRAY_TASK_ID"
echo "collection=$COLL features=$FEATURES layer=$LAYER"

python -m classifiers.scripts.run_per_dataset \
    --data $COLL --raw last --features $FEATURES --layers $LAYER --n-jobs 5
