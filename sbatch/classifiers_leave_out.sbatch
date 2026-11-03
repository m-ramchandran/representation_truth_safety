#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a_%A.out

# Leave-one-out, one collection, both holdout kinds in one array.
# Submit through sbatch/submit_all.sh; sbatch/classifiers_grid.sh maps
# the index to a cell.
#
# One task = one (holdout kind, feature set, layer): first half dataset
# mode, second half task mode. The csvs are the resume record.
#
# The collection's pool_excluded units are in no pool ever, so the runner
# scores all of them against every holdout on top of its own members.

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

# two halves: dataset mode, then task mode
KINDS=(dataset task)
PER_KIND=$(( $(feats_of "$COLL" pooled | wc -w) * $(layers_of "$COLL" | wc -w) ))
KIND=${KINDS[$((SLURM_ARRAY_TASK_ID / PER_KIND))]}
cell_of "$COLL" pooled "$((SLURM_ARRAY_TASK_ID % PER_KIND))"
echo "collection=$COLL holdout=$KIND features=$FEATURES layer=$LAYER"

python -m classifiers.scripts.run_leave_out \
    --data $COLL --holdout $KIND --raw last \
    --features $FEATURES --layers $LAYER --n-jobs 5
