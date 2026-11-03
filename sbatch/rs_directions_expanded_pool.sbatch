#!/bin/bash
#SBATCH --job-name=rs_dir_expanded_pool
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%A.out

# Coefficient geometry for the expanded pool: the cosine
# between each part of the pooled random-slopes fit and the direction
# fitted on that dataset alone. figure4_pooled_training panel b currently
# reads the 10-dataset base pool, so VDS and WJB-adversarial have no
# marks; in the expanded-pool fit both ARE groups and do have deviations.
#
# The expanded-pool runs only ever wrote merged/, and rs_directions wants
# merged/ and per_dataset/ under one root. So this stitches a scratch
# root per collection whose merged/ points at the expanded-pool fit and
# whose per_dataset/ points at the base collection's single fits -- same
# activations, same layer, so the single-fit side is unchanged. Nothing
# under results/<collection>/ is written to.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
set -eo pipefail
mkdir -p logs

R=$PWD/classifiers/results
OUT=$R/_rs_dir_expanded_pool
mkdir -p "$OUT"

for spec in gemma_original:29 gemma_tf_a:29 qwen_original:24 qwen_tf_a:24; do
    coll=${spec%%:*}
    layer=${spec##*:}
    root=$OUT/$coll
    mkdir -p "$root"
    ln -sfn "$R/${coll}_expanded_pool/merged" "$root/merged"
    ln -sfn "$R/$coll/per_dataset"         "$root/per_dataset"
    echo "=== $coll  layer $layer"
    python -m classifiers.analysis_scripts.rs_directions \
        --tag raw_last --layer "$layer" --methods l2reg \
        --data "$coll" --root "$root"
done

echo "== outputs =="
ls -l "$OUT"/*/coefficients/tables/rs_vs_single_*.csv
echo done
