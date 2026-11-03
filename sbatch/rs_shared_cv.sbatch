#!/bin/bash
#SBATCH --job-name=rs_shared_cv
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A.out

# Cross-validated AUC of the pooled random-slopes SHARED direction alone
# on the expanded pool at the 88%-depth layer: refits each
# stored outer fold at its stored hyperparameters and scores the held-out
# rows shared-only (paper_final_figures/scripts/rs_shared_cv.py).
# Five fits per collection, so a fraction of the original merged run.
# Outputs: classifiers/results/_rs_shared_cv/<coll>_expanded_pool_layer<L>.csv

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
set -eo pipefail
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for spec in gemma_tf_a:29 qwen_tf_a:24 gemma_original:29 qwen_original:24; do
    echo "=== ${spec%%:*} layer ${spec##*:}  $(date)"
    python -u paper_final_figures/scripts/rs_shared_cv.py \
        "${spec%%:*}" "${spec##*:}"
done

echo "== outputs =="
ls -l classifiers/results/_rs_shared_cv/
echo done
