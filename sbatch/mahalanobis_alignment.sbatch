#!/bin/bash
#SBATCH --job-name=mahalanobis_alignment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%A.out

# Figure 4 panel b under the regularized Mahalanobis cosine instead of the
# plain one: each part of the expanded-pool fit against the direction fitted
# on that dataset alone, whitened by the cached Ledoit-Wolf covariances.
# Reads only; no fit is redone and no covariance is re-estimated (the cached
# cov_pool's 13-dataset roster is printed in the log). The plain cosine is
# written alongside as an unweighted diagnostic.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
set -eo pipefail
mkdir -p logs

for spec in gemma_original:29 gemma_tf_a:29 qwen_original:24 qwen_tf_a:24; do
    echo "=============================== ${spec%%:*}  layer ${spec##*:}"
    python paper_final_figures/scripts/mahalanobis_alignment.py \
        "${spec%%:*}" "${spec##*:}"
done

echo "== outputs =="
ls -l classifiers/results/_mahalanobis_alignment/
echo done
