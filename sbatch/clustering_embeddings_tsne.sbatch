#!/bin/bash
#SBATCH --job-name=embeddings_tsne
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%a_%A.out

# One task per layer x perplexity (5 seeds each); openTSNE uses the 5 cores.
# CPU-only. Requires the finished run_clustering results (reads the saved
# pca__pc=15.npz). The stability-curve json is rebuilt from per-value files.

module load anaconda3
conda activate representation

LAYERS=(9 17 22 29)
PERPS=(10 30 50 100)
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.run_embeddings \
    clustering/configs/config_merged_last_L${LAYERS[$((SLURM_ARRAY_TASK_ID / 4))]}.yaml \
    tsne ${PERPS[$((SLURM_ARRAY_TASK_ID % 4))]}
