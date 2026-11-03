#!/bin/bash
#SBATCH --job-name=embeddings_umap
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x_%a_%A.out

# Seeded UMAP is single-threaded (umap-learn forces it for reproducibility),
# so at N=163k one task per layer x n_neighbors value; each runs its 5 seeds
# serially. The GPU itself is unused. Curve JSON is rebuilt from the
# per-value NESS files.

module load anaconda3
conda activate representation

LAYERS=(9 17 22 29)
NNS=(15 30 60 120)
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.run_embeddings \
    clustering/configs/config_merged_last_L${LAYERS[$((SLURM_ARRAY_TASK_ID / 4))]}.yaml \
    umap ${NNS[$((SLURM_ARRAY_TASK_ID % 4))]}
