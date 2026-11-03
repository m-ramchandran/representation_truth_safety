#!/bin/bash
#SBATCH --job-name=clustering_merged
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_L%a_%A.out

# One array task per layer, so the four layers run on four GPUs in parallel.
# All tasks write into the same results/merged_last/ tree (one L<layer>/ each).
# Each task requests five CPU cores and one GPU.

module load anaconda3
conda activate representation

LAYERS=(9 17 22 29)
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.run_clustering \
    clustering/configs/config_merged_last_L${LAYERS[$SLURM_ARRAY_TASK_ID]}.yaml
