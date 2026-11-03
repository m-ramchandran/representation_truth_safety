#!/bin/bash
#SBATCH --job-name=clustering_merged
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.run_clustering clustering/configs/config_merged_last.yaml
