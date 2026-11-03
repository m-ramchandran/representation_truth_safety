#!/bin/bash
#SBATCH --job-name=clusters_on_pcs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%A.out

# Renders the clusters_on_pcs figure group from the finished merged_last run
# (15 panels-per-k grids + 15 featured atlases per layer x 4 layers; ~25 min
# measured). Reads assignments.npz + model.joblib + activations; nothing
# re-fits. MFA models are torch tensors, hence the GPU node. The other
# figure groups run locally from the light artifacts.

module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.make_figures clusters_on_pcs
