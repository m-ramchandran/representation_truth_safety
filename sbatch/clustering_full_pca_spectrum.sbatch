#!/bin/bash
#SBATCH --job-name=full_pca_spectrum
#SBATCH --cpus-per-task=5
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%A.out

# Full 2560-eigenvalue PCA spectrum of the merged last-token data, all four
# layers in one task (each is one covariance + eigh, a couple of minutes).
# Writes pca__full_spectrum.npz into each results/merged_last/L<layer>/.

module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.full_pca_spectrum
