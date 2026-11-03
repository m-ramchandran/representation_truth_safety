#!/bin/bash
#SBATCH --job-name=agreement_figures
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A.out

# Renders the five agreement/composition figure groups from the finished
# merged_last run: ARI across seeds/bootstraps per setting, ARI between
# methods at matched k, cluster composition bars, cross-layer ARI heatmaps,
# and the stability-best t-SNE/UMAP maps. Reads assignments.npz, samples.npz
# and the saved embedding files; nothing re-fits and no activations load.

module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -m clustering.scripts.make_figures ari_stability method_agreement composition \
    cross_layer_ari embedding_maps
