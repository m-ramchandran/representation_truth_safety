#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24000M
#SBATCH --time=0:40:00
#SBATCH --job-name=proj_dirs
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out

# Build the projected-coefficient direction files:
# each saved regression vector kept only inside the top 2 / 10 / 50
# principal components of the rows it was trained on. CPU only, nothing refit.
CELLS=("gemma_tf_a 29 gemmaproj" "qwen_tf_a 24 qwenproj")
read -r COLL LAYER TAG <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/projected_directions.py --collection "$COLL" --layer "$LAYER" --tag "$TAG"
