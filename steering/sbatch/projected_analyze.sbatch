#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=0:20:00
#SBATCH --job-name=proj_analyze
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out

# Read the finished projected-coefficient runs under both answer
# vocabularies. Submit with --dependency=afterok:<gemma>:<qwen>.
CELLS=("gemmaproj 29 true" "gemmaproj 29 safe" "qwenproj 24 true" "qwenproj 24 safe")
read -r TAG LAYER PAIR <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/analyze.py \
    --run-dir steering/out/runs_${TAG}_layer${LAYER} \
    --manifest steering/out/manifest_${TAG}_layer${LAYER}.csv \
    --pair "$PAIR"
