#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=2:00:00
#SBATCH --job-name=qwen_analyze_L18
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out

# Read the finished Qwen sweep under two answer vocabularies: True/False
# (the primary read) and safe/unsafe (the check against the vocabulary
# artifact that made Gemma's xstest look like chance). CPU only.
PAIRS=(true safe)
PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/analyze.py \
    --run-dir steering/out/runs_qwen_layer18 \
    --manifest steering/out/manifest_qwen_layer18.csv \
    --pair "$PAIR"
