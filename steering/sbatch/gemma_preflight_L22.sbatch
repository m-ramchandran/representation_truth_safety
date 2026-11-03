#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=0:20:00
#SBATCH --job-name=gemma_preflight_L22
#SBATCH --output=logs/%x_%A.out

# Does the Gemma layer-22 saved-fit set cover all six datasets? Opens every
# file the build reads and reports what is missing. Two minutes here instead
# of a failure an hour into a GPU job. Login nodes kill even light python,
# so this runs as a job like everything else.

if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null \
        || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
module load anaconda3
conda activate representation

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python -u steering/preflight.py --collection gemma_tf_a --layer 22 --targets common_claim,wildjailbreak_vanilla,xstest,pubmedqa_labeled
