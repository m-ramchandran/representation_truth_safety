#!/bin/bash
# Submit the final classifier runs: four collections x three experiments,
# gated on the smoke test, each collection's analyses chained after its
# own arrays.
#
#   bash sbatch/submit_all.sh                  # everything
#   bash sbatch/submit_all.sh --dry-run        # print the sbatch calls
#   bash sbatch/submit_all.sh gemma_tf_a       # one or more collections
#   bash sbatch/submit_all.sh --no-smoke ...   # skip the smoke gate
#
# This file decides WHAT is submitted and in what order;
# sbatch/classifiers_grid.sh decides which cells exist.
#
# Arrays are throttled to %20 so four collections can be in flight
# together. Walltime is 24h; the metrics csvs are the resume record, so
# an overrun is a resubmit of the same command, not a loss.
set -euo pipefail

cd "$(dirname "$0")/.."

ALL_COLLECTIONS=(gemma_original gemma_tf_a qwen_original qwen_tf_a)
DRY=0
SMOKE=1
COLLECTIONS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        --no-smoke) SMOKE=0 ;;
        -*) echo "unknown flag $arg" >&2; exit 2 ;;
        *) COLLECTIONS+=("$arg") ;;
    esac
done
[ ${#COLLECTIONS[@]} -eq 0 ] && COLLECTIONS=("${ALL_COLLECTIONS[@]}")

# Commentary goes to stderr so the captured stdout is only ever a job id
# (or the DRYRUN sentinel) — otherwise --dry-run feeds its own echoes
# back into the next command's --dependency.
run() {
    printf '    $ %s\n' "$*" >&2
    if [ "$DRY" = 1 ]; then echo "DRYRUN"; return; fi
    "$@"
}

# --- the smoke-test gate ---------------------------------------------
# Every array depends on it with afterok, so a broken scripts/ change
# cannot burn GPU hours across 200+ tasks before anyone notices.
SMOKE_ID=""
if [ "$SMOKE" = 1 ]; then
    echo "smoke test:"
    SMOKE_ID=$(run sbatch --parsable -J clf_smoke sbatch/classifiers_smoke.sbatch)
    echo "  -> job $SMOKE_ID  (every array below waits on it with afterok)"
fi
dep()  { [ -n "$SMOKE_ID" ] && [ "$SMOKE_ID" != "DRYRUN" ] \
             && echo "--dependency=afterok:$SMOKE_ID" || echo ""; }

# layers_of / feats_of / cell_of — shared with the sbatch scripts, so the
# array bounds computed here and the cell each task runs cannot drift.
source sbatch/classifiers_grid.sh

short() { echo "$1" | sed -e 's/gemma_/g/' -e 's/qwen_/q/' \
                          -e 's/original/orig/' -e 's/tf_a/tfa/'; }

for COLL in "${COLLECTIONS[@]}"; do
    LAYERS=$(layers_of "$COLL")
    NL=$(echo "$LAYERS" | wc -w)
    PD_FEATS=$(feats_of "$COLL" per_dataset); N_PD=$(echo "$PD_FEATS" | wc -w)
    PL_FEATS=$(feats_of "$COLL" pooled);      N_PL=$(echo "$PL_FEATS" | wc -w)
    S=$(short "$COLL")
    echo
    echo "=== $COLL  layers[$LAYERS]"
    echo "    per_dataset features: $PD_FEATS"
    echo "    pooled      features: $PL_FEATS"

    IDS=()
    IDS+=("$(run sbatch --parsable $(dep) -J "pd_$S" \
        --array=0-$((N_PD * NL - 1))%20 --export=ALL,COLL="$COLL" \
        sbatch/classifiers_per_dataset.sbatch)")
    # leave_out carries both holdout kinds, hence the doubled array
    IDS+=("$(run sbatch --parsable $(dep) -J "lo_$S" \
        --array=0-$((2 * N_PL * NL - 1))%20 --export=ALL,COLL="$COLL" \
        sbatch/classifiers_leave_out.sbatch)")
    IDS+=("$(run sbatch --parsable $(dep) -J "mg_$S" \
        --array=0-$((N_PL * NL - 1))%20 --export=ALL,COLL="$COLL" \
        sbatch/classifiers_merged.sbatch)")
    echo "    arrays: ${IDS[*]}  ($((N_PD*NL)) + $((2*N_PL*NL)) + $((N_PL*NL)) tasks)"

    # afterany, not afterok: a timed-out array still leaves real results,
    # and the analyses are re-runnable. They DO need every array to have
    # stopped, though — see the placeholder-row warning in the analysis
    # script's header.
    if [ "$DRY" = 1 ]; then
        AFTER=""
    else
        AFTER="--dependency=afterany:$(IFS=:; echo "${IDS[*]}")"
    fi
    ANALYSIS=$(run sbatch --parsable $AFTER -J "an_$S" \
        --export=ALL,COLL="$COLL" sbatch/classifiers_analysis.sbatch)
    echo "    analysis: $ANALYSIS"
done

echo
echo "submitted. watch with: squeue -u \$USER"
echo "results land in classifiers/results/<collection>/"
