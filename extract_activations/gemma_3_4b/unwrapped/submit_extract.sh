#!/bin/bash

# Convenience wrapper around `sbatch extract_activations.sh`.
#
# Env: DATASET. Or pass the dataset as $1; 'all' submits every dataset.

set -euo pipefail

UNWRAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${UNWRAP_ROOT}"

VALID_DATASETS=(
    geometry_truth harmful_harmless xstest social_iqa social_iqa_binary wildjailbreak
    pubmedqa_artificial pubmedqa_labeled
    claude_anatomy claude_blood_test_ranges claude_vds claude_vds_300
    rm_bench
)

_usage() {
  cat <<EOF
Usage:
  bash submit_extract.sh <dataset>
  DATASET=<dataset> bash submit_extract.sh
  bash submit_extract.sh all

Datasets: ${VALID_DATASETS[*]}
EOF
}

DATASET="${1:-${DATASET:-}}"
if [[ -z "${DATASET}" ]]; then
  _usage >&2
  exit 1
fi

if [[ "${DATASET}" == "all" ]]; then
  exec bash "${UNWRAP_ROOT}/submit_all_extract.sh"
fi

valid=0
for d in "${VALID_DATASETS[@]}"; do
  [[ "${d}" == "${DATASET}" ]] && valid=1 && break
done
if [[ "${valid}" -ne 1 ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  _usage >&2
  exit 1
fi

mkdir -p "${UNWRAP_ROOT}/results"
echo "Submitting unwrapped extract: ${DATASET}"
sbatch \
  --job-name="extract_${DATASET}" \
  --output="${UNWRAP_ROOT}/results/extract-${DATASET}-%j.out" \
  --error="${UNWRAP_ROOT}/results/extract-${DATASET}-%j.err" \
  --export=ALL,DATASET="${DATASET}" \
  "${UNWRAP_ROOT}/extract_activations.sh"
