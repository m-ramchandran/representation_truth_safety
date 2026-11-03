#!/bin/bash

# Convenience wrapper around `sbatch extract_activations.sh`.
#
# Env: DATASET, VARIANT=A|B. Or pass the dataset as $1.

set -euo pipefail

QWEN_WRAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${QWEN_WRAP_ROOT}"

VALID_DATASETS=(
    geometry_truth harmful_harmless xstest social_iqa social_iqa_binary wildjailbreak
    pubmedqa_artificial pubmedqa_labeled
    claude_anatomy claude_blood_test_ranges claude_vds claude_vds_300
    rm_bench
)

_usage() {
  cat <<EOF
Usage:
  bash submit_extract.sh <dataset> [variant]
  DATASET=<dataset> VARIANT=<A|B> bash submit_extract.sh
  bash submit_extract.sh all

Datasets: ${VALID_DATASETS[*]}
Variants: A B (default A)
EOF
}

DATASET="${1:-${DATASET:-}}"
VARIANT="${2:-${VARIANT:-A}}"
if [[ -z "${DATASET}" ]]; then
  _usage >&2
  exit 1
fi

if [[ "${DATASET}" == "all" ]]; then
  exec bash "${QWEN_WRAP_ROOT}/submit_all_extract.sh"
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
if [[ "${VARIANT}" != "A" && "${VARIANT}" != "B" ]]; then
  echo "Unknown variant: ${VARIANT}" >&2
  _usage >&2
  exit 1
fi

mkdir -p "${QWEN_WRAP_ROOT}/results"
echo "Submitting Qwen wrapped extract: ${DATASET} variant=${VARIANT}"
sbatch \
  --job-name="extract_qwen_wrapped_${DATASET}_${VARIANT}" \
  --output="${QWEN_WRAP_ROOT}/results/extract-${DATASET}-${VARIANT}-%j.out" \
  --error="${QWEN_WRAP_ROOT}/results/extract-${DATASET}-${VARIANT}-%j.err" \
  --export=ALL,DATASET="${DATASET}",VARIANT="${VARIANT}" \
  "${QWEN_WRAP_ROOT}/extract_activations.sh"
