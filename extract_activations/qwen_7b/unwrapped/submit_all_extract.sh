#!/bin/bash

# Submit extract_activations.sh for every unwrapped dataset.

set -euo pipefail

QWEN_UNWRAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${QWEN_UNWRAP_ROOT}"
mkdir -p "${QWEN_UNWRAP_ROOT}/results"

DATASETS=(
    geometry_truth
    harmful_harmless
    xstest
    social_iqa
    social_iqa_binary
    wildjailbreak
    pubmedqa_artificial
    pubmedqa_labeled
    claude_anatomy
    claude_blood_test_ranges
    claude_vds
    claude_vds_300
)

for dataset in "${DATASETS[@]}"; do
  echo "Submitting ${dataset} ..."
  sbatch \
    --job-name="extract_qwen_${dataset}" \
    --output="${QWEN_UNWRAP_ROOT}/results/extract-${dataset}-%j.out" \
    --error="${QWEN_UNWRAP_ROOT}/results/extract-${dataset}-%j.err" \
    --export=ALL,DATASET="${dataset}" \
    "${QWEN_UNWRAP_ROOT}/extract_activations.sh"
done

echo "Submitted ${#DATASETS[@]} jobs."
