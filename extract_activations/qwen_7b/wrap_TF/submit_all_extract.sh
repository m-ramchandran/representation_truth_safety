#!/bin/bash

# Submit extract_activations.sh for every wrapped dataset.

set -euo pipefail

QWEN_WRAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${QWEN_WRAP_ROOT}"
mkdir -p "${QWEN_WRAP_ROOT}/results"

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

VARIANTS=(A B)

for variant in "${VARIANTS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Submitting ${dataset} variant=${variant} ..."
    sbatch \
      --job-name="extract_qwen_wrapped_${dataset}_${variant}" \
      --output="${QWEN_WRAP_ROOT}/results/extract-${dataset}-${variant}-%j.out" \
      --error="${QWEN_WRAP_ROOT}/results/extract-${dataset}-${variant}-%j.err" \
      --export=ALL,DATASET="${dataset}",VARIANT="${variant}" \
      "${QWEN_WRAP_ROOT}/extract_activations.sh"
  done
done

echo "Submitted $(( ${#DATASETS[@]} * ${#VARIANTS[@]} )) jobs."
