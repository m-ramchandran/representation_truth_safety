# The classifier grid. Sourced by submit_all.sh to size the arrays and by
# each sbatch task to map its index onto a cell — the same file on both
# sides, so the bounds and the work cannot drift apart. Only COLL crosses
# the sbatch boundary.

# The collection's Scope layers, from the registry.
layers_of() {
    python -c "from classifiers.data import COLLECTIONS
print(' '.join(str(l) for l in COLLECTIONS['$1'].layers))"
}

# Feature sets, by model and by experiment.
#
# Qwen has no sae/transcoder/pca_dict — it was extracted at the last
# prompt token only, with no Gemma Scope decompositions. That is the
# data, not a choice.
#
# The pooled runs skip sparse sets because their densified forms have high
# memory use. Per-dataset runs retain them for the feature-set comparison.
#   $1 collection, $2 per_dataset|pooled
feats_of() {
    local pd
    case "$1" in
        gemma_*) pd="raw pca binned_activations sae transcoder pca_dict" ;;
        qwen_*)  pd="raw pca binned_activations" ;;
        *) echo "feats_of: unknown collection $1" >&2; return 2 ;;
    esac
    if [ "$2" = per_dataset ]; then echo "$pd"; else
        echo "raw pca binned_activations"; fi
}

# Array index -> (feature set, layer). Sets FEATURES and LAYER.
#   $1 collection, $2 per_dataset|pooled, $3 array index
cell_of() {
    local feats layers nl
    feats=($(feats_of "$1" "$2"))
    layers=($(layers_of "$1"))
    nl=${#layers[@]}
    FEATURES=${feats[$(($3 / nl))]}
    LAYER=${layers[$(($3 % nl))]}
    if [ -z "${FEATURES:-}" ] || [ -z "${LAYER:-}" ]; then
        echo "cell_of: index $3 is outside the $1 $2 grid" >&2
        return 2
    fi
}
