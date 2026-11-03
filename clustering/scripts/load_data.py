"""Input contract for a clustering run.

One layer of one dataset = a single `.npz`. The activation matrix is either
dense under `X` / `activations`, or sparse COO (`row`/`col`/`val`/`shape`, the
sae / transcoder files) which is densified on load:
    X / activations  (N, D) float   activations                 [required]
    labels           (N,)           evaluation labels           [optional]
    dataset_id       (N,)           source dataset per sample   [optional, multi]
    topic / types    (N,)           topic per sample            [optional]
    prompts          (N,)           prompt text                 [optional]

Every loaded sample also gets a stable `sample_id`, "<source>:<row>" (e.g.
"cities:42") — the source file's name and the row number within it. Row order
is the same across layer files, so a sample keeps its ID at every layer.

labels / dataset_id / topic / prompts are used only for evaluation and for
inspecting what a cluster contains — never for fitting. `load` validates shapes
and returns a plain dict.

`load_hf` reads the same files from the Hugging Face dataset repository named
by `ACTIVATIONS_REPO` instead of a local path; `load_merged` concatenates
several datasets and adds `dataset_id`. Downloads go through
`hf_hub_download`, which keeps a local cached copy.

Layers are always Scope layers (the convention in README.md and the configs);
the files are named by HF hidden_states index, Scope layer s -> file layer{s+1}.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

HF_REPO = os.environ.get("ACTIVATIONS_REPO", "anqiwang02/Activations")

_GEOMETRY_TRUTH_PARTS = [
    "cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans",
    "larger_than", "smaller_than", "companies_true_false",
    "common_claim_true_false", "counterfact_true_false",
]

# dataset name -> the (hf_dir, file_stem) parts it is built from. geometry_truth
# is one dataset made of 9 files (the sub-dataset name goes into `topic`);
# pubmedqa's two subsets are separate datasets.
DATASETS = {
    "geometry_truth": [(f"geometry_truth/{p}", f"got_{p}") for p in _GEOMETRY_TRUTH_PARTS],
    "harmful_harmless": [("harmful_harmless", "harmful_harmless")],
    "pubmedqa_artificial": [("pubmedqa/pqa_artificial", "pubmedqa_artificial")],
    "pubmedqa_labeled": [("pubmedqa/pqa_labeled", "pubmedqa_labeled")],
    "social_iqa": [("social_iqa", "social_iqa")],
    "wildjailbreak": [("wildjailbreak", "wildjailbreak")],
    "xstest": [("xstest", "xstest")],
}

VARIANTS = ("last", "mean", "sae", "transcoder")


def load(path, source: str = None) -> dict:
    """Load one .npz; `source` names the sample_id prefix (default: file stem)."""
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    if "row" in d.files and "col" in d.files and "val" in d.files:
        X = np.zeros(tuple(d["shape"]), dtype=np.float32)  # sparse COO -> dense
        X[d["row"], d["col"]] = d["val"]
    else:
        x_key = "X" if "X" in d.files else "activations"
        if x_key not in d.files:
            raise KeyError(f"{path}: missing activations ('X' or 'activations')")
        X = np.asarray(d[x_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{path}: activations must be 2-D (N, D), got {X.shape}")

    out = {"X": X}
    for key in ("labels", "dataset_id", "topic", "types", "prompts"):
        if key in d.files:
            arr = np.asarray(d[key])
            if len(arr) != len(X):
                raise ValueError(f"{path}: '{key}' length {len(arr)} != N {len(X)}")
            out["topic" if key == "types" else key] = arr
    source = source or path.stem
    out["sample_id"] = np.array([f"{source}:{i}" for i in range(len(X))])
    return out


def load_hf(dataset: str, variant: str, layer: int) -> dict:
    """One dataset at one Scope layer from the HF repo (multi-part -> concatenated)."""
    from huggingface_hub import hf_hub_download

    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {list(DATASETS)}")
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {list(VARIANTS)}")

    parts = []
    for hf_dir, stem in DATASETS[dataset]:
        path = hf_hub_download(
            HF_REPO, f"{hf_dir}/{stem}_{variant}_layer{layer + 1}.npz", repo_type="dataset"
        )
        part = load(path, source=hf_dir.rsplit("/", 1)[-1])  # e.g. cities, pqa_labeled
        if len(DATASETS[dataset]) > 1:  # keep the sub-dataset name (e.g. cities)
            part["topic"] = np.full(len(part["X"]), hf_dir.rsplit("/", 1)[-1])
        parts.append(part)
    out = parts[0] if len(parts) == 1 else _concat(parts)
    if dataset == "harmful_harmless" and "labels" in out:
        # The HF file stores 1 = harmless; xstest and wildjailbreak store
        # 1 = harmful/unsafe. Flip so the positive class means the same
        # thing across the safety datasets (verified on prompt text).
        out["labels"] = 1 - out["labels"].astype(int)
    return out


def load_merged(datasets: list, variant: str, layer: int) -> dict:
    """Several datasets concatenated, with `dataset_id` marking the source."""
    parts = []
    for name in datasets:
        part = load_hf(name, variant, layer)
        part["dataset_id"] = np.full(len(part["X"]), name)
        part.setdefault("topic", np.full(len(part["X"]), name))
        parts.append(part)
    return _concat(parts)


def _concat(parts: list) -> dict:
    keys = set.intersection(*(set(p) for p in parts))
    dropped = set.union(*(set(p) for p in parts)) - keys
    if dropped:
        print(f"  (dropping {sorted(dropped)}: not present in every merged part)")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}
