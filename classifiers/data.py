"""Activation loading for the classifier experiments.

This module is import-safe on any machine (numpy/scipy only — no cuML,
no torch), which is why it lives at the package root: `scripts/` (GPU
nodes) and `analysis/` (the Mac) both import it, so there is exactly one
copy of the label conventions and the dataset roster. The clustering
package keeps its own older copy of the loader with the OLD label
convention (1 = harmful on the safety sets) — do not mix the two.

The activations come in named COLLECTIONS — independent snapshots of the
HF repo differing in model, prompt wording, roster, and label storage. The
four current ones are the 2026-08 chat-templated re-extraction, one per
(model, perturbation): `gemma_original` / `gemma_tf_a` and
`qwen_original` / `qwen_tf_a`, where "original" is the bare prompt and
"tf_a" is the same content rephrased as a True/False question (wrapper
variant A). A collection declares where its files live, which experiment
units exist, their task groups, which units are held out of pooled
training, and any label flips; the loading machinery below is shared.
Every loader takes `data=<collection name>` (default DEFAULT_COLLECTION),
the runners expose it as --data, and results are kept apart under
results/<collection>/. Adding a new collection = one COLLECTIONS entry.

The pre-2026-08 collections (`original`, `tf_wrapped_a`, `data_unwrap`)
are gone: their HF trees moved under `archive_no_chat_template/` when the
prompts were re-extracted with the chat template applied, so their paths
would 404 and their results are not comparable with anything here.

One layer of one dataset = a single `.npz`. The activation matrix is either
dense under `X` / `activations`, or sparse COO (`row`/`col`/`val`/`shape`, the
sae / transcoder files) which loads as a scipy CSR matrix (kept sparse —
the classifier code densifies only where a method needs it):
    X / activations  (N, D) float   activations                 [required]
    labels           (N,)           evaluation labels           [optional]
    dataset_id       (N,)           source dataset per sample   [optional, multi]
    topic / types /
    data_types       (N,)           topic or subtype per sample [optional]
    prompts          (N,)           prompt text                 [optional]

Every loaded sample also gets a stable `sample_id`, "<source>:<row>" (e.g.
"cities:42") — the source file's name and the row number within it. Row order
is the same across layer files, so a sample keeps its ID at every layer.

`load_hf` reads files from the Hugging Face dataset named by
`ACTIVATIONS_REPO`;
downloads go through `hf_hub_download`, which keeps a local copy under
~/.cache/huggingface — each file is fetched from the network once ever.
Layers are always zero-indexed Gemma Scope checkpoint layers;
the files are named by HF hidden_states index, Scope layer s -> file layer{s+1}.

On top of the file loaders sit the experiment-unit loaders: `DATASETS`
names the binary units (geometry_truth split into got / common_claim /
counterfact following the original paper), `binary_labels` casts each
unit's labels to {0,1} ints, and `load_datasets` returns unit ->
{y, sample_id, topic, blocks} with one matrix per requested cache role.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from huggingface_hub.errors import EntryNotFoundError

HF_REPO = os.environ.get("ACTIVATIONS_REPO", "anqiwang02/Activations")

VARIANTS = ("last", "mean", "sae", "transcoder")


@dataclass(frozen=True)
class Collection:
    """One snapshot of the activation data: where its HF files live and
    what the experiment looks like on it. All label conventions here are
    1 = true / yes / safe / harmless.

    Three roles a unit can have, in decreasing order of participation:

      ordinary       trained on and evaluated everywhere.
      pool_excluded  trained on ONLY as a single-dataset learner (its own
                     nested CV in run_per_dataset); never enters a pooled
                     training corpus, and never produces a transfer row.
                     Still evaluated everywhere — as a test column in
                     run_per_dataset, by run_merged's whole-corpus fit, and
                     against every run_leave_out pool. These are the real
                     held-out sets.
      eval_only      never trained on at all, only ever scored.
    """
    hf_files: dict          # HF source -> [(hf_dir, file_stem) parts]
    datasets: tuple         # the experiment units, in display order
    tasks: dict             # task -> member units (see task_of)
    layers: tuple           # default Scope layers for runs (--layers overrides)
    model: str = "google/gemma-3-4b-it"   # provenance, for tables and figures
    eval_only: tuple = ()   # units only ever evaluated on, never trained on
    pool_excluded: tuple = ()   # units kept out of every pooled training corpus
    flip_labels: tuple = () # sources stored with the opposite convention
    label_map: dict = field(default_factory=dict)   # source -> {raw value: 0/1}

    @property
    def train_datasets(self) -> tuple:
        """Units any runner may fit a model on. pool_excluded units are
        here: run_per_dataset trains them for their own diagonal."""
        return tuple(d for d in self.datasets if d not in self.eval_only)

    @property
    def pool_datasets(self) -> tuple:
        """Units that may enter a POOLED training corpus (run_merged's
        pool, run_leave_out's remainder). The held-out sets are absent."""
        return tuple(d for d in self.train_datasets
                     if d not in self.pool_excluded)

    def task_of(self, dataset: str) -> str:
        """The task a unit belongs to; a unit in no task is its own."""
        for task, members in self.tasks.items():
            if dataset in members:
                return task
        return dataset


_GEOMETRY_TRUTH_PARTS = [
    "cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans",
    "larger_than", "smaller_than", "companies_true_false",
    "common_claim_true_false", "counterfact_true_false",
]

# The 10 HF sources of the chat-templated re-extraction (2026-08), shared by
# all four collections: the same prompts through two models and two
# perturbations. `claude_diagnosis_severity` (n=40) is gone — the upstream
# extraction dropped it, which is the cut we wanted anyway. Two sources were
# renamed at the same time: social_iqa -> social_iqa_binary (one wrong choice
# dropped, so two remain) and claude_vds -> claude_vds_300 (n grew 126 -> 300).
# The unit names stay short; only the HF directory carries the long name.
_SOURCES = (
    "geometry_truth", "harmful_harmless", "pubmedqa_artificial",
    "pubmedqa_labeled", "social_iqa", "wildjailbreak", "xstest",
    "claude_anatomy", "claude_blood_test_ranges", "claude_vds",
)

# unit name -> HF directory name, where the two differ.
_SOURCE_DIRS = {"social_iqa": "social_iqa_binary",
                "claude_vds": "claude_vds_300"}

# The 13 experiment units every collection has, in display order:
# geometry_truth splits three ways and wildjailbreak two (see _SPLIT_SOURCES).
_UNITS = ("got", "common_claim", "counterfact", "harmful_harmless",
          "pubmedqa_artificial", "pubmedqa_labeled", "social_iqa",
          "wildjailbreak_vanilla", "wildjailbreak_adversarial", "xstest",
          "claude_anatomy", "claude_blood_test_ranges", "claude_vds")

_TASKS = {
    "tf": ("got", "common_claim", "counterfact"),
    "medical": ("pubmedqa_artificial", "pubmedqa_labeled", "claude_anatomy",
                "claude_blood_test_ranges"),
    "jailbreak": ("harmful_harmless", "wildjailbreak_vanilla",
                  "wildjailbreak_adversarial", "xstest"),
}

# The held-out sets. Each is trained on only
# as a single-dataset learner, to show whether the property is separable at
# all within its own data; none of them ever enters a pooled training corpus,
# and all three are evaluated in every setting. wildjailbreak_adversarial is
# the clean safety target; claude_vds and claude_blood_test_ranges are the
# two sets nothing has ever predicted well.
_POOL_EXCLUDED = ("claude_blood_test_ranges", "claude_vds",
                  "wildjailbreak_adversarial")

# Gemma-3-4B has 34 layers, Qwen2.5-7B has 28. Qwen's layers are Gemma's at
# matched relative depth (file index / n_layers), so the two models are
# compared at the same place in the stack rather than the same integer:
#   Gemma scope 17/22/29/33 -> files 18/23/30/34 -> depth .529/.676/.882/1.0
#   Qwen  scope 14/18/24/27 -> files 15/19/25/28 -> depth .536/.679/.893/1.0
# Largest mismatch 1.1 percentage points.
_GEMMA_LAYERS = (17, 22, 29, 33)
_QWEN_LAYERS = (14, 18, 24, 27)


def _hf_files(root: str, suffix: str = "") -> dict:
    """source -> [(hf_dir, file_stem)] for one model/perturbation tree.
    Files are <dir>/<stem>_<variant>_layer{L+1}.npz."""
    return {src: [(f"{root}/{_SOURCE_DIRS.get(src, src)}",
                   f"{_SOURCE_DIRS.get(src, src)}{suffix}")]
            for src in _SOURCES}


def _wrapped_social_iqa(model: str) -> dict:
    """social_iqa from the perturbation-A tree, for the bare collections
    to merge over their own entry (see gemma_original)."""
    return {"social_iqa": [(f"{model}/perturbations/A/social_iqa_binary",
                            "social_iqa_binary_tf_a")]}


COLLECTIONS = {
    # ---- the bare prompts, chat-templated ---------------------------
    #
    # Each source keeps its own stored label convention here (perturbation
    # A has one shared convention instead), so wildjailbreak and xstest,
    # which store True = harmful/unsafe, are flipped at load.
    #
    # social_iqa comes from the WRAPPED tree. There is no unwrapped file
    # that asks a truth question: the bare extraction labels each question
    # with the INDEX of the correct choice, a property of ordering. It was
    # first run that way, and nothing transferred to it (median AUC 0.500
    # over 360 raw_last cells, against 0.588-0.853 for every other unit),
    # so it was switched on 2026-08-07. Those results are kept under
    # results/*_answerindex_social_iqa/; reverting means dropping the
    # _wrapped_social_iqa merge and restoring
    # label_map={"social_iqa": {1: 1, 2: 0}}.
    #
    # The cost: one of the ten pooled datasets now carries the "Determine
    # whether the following is True or False." line the other nine do not,
    # so a pooled model can separate it on surface form alone.
    "gemma_original": Collection(
        hf_files=_hf_files("gemma_3_4B/original")
                 | _wrapped_social_iqa("gemma_3_4B"),
        datasets=_UNITS,
        tasks=_TASKS,
        layers=_GEMMA_LAYERS,
        model="google/gemma-3-4b-it",
        pool_excluded=_POOL_EXCLUDED,
        flip_labels=("wildjailbreak", "xstest"),
    ),
    # ---- perturbation A: every prompt rephrased as a True/False question --
    #
    # One shared convention in the files themselves ("True = factually
    # correct or safe to answer"), so no flips anywhere. The wrapping expands
    # each pubmedqa question and each social_iqa question into a true and a
    # false statement, so those three units are twice the size of their bare
    # counterparts (pubmedqa_artificial 60,500, pubmedqa_labeled 1,352,
    # social_iqa 70,728) and the two rows of a pair share a fold group.
    "gemma_tf_a": Collection(
        hf_files=_hf_files("gemma_3_4B/perturbations/A", "_tf_a"),
        datasets=_UNITS,
        tasks=_TASKS,
        layers=_GEMMA_LAYERS,
        model="google/gemma-3-4b-it",
        pool_excluded=_POOL_EXCLUDED,
    ),
    # ---- the same two trees for Qwen --------------------------------------
    #
    # Qwen was extracted at the last prompt token only: no `mean` pooling and
    # no Gemma Scope decompositions, so the sae / transcoder / pca_dict
    # feature sets do not exist here. That is a property of the data, not a
    # choice — the runners are given raw / pca / binned_activations only.
    "qwen_original": Collection(
        hf_files=_hf_files("qwen_7B/original")
                 | _wrapped_social_iqa("qwen_7B"),
        datasets=_UNITS,
        tasks=_TASKS,
        layers=_QWEN_LAYERS,
        model="Qwen/Qwen2.5-7B-Instruct",
        pool_excluded=_POOL_EXCLUDED,
        flip_labels=("wildjailbreak", "xstest"),
    ),
    "qwen_tf_a": Collection(
        hf_files=_hf_files("qwen_7B/perturbations/A", "_tf_a"),
        datasets=_UNITS,
        tasks=_TASKS,
        layers=_QWEN_LAYERS,
        model="Qwen/Qwen2.5-7B-Instruct",
        pool_excluded=_POOL_EXCLUDED,
    ),
}

DEFAULT_COLLECTION = "gemma_tf_a"

# The default collection's tables, importable by name — callers that
# predate the registry keep working unchanged.
HF_FILES = COLLECTIONS[DEFAULT_COLLECTION].hf_files
DATASETS = COLLECTIONS[DEFAULT_COLLECTION].datasets
TASKS = COLLECTIONS[DEFAULT_COLLECTION].tasks


def task_of(dataset: str, data: str = DEFAULT_COLLECTION) -> str:
    """The task a unit belongs to; a unit in no task is its own."""
    return COLLECTIONS[data].task_of(dataset)


# Array names that mean the same thing under different spellings. The
# unwrapped and wrapped extraction pipelines disagree on the plural (the HF
# README says so outright), and `types` / `data_types` are both the subtype
# column we call `topic`.
_CANONICAL_KEYS = {
    "types": "topic", "data_types": "topic",
    "source_row_ids": "source_row_id",
    "source_prompt_ids": "source_prompt_id",
}


def load(path, source: str = None) -> dict:
    """Load one .npz; `source` names the sample_id prefix (default: file stem)."""
    path = Path(path)
    d = np.load(path, allow_pickle=True)
    if "row" in d.files and "col" in d.files and "val" in d.files:
        X = sp.csr_matrix(  # sparse COO -> scipy CSR, stays sparse
            (d["val"].astype(np.float32), (d["row"], d["col"])),
            shape=tuple(d["shape"]))
    else:
        x_key = "X" if "X" in d.files else "activations"
        if x_key not in d.files:
            raise KeyError(f"{path}: missing activations ('X' or 'activations')")
        X = np.asarray(d[x_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"{path}: activations must be 2-D (N, D), got {X.shape}")

    n = X.shape[0]
    out = {"X": X}
    for key in ("labels", "dataset_id", "topic", "types", "data_types",
                "prompts", "source_row_ids", "source_row_id",
                "source_prompt_ids", "source_prompt_id"):
        if key in d.files:
            arr = np.asarray(d[key])
            if len(arr) != n:
                raise ValueError(f"{path}: '{key}' length {len(arr)} != N {n}")
            out[_CANONICAL_KEYS.get(key, key)] = arr
    source = source or path.stem
    out["sample_id"] = np.array([f"{source}:{i}" for i in range(n)])
    return out


def load_hf(dataset: str, variant: str, layer: int,
            data: str = DEFAULT_COLLECTION) -> dict:
    """One HF source at one Scope layer (multi-part -> concatenated)."""
    from huggingface_hub import hf_hub_download

    c = COLLECTIONS[data]
    if dataset not in c.hf_files:
        raise KeyError(f"unknown dataset {dataset!r}; known: {list(c.hf_files)}")
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {list(VARIANTS)}")

    parts = []
    for hf_dir, stem in c.hf_files[dataset]:
        path = hf_hub_download(
            HF_REPO, f"{hf_dir}/{stem}_{variant}_layer{layer + 1}.npz", repo_type="dataset"
        )
        part = load(path, source=hf_dir.rsplit("/", 1)[-1])  # e.g. cities, pqa_labeled
        if len(c.hf_files[dataset]) > 1:  # keep the sub-dataset name (e.g. cities)
            part["topic"] = np.full(part["X"].shape[0], hf_dir.rsplit("/", 1)[-1])
        parts.append(part)
    out = parts[0] if len(parts) == 1 else _concat(parts)
    if dataset in c.flip_labels and "labels" in out:
        out["labels"] = 1 - out["labels"].astype(int)
    return out


def load_prompts(source: str, layer: int, data: str = DEFAULT_COLLECTION) -> dict:
    """sample_id -> prompt text for one HF source, decompressing only
    the prompts array — the activation matrices (GBs across a
    collection) stay untouched. Sources without a prompts key return
    {}. Prompts are identical across variants and layers; the 'last'
    file is read."""
    from huggingface_hub import hf_hub_download

    c = COLLECTIONS[data]
    out = {}
    for hf_dir, stem in c.hf_files[source]:
        path = hf_hub_download(
            HF_REPO, f"{hf_dir}/{stem}_last_layer{layer + 1}.npz",
            repo_type="dataset")
        with np.load(path, allow_pickle=True) as d:
            if "prompts" not in d.files:
                continue
            src = hf_dir.rsplit("/", 1)[-1]     # sample_id prefix, as in load()
            out.update({f"{src}:{i}": p
                        for i, p in enumerate(d["prompts"])})
    return out


def load_merged(datasets: list, variant: str, layer: int,
                data: str = DEFAULT_COLLECTION) -> dict:
    """Several HF sources concatenated, with `dataset_id` marking the source."""
    parts = []
    for name in datasets:
        part = load_hf(name, variant, layer, data)
        part["dataset_id"] = np.full(part["X"].shape[0], name)
        part.setdefault("topic", np.full(part["X"].shape[0], name))
        parts.append(part)
    return _concat(parts)


def _concat(parts: list) -> dict:
    keys = set.intersection(*(set(p) for p in parts))
    dropped = set.union(*(set(p) for p in parts)) - keys
    if dropped:
        print(f"  (dropping {sorted(dropped)}: not present in every merged part)")
    return {k: (sp.vstack([p[k] for p in parts], format="csr")
                if sp.issparse(parts[0][k])
                else np.concatenate([p[k] for p in parts])) for k in keys}


def binary_labels(labels: np.ndarray, dataset: str,
                  label_map: dict | None = None) -> np.ndarray:
    """Labels as {0,1} ints. Bools cast directly; pubmedqa stores 'yes'/'no'
    strings (positive = 'yes'). Anything else is an error, not a guess —
    unless the collection declares an explicit `label_map` for this source,
    which is the escape hatch for a stored encoding that is binary but not
    {0,1}. The map must be total over the values present and land in {0,1};
    a partial or non-binary map is an error, so a silent miscoding is not
    possible.

    The one map in use: the bare tree's social_iqa stores the INDEX of the
    correct answer among the two remaining choices, {1, 2}. `{1: 1, 2: 0}`
    makes y=1 mean "the first presented choice is the correct one" — a
    one-to-one translation, since exactly one of the two is right."""
    labels = np.asarray(labels)
    if label_map:
        vals = set(np.unique(labels).tolist())
        missing = vals - set(label_map)
        if missing:
            raise ValueError(f"{dataset}: label_map has no entry for "
                             f"{sorted(missing)} (map: {label_map})")
        if not set(label_map.values()) <= {0, 1}:
            raise ValueError(f"{dataset}: label_map must send values to 0/1, "
                             f"got {sorted(set(label_map.values()))}")
        return np.array([label_map[v] for v in labels.tolist()], dtype=int)
    if labels.dtype.kind in "biu":
        vals = set(np.unique(labels).tolist())
        if not vals <= {0, 1, False, True}:
            raise ValueError(f"{dataset}: labels {sorted(vals)} are not binary")
        return labels.astype(int)
    if labels.dtype.kind in "US":
        vals = set(np.unique(labels).tolist())
        if vals != {"no", "yes"}:
            raise ValueError(f"{dataset}: string labels {sorted(vals)} != no/yes")
        return (labels == "yes").astype(int)
    raise ValueError(f"{dataset}: unsupported label dtype {labels.dtype}")


# --- fold groups: rows that must not be split across CV folds ---------------
#
# Several sources put more than one row on the same underlying item. Split
# those across folds and a near duplicate of a test row sits in the
# training set. --group-folds (on by default) keeps them together.
#
# Prefer the STORED id: the 2026-08 files carry `source_row_id` everywhere
# and `source_prompt_id` on wildjailbreak. Exact, and it avoids
# decompressing the `prompts` array (~1 GB on wildjailbreak).
# GROUP_ID_KEYS names them; GROUP_RULES holds the prompt-text fallbacks.
#
# geometry_truth deliberately stays on a text rule: its nine parts are
# concatenated and each restarts its row numbering, so stored ids collide
# across parts, while the negation pairs the rule exists to catch (cities
# vs neg_cities) live in different parts.
#
# Over-grouping is the safe direction, so the rules err coarse.

# The wrapper line tf_wrapped_a puts between context and statement. Absent
# in data_unwrap, where the prompt is the bare context/statement already.
_TF_WRAPPER = re.compile(
    r"\s*Determine whether the following is True or False\.\s*")

# Negation / direction variants of one underlying fact, mapped together.
_GOT_NORMALISE = (
    (" is not in ", " is in "),                  # cities / neg_cities
    (" does not mean ", " means "),              # sp_en_trans / neg_*
    (" is smaller than ", " is larger than "),   # smaller_than / larger_than
)
# Words of the normalised statement that make the group key. Long enough to
# keep unrelated facts apart, short enough that the differing tail (the
# false object, the wrong country) falls off the end.
_GOT_KEY_WORDS = 6


def _split_wrapper(prompt: str) -> tuple[str, str]:
    """(context, statement). The wrapper line separates them where it is
    present; without it there is no context and the whole prompt is the
    statement."""
    parts = _TF_WRAPPER.split(prompt, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", prompt.strip()


def _group_social_iqa(context: str, statement: str) -> str:
    """The scenario paragraph — shared by every answer choice expanded from
    one question, which is exactly the set of rows to keep together."""
    return context


def _group_pubmedqa(context: str, statement: str) -> str:
    """The abstract. Wrapped, it is the context; unwrapped, the prompt is
    "<abstract>\\n\\n<question>" and the question is the last block."""
    return context or statement.rsplit("\n\n", 1)[0]


def _group_geometry_truth(context: str, statement: str) -> str:
    """The fact behind the statement: direction-normalised, then truncated
    so the part that differs between the true and the false version (the
    swapped country, the wrong translation) is not in the key."""
    s = statement
    for variant, canonical in _GOT_NORMALISE:
        s = s.replace(variant, canonical)
    return " ".join(s.split()[:_GOT_KEY_WORDS])


# source -> the stored array to use as its fold-group id (see the section
# comment above). Checked first; GROUP_RULES is the fallback.
GROUP_ID_KEYS = {
    "wildjailbreak": "source_prompt_id",
    "pubmedqa_artificial": "source_row_id",
    "pubmedqa_labeled": "source_row_id",
    "social_iqa": "source_row_id",
}

# The text rules. geometry_truth uses one by design; the other three are
# kept only as a fallback for a file with no stored id, so that a missing
# array degrades to the old behaviour instead of silently becoming one
# group per row (which for perturbation A's doubled pubmedqa / social_iqa
# would put a statement and its negation in different folds).
GROUP_RULES = {
    "geometry_truth": _group_geometry_truth,
    "social_iqa": _group_social_iqa,
    "pubmedqa_artificial": _group_pubmedqa,
    "pubmedqa_labeled": _group_pubmedqa,
}


def _stored_group_key(source: str, loaded: dict) -> np.ndarray | None:
    """The stored fold-group id for a just-loaded source, or None if this
    source does not use one (or the file predates it). Ids are namespaced
    with the source name: they are row numbers into the upstream dataset,
    so pubmedqa_artificial's 0 and pubmedqa_labeled's 0 are unrelated and
    must not collide once the pooled corpus concatenates them."""
    key = GROUP_ID_KEYS.get(source)
    if key is None or loaded.get(key) is None:
        return None
    return np.char.add(f"{source}:", np.asarray(loaded[key]).astype(str))


@lru_cache(maxsize=None)
def _prompts_cached(source: str, layer: int, data: str) -> dict:
    """`load_prompts` memoised: `load_datasets` derives groups once per role
    and the prompts are the same every time. Read-only — callers wanting a
    dict of their own should use `load_prompts`."""
    return load_prompts(source, layer, data)


def group_ids(source: str, sample_id: np.ndarray, layer: int,
              data: str = DEFAULT_COLLECTION,
              stored: np.ndarray | None = None) -> np.ndarray:
    """Fold-group id per sample for one HF source, aligned to `sample_id`.

    `stored` is the source's own id array where it has one (see
    GROUP_ID_KEYS) — exact, and it costs no prompt read, so it wins.
    Otherwise a GROUP_RULES text rule reconstructs the id from the prompt,
    read from the source's `last` cache file (the `_topic_from_last`
    pattern: same rows, same order, and the only variant guaranteed to
    carry them), so a run on another variant pays one extra cached-file
    read here. A source with neither gets its sample_id back, i.e. one
    group per row — the identity case, which splits exactly as an ungrouped
    run would."""
    if stored is not None:
        if len(stored) != len(sample_id):
            raise ValueError(f"{source}: stored group id has {len(stored)} "
                             f"rows, sample_id has {len(sample_id)}")
        return np.asarray(stored)
    rule = GROUP_RULES.get(source)
    if rule is None:
        return np.asarray(sample_id)
    prompts = _prompts_cached(source, layer, data)
    missing = [s for s in sample_id if s not in prompts]
    if missing:
        raise KeyError(f"{source}: no prompt for {len(missing)} samples "
                       f"(e.g. {missing[:3]}) — cannot build fold groups")
    out = np.array([rule(*_split_wrapper(str(prompts[s]))) for s in sample_id])
    # A text rule that collapses everything into one group would put the
    # whole source in a single fold — CV would silently stop being CV.
    # _group_social_iqa keys on the wrapper's context block, so it does
    # exactly this on a tree with no wrapper; fail loudly instead.
    if len(out) > 1 and len(np.unique(out)) == 1:
        raise ValueError(
            f"{source}: the {rule.__name__} fold-group rule put all "
            f"{len(out)} rows in one group — it does not fit this "
            f"collection's prompt format")
    return out


# Sources that carry more than one experiment unit, split on their `topic`
# array. Each entry is (source -> {unit: the topic values that belong to it},
# default unit for every topic value not claimed above, or None to require
# that every value is claimed). Applies in any collection whose roster
# contains the units.
#
#   geometry_truth  the paper's two uncurated test sets stand alone; the
#                   seven curated parts merge into got.
#   wildjailbreak   `types` is {vanilla,adversarial} x {benign,harmful},
#                   12,500 rows each. The two halves behave completely
#                   differently under transfer — a pooled wildjailbreak AUC
#                   averages over a ~0.3 gap — so they are separate units
#                   and the adversarial half is a held-out set
#                   (_POOL_EXCLUDED).
_SPLIT_SOURCES = {
    "geometry_truth": ({"common_claim": ("common_claim_true_false",),
                        "counterfact": ("counterfact_true_false",)},
                       "got"),
    "wildjailbreak": ({"wildjailbreak_vanilla": ("vanilla_benign",
                                                 "vanilla_harmful"),
                       "wildjailbreak_adversarial": ("adversarial_benign",
                                                     "adversarial_harmful")},
                      None),
}

# every unit produced by a split, so _load_units knows which roster entries
# are units rather than sources
_SPLIT_UNITS = tuple(
    u for claimed, default in _SPLIT_SOURCES.values()
    for u in (*claimed, *([default] if default else ()))
)


def unit_source(unit: str) -> str:
    """The HF source an experiment unit's rows come from (identity for a
    unit that is its own source). Fold-group namespacing must use the
    SOURCE: wildjailbreak's two units share one source_prompt_id space —
    each adversarial prompt is a rewrite of one vanilla prompt, same
    label — so a pooled corpus containing both halves has to keep such a
    pair in a single CV fold, which a per-unit namespace would sever."""
    for src, (claimed, default) in _SPLIT_SOURCES.items():
        if unit in claimed or unit == default:
            return src
    return unit


def _topic_from_last(source: str, variant: str, layer: int, data: str,
                     n: int) -> np.ndarray:
    """A source's `topic` array read from its `last` cache file.

    Some collections' sparse caches omit `topic` even though the dense one
    carries it: data_unwrap/geometry_truth is the case that forced this —
    its sae and transcoder files have no topic array, while
    tf_wrapped_update/A's do. geometry_truth cannot be split into
    got / common_claim / counterfact without it, so read it from the
    `last` file, whose rows are in the same order. That ordering is not
    an assumption: load_datasets refuses any pair of cache variants whose
    labels or sample_ids disagree, so a mismatch would be caught there —
    and the row count is checked here.
    """
    from huggingface_hub import hf_hub_download

    c = COLLECTIONS[data]
    parts = []
    for hf_dir, stem in c.hf_files[source]:
        path = hf_hub_download(
            HF_REPO, f"{hf_dir}/{stem}_last_layer{layer + 1}.npz",
            repo_type="dataset")
        with np.load(path, allow_pickle=True) as d:
            key = next((k for k in ("topic", "types", "data_types")
                        if k in d.files), None)
            if key is None:
                raise KeyError(
                    f"{source}: no 'topic' in the {variant} cache and none "
                    f"in the last cache either — cannot split it")
            parts.append(np.asarray(d[key]))
    topic = parts[0] if len(parts) == 1 else np.concatenate(parts)
    if len(topic) != n:
        raise ValueError(f"{source}: topic from the last cache has "
                         f"{len(topic)} rows, this variant has {n}")
    return topic


def _load_units(variant: str, layer: int, data: str,
                groups: bool = True) -> dict:
    """unit -> {X, y, sample_id, topic, group} for one cache variant.
    A source in _SPLIT_SOURCES (geometry_truth, wildjailbreak) is loaded
    once and split by its saved `topic` array into its units; missing HF
    files are skipped with a note. `groups=False` skips the fold-group
    derivation, and with it any prompt read `group_ids` does — worth it
    only on a variant whose `last` cache is not already downloaded."""
    c = COLLECTIONS[data]
    sources = [d for d in c.datasets if d not in _SPLIT_UNITS]
    for src, (claimed, default) in _SPLIT_SOURCES.items():
        units = (*claimed, *([default] if default else ()))
        if any(u in c.datasets for u in units):
            sources.insert(0, src)
    out = {}
    for name in sources:
        try:
            d = load_hf(name, variant, layer, data)
        except EntryNotFoundError:
            print(f"  {name} {variant} layer{layer}: not on HF yet, skipped")
            continue
        d = {"X": d["X"],
             "y": binary_labels(d["labels"], name, c.label_map.get(name)),
             "sample_id": d["sample_id"], "topic": d.get("topic"),
             "group": (group_ids(name, d["sample_id"], layer, data,
                                 stored=_stored_group_key(name, d))
                       if groups else d["sample_id"])}
        if name not in _SPLIT_SOURCES:
            out[name] = d
            continue

        claimed, default = _SPLIT_SOURCES[name]
        if d["topic"] is None:   # sparse caches of some collections omit it
            d["topic"] = _topic_from_last(name, variant, layer, data,
                                          len(d["y"]))
        topic = d["topic"].astype(str)
        masks = {unit: np.isin(topic, list(values))
                 for unit, values in claimed.items()}
        spoken = [v for values in claimed.values() for v in values]
        if default is None:
            unclaimed = set(np.unique(topic).tolist()) - set(spoken)
            if unclaimed:
                raise ValueError(
                    f"{name}: topic values {sorted(unclaimed)} belong to no "
                    f"unit; _SPLIT_SOURCES must claim every value when it "
                    f"declares no default unit")
        else:
            masks[default] = ~np.isin(topic, spoken)
        for unit, m in masks.items():
            if unit in c.datasets:
                out[unit] = {k: v[m] for k, v in d.items()}
    return out


def load_datasets(layer: int, raw: str, roles, data: str = DEFAULT_COLLECTION,
                  groups: bool = False) -> dict:
    """unit -> {y, sample_id, topic, group, blocks}. `blocks` maps each
    requested role ("raw" -> the --raw variant, "sae", "transcoder") to that
    unit's matrix. A unit missing a role's HF file simply lacks that block,
    and feature sets that need it skip the unit. Labels and row order must
    agree across the roles' cache files.

    `group` is the fold-group id (see the fold-groups section above). It
    costs a prompt read per rule-carrying source, so it is opt-in: the CV
    runners pass `groups=True` (--group-folds), and callers that only want
    matrices — the analysis scripts — leave it off and get the sample_id,
    i.e. one group per row."""
    units = {}
    for role in roles:
        variant = raw if role == "raw" else role
        for unit, d in _load_units(variant, layer, data, groups=groups).items():
            u = units.setdefault(unit, {"y": d["y"], "sample_id": d["sample_id"],
                                        "topic": d["topic"],
                                        "group": d["group"], "blocks": {}})
            if not (np.array_equal(u["y"], d["y"])
                    and np.array_equal(u["sample_id"], d["sample_id"])):
                raise ValueError(f"{unit}: rows disagree between cache variants")
            u["blocks"][role] = d["X"]
    return units
