"""Activation loading for the classifier experiments.

This module is import-safe on any machine (numpy/scipy only — no cuML,
no torch), which is why it lives at the package root: `experiments/`
(GPU) and `analysis/` (laptop) both import it, so there is exactly one
copy of the label conventions and the dataset roster. The clustering
package keeps its own older copy of the loader, with the OLD label
convention (1 = harmful on the safety sets) — do not mix the two.

The activations come in named COLLECTIONS — independent snapshots of the
HF repo differing in prompt wording, roster, and label storage (e.g.
"original" = the bare prompts, "tf_wrapped_a" = everything rephrased as
a True/False question, wrapper variant A). A collection declares where
its files live, which experiment units exist, their task groups, and any
label flips; the loading machinery below is shared. Every loader takes
`data=<collection name>` (default "original"), the runners expose it as
--data, and results are kept apart under results/<collection>/. Adding a
new collection = one COLLECTIONS entry.

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

`load_hf` reads the files from the HF repo (anqiwang02/Activations);
downloads go through `hf_hub_download`, which keeps a local copy under
~/.cache/huggingface — each file is fetched from the network once ever.
Layers are always Scope layers (see docs/data.md);
the files are named by HF hidden_states index, Scope layer s -> file layer{s+1}.

On top of the file loaders sit the experiment-unit loaders: `DATASETS`
names the binary units (geometry_truth split into got / common_claim /
counterfact following the original paper), `binary_labels` casts each
unit's labels to {0,1} ints, and `load_datasets` returns unit ->
{y, sample_id, topic, blocks} with one matrix per requested cache role.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from huggingface_hub.errors import EntryNotFoundError

HF_REPO = "anqiwang02/Activations"

VARIANTS = ("last", "mean", "sae", "transcoder")


@dataclass(frozen=True)
class Collection:
    """One snapshot of the activation data: where its HF files live and
    what the experiment looks like on it. All label conventions here are
    1 = true / yes / safe / harmless."""
    hf_files: dict          # HF source -> [(hf_dir, file_stem) parts]
    datasets: tuple         # the experiment units, in display order
    tasks: dict             # task -> member units (see task_of)
    layers: tuple           # default Scope layers for runs (--layers overrides)
    eval_only: tuple = ()   # units only ever evaluated on, never trained on
    flip_labels: tuple = () # sources stored with the opposite convention

    @property
    def train_datasets(self) -> tuple:
        return tuple(d for d in self.datasets if d not in self.eval_only)

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

# The 11 HF sources of the re-extracted era, shared by tf_wrapped_a and
# data_unwrap: same prompts, one wrapped and one not. social_iqa is a
# unit only in the wrapped collection, where the T/F expansion turned
# its answer index into binary labels; it is loadable in both.
_EXTENDED_SOURCES = (
    "geometry_truth", "harmful_harmless", "pubmedqa_artificial",
    "pubmedqa_labeled", "social_iqa", "wildjailbreak", "xstest",
    "claude_anatomy", "claude_blood_test_ranges",
    "claude_diagnosis_severity", "claude_vds",
)

COLLECTIONS = {
    # The bare prompts, cached at Scope layers 9/17/22/29 only.
    # geometry_truth is one source made of 9 files (the sub-dataset name
    # goes into `topic`); pubmedqa's two subsets are separate sources.
    # Units: geometry_truth split into got + the paper's two uncurated
    # test sets, social_iqa excluded (labels are an answer index, not
    # binary). wildjailbreak stores 1 = harmful intent and xstest stores
    # 1 = unsafe, so both are flipped at load; harmful_harmless already
    # stores 1 = harmless, the geometry_truth parts store 1 = true, and
    # pubmedqa stores 'yes'/'no' strings (mapped in binary_labels,
    # positive = 'yes'). All raw conventions were verified against the
    # prompt text.
    "original": Collection(
        hf_files={
            "geometry_truth": [(f"geometry_truth/{p}", f"got_{p}")
                               for p in _GEOMETRY_TRUTH_PARTS],
            "harmful_harmless": [("harmful_harmless", "harmful_harmless")],
            "pubmedqa_artificial": [("pubmedqa/pqa_artificial",
                                     "pubmedqa_artificial")],
            "pubmedqa_labeled": [("pubmedqa/pqa_labeled", "pubmedqa_labeled")],
            "social_iqa": [("social_iqa", "social_iqa")],
            "wildjailbreak": [("wildjailbreak", "wildjailbreak")],
            "xstest": [("xstest", "xstest")],
        },
        datasets=("got", "common_claim", "counterfact", "harmful_harmless",
                  "pubmedqa_artificial", "pubmedqa_labeled", "wildjailbreak",
                  "xstest"),
        tasks={
            "tf": ("got", "common_claim", "counterfact"),
            "medical": ("pubmedqa_artificial", "pubmedqa_labeled"),
            "jailbreak": ("harmful_harmless", "wildjailbreak", "xstest"),
        },
        layers=(9, 17, 22, 29),
        flip_labels=("wildjailbreak", "xstest"),
    ),
    # Every prompt rephrased as a True/False question (wrapper variant A),
    # under tf_wrapped_update/A/ — one file per source (geometry_truth's 9
    # parts pre-merged, with `topic` intact), all 34 layers cached, and
    # one shared label convention stored in the files themselves ("True =
    # factually correct or safe to answer"), so no flips. social_iqa is
    # now a unit (its labels became binary T/F), and four new claude_*
    # sets join; claude_diagnosis_severity (n=40) is evaluated on but
    # never trained on. Conventions were verified against the files'
    # meta_label_convention and prompt text.
    "tf_wrapped_a": Collection(
        hf_files={s: [(f"tf_wrapped_update/A/{s}", f"{s}_tf_a")]
                  for s in _EXTENDED_SOURCES},
        datasets=("got", "common_claim", "counterfact", "harmful_harmless",
                  "pubmedqa_artificial", "pubmedqa_labeled", "wildjailbreak",
                  "xstest", "social_iqa", "claude_anatomy",
                  "claude_blood_test_ranges", "claude_diagnosis_severity",
                  "claude_vds"),
        tasks={
            "tf": ("got", "common_claim", "counterfact"),
            "medical": ("pubmedqa_artificial", "pubmedqa_labeled",
                        "claude_anatomy", "claude_blood_test_ranges",
                        "claude_diagnosis_severity"),
            "jailbreak": ("harmful_harmless", "wildjailbreak", "xstest"),
        },
        layers=(17, 22, 29, 33),
        eval_only=("claude_diagnosis_severity",),
    ),
    # The bare prompts again, re-extracted under data_unwrap/ in the
    # tf_wrapped_a layout: one file per source (geometry_truth's 9 parts
    # pre-merged, with `topic` intact), all 34 layers cached, the same
    # four claude_* sets. This is tf_wrapped_a's unwrapped counterpart —
    # same prompts, same roster, no True/False wrapper (with one
    # deliberate exception, social_iqa; see below) — so it runs at
    # tf_wrapped_a's layers, not `original`'s, to keep the comparison at
    # matched depths.
    #
    # Labels are stored per source here (there is no one shared
    # convention as in tf_wrapped_a), and they are `original`'s: the
    # geometry_truth parts and all four claude_* sets store 1 = true
    # statement, harmful_harmless stores 1 = harmless, pubmedqa stores
    # 'yes'/'no' strings (mapped in binary_labels, positive = 'yes'),
    # and wildjailbreak ('True=harmful') and xstest ('True=unsafe')
    # store the opposite, so both are flipped at load.
    # claude_diagnosis_severity (n=40) is evaluated on but never trained
    # on. Conventions were verified against each file's
    # meta_label_convention and prompt text; the claude_* sets carry no
    # convention string, so their polarity was read off the prompts —
    # and blood_test_ranges' labels match its `perturbation` field
    # exactly, none <-> 1 on all 450 rows.
    #
    # ONE EXCEPTION, and it is deliberate:
    # social_iqa is loaded from the WRAPPED tree, the same files
    # tf_wrapped_a uses. There is no unwrapped equivalent to load. The
    # two extractions have different rows, not just different wording:
    # data_unwrap/social_iqa is one activation per QUESTION (n=35,364,
    # all of Social IQa), whose prompt lists choices A/B/C and asserts
    # nothing, labelled with the correct answer index (1/2/3) — not a
    # binary property of the prompt, which is why `original` has no
    # social_iqa unit either. The wrapped tree instead expands each
    # question into one row per CHOICE ("The correct answer for the
    # question ... is ...", n=50,000 sampled from 3 x 35,364, balanced),
    # so each activation belongs to a single assertion that is true or
    # false. Nothing computable from the 35,364 unwrapped vectors
    # recovers those; the model never saw a prompt asserting one answer.
    # Keeping the wrapped file keeps the roster identical to
    # tf_wrapped_a's 13 units, at the cost of one dataset in this corpus
    # carrying the "Determine whether the following is True or False."
    # line the other eleven do not — a surface-form difference a pooled
    # model can see. The clean fix is variant B of the wrapper plan (the
    # same per-choice expansion, wrapper line dropped), which is
    # specified in extract_activations/TF_prompt_wrappers_plan.md but
    # has never been extracted — there is no tf_wrapped_update/B/ on HF.
    "data_unwrap": Collection(
        hf_files={
            **{s: [(f"data_unwrap/{s}", s)]
               for s in _EXTENDED_SOURCES if s != "social_iqa"},
            "social_iqa": [("tf_wrapped_update/A/social_iqa",
                            "social_iqa_tf_a")],
        },
        datasets=("got", "common_claim", "counterfact", "harmful_harmless",
                  "pubmedqa_artificial", "pubmedqa_labeled", "wildjailbreak",
                  "xstest", "social_iqa", "claude_anatomy",
                  "claude_blood_test_ranges", "claude_diagnosis_severity",
                  "claude_vds"),
        tasks={
            "tf": ("got", "common_claim", "counterfact"),
            "medical": ("pubmedqa_artificial", "pubmedqa_labeled",
                        "claude_anatomy", "claude_blood_test_ranges",
                        "claude_diagnosis_severity"),
            "jailbreak": ("harmful_harmless", "wildjailbreak", "xstest"),
        },
        layers=(17, 22, 29, 33),
        eval_only=("claude_diagnosis_severity",),
        flip_labels=("wildjailbreak", "xstest"),
    ),
}

# The original collection's tables, importable by name — analysis code
# written against the pre-collection module keeps working unchanged.
HF_FILES = COLLECTIONS["original"].hf_files
DATASETS = COLLECTIONS["original"].datasets
TASKS = COLLECTIONS["original"].tasks


def task_of(dataset: str, data: str = "original") -> str:
    """The task a unit belongs to; a unit in no task is its own."""
    return COLLECTIONS[data].task_of(dataset)


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
                "prompts"):
        if key in d.files:
            arr = np.asarray(d[key])
            if len(arr) != n:
                raise ValueError(f"{path}: '{key}' length {len(arr)} != N {n}")
            out["topic" if key in {"types", "data_types"} else key] = arr
    source = source or path.stem
    out["sample_id"] = np.array([f"{source}:{i}" for i in range(n)])
    return out


def load_hf(dataset: str, variant: str, layer: int,
            data: str = "original") -> dict:
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


def load_prompts(source: str, layer: int, data: str = "original") -> dict:
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
                data: str = "original") -> dict:
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


def binary_labels(labels: np.ndarray, dataset: str) -> np.ndarray:
    """Labels as {0,1} ints. Bools cast directly; pubmedqa stores 'yes'/'no'
    strings (positive = 'yes'). Anything else is an error, not a guess."""
    labels = np.asarray(labels)
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


# geometry_truth parts -> units: the paper's uncurated sets stand alone,
# everything else merges into got. Applies in any collection whose
# roster contains the split units.
_UNCURATED = {"common_claim": "common_claim_true_false",
              "counterfact": "counterfact_true_false"}
_SPLIT_UNITS = ("got", *_UNCURATED)


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


def _load_units(variant: str, layer: int, data: str) -> dict:
    """unit -> {X, y, sample_id, topic} for one cache variant. geometry_truth
    is loaded once and split by its saved part label into got / common_claim /
    counterfact; missing HF files are skipped with a note."""
    c = COLLECTIONS[data]
    sources = [d for d in c.datasets if d not in _SPLIT_UNITS]
    if any(u in _SPLIT_UNITS for u in c.datasets):
        sources = ["geometry_truth"] + sources
    out = {}
    for name in sources:
        try:
            d = load_hf(name, variant, layer, data)
        except EntryNotFoundError:
            print(f"  {name} {variant} layer{layer}: not on HF yet, skipped")
            continue
        d = {"X": d["X"], "y": binary_labels(d["labels"], name),
             "sample_id": d["sample_id"], "topic": d.get("topic")}
        if name != "geometry_truth":
            out[name] = d
            continue
        if d["topic"] is None:   # sparse caches of some collections omit it
            d["topic"] = _topic_from_last(name, variant, layer, data,
                                          len(d["y"]))
        topic = d["topic"].astype(str)
        masks = {unit: topic == part for unit, part in _UNCURATED.items()}
        masks["got"] = ~np.isin(topic, list(_UNCURATED.values()))
        for unit, m in masks.items():
            if unit in c.datasets:
                out[unit] = {k: v[m] for k, v in d.items()}
    return out


def load_datasets(layer: int, raw: str, roles, data: str = "original") -> dict:
    """unit -> {y, sample_id, topic, blocks}. `blocks` maps each requested
    role ("raw" -> the --raw variant, "sae", "transcoder") to that unit's
    matrix. A unit missing a role's HF file simply lacks that block, and
    feature sets that need it skip the unit. Labels and row order must
    agree across the roles' cache files."""
    units = {}
    for role in roles:
        variant = raw if role == "raw" else role
        for unit, d in _load_units(variant, layer, data).items():
            u = units.setdefault(unit, {"y": d["y"], "sample_id": d["sample_id"],
                                        "topic": d["topic"], "blocks": {}})
            if not (np.array_equal(u["y"], d["y"])
                    and np.array_equal(u["sample_id"], d["sample_id"])):
                raise ValueError(f"{unit}: rows disagree between cache variants")
            u["blocks"][role] = d["X"]
    return units
