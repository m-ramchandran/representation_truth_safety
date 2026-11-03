"""Load T/F-wrapped prompts for Gemma activation extraction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SHARED_CORE = Path(__file__).resolve().parents[2] / "shared_core"
if str(_SHARED_CORE) not in sys.path:
    sys.path.insert(0, str(_SHARED_CORE))

from tf_source_loaders import ALL_DATASETS, WILDMJAILBREAK_CATEGORIES, load_records
from tf_wrappers import (
    OPTIONAL_EXAMPLE_KEYS,
    build_all_tf_examples,
    records_to_tf_examples,
)


@dataclass(frozen=True)
class ExtractionDefaults:
    """Row counts aligned with the unwrapped pipelines."""

    balance: bool = False
    n: int | None = None  # random / balanced subsample cap (after other filters)
    # WildJailbreak only: False → per-category subsample; True → all rows.
    # Ignored for other datasets (use balance / n instead).
    full_dataset: bool = True
    n_per_category: int = 12_500  # wildjailbreak only


DEFAULTS: dict[str, ExtractionDefaults] = {
    "geometry_truth": ExtractionDefaults(),
    "harmful_harmless": ExtractionDefaults(),
    # balance/n unused: row selection comes from shared_sampling.py's canonical
    # balanced sample (200 safe + 200 unsafe = 400 of 450) -- see its docstring.
    "xstest": ExtractionDefaults(),
    # balance/n are unused for this dataset: question selection now comes from
    # shared_sampling.py's canonical row-ID sample (30,250 questions, balanced
    # by true final_decision), applied *before* the yes/no claim expansion so
    # both claim rows always travel together -> 60,500 expanded rows. See
    # shared_sampling.py docstring for why (keeps this in sync with the
    # unwrapped pipeline, which draws from the same file).
    "pubmedqa_artificial": ExtractionDefaults(),
    # balance/n unused: row selection comes from shared_sampling.py's canonical
    # balanced sample (338 yes + 338 no = 676 of 890), applied *before* the
    # yes/no claim expansion -> 1,352 expanded rows.
    "pubmedqa_labeled": ExtractionDefaults(),
    "social_iqa": ExtractionDefaults(),
    # One wrong choice dropped per question before T/F-claim building (see
    # tf_wrappers.expand_social_iqa_binary_record) -- 2 rows/question, always
    # kept in full (35,364 questions x 2 = 70,728 rows), no balance/n applied.
    "social_iqa_binary": ExtractionDefaults(),
    "wildjailbreak": ExtractionDefaults(
        full_dataset=False,
        n_per_category=12_500,
    ),
    "claude_anatomy": ExtractionDefaults(),
    "claude_blood_test_ranges": ExtractionDefaults(),
    "claude_vds": ExtractionDefaults(),
    "claude_vds_300": ExtractionDefaults(),
    # 1327 x 3 formats x chosen/rejected = 7962 rows, already exactly balanced.
    "rm_bench": ExtractionDefaults(),
}


def _balanced_subsample(
    indices: list[int],
    labels: list[bool],
    n: int,
    seed: int,
) -> list[int]:
    by_label: dict[bool, list[int]] = {True: [], False: []}
    for i in indices:
        by_label[bool(labels[i])].append(i)

    rng = np.random.default_rng(seed)
    per_class = n // 2
    minority = min(len(by_label[True]), len(by_label[False]))
    if per_class > minority:
        per_class = minority

    chosen: list[int] = []
    for grp in (True, False):
        idxs = by_label[grp]
        if len(idxs) <= per_class:
            chosen.extend(idxs)
        else:
            chosen.extend(rng.choice(idxs, size=per_class, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def _canonical_row_indices(
    dataset: str,
    examples: list[dict],
    *,
    seed: int,
) -> list[int]:
    """Positions in `examples` whose 'row' is in the shared canonical sample.

    For pubmedqa_artificial this runs after yes/no claim expansion, so both
    claim rows of a canonical question share a 'row' and are both kept.
    Fingerprint verification happens in shared_sampling.filter_to_canonical.
    """
    from shared_sampling import filter_to_canonical

    kept = filter_to_canonical(dataset, examples, text_fn=lambda r: r["original_prompt"])
    kept_rows = {r["row"] for r in kept}
    idx = [i for i, ex in enumerate(examples) if ex.get("row") in kept_rows]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx


def _wildjailbreak_indices(
    examples: list[dict],
    *,
    full_dataset: bool,
    n_per_category: int,
    seed: int,
) -> list[int]:
    if not full_dataset and n_per_category == 12_500:
        return _canonical_row_indices("wildjailbreak", examples, seed=seed)

    by_type: dict[str, list[int]] = {cat: [] for cat in WILDMJAILBREAK_CATEGORIES}
    for i, ex in enumerate(examples):
        dt = str(ex.get("type", ""))
        if dt in by_type:
            by_type[dt].append(i)

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cat in WILDMJAILBREAK_CATEGORIES:
        idxs = by_type[cat]
        if not idxs:
            continue
        n_take = len(idxs) if full_dataset else min(n_per_category, len(idxs))
        if n_take < len(idxs):
            chosen.extend(rng.choice(idxs, size=n_take, replace=False).tolist())
        else:
            chosen.extend(idxs)
    rng.shuffle(chosen)
    return chosen


def _select_indices(
    dataset: str,
    examples: list[dict],
    *,
    seed: int,
    balance: bool,
    n: int | None,
    full_dataset: bool,
    n_per_category: int,
) -> list[int]:
    all_idx = list(range(len(examples)))
    if dataset in ("pubmedqa_artificial", "pubmedqa_labeled", "xstest"):
        return _canonical_row_indices(dataset, examples, seed=seed)

    if dataset == "wildjailbreak" and not full_dataset:
        return _wildjailbreak_indices(
            examples,
            full_dataset=full_dataset,
            n_per_category=n_per_category,
            seed=seed,
        )

    labels = [bool(ex["label_tf"]) for ex in examples]
    if balance and n is not None and n > 0:
        return _balanced_subsample(all_idx, labels, n, seed)

    if n is not None and n > 0 and n < len(all_idx):
        rng = np.random.default_rng(seed)
        return rng.choice(all_idx, size=n, replace=False).tolist()
    return all_idx


def _extra_arrays(examples: list[dict], indices: list[int]) -> dict[str, np.ndarray]:
    selected = [examples[i] for i in indices]
    extras: dict[str, np.ndarray] = {
        "statements": np.array([ex["statement"] for ex in selected]),
        "original_prompts": np.array([ex.get("original_prompt", "") for ex in selected]),
        "source_row_ids": np.array([ex["row"] for ex in selected], dtype=np.int64),
        "meta_wrapper_variant": np.array(selected[0]["wrapper_variant"]),
        "meta_label_convention": np.array(
            "labels[i]=label_tf (bool): True = factually correct or safe to answer"
        ),
        "meta_original_prompts_convention": np.array(
            "original_prompts[i] is the unwrapped-pipeline prompt for source_row_ids[i] "
            "(no T/F wrapper/claim template) -- for social_iqa multiple wrapped rows "
            "share one original_prompt/source_row_id (one per answer choice)."
        ),
    }
    if "topic" in selected[0]:
        extras["topic"] = np.array([ex.get("topic", "") for ex in selected])
    if "type" in selected[0]:
        extras["types"] = np.array([ex.get("type", "") for ex in selected])
    if "choice" in selected[0]:
        extras["choice"] = np.array([ex.get("choice", "") for ex in selected])
    for key in OPTIONAL_EXAMPLE_KEYS:
        if key in ("topic", "type", "choice"):
            continue
        if key in selected[0]:
            extras[key] = np.array([ex.get(key, "") for ex in selected])
    return extras


def load_wrapped_for_extraction(
    dataset: str,
    variant: str,
    *,
    seed: int = 0,
    balance: bool | None = None,
    n: int | None = None,
    full_dataset: bool | None = None,
    n_per_category: int | None = None,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Return (prompts, label_tf bool array, extra npz fields)."""
    if dataset not in ALL_DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {ALL_DATASETS}")
    if variant not in ("A", "B"):
        raise ValueError(f"extraction supports variants A and B only (got {variant!r})")

    defaults = DEFAULTS[dataset]
    balance = defaults.balance if balance is None else balance
    n = defaults.n if n is None else n
    full_dataset = defaults.full_dataset if full_dataset is None else full_dataset
    n_per_category = defaults.n_per_category if n_per_category is None else n_per_category

    print(
        f"Building T/F-wrapped examples: dataset={dataset} variant={variant} ...",
        flush=True,
    )
    examples = build_all_tf_examples(dataset, variant, seed=seed)
    indices = _select_indices(
        dataset,
        examples,
        seed=seed,
        balance=balance,
        n=n,
        full_dataset=full_dataset,
        n_per_category=n_per_category,
    )
    selected = [examples[i] for i in indices]
    prompts = [ex["prompt_wrapped"] for ex in selected]
    labels = np.array([bool(ex["label_tf"]) for ex in selected], dtype=bool)
    extras = _extra_arrays(examples, indices)

    n_true = int(labels.sum())
    print(
        f"Selected {len(prompts)} wrapped prompts | label_tf True={n_true} "
        f"False={len(prompts) - n_true}",
        flush=True,
    )
    return prompts, labels, extras


def materialize_wrapped_subset(
    dataset: str,
    variant: str,
    records: list[dict],
) -> list[dict]:
    """Wrap an in-memory record list (for tests / pilots)."""
    return records_to_tf_examples(dataset, records, variant)
