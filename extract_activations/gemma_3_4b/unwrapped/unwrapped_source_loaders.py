"""Per-dataset prompt/label loading for the unwrapped extraction pipeline.

Record loading and prompt formatting live in build_prompts.py. The canonical
row-ID sample for pubmedqa_artificial and wildjailbreak comes from
shared_core/shared_sampling.py, shared with the wrapped pipeline.

geometry_truth is always the merged 9-part dataset; the topic column
distinguishes sub-parts.

Label conventions: build_prompts' label_raw is a bool for
geometry_truth/harmful_harmless/wildjailbreak/claude_*, a "yes"/"no" string
for pubmedqa, a "safe"/"unsafe" string for xstest (converted here to bool,
True=unsafe), and an int 1/2/3 for social_iqa (cast to int8 here).

social_iqa_binary is a derived variant (see _social_iqa_binary_prompts): one
wrong choice dropped per row, survivors relabeled 1/2 in a seeded random
order, so int8 labels are in {1, 2}.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UNWRAP_ROOT = Path(__file__).resolve().parent
_SHARED_CORE = _UNWRAP_ROOT.parents[1] / "shared_core"
for _p in (_UNWRAP_ROOT, _SHARED_CORE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_prompts as bp  # noqa: E402

UNWRAPPED_DATASETS = [
    "geometry_truth",
    "harmful_harmless",
    "xstest",
    "social_iqa",
    "social_iqa_binary",
    "wildjailbreak",
    "pubmedqa_artificial",
    "pubmedqa_labeled",
    "claude_anatomy",
    "claude_blood_test_ranges",
    "claude_vds",
    "claude_vds_300",
    "rm_bench",
]

XSTEST_HF_PRIMARY = "walledai/XSTest"
XSTEST_HF_FALLBACK = "Paul/XSTest"
WILDJAILBREAK_HF_REPO = "allenai/wildjailbreak"
WILDJAILBREAK_CATEGORIES = bp.WILDJAILBREAK_CATEGORIES
DEFAULT_N_PER_CATEGORY = 12_500


@dataclass(frozen=True)
class UnwrappedDefaults:
    """Per-dataset extraction defaults."""

    balance: bool = False
    n: int | None = None
    full_dataset: bool = True
    n_per_category: int = DEFAULT_N_PER_CATEGORY


UNWRAPPED_DEFAULTS: dict[str, UnwrappedDefaults] = {
    "geometry_truth": UnwrappedDefaults(),
    # Canonical sample from shared_sampling.py bypasses n/balance -- see below
    # (only applies to the default --hf-dataset repo; an explicit override skips it).
    "xstest": UnwrappedDefaults(),
    "social_iqa": UnwrappedDefaults(),
    "social_iqa_binary": UnwrappedDefaults(),
    "harmful_harmless": UnwrappedDefaults(),
    # Canonical sample from shared_sampling.py bypasses n/balance -- see below.
    "pubmedqa_artificial": UnwrappedDefaults(balance=True, n=30_250),
    "pubmedqa_labeled": UnwrappedDefaults(n=0, balance=False),
    # Canonical sample from shared_sampling.py bypasses n_per_category -- see below.
    "wildjailbreak": UnwrappedDefaults(full_dataset=False, n_per_category=DEFAULT_N_PER_CATEGORY),
    "claude_anatomy": UnwrappedDefaults(),
    "claude_blood_test_ranges": UnwrappedDefaults(),
    "claude_vds": UnwrappedDefaults(),
    "claude_vds_300": UnwrappedDefaults(),
    # 1327 x 3 formats x chosen/rejected = 7962 rows, already exactly balanced.
    "rm_bench": UnwrappedDefaults(),
}


def log(msg: str) -> None:
    from datetime import datetime

    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _labels_for(dataset: str, records: list[dict]) -> np.ndarray:
    if dataset == "xstest":
        return np.array([r["label_raw"] == "unsafe" for r in records], dtype=bool)
    if dataset == "social_iqa":
        return np.array([r["label_raw"] for r in records], dtype=np.int8)
    if dataset.startswith("pubmedqa"):
        return np.array([r["label_raw"] for r in records], dtype="<U8")
    if dataset == "rm_bench":
        return np.array([r["label_raw"] for r in records], dtype="<U8")
    return np.array([bool(r["label_raw"]) for r in records], dtype=bool)


def _random_subsample(
    prompts: list[str],
    labels: np.ndarray,
    row_ids: np.ndarray,
    n: int | None,
    seed: int,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    extra_arrays = extra_arrays or {}
    n_use = len(prompts) if n is None or n <= 0 else min(n, len(prompts))
    if n_use >= len(prompts):
        return prompts, labels, row_ids, extra_arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prompts), size=n_use, replace=False)
    return (
        [prompts[i] for i in idx],
        labels[idx],
        row_ids[idx],
        {k: v[idx] for k, v in extra_arrays.items()},
    )


def _simple_dataset(dataset: str, *, n: int, seed: int) -> tuple[list, np.ndarray, np.ndarray, dict]:
    """geometry_truth, harmful_harmless, social_iqa, claude_* -- no dataset-
    specific subsampling logic, just a plain random subsample (usually n=None
    -> full set, per UNWRAPPED_DEFAULTS)."""
    records = bp.load_records(dataset)
    prompts = [bp.build_prompt(dataset, r) for r in records]
    labels = _labels_for(dataset, records)
    row_ids = np.array([r["row"] for r in records], dtype=np.int64)

    extra: dict[str, np.ndarray] = {}
    if dataset == "geometry_truth":
        extra["topic"] = np.array([r["topic"] for r in records])
    if dataset.startswith("claude_"):
        skip = {"statement", "label_raw", "row"}
        extra_keys = sorted({k for r in records for k in r if k not in skip})
        for key in extra_keys:
            extra[key] = np.array([r.get(key) for r in records], dtype=object)
    if dataset == "rm_bench":
        skip = {"prompt", "response", "label_raw", "row"}
        extra_keys = sorted({k for r in records for k in r if k not in skip})
        for key in extra_keys:
            extra[key] = np.array([r.get(key) for r in records], dtype=object)

    prompts, labels, row_ids, extra = _random_subsample(prompts, labels, row_ids, n, seed, extra)
    log(f"{dataset}: loaded {len(prompts)} prompts (n={n})")
    return prompts, labels, row_ids, extra


def social_iqa_binary_selections(seed: int) -> list[dict]:
    """Per-row binarization: drop one wrong choice, relabel survivors 1/2.

    Both stages draw from one np.random.default_rng(seed). The wrong choice is
    dropped uniformly at random, not by A/B/C position. Positions come from one
    global shuffle split into exact halves, so each position holds n//2 rows.

    Single source of truth: wrap_TF/tf_wrappers.py calls this directly, so both
    pipelines binarize identically for a given seed.
    """
    records = bp.load_records("social_iqa")
    rng = np.random.default_rng(seed)

    # Pass 1: per-row wrong-choice draw (in row order).
    rows: list[dict] = []
    for r in records:
        correct_idx = int(r["label_raw"])  # 1, 2, or 3 (A/B/C)
        answers = {1: r["answerA"], 2: r["answerB"], 3: r["answerC"]}
        wrong_idx = int(rng.choice([i for i in (1, 2, 3) if i != correct_idx]))
        rows.append({"r": r, "correct_idx": correct_idx, "wrong_idx": wrong_idx, "answers": answers})

    # Pass 2: global shuffle -> exact first-half/second-half position split.
    # Index-membership-tested (not built by iterating `order` directly) so
    # output stays in source-row order -- only the label assignment is
    # shuffled, not the list order downstream code depends on.
    n = len(rows)
    order = rng.permutation(n)
    first_half = set(order[: n // 2].tolist())

    selections: list[dict] = []
    for i, row in enumerate(rows):
        correct_idx, wrong_idx, answers = row["correct_idx"], row["wrong_idx"], row["answers"]
        if i in first_half:
            answer1, answer2, label = answers[correct_idx], answers[wrong_idx], 1
        else:
            answer1, answer2, label = answers[wrong_idx], answers[correct_idx], 2
        selections.append({
            **row["r"],
            "answer1": answer1,
            "answer2": answer2,
            "label": label,
            "wrong_idx": wrong_idx,
        })
    return selections


def _social_iqa_binary_prompts(*, n: int | None, seed: int) -> tuple[list, np.ndarray, np.ndarray, dict]:
    selections = social_iqa_binary_selections(seed)
    prompts = [bp.build_prompt("social_iqa_binary", s) for s in selections]
    labels_arr = np.array([s["label"] for s in selections], dtype=np.int8)
    row_ids_arr = np.array([s["row"] for s in selections], dtype=np.int64)
    prompts, labels_arr, row_ids_arr, _ = _random_subsample(prompts, labels_arr, row_ids_arr, n, seed)
    n1 = int((labels_arr == 1).sum())
    log(f"social_iqa_binary: {len(prompts)} rows | label1={n1} label2={len(prompts) - n1}")
    return prompts, labels_arr, row_ids_arr, {}


def _xstest_prompts(*, n: int | None, seed: int, hf_dataset: str | None) -> tuple[list, np.ndarray, np.ndarray, dict]:
    if hf_dataset is None or hf_dataset == XSTEST_HF_PRIMARY:
        # Canonical sample shared with the wrapped pipeline -- see shared_sampling.py.
        from shared_sampling import filter_to_canonical

        records = bp.load_records("xstest")  # already tries walledai -> Paul, same as the old default
        records = filter_to_canonical("xstest", records, text_fn=lambda r: bp.build_prompt("xstest", r))
        prompts = [bp.build_prompt("xstest", r) for r in records]
        labels = _labels_for("xstest", records)
        row_ids = np.array([r["row"] for r in records], dtype=np.int64)
        extra = {"types": np.array([r["type"] for r in records])}
        log(f"xstest: {len(prompts)} rows (canonical sample)")
        return prompts, labels, row_ids, extra

    # Explicit non-default --hf-dataset override: load that repo directly,
    # no canonical filter (the canonical ids were computed against the
    # default repo, so they wouldn't necessarily align with a different one).
    from datasets import load_dataset

    log(f"Loading XSTest ({hf_dataset}) [non-default --hf-dataset override] ...")
    ds = None
    last_err = None
    for split in ("train", "test"):
        try:
            ds = load_dataset(hf_dataset, split=split)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load XSTest from {hf_dataset}: {last_err}")
    prompts = [ex["prompt"].strip() for ex in ds]
    labels = np.array([ex["label"] == "unsafe" for ex in ds], dtype=bool)
    row_ids = np.arange(len(prompts), dtype=np.int64)
    extra = {"types": np.array([ex["type"] for ex in ds])}

    prompts, labels, row_ids, extra = _random_subsample(prompts, labels, row_ids, n, seed, extra)
    log(f"xstest: loaded {len(prompts)} prompts (n={n})")
    return prompts, labels, row_ids, extra


def _uses_default_wildjailbreak_sample(
    categories, n_per_category: int, full_dataset: bool,
) -> bool:
    return (
        not full_dataset
        and n_per_category == DEFAULT_N_PER_CATEGORY
        and (categories is None or tuple(categories) == WILDJAILBREAK_CATEGORIES)
    )


def _wildjailbreak_prompts(
    *, seed: int, categories: list[str] | None, n_per_category: int, full_dataset: bool,
) -> tuple[list, np.ndarray, np.ndarray, dict]:
    if _uses_default_wildjailbreak_sample(categories, n_per_category, full_dataset):
        # Canonical sample shared with the wrapped pipeline -- see shared_sampling.py.
        from shared_sampling import filter_to_canonical

        records = bp.load_records("wildjailbreak")  # already filtered to the 4 default categories
        records = filter_to_canonical(
            "wildjailbreak", records, text_fn=lambda r: bp.build_prompt("wildjailbreak", r),
        )
        prompts = [bp.build_prompt("wildjailbreak", r) for r in records]
        labels = _labels_for("wildjailbreak", records)
        row_ids = np.array([r["row"] for r in records], dtype=np.int64)
        data_types = np.array([r["type"] for r in records])
        source_prompt_ids = np.array([r["source_prompt_id"] for r in records])
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(prompts))
        prompts = [prompts[i] for i in order]
        labels, row_ids, data_types, source_prompt_ids = (
            labels[order], row_ids[order], data_types[order], source_prompt_ids[order]
        )
        log(f"wildjailbreak: {len(prompts)} rows (canonical sample)")
        return prompts, labels, row_ids, {
            "data_types": data_types,
            "source_prompt_ids": source_prompt_ids,
        }

    # Non-default categories/n_per_category/full_dataset: legacy dynamic sampling
    # over the raw HF dataset (not restricted to build_prompts' fixed 4-category filter).
    from datasets import load_dataset

    cats = list(categories) if categories else list(WILDJAILBREAK_CATEGORIES)
    log(f"Loading WildJailbreak ({WILDJAILBREAK_HF_REPO}) [non-default sampling] ...")
    ds = load_dataset(WILDJAILBREAK_HF_REPO, "train", delimiter="\t", keep_default_na=False)["train"]

    by_type: dict[str, list[int]] = {cat: [] for cat in cats}
    for i, ex in enumerate(ds):
        if ex["data_type"] in by_type:
            by_type[ex["data_type"]].append(i)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cat in cats:
        idxs = by_type[cat]
        if not idxs:
            continue
        n_take = len(idxs) if full_dataset else min(n_per_category, len(idxs))
        chosen = rng.choice(idxs, size=n_take, replace=False).tolist() if n_take < len(idxs) else idxs
        selected.extend(chosen)
    rng.shuffle(selected)

    prompts, labels, row_ids, data_types = [], [], [], []
    for i in selected:
        ex = ds[i]
        text = ex["vanilla"].strip() if ex["data_type"].startswith("vanilla_") else ex["adversarial"].strip()
        if not text:
            continue
        prompts.append(text)
        dt = ex["data_type"]
        labels.append(True if dt.endswith("_harmful") else False if dt.endswith("_benign") else bool(ex.get("label")))
        row_ids.append(i)
        data_types.append(dt)

    log(f"wildjailbreak: {len(prompts)} rows (custom categories={cats}, n_per_category={n_per_category}, full_dataset={full_dataset})")
    return (
        prompts,
        np.array(labels, dtype=bool),
        np.array(row_ids, dtype=np.int64),
        {"data_types": np.array(data_types)},
    )


def _pubmedqa_prompts(dataset: str, *, n: int | None, seed: int, balance: bool) -> tuple[list, np.ndarray, np.ndarray, dict]:
    # Both pubmedqa_artificial and pubmedqa_labeled (the only two keys in
    # _PUBMEDQA_CONFIG / dataset.startswith("pubmedqa") callers) always use the
    # canonical sample shared with the wrapped pipeline -- see shared_sampling.py.
    from shared_sampling import filter_to_canonical

    records = bp.load_records(dataset)  # build_prompts keys pubmedqa_artificial/pubmedqa_labeled separately
    records = filter_to_canonical(dataset, records, text_fn=lambda r: bp.build_prompt(dataset, r))
    prompts = [bp.build_prompt(dataset, r) for r in records]
    labels = _labels_for(dataset, records)
    row_ids = np.array([r["row"] for r in records], dtype=np.int64)
    log(f"{dataset}: {len(prompts)} rows (canonical sample)")
    return prompts, labels, row_ids, {}


def load_prompts(
    dataset: str,
    *,
    split: str = "all",
    n: int | None = None,
    seed: int = 0,
    balance: bool | None = None,
    hf_dataset: str | None = None,
    categories: list[str] | None = None,
    n_per_category: int | None = None,
    full_dataset: bool | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (prompts, labels, row_ids, extra_meta).

    `split` is accepted for CLI-interface compatibility but is a no-op:
    build_prompts' loaders always load every split relevant to a dataset
    (matching the old scripts' actual default usage, which was always
    --split all in practice).
    """
    if dataset not in UNWRAPPED_DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {UNWRAPPED_DATASETS}")
    if split != "all":
        log(f"WARNING: --split {split!r} has no effect for {dataset!r} in the consolidated loader (always loads all rows).")

    defaults = UNWRAPPED_DEFAULTS[dataset]
    n = defaults.n if n is None else n
    balance = defaults.balance if balance is None else balance

    if dataset == "social_iqa_binary":
        return _social_iqa_binary_prompts(n=n, seed=seed)
    if dataset == "xstest":
        return _xstest_prompts(n=n, seed=seed, hf_dataset=hf_dataset)
    if dataset == "wildjailbreak":
        return _wildjailbreak_prompts(
            seed=seed,
            categories=categories,
            n_per_category=defaults.n_per_category if n_per_category is None else n_per_category,
            full_dataset=defaults.full_dataset if full_dataset is None else full_dataset,
        )
    if dataset.startswith("pubmedqa"):
        return _pubmedqa_prompts(dataset, n=n, seed=seed, balance=balance)
    return _simple_dataset(dataset, n=n, seed=seed)
