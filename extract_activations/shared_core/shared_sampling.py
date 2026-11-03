"""Canonical row-ID selection for pubmedqa_artificial, pubmedqa_labeled,
xstest, and wildjailbreak.

Single source of truth for which rows these four datasets contribute. Samples
once from the raw HF dataset and writes the selected positional row indices to
JSON under shared_sample_ids/. Both the unwrapped and wrapped pipelines read
that file via filter_to_canonical() instead of resampling, so the two extract
the same underlying rows.

Each saved row also carries a fingerprint of its own text. filter_to_canonical()
recomputes that fingerprint and raises on a mismatch, so a change in HF row
order is caught rather than silently accepted.

Run this module directly to regenerate the JSON for a different N or seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UNWRAPPED_ROOT = _REPO_ROOT / "gemma_3_4b" / "unwrapped"
if str(_UNWRAPPED_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNWRAPPED_ROOT))
import build_prompts as _unwrapped_build_prompts  # noqa: E402

SAMPLE_IDS_DIR = Path(__file__).resolve().parent / "shared_sample_ids"

PUBMEDQA_HF_REPO = "qiaojin/PubMedQA"
PUBMEDQA_ARTIFICIAL_CONFIG = "pqa_artificial"
PUBMEDQA_ARTIFICIAL_SEED = 0
PUBMEDQA_ARTIFICIAL_N = 30_250  # balanced 15,125 yes + 15,125 no (matches prior unwrapped default)

XSTEST_HF_REPO = "walledai/XSTest"
XSTEST_HF_FALLBACK = "Paul/XSTest"
XSTEST_SEED = 0

PUBMEDQA_LABELED_CONFIG = "pqa_labeled"
PUBMEDQA_LABELED_SEED = 0

WILDJAILBREAK_HF_REPO = "allenai/wildjailbreak"
WILDJAILBREAK_CONFIG = "train"
WILDJAILBREAK_SEED = 0
WILDJAILBREAK_N_PER_CATEGORY = 12_500
WILDJAILBREAK_CATEGORIES = (
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
)


def _ids_path(name: str) -> Path:
    return SAMPLE_IDS_DIR / f"{name}_row_ids.json"


def _row_fingerprint(text: str) -> str:
    """Short content hash of a row's own identifying text (the same text the
    downstream pipeline builds its prompt from), used by filter_to_canonical
    to detect a positional row id silently pointing at a different example
    than the one selected here (dataset drift, wrong fallback repo, etc.)."""
    return hashlib.sha1(text.strip().encode()).hexdigest()[:16]


def _best_effort_revision(repo_id: str, repo_type: str = "dataset") -> str | None:
    """Current commit sha for `repo_id`, or None if unreachable/unavailable.

    Advisory only -- see module docstring's "What's NOT covered" section.
    Never raises: a revision-lookup failure shouldn't block sample
    generation, which already needs network access to load_dataset anyway.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id) if repo_type == "dataset" else None
        return info.sha if info is not None else None
    except Exception:
        return None


def _balance_by_min_class(by_label: dict[str, list[int]], seed: int) -> tuple[list[int], int]:
    """Keep all of the minority label's rows + an equal-size random (without
    replacement) draw from every other label -- same method as
    representation_ai_safety/1B/load_datasets.py's `_subsample_data` when
    n >= 2 * minority (its default for xstest). Returns (sorted row ids, per_class)."""
    rng = np.random.default_rng(seed)
    per_class = min(len(v) for v in by_label.values())
    chosen: list[int] = []
    for label in sorted(by_label):
        idxs = by_label[label]
        if len(idxs) <= per_class:
            chosen.extend(idxs)
        else:
            chosen.extend(rng.choice(idxs, size=per_class, replace=False).tolist())
    chosen.sort()
    return chosen, per_class


def select_pubmedqa_artificial_ids(
    seed: int = PUBMEDQA_ARTIFICIAL_SEED,
    n: int = PUBMEDQA_ARTIFICIAL_N,
) -> dict:
    """Balanced sample of raw dataset row indices, by true final_decision."""
    from datasets import load_dataset

    ds = load_dataset(PUBMEDQA_HF_REPO, PUBMEDQA_ARTIFICIAL_CONFIG, split="train")
    by_label: dict[str, list[int]] = {}
    for row_idx, ex in enumerate(ds):
        decision = str(ex["final_decision"]).strip().lower()
        if decision not in ("yes", "no"):
            continue
        if not ex["question"].strip():
            # Matches build_prompts.py's load_pubmedqa_records, which also
            # drops empty-question rows -- keeps this selector's eligible set
            # identical to what the downstream loaders will actually produce.
            continue
        by_label.setdefault(decision, []).append(row_idx)

    rng = np.random.default_rng(seed)
    per_class = n // 2
    minority = min(len(v) for v in by_label.values())
    if per_class > minority:
        per_class = minority

    chosen: list[int] = []
    for label in sorted(by_label):  # deterministic: "no" before "yes"
        idxs = by_label[label]
        if len(idxs) <= per_class:
            chosen.extend(idxs)
        else:
            chosen.extend(rng.choice(idxs, size=per_class, replace=False).tolist())
    chosen.sort()

    # Fingerprint text (full context + question) only for the ~n rows actually
    # chosen, not all ~211k eligible rows -- pqa_artificial's context field is
    # large enough that eagerly building this for every row risked real memory
    # pressure. HF Datasets supports cheap single-row random access here.
    row_fingerprints = {}
    for row_idx in chosen:
        ex = ds[row_idx]
        context = " ".join(ex["context"]["contexts"])
        text = _unwrapped_build_prompts.build_prompt(
            "pubmedqa_artificial", {"context": context, "question": ex["question"].strip()},
        )
        row_fingerprints[str(row_idx)] = _row_fingerprint(text)

    return {
        "dataset": "pubmedqa_artificial",
        "hf_repo": PUBMEDQA_HF_REPO,
        "hf_config": PUBMEDQA_ARTIFICIAL_CONFIG,
        "hf_split": "train",
        "hf_revision": _best_effort_revision(PUBMEDQA_HF_REPO),
        "seed": seed,
        "requested_n": n,
        "per_class": per_class,
        "label_counts": {label: len(idxs) for label, idxs in by_label.items()},
        "row_ids": chosen,
        "row_fingerprints": row_fingerprints,
    }


def select_xstest_ids(seed: int = XSTEST_SEED) -> dict:
    """Balanced sample of raw dataset row indices, by safe/unsafe label."""
    from datasets import load_dataset

    ds = None
    used_repo = used_split = None
    last_err = None
    for repo in (XSTEST_HF_REPO, XSTEST_HF_FALLBACK):
        for split in ("train", "test"):
            try:
                ds = load_dataset(repo, split=split)
                used_repo, used_split = repo, split
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if ds is not None:
            break
    if ds is None:
        raise RuntimeError(f"Could not load XSTest: {last_err}")

    by_label: dict[str, list[int]] = {}
    row_text: dict[int, str] = {}
    for row_idx, ex in enumerate(ds):
        label = str(ex["label"]).strip().lower()
        by_label.setdefault(label, []).append(row_idx)
        # Same text build_prompts.py's build_prompt("xstest", ...) produces.
        row_text[row_idx] = _unwrapped_build_prompts.build_prompt(
            "xstest", {"request": ex["prompt"].strip()},
        )

    chosen, per_class = _balance_by_min_class(by_label, seed)

    return {
        "dataset": "xstest",
        "hf_repo": used_repo,
        "hf_split": used_split,
        "hf_revision": _best_effort_revision(used_repo),
        "seed": seed,
        "per_class": per_class,
        "label_counts": {label: len(idxs) for label, idxs in by_label.items()},
        "row_ids": chosen,
        "row_fingerprints": {str(r): _row_fingerprint(row_text[r]) for r in chosen},
    }


def select_pubmedqa_labeled_ids(seed: int = PUBMEDQA_LABELED_SEED) -> dict:
    """Balanced sample of raw dataset row indices, by true final_decision (yes/no)."""
    from datasets import load_dataset

    ds = load_dataset(PUBMEDQA_HF_REPO, PUBMEDQA_LABELED_CONFIG, split="train")
    by_label: dict[str, list[int]] = {}
    for row_idx, ex in enumerate(ds):
        decision = str(ex["final_decision"]).strip().lower()
        if decision not in ("yes", "no"):
            continue
        if not ex["question"].strip():
            continue  # matches build_prompts.py's load_pubmedqa_records
        by_label.setdefault(decision, []).append(row_idx)

    chosen, per_class = _balance_by_min_class(by_label, seed)

    # See select_pubmedqa_artificial_ids: fingerprint only the chosen rows.
    row_fingerprints = {}
    for row_idx in chosen:
        ex = ds[row_idx]
        context = " ".join(ex["context"]["contexts"])
        text = _unwrapped_build_prompts.build_prompt(
            "pubmedqa_labeled", {"context": context, "question": ex["question"].strip()},
        )
        row_fingerprints[str(row_idx)] = _row_fingerprint(text)

    return {
        "dataset": "pubmedqa_labeled",
        "hf_repo": PUBMEDQA_HF_REPO,
        "hf_config": PUBMEDQA_LABELED_CONFIG,
        "hf_split": "train",
        "hf_revision": _best_effort_revision(PUBMEDQA_HF_REPO),
        "seed": seed,
        "per_class": per_class,
        "label_counts": {label: len(idxs) for label, idxs in by_label.items()},
        "row_ids": chosen,
        "row_fingerprints": row_fingerprints,
    }


def _wildjailbreak_prompt_text(ex: dict) -> str:
    data_type = ex["data_type"]
    if data_type.startswith("vanilla_"):
        return ex["vanilla"].strip()
    return ex["adversarial"].strip()


def _source_prompt_id(vanilla_text: str) -> str:
    """Deterministic id shared by a vanilla row and every adversarial row
    rewritten from it: both carry the identical stripped text in their own
    `vanilla` field. Stateless, needs no lookup.
    """
    return hashlib.sha1(vanilla_text.strip().encode()).hexdigest()[:12]


def select_wildjailbreak_ids(
    seed: int = WILDJAILBREAK_SEED,
    n_per_category: int = WILDJAILBREAK_N_PER_CATEGORY,
) -> dict:
    """Sample matched vanilla/adversarial pairs for harmful and benign prompts.

    For each label (harmful/benign), sample up to n_per_category vanilla prompts with
    rewrites and select exactly one rewrite per prompt, using seed.
    Pairs are matched by their stripped `vanilla` field.

    Defaults yield 12,500 rows per category (50,000 total).
    Returned source_prompt_id values support inspection; downstream loaders
    recompute them from raw records.
    """
    from datasets import load_dataset

    ds = load_dataset(
        WILDJAILBREAK_HF_REPO, WILDJAILBREAK_CONFIG,
        delimiter="\t", keep_default_na=False,
    )["train"]

    polarities = ("harmful", "benign")
    vanilla_row_by_text: dict[str, dict[str, int]] = {p: {} for p in polarities}
    adversarial_rows_by_vanilla_text: dict[str, dict[str, list[int]]] = {p: {} for p in polarities}
    row_vanilla_field: dict[int, str] = {}  # row_idx -> its own vanilla field text, every row seen below

    for row_idx, ex in enumerate(ds):
        dt = ex["data_type"]
        if dt.startswith("vanilla_"):
            polarity = dt[len("vanilla_"):]
            if polarity not in polarities:
                continue
            text = ex["vanilla"].strip()
            if not text:
                continue
            vanilla_row_by_text[polarity][text] = row_idx
            row_vanilla_field[row_idx] = text
        elif dt.startswith("adversarial_"):
            polarity = dt[len("adversarial_"):]
            if polarity not in polarities:
                continue
            source_text = ex["vanilla"].strip()
            adversarial_text = ex["adversarial"].strip()
            if not source_text or not adversarial_text:
                continue
            adversarial_rows_by_vanilla_text[polarity].setdefault(source_text, []).append(row_idx)
            row_vanilla_field[row_idx] = source_text

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    category_counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    for polarity in polarities:
        v_by_text = vanilla_row_by_text[polarity]
        a_by_text = adversarial_rows_by_vanilla_text[polarity]
        eligible_texts = sorted(t for t in v_by_text if t in a_by_text)
        eligible_counts[polarity] = len(eligible_texts)
        n_take = min(n_per_category, len(eligible_texts))
        sample_pos = rng.choice(len(eligible_texts), size=n_take, replace=False)
        sampled_texts = [eligible_texts[i] for i in sample_pos]

        vanilla_ids = [v_by_text[t] for t in sampled_texts]
        adversarial_ids = [int(rng.choice(a_by_text[t])) for t in sampled_texts]
        chosen.extend(vanilla_ids)
        chosen.extend(adversarial_ids)
        category_counts[f"vanilla_{polarity}"] = len(vanilla_ids)
        category_counts[f"adversarial_{polarity}"] = len(adversarial_ids)
    chosen.sort()

    source_prompt_ids = {
        str(row_idx): _source_prompt_id(row_vanilla_field[row_idx]) for row_idx in chosen
    }

    # Fingerprint text (own request text -- vanilla or adversarial) only for
    # the ~50k chosen rows via single-row random access, not by holding every
    # one of the ~262k eligible rows' text in memory at once (adversarial
    # rewrites can be long; see select_pubmedqa_artificial_ids for the same
    # pattern and why).
    row_fingerprints = {
        str(row_idx): _row_fingerprint(_wildjailbreak_prompt_text(ds[row_idx]))
        for row_idx in chosen
    }

    return {
        "dataset": "wildjailbreak",
        "hf_repo": WILDJAILBREAK_HF_REPO,
        "hf_config": WILDJAILBREAK_CONFIG,
        "hf_revision": _best_effort_revision(WILDJAILBREAK_HF_REPO),
        "seed": seed,
        "n_per_category": n_per_category,
        "categories": list(WILDJAILBREAK_CATEGORIES),
        "category_counts": category_counts,
        "eligible_vanilla_with_rewrite": {
            f"vanilla_{p}": n for p, n in eligible_counts.items()
        },
        "row_ids": chosen,
        "source_prompt_ids": source_prompt_ids,
        "row_fingerprints": row_fingerprints,
    }


SELECTORS = {
    "pubmedqa_artificial": select_pubmedqa_artificial_ids,
    "pubmedqa_labeled": select_pubmedqa_labeled_ids,
    "xstest": select_xstest_ids,
    "wildjailbreak": select_wildjailbreak_ids,
}


def save_ids(payload: dict) -> Path:
    SAMPLE_IDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _ids_path(payload["dataset"])
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _load_payload(name: str) -> dict:
    path = _ids_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"No shared sample IDs at {path}. Generate with: "
            f"python {Path(__file__).name} --dataset {name}"
        )
    with open(path) as f:
        return json.load(f)


def load_ids(name: str) -> set[int]:
    """Return the bare canonical set of raw HF split row indices for `name`,
    with no content/repo/split verification. Prefer filter_to_canonical()
    for actually loading pipeline rows -- this is for callers that only need
    the id set itself (e.g. a quick count)."""
    return set(_load_payload(name)["row_ids"])


def filter_to_canonical(
    dataset: str,
    records: list[dict],
    text_fn: Callable[[dict], str],
    row_key: str = "row",
) -> list[dict]:
    """Filter `records` to the canonical sample for `dataset`, verifying content.

    `text_fn(record) -> str` must reproduce the exact text the selector hashed
    into row_fingerprints, or every row mismatches. All four selectors
    fingerprint via build_prompts.build_prompt(); the wrapped side can pass its
    already-identical "original_prompt" field.

    Raises ValueError if a canonical row id is missing or its fingerprint
    differs, meaning the loaded dataset has drifted from the one that generated
    the JSON.
    """
    payload = _load_payload(dataset)
    canonical_ids = set(payload["row_ids"])
    fingerprints: dict[str, str] = payload.get("row_fingerprints", {})

    kept: list[dict] = []
    mismatched: list[int] = []
    for r in records:
        row = r.get(row_key)
        if row not in canonical_ids:
            continue
        expected_fp = fingerprints.get(str(row))
        if expected_fp is not None and _row_fingerprint(text_fn(r)) != expected_fp:
            mismatched.append(row)
            continue
        kept.append(r)

    if mismatched:
        raise ValueError(
            f"{dataset}: {len(mismatched)} canonical row id(s) found at the expected "
            f"position but with different content than shared_sampling.py selected "
            f"(e.g. rows {sorted(mismatched)[:5]}) -- the loaded dataset has likely "
            f"drifted (wrong fallback repo, revision change, reprocessing) from the one "
            f"used to generate shared_sample_ids/{dataset}_row_ids.json. Regenerate the "
            "canonical sample, or investigate the dataset source, before proceeding."
        )

    seen_rows = {r[row_key] for r in kept}
    missing = canonical_ids - seen_rows
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} canonical row id(s) not found in loaded records "
            f"(e.g. {sorted(missing)[:5]}) -- dataset drift since shared_sampling.py was "
            f"run? Regenerate shared_sample_ids/{dataset}_row_ids.json, or investigate "
            "before proceeding."
        )

    # Resolved hf_repo/hf_split aren't threaded through `records` today, so they
    # can't be cross-checked against payload["hf_repo"]/["hf_split"] directly here
    # -- but a repo/split divergence (e.g. XSTest falling back to Paul/XSTest)
    # almost always changes row text too, which the fingerprint check above does
    # catch in practice.
    return kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[*SELECTORS, "all"], default="all")
    args = p.parse_args()
    names = list(SELECTORS) if args.dataset == "all" else [args.dataset]
    for name in names:
        payload = SELECTORS[name]()
        path = save_ids(payload)
        print(f"{name}: {len(payload['row_ids'])} row ids -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
