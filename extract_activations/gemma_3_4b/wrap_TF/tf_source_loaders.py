"""Load upstream dataset rows for T/F wrapper conversion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CLAUDE_DATA_ROOT = Path(__file__).resolve().parents[2] / "generated_data"
CLAUDE_DATASET_FILES: dict[str, str] = {
    "claude_anatomy": "anatomy_true_false.csv",
    "claude_blood_test_ranges": "blood_test_ranges_true_false.csv",
    "claude_vds": "VDS_TF_datasets.csv",
    "claude_vds_300": "VDS_TF_datasets_300.csv",
}
CLAUDE_DATASETS = frozenset(CLAUDE_DATASET_FILES)

GOT_BASE = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets"
GOT_PARTS = [
    "cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans",
    "larger_than", "smaller_than", "companies_true_false",
    "common_claim_true_false", "counterfact_true_false",
]

WILDMJAILBREAK_CATEGORIES = (
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
)

RM_BENCH_FORMATS = ("concise", "detailed_plain", "detailed_markdown")

ALL_DATASETS = [
    "geometry_truth",
    "harmful_harmless",
    "xstest",
    "pubmedqa_artificial",
    "pubmedqa_labeled",
    "social_iqa",
    "social_iqa_binary",
    "wildjailbreak",
    *sorted(CLAUDE_DATASET_FILES),
    "rm_bench",
]


def _reservoir_add(rng, bucket: list, item, k: int, seen: int) -> None:
    if len(bucket) < k:
        bucket.append(item)
    else:
        j = int(rng.integers(0, seen))
        if j < k:
            bucket[j] = item


def _parse_claude_label(val) -> bool:
    """CSV labels may be 0/1 (anatomy) or True/False strings/bools (VDS)."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val) == 1
    if isinstance(val, float) and not np.isnan(val):
        return int(val) == 1
    s = str(val).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    raise ValueError(f"Unrecognized Claude CSV label: {val!r}")


def load_claude_records(dataset: str) -> list[dict]:
    import pandas as pd

    if dataset not in CLAUDE_DATASET_FILES:
        raise KeyError(f"not a Claude CSV dataset: {dataset!r}")
    path = CLAUDE_DATA_ROOT / CLAUDE_DATASET_FILES[dataset]
    if not path.is_file():
        raise FileNotFoundError(f"Missing Claude data CSV: {path}")

    df = pd.read_csv(path)
    records: list[dict] = []
    for row_idx, row in df.iterrows():
        statement = str(row["statement"]).strip()
        if not statement:
            continue
        if not statement.endswith((".", "?")):
            statement += "."
        rec: dict = {
            "row": int(row_idx),
            "statement": statement,
            "label_raw": _parse_claude_label(row["label"]),
        }
        for col in row.index:
            if col in ("statement", "label"):
                continue
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            rec[col] = val.item() if hasattr(val, "item") else val
        records.append(rec)
    return records


def load_geometry_truth_records() -> list[dict]:
    import pandas as pd

    records: list[dict] = []
    for part in GOT_PARTS:
        df = pd.read_csv(f"{GOT_BASE}/{part}.csv")
        for row_idx, row in df.iterrows():
            statement = str(row["statement"]).strip()
            if not statement:
                continue
            if not statement.endswith((".", "?")):
                statement += "."
            records.append({
                "row": int(row_idx),
                "topic": part,
                "statement": statement,
                "label_raw": bool(int(row["label"]) == 1),
            })
    return records


def load_harmful_harmless_records() -> list[dict]:
    from datasets import concatenate_datasets, load_dataset

    ds = concatenate_datasets([
        load_dataset("justinphan3110/harmful_harmless_instructions", split="train"),
        load_dataset("justinphan3110/harmful_harmless_instructions", split="test"),
    ])
    records: list[dict] = []
    row = 0
    for ex in ds:
        sents, labs = ex["sentence"], ex["label"]
        if isinstance(sents, str):
            sents, labs = [sents], [labs]
        for s, lab in zip(sents, labs):
            text = s.strip() if isinstance(s, str) else str(s).strip()
            if text:
                records.append({
                    "row": row,
                    "request": text,
                    "label_raw": bool(lab),
                })
                row += 1
    return records


def load_xstest_records() -> list[dict]:
    from datasets import load_dataset

    last_err = None
    for repo in ("walledai/XSTest", "Paul/XSTest"):
        for split in ("train", "test"):
            try:
                ds = load_dataset(repo, split=split)
                return [
                    {
                        "row": i,
                        "request": ex["prompt"].strip(),
                        "label_raw": ex["label"],
                        "type": ex["type"],
                    }
                    for i, ex in enumerate(ds)
                ]
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Could not load XSTest: {last_err}")


def load_pubmedqa_records(config: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", config, split="train")
    records: list[dict] = []
    for row_idx, ex in enumerate(ds):
        decision = str(ex["final_decision"]).strip().lower()
        if decision not in ("yes", "no"):
            continue
        context = " ".join(ex["context"]["contexts"])
        question = ex["question"].strip()
        if not question:
            continue
        records.append({
            "row": row_idx,
            "context": context,
            "question": question,
            "label_raw": decision,
        })
    return records


def load_social_iqa_records() -> list[dict]:
    from datasets import concatenate_datasets, load_dataset

    repo, rev = "allenai/social_i_qa", "refs/convert/parquet"
    ds = concatenate_datasets([
        load_dataset(repo, revision=rev, split="train"),
        load_dataset(repo, revision=rev, split="validation"),
    ])
    return [
        {
            "row": i,
            "context": ex["context"].strip(),
            "question": ex["question"].strip(),
            "answerA": ex["answerA"].strip(),
            "answerB": ex["answerB"].strip(),
            "answerC": ex["answerC"].strip(),
            "label_raw": int(ex["label"]),
        }
        for i, ex in enumerate(ds)
    ]


def _wildjailbreak_request(ex: dict) -> str:
    dt = ex["data_type"]
    if dt.startswith("vanilla_"):
        return ex["vanilla"].strip()
    return ex["adversarial"].strip()


def _wildjailbreak_harmful(data_type: str, label) -> bool:
    if data_type.endswith("_harmful"):
        return True
    if data_type.endswith("_benign"):
        return False
    if label is not None:
        return bool(label)
    raise ValueError(f"Unknown data_type: {data_type!r}")


def load_wildjailbreak_records() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False
    )["train"]
    records: list[dict] = []
    for row_idx, ex in enumerate(ds):
        data_type = ex["data_type"]
        if data_type not in WILDMJAILBREAK_CATEGORIES:
            continue
        text = _wildjailbreak_request(ex)
        if not text:
            continue
        # ex["vanilla"] is a row's own text for vanilla_* rows, and the
        # *source* text it was rewritten from for adversarial_* rows -- so
        # hashing it gives a vanilla row and every rewrite of it the same id
        # (see shared_sampling.py's _source_prompt_id / select_wildjailbreak_ids).
        vanilla_text = ex["vanilla"].strip()
        records.append({
            "row": row_idx,
            "request": text,
            "label_raw": _wildjailbreak_harmful(data_type, ex.get("label")),
            "type": data_type,
            "source_prompt_id": hashlib.sha1(vanilla_text.encode()).hexdigest()[:12],
        })
    return records


def load_rm_bench_records() -> list[dict]:
    """Binarize each RM-Bench example into 6 records: one per (format,
    chosen/rejected), all sharing an `rm_bench_id` so the 3 pairs can be
    regrouped downstream.

    Field names match unwrapped/build_prompts.load_rm_bench_records exactly.
    """
    from datasets import load_dataset

    ds = load_dataset("THU-KEG/RM-Bench", split="train")
    records: list[dict] = []
    row = 0
    for ex in ds:
        prompt = ex["prompt"].strip()
        for fmt, chosen_text, rejected_text in zip(RM_BENCH_FORMATS, ex["chosen"], ex["rejected"]):
            for response, label in ((chosen_text, "chosen"), (rejected_text, "rejected")):
                response = response.strip()
                if not response:
                    continue
                records.append({
                    "row": row,
                    "prompt": prompt,
                    "response": response,
                    "label_raw": label,
                    "rm_bench_id": ex["id"],
                    "domain": ex["domain"],
                    "format": fmt,
                })
                row += 1
    return records


LOADERS = {
    "geometry_truth": load_geometry_truth_records,
    "harmful_harmless": load_harmful_harmless_records,
    "xstest": load_xstest_records,
    "pubmedqa_artificial": lambda: load_pubmedqa_records("pqa_artificial"),
    "pubmedqa_labeled": lambda: load_pubmedqa_records("pqa_labeled"),
    "social_iqa": load_social_iqa_records,
    # social_iqa_binary's actual expansion (tf_wrappers._expand_records)
    # ignores this raw load and pulls the binarization decision straight from
    # unwrapped_source_loaders.social_iqa_binary_selections instead -- this
    # entry only exists so load_records("social_iqa_binary") doesn't KeyError.
    "social_iqa_binary": load_social_iqa_records,
    "wildjailbreak": load_wildjailbreak_records,
    "rm_bench": load_rm_bench_records,
    **{name: (lambda d=name: load_claude_records(d)) for name in CLAUDE_DATASET_FILES},
}


def load_records(dataset: str) -> list[dict]:
    if dataset not in LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {ALL_DATASETS}")
    return LOADERS[dataset]()


def reservoir_sample_records(
    records: list[dict],
    *,
    group_key: str,
    n_per_group: int,
    seed: int = 0,
) -> list[dict]:
    """Reservoir-sample up to n_per_group rows per group value."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}

    for rec in records:
        grp = str(rec[group_key])
        seen[grp] = seen.get(grp, 0) + 1
        buckets.setdefault(grp, [])
        _reservoir_add(rng, buckets[grp], rec, n_per_group, seen[grp])

    out: list[dict] = []
    for grp in sorted(buckets):
        out.extend(buckets[grp])
    return out
