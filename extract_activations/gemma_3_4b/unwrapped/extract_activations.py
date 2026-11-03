#!/usr/bin/env python3
"""Extract Gemma-3-4B-it activations from unwrapped prompts.

Runs shared_core/shared_extract_engine.py over one dataset: all layers, mean
and last-token pooling, greedy 10-token decoding, and SAE, crosscoder, and
transcoder decompositions. Also writes next-token concept logits to
{dataset}_tf_logits.npz. Prompts and labels come from
unwrapped_source_loaders.py.

Dataset notes:
- geometry_truth merges all 9 GoT parts.
- rm_bench produces 6 rows per example: 3 formats x chosen/rejected.
- social_iqa_binary keeps the correct answer and one random wrong answer,
  then shuffles position so each label holds 17,682 rows. One RNG seeded by
  --seed drives both steps.
"""
from __future__ import annotations

import argparse
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np

UNWRAP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = UNWRAP_ROOT.parents[1]
# Large .npz outputs default here (not under ~/rare_event_proj) — override with --outdir
SCRATCH_SUBDIR = "dataset_unwrap"


def default_scratch_root() -> Path:
    """$SCRATCH_ROOT/SCRATCH_SUBDIR. Required unless --outdir is given."""
    root = os.environ.get("SCRATCH_ROOT")
    if not root:
        raise SystemExit(
            "Set SCRATCH_ROOT to a writable scratch directory, or pass --outdir."
        )
    return Path(root) / SCRATCH_SUBDIR
ENGINE_PATH = REPO_ROOT / "shared_core" / "shared_extract_engine.py"

if str(UNWRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(UNWRAP_ROOT))

from unwrapped_source_loaders import UNWRAPPED_DATASETS, load_prompts  # noqa: E402

_EXTRA_NPZ: dict[str, object] = {}
_LOAD_KWARGS: dict[str, object] = {}
_LABEL_CONVENTIONS: dict[str, str] = {
    "geometry_truth": "True=statement is true (label 1), False=statement is false (label 0)",
    "harmful_harmless": "True=harmless, False=harmful",
    "xstest": "True=unsafe, False=safe",
    "social_iqa": "labels = correct answer index (1, 2, or 3) for Social IQa",
    "social_iqa_binary": "labels = correct answer index (1 or 2) after dropping one wrong choice",
    "wildjailbreak": "True=harmful, False=benign",
    "pubmedqa_artificial": 'labels[i] is final_decision string: "yes" or "no"',
    "pubmedqa_labeled": 'labels[i] is final_decision string: "yes" or "no"',
    "claude_anatomy": "True=factually correct, False=factually incorrect",
    "claude_blood_test_ranges": "True=factually correct, False=factually incorrect",
    "claude_vds": "True=factually correct, False=factually incorrect",
    "claude_vds_300": "True=factually correct, False=factually incorrect",
    "rm_bench": 'labels[i] is "chosen" or "rejected"',
}


def default_outdir(dataset: str, scratch_root: Path | None = None) -> Path:
    """`/scratch/.../dataset_unwrap/{dataset}_activations/`."""
    root = scratch_root or default_scratch_root()
    return root / f"{dataset}_activations"


def _load_engine():
    spec = importlib.util.spec_from_file_location("extract_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extraction engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks up sys.modules[cls.__module__] while the class body runs;
    # without this pre-registration that lookup is None and import crashes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_unwrapped_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Gemma-3-4B activations for unwrapped prompts.",
    )
    p.add_argument("--dataset", required=True, choices=UNWRAPPED_DATASETS,
                   help="Dataset to extract.")

    # Pass-through flags for the shared engine (same names/defaults as harmful_harmless script).
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--layers", nargs="*", default=["all"])
    p.add_argument("--pool", choices=["mean", "last", "both"], default="both")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--split", default="all", choices=["train", "test", "validation", "all"],
                   help="No-op for most datasets — see unwrapped_source_loaders.load_prompts docstring.")
    p.add_argument("--n", type=int, default=None,
                   help="Subsample cap (default: dataset-specific — see UNWRAPPED_DEFAULTS).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=0,
                   help="Max prompt tokens (0 = no truncation; matches old per-dataset defaults).")
    p.add_argument("--gen-tokens", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None)
    p.add_argument("--outdir", default=None,
                   help="Output dir for .npz files. Default: $SCRATCH_ROOT/dataset_unwrap/{dataset}_activations/")
    p.add_argument("--scratch-root", type=Path, default=None,
                   help="Base scratch folder for default outdir (default: $SCRATCH_ROOT/dataset_unwrap). Ignored if --outdir is set.")
    p.add_argument("--decomps", nargs="*", default=["sae", "crosscoder", "transcoder"],
                    choices=["sae", "transcoder", "crosscoder", "clt"])
    p.add_argument("--crosscoder-path", default=os.environ.get("CROSSCODER_PATH"))
    p.add_argument("--no-decomps", action="store_true")
    p.add_argument("--no-sae", action="store_true")
    p.add_argument("--allow-missing-decomps", action="store_true",
                   help="Warn and drop an explicitly-requested --decomps entry that "
                        "fails to load instead of failing the run.")
    p.add_argument("--save-tf-logits", action=argparse.BooleanOptionalAction, default=True,
                   help="Write {dataset}_tf_logits.npz. Default: on.")
    p.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True,
                   help="Wrap each prompt via tokenizer.apply_chat_template before tokenizing "
                        "(default: on -- gemma-3-4b-it is instruction-tuned and expects this).")

    # Dataset-family-specific overrides (no-ops for datasets they don't apply to).
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                   help="PubMedQA only: balanced yes/no subsample (default: dataset-specific).")
    p.add_argument("--hf-dataset", default=None,
                   help="XSTest only: override HF repo id (default: walledai/XSTest, falls back to Paul/XSTest).")
    p.add_argument("--categories", nargs="*", default=None,
                   help="WildJailbreak only: override the 4 default data_type categories.")
    p.add_argument("--n-per-category", type=int, default=None,
                   help="WildJailbreak only: rows per category (default 12500 -> 50k total canonical sample).")
    p.add_argument("--full-dataset", action=argparse.BooleanOptionalAction, default=None,
                   help="WildJailbreak only: use all rows in every category instead of --n-per-category.")

    return p.parse_args(argv)


def _patch_engine(engine):
    _orig_build_meta = engine.build_shared_meta

    def engine_load_prompts(split: str, n: int | None, seed: int):
        del split, n, seed  # consolidated loader reads from module-level _LOAD_KWARGS instead
        prompts, labels, row_ids, extra = load_prompts(**_LOAD_KWARGS)
        _EXTRA_NPZ.clear()
        _EXTRA_NPZ.update(extra)
        _EXTRA_NPZ["source_row_ids"] = np.asarray(row_ids, dtype=np.int64)
        convention = _LABEL_CONVENTIONS.get(_LOAD_KWARGS["dataset"])
        if convention:
            _EXTRA_NPZ["meta_label_convention"] = np.array(convention)
        return prompts, labels

    def build_shared_meta(args, prompts, labels, responses, response_ids, hidden_dim, decomps, tok, num_hidden_layers, **kw):
        meta = _orig_build_meta(
            args, prompts, labels, responses, response_ids, hidden_dim, decomps, tok, num_hidden_layers, **kw
        )
        meta.update(_EXTRA_NPZ)
        return meta

    engine.load_prompts = engine_load_prompts
    engine.build_shared_meta = build_shared_meta


def main(argv: list[str] | None = None) -> int:
    args = parse_unwrapped_args(argv)

    if args.no_decomps or args.no_sae:
        args.decomps = []

    if args.dataset_name is None:
        args.dataset_name = args.dataset

    if args.outdir is None:
        args.outdir = str(default_outdir(args.dataset, args.scratch_root))

    global _LOAD_KWARGS
    _LOAD_KWARGS = {
        "dataset": args.dataset,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "balance": args.balance,
        "hf_dataset": args.hf_dataset,
        "categories": args.categories,
        "n_per_category": args.n_per_category,
        "full_dataset": args.full_dataset,
    }

    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing extraction engine: {ENGINE_PATH}\n"
            "The canonical copy lives in shared_core/shared_extract_engine.py."
        )

    engine = _load_engine()
    _patch_engine(engine)
    engine.parse_args = lambda: args  # noqa: E731
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
