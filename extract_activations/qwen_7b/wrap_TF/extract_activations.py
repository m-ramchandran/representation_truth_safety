#!/usr/bin/env python3
"""Extract Qwen2.5-7B-Instruct activations for T/F-wrapped prompts.

Sibling of gemma_3_4b/wrap_TF/extract_activations.py, importing its
prompt-wrapping code directly so both models see identical wrapped prompts.

Differences from the Gemma wrapped pipeline:
- --model defaults to Qwen/Qwen2.5-7B-Instruct
- --pool defaults to "last" only
- No Gemma Scope decompositions, since they are trained on Gemma's
  activation space. Decomps is hardcoded off.
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import sys
from pathlib import Path

import numpy as np

QWEN_WRAP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QWEN_WRAP_ROOT.parents[1]
GEMMA_WRAP_ROOT = REPO_ROOT / "gemma_3_4b" / "wrap_TF"
SHARED_CORE = REPO_ROOT / "shared_core"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Separate scratch tree from the Gemma pipeline's dataset_wrap_TF/ -- different
# model, different activation shapes/content, shouldn't share a directory.
SCRATCH_SUBDIR = "dataset_wrap_TF_qwen7b"


def default_scratch_root() -> Path:
    """$SCRATCH_ROOT/SCRATCH_SUBDIR. Required unless --outdir is given."""
    root = os.environ.get("SCRATCH_ROOT")
    if not root:
        raise SystemExit(
            "Set SCRATCH_ROOT to a writable scratch directory, or pass --outdir."
        )
    return Path(root) / SCRATCH_SUBDIR
ENGINE_PATH = SHARED_CORE / "shared_extract_engine.py"

# Reuse gemma_3_4b/wrap_TF's T/F prompt-wrapping + dataset loading directly
# (not copied) -- see module docstring. Those modules resolve generated_data/
# and gemma_3_4b/unwrapped/build_prompts.py relative to their *own* file
# locations, so they stay correct even though this script lives elsewhere.
for _p in (GEMMA_WRAP_ROOT, SHARED_CORE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tf_extract_data import load_wrapped_for_extraction  # noqa: E402
from tf_source_loaders import ALL_DATASETS  # noqa: E402

_EXTRA_NPZ: dict[str, object] = {}
_LOAD_KWARGS: dict[str, object] = {}


def default_outdir(dataset: str, variant: str, scratch_root: Path | None = None) -> Path:
    """`/scratch/.../dataset_wrap_TF_qwen7b/{dataset}_activations/{variant}/`."""
    root = scratch_root or default_scratch_root()
    return root / f"{dataset}_activations" / variant


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


def parse_tf_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Qwen2.5-7B-Instruct activations for T/F-wrapped prompts.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=ALL_DATASETS,
        help="Dataset to wrap and extract.",
    )
    p.add_argument(
        "--variant",
        default="A",
        choices=["A", "B"],
        help=(
            "Wrapper variant (A = outer wrapper on every dataset; "
            "B = omit wrapper on pubmedqa/safety/social_iqa, keep on geometry/Claude)."
        ),
    )
    p.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Balanced label_tf subsample. Ignored for pubmedqa_artificial, "
            "pubmedqa_labeled, and xstest (all three always use their canonical "
            "shared_sampling.py sample, already balanced)."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Subsample cap after dataset-specific filters (see tf_extract_data.DEFAULTS).",
    )
    p.add_argument(
        "--full-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "WildJailbreak only: use all rows in every category "
            "(otherwise --n-per-category). "
            "Does not change pubmedqa_artificial: it always uses the canonical "
            "shared_sampling.py sample (30,250 questions -> 60,500 expanded rows), "
            "ignoring --balance/--n entirely (see tf_extract_data._select_indices)."
        ),
    )
    p.add_argument(
        "--n-per-category",
        type=int,
        default=None,
        help="WildJailbreak: rows per data_type (default 12500 -> 50k total).",
    )

    # Pass-through flags for the shared engine (same names as the Gemma script).
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layers", nargs="*", default=["all"])
    p.add_argument("--pool", choices=["mean", "last", "both"], default="last",
                   help="Default: last only (no mean-pooled activations in this pipeline).")
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="Max prompt tokens (0 = no truncation; recommended for wrapped PubMedQA).",
    )
    p.add_argument("--gen-tokens", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None)
    p.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output dir for .npz files. Default: "
            "$SCRATCH_ROOT/dataset_wrap_TF_qwen7b/{dataset}_activations/{variant}/"
        ),
    )
    p.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help=(
            "Base scratch folder for default outdir "
            "(default: $SCRATCH_ROOT/dataset_wrap_TF_qwen7b). Ignored if --outdir is set."
        ),
    )
    p.add_argument(
        "--save-tf-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write {dataset}_tf_logits.npz (T/F, Yes/No, Safe/Unsafe, Harmless/Harmful logits). Default: on.",
    )
    p.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap each prompt via tokenizer.apply_chat_template before tokenizing "
             "(default: on -- Qwen2.5-7B-Instruct is instruction-tuned and expects this).",
    )
    args = p.parse_args(argv)

    # Gemma Scope decompositions (SAE/crosscoder/transcoder) are trained on
    # Gemma's activation space specifically -- not exposed as flags here,
    # always off for this pipeline (same as qwen_7b/unwrapped/).
    args.decomps = []
    args.no_decomps = True
    args.no_sae = True
    args.crosscoder_path = os.environ.get("CROSSCODER_PATH")
    return args


def _patch_engine(engine):
    _orig_build_meta = engine.build_shared_meta

    def load_prompts(split: str, n: int | None, seed: int):
        del split, n  # TF loader uses module-level _LOAD_KWARGS
        prompts, labels, extras = load_wrapped_for_extraction(**_LOAD_KWARGS)
        _EXTRA_NPZ.clear()
        _EXTRA_NPZ.update(extras)
        return prompts, labels

    def build_shared_meta(args, prompts, labels, responses, response_ids, hidden_dim, decomps, tok, num_hidden_layers, **kw):
        meta = _orig_build_meta(
            args, prompts, labels, responses, response_ids, hidden_dim, decomps, tok, num_hidden_layers, **kw
        )
        meta.update(_EXTRA_NPZ)
        meta["meta_tf_dataset"] = np.array(_LOAD_KWARGS["dataset"])
        meta["meta_prompt_format"] = np.array("T/F wrapped prompt_wrapped text")
        return meta

    engine.load_prompts = load_prompts
    engine.build_shared_meta = build_shared_meta


def main(argv: list[str] | None = None) -> int:
    args = parse_tf_args(argv)

    if args.dataset_name is None:
        args.dataset_name = f"{args.dataset}_tf_{args.variant.lower()}"

    if args.outdir is None:
        args.outdir = str(
            default_outdir(args.dataset, args.variant, args.scratch_root)
        )

    global _LOAD_KWARGS
    _LOAD_KWARGS = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "balance": args.balance,
        "n": args.n,
        "full_dataset": args.full_dataset,
        "n_per_category": args.n_per_category,
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
