#!/usr/bin/env python3
"""Backfill {dataset}_tf_logits.npz for unwrapped activation trees.

Reads saved prompts and meta_dataset from any layer .npz and tokenizes them the
same way the extract scripts do: plain text, left-padded batches.

Stores next-token logits for four concept pairs in one forward pass: True/False,
Yes/No, Safe/Unsafe, Harmless/Harmful. Same schema as wrapped runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

SHARED_CORE = Path(__file__).resolve().parents[2] / "shared_core"
if str(SHARED_CORE) not in sys.path:
    sys.path.insert(0, str(SHARED_CORE))

from tf_next_token_logits import (  # noqa: E402
    META_SCORING,
    concept_logit_chunk_keys,
    concept_meta_token_id_arrays,
    concept_token_tensors,
    score_last_position_concept_logits,
    stacks_to_numpy,
)


def _pick_npz(outdir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(outdir, "*_mean_layer1.npz")))
    if not candidates:
        candidates = sorted(
            p
            for p in glob.glob(os.path.join(outdir, "*.npz"))
            if not p.endswith("_tf_logits.npz")
        )
    if not candidates:
        raise FileNotFoundError(f"No .npz files under {outdir}")
    return candidates[0]


def tf_logits_output_path(outdir: str, dataset_name: str) -> str:
    return os.path.join(outdir, f"{dataset_name}_tf_logits.npz")


def _meta_scalar(data, key: str):
    """Read a scalar meta_* field saved as np.array(value); None if absent."""
    if key not in data.files:
        return None
    return np.asarray(data[key]).item()



_EXCLUDE_SOURCE_KEYS = {
    "activations", "meta_pool", "meta_hf_layer", "meta_scope_layer",
    "meta_last_layer_used_raw_hook", "meta_feature_set",
    "prompts", "labels", "meta_dataset", "meta_model",
    "meta_chat_template", "meta_add_special_tokens", "meta_max_len",
    "meta_truncation_side",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score multi-concept next-token logits for unwrapped activation dirs.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", help="Any layer .npz with prompts + meta_dataset.")
    g.add_argument("--outdir", help="Directory containing activation .npz files.")
    p.add_argument(
        "--model", default=None,
        help="Default: read from the source .npz's meta_model (falls back to "
             "google/gemma-3-4b-it if that metadata is absent). Passing this "
             "explicitly only overrides when it matches meta_model -- see "
             "--allow-metadata-mismatch.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-len", type=int, default=None,
        help="Default: read from the source .npz's meta_max_len (falls back to 0 "
             "= no truncation if absent).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing tf_logits file.")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if CUDA is unavailable (use for sbatch GPU jobs).",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback (slow; for tiny tests only).",
    )
    p.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wrap each prompt via tokenizer.apply_chat_template before tokenizing. "
             "Default: read from the source .npz's meta_chat_template (falls back "
             "to on if absent) -- must match how the activations being scored were "
             "extracted.",
    )
    p.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Allow an explicitly-passed --model/--chat-template/--max-len to "
             "differ from the source activation file's saved metadata instead of "
             "failing. Only for deliberately rescoring under different settings.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.npz:
        src = args.npz
        outdir = os.path.dirname(src)
    else:
        outdir = args.outdir
        src = _pick_npz(outdir)

    data = np.load(src, allow_pickle=True)
    prompts = [str(p) for p in data["prompts"]]
    labels = np.asarray(data["labels"])
    dataset_name = str(np.asarray(data["meta_dataset"]).item())
    out_path = tf_logits_output_path(outdir, dataset_name)

    if os.path.exists(out_path) and not args.force:
        print(f"Exists (use --force): {out_path}")
        return 0


    meta_model_saved = _meta_scalar(data, "meta_model")
    meta_chat_template_saved = _meta_scalar(data, "meta_chat_template")
    meta_max_len_saved = _meta_scalar(data, "meta_max_len")

    def _resolve(cli_value, saved_value, default, field_name):
        if cli_value is None:
            return saved_value if saved_value is not None else default
        if (
            saved_value is not None
            and cli_value != saved_value
            and not args.allow_metadata_mismatch
        ):
            raise SystemExit(
                f"ERROR: --{field_name.replace('_', '-')}={cli_value!r} does not match "
                f"source activation file's meta_{field_name}={saved_value!r} ({src}). "
                "Pass --allow-metadata-mismatch to deliberately rescore under "
                "different settings."
            )
        return cli_value

    model_id = _resolve(args.model, meta_model_saved, "google/gemma-3-4b-it", "model")
    chat_template = bool(
        _resolve(args.chat_template, meta_chat_template_saved, True, "chat_template")
    )
    max_len = int(_resolve(args.max_len, meta_max_len_saved, 0, "max_len"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    if args.require_gpu and not args.allow_cpu and not torch.cuda.is_available():
        print("ERROR: --require-gpu set but CUDA is not available.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        print(
            "ERROR: running on CPU. Use sbatch on a GPU node or pass --allow-cpu for tests.",
            file=sys.stderr,
        )
        return 1
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # Match shared_extract_engine.py: right-truncation would silently strip
    # the *end* of a chat-templated prompt instead of trimming from the start.
    tok.truncation_side = "left"

    # Must match how the activations being (re)scored were extracted 
    if chat_template:
        model_input_prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
    else:
        model_input_prompts = prompts

    concept_tensors, id_lists = concept_token_tensors(tok, device)
    print(
        f"Token ids — true={id_lists['true']} false={id_lists['false']} | "
        f"yes={id_lists['yes']} no={id_lists['no']} | "
        f"safe={id_lists['safe']} unsafe={id_lists['unsafe']} | "
        f"harmless={id_lists['harmless']} harmful={id_lists['harmful']}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    chunks: dict[str, list[np.ndarray]] = {k: [] for k in concept_logit_chunk_keys()}

    with torch.no_grad():
        for i in tqdm(range(0, len(model_input_prompts), args.batch_size), desc="concept-logits"):
            batch = model_input_prompts[i : i + args.batch_size]
            if max_len > 0:
                enc = tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=not chat_template,
                ).to(device)
            else:
                enc = tok(
                    batch, return_tensors="pt", padding=True, truncation=False,
                    add_special_tokens=not chat_template,
                ).to(device)
            out = model(**enc)
            logits = out.logits[:, -1, :].float()
            scored = score_last_position_concept_logits(logits, concept_tensors)
            np_scored = stacks_to_numpy(scored)
            for k in chunks:
                chunks[k].append(np_scored[k])

    payload = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    top_tokens = [
        tok.decode([int(tid)], skip_special_tokens=True)
        for tid in payload["top_next_token_id"]
    ]


    copied_source_fields = {
        k: data[k] for k in data.files if k not in _EXCLUDE_SOURCE_KEYS
    }

    np.savez_compressed(
        out_path,
        prompts=np.array(prompts),
        labels=labels,
        top_next_token=np.array(top_tokens),
        meta_dataset=np.array(dataset_name),
        meta_model=np.array(model_id),
        meta_chat_template=np.array(chat_template),
        meta_add_special_tokens=np.array(not chat_template),
        meta_max_len=np.array(max_len),
        meta_truncation_side=np.array(tok.truncation_side),
        meta_scoring=np.array(META_SCORING),
        meta_backfilled_from=np.array(os.path.abspath(src)),
        **copied_source_fields,
        **concept_meta_token_id_arrays(id_lists),
        **payload,
    )
    print(f"Saved {out_path} ({len(prompts)} rows)")
    if "p_true" in payload and labels.dtype == bool:
        print(
            f"  p_true mean={payload['p_true'].mean():.3f} | "
            f"acc@0.5 vs labels={((payload['p_true'] > 0.5) == labels).mean():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
