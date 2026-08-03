#!/usr/bin/env python
"""
Step 1 (GPU): extract Gemma-3-4B activations for XSTest.

Extraction settings:
  - Model: google/gemma-3-4b-it
  - Layers: all transformer layers (for raw activations)
  - Raw activations: mean-pooled **and** last prompt token (both saved per run by default)
  - Greedy decode: save up to 10 generated tokens per prompt
  - Gemma Scope 2 decompositions (sparse COO), streamed inline:
      * SAE (residual stream, every layer)
      * Single-layer transcoder (MLP input, every layer) — SAELens transcoders-all
      * Cross-layer crosscoder (4-depth residual concat: layers 9/17/22/29) (layer indices in HF are different from the Scope layer indices, need to add 1 to the scope layer indices to get the HF layer indices)
      * Cross-layer transcoder (CLT): NOT released for 4B — only 270M/1B

Dataset (450 prompts, single train split):
  https://huggingface.co/datasets/walledai/XSTest
  (falls back to Paul/XSTest if walledai is gated — same 450 prompts)
  Fields: prompt (str), type (str), label ("safe" | "unsafe"), focus, note
  250 safe + 200 unsafe test questions. Only the raw prompt text goes to the model;
  labels are metadata (saved as bool: True = unsafe).

Hugging Face auth: google/gemma-3-4b-it is gated. Authenticate first, e.g.
    hf auth login

Decompositions require sae-lens:
    pip install sae-lens

Usage:
    python extract_activations.py
    python extract_activations.py --decomps sae crosscoder transcoder
    python extract_activations.py --no-decomps
    python extract_activations.py --crosscoder-path $GEMMA_SCOPE_DIR/crosscoder

Outputs (default) under activations/ — one self-contained file per layer per feature set:
    activations/xstest_mean_layer12.npz
    activations/xstest_last_layer12.npz
    activations/xstest_sae_layer12.npz
    activations/xstest_transcoder_layer12.npz
    activations/xstest_crosscoder_layer9_17_22_29.npz

Naming: {dataset}_{feature_set}_{layer}.npz  (see --dataset-name)

Layer numbering — THREE index systems (easy to mix up):
  1. HF hidden_states index (k): what we index in code via out.hidden_states[k].
       k=0 → token embeddings (not a transformer block).
       k>=1 → output after transformer block k (1-indexed block count).
       len(hidden_states) = num_hidden_layers + 1  (e.g. 35 for Gemma-3-4B).

  2. Gemma Scope layer (s, 0-indexed): SAE checkpoint names layer_{s}.
       Hooks model.layers[s].output (resid_post after block s+1).
       Mapping:  hidden_states[k]  <->  Scope layer s = k - 1
                 Scope layer s      ->  hidden_states[s + 1]

  3. Saved filenames ({layer}): L is the HF hidden_states index (social_iqa convention).
       Default --layers all → layer1, layer2, …, layer34 in filenames.
       Each .npz is self-contained (activations + prompts + labels + responses).

  Per artifact in this script:
    Raw activations     → HF index L; saved as xstest_{mean|last}_layer{L}.npz
    SAE decompositions  → read hidden_states[L]; load Scope checkpoint layer_{L-1}
    Transcoder decomps  → hook pre_feedforward_layernorm at Scope layer L-1; load
                          Scope checkpoint layer_{L-1} from transcoders-all
    Crosscoder          → Scope layers (9,17,22,29) → hidden_states (10,18,23,30)

  Pitfall: Scope SAE layer_12 requires hidden_states[13], not hidden_states[12].
  See docs/data.md, "Layer numbering", for the full table.

Gemma Scope 2 SAELens naming (verified against the google/gemma-scope-2-4b-it
HF repo, July 2026):
  - The repo is organized into top-level "site" folders: resid_post,
    resid_post_all, attn_out(_all), mlp_out(_all), transcoder(_all),
    crosscoder. (No `clt/` folder for 4B — confirmed CLT really is
    270M/1B only.)
  - The *site* goes into the SAELens `release` string, NOT into `sae_id`.
  - SAELens's registry uses short site codes (res/mlp/att/transcoders),
    not the HF folder names -- confirmed via get_pretrained_saes_directory():
    the per-layer releases for this model are gemma-scope-2-4b-it-{res,
    res-all, mlp, mlp-all, att, att-all, transcoders, transcoders-all}.
  - resid_post_all only ships widths {16k, 262k} (note: 262k, not 256k)
    with L0 in {small, big} — no "medium" at 16k, matching the
    SAE_ID_TEMPLATE below.

Crosscoder loading — NOT via SAELens:
  The crosscoder is absent from SAELens's registry (see list above) and
  its multi-layer architecture isn't one of SAELens's supported SAE
  classes, so SAE.from_pretrained can't load it. Instead this script
  loads the raw Gemma Scope weight file (params.npz) directly and
  implements the JumpReLU encode itself — the same approach as Google's
  own Gemma Scope loading examples:

      pre  = x @ W_enc + b_enc          # x = concat of 4 resid layers
      f    = pre * (pre > threshold)    # JumpReLU

  where x is the depth-ordered concatenation of resid_post at Scope
  layers 9,17,22,29 (so d_in = 4 * hidden_dim). Only encode() is needed
  here (we save sparse features), so W_dec/b_dec are loaded but unused.

  Weight location: pre-downloaded locally (default):
    $GEMMA_SCOPE_DIR/crosscoder/
      layer_9_17_22_29_width_65k_l0_medium/params_layer_{0..3}.safetensors
  Resolution order:
    1. --crosscoder-path (default: the crosscoder/ dir above)
    2. huggingface_hub.hf_hub_download(..., local_files_only=True)
    3. same call with network allowed, as a last resort

Left-padding fix: tok.padding_side is "left" (required for clean batched
generation), which means every prompt's real tokens are right-aligned —
the last real token is always at the final sequence position, the same
index for every row in the batch. The first version of this script
computed last-token position as attention_mask.sum(1) - 1, which is only
correct for right-padding; with left-padding it pointed at an early
token (or into the pad region) for every row shorter than the batch max.
The same mistake showed up in response-token extraction, which sliced
generated output starting at each row's real prompt length instead of
the padded input length. Both are fixed below.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np


DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_DATASET_NAME = "xstest"
XSTEST_HF_PRIMARY = "walledai/XSTest"
XSTEST_HF_FALLBACK = "Paul/XSTest"

GEMMA_SCOPE_REPO = "google/gemma-scope-2-4b-it"  # HF repo id (for reference / hf auth); NOT the SAELens release string
GEMMA_SCOPE_MODEL_TAG = "gemma-scope-2-4b-it"     # substring used to find the right release in SAELens's registry
# --- Layer index conventions (see module docstring) ---
# HF hidden_states[k] for k>=1 equals model.layers[k-1].output (Scope resid_post).
# Scope checkpoint layer_{s} hooks model.layers[s].output → use hidden_states[s+1].
# Gemma Scope 2 weights: https://huggingface.co/google/gemma-scope-2-4b-it
SAE_SITE = "res-all"  # confirmed via get_pretrained_saes_directory(): 'gemma-scope-2-4b-it-res-all'
                       # (NOT 'resid_post_all' -- SAELens's registry uses short site
                       # codes: res, mlp, att, transcoders -- not the HF folder names.)

# if storage allows, consider using 262k instead of 16k width (folder is
# literally named "262k", not "256k" -- the blog post's "256k" is a rounded
# marketing figure, not the folder name)
SAE_ID_TEMPLATE = "layer_{layer}_width_16k_l0_small"  # 16k only has small/big, not medium

# Single-layer transcoders: SAELens transcoders-all; hook MLP input (pre-MLP), not resid_post.
TRANSCODER_SITE = "transcoders-all"
TRANSCODER_ID_TEMPLATE = SAE_ID_TEMPLATE

# Crosscoder: loaded from raw Gemma Scope weights, NOT via SAELens (see docstring).
CROSSCODER_SITE = "crosscoder"
CROSSCODER_ID = "layer_9_17_22_29_width_65k_l0_medium"
CROSSCODER_LAYERS = (9, 17, 22, 29)  # Scope 0-indexed; HF indices = each + 1; confirmed real folder name
CROSSCODER_HF_SUBPATH = f"{CROSSCODER_SITE}/{CROSSCODER_ID}"
DEFAULT_CROSSCODER_PATH = os.environ.get(
    "GEMMA_SCOPE_DIR", "gemma_scope_weights") + "/crosscoder"

# CLT exists only for gemma-scope-2-{270m,1b}-it, not 4B.
CLT_AVAILABLE_MODELS = ("google/gemma-3-270m-it", "google/gemma-3-1b-it")


@dataclass(frozen=True)
class DecompSpec:
    name: str
    release: str
    sae_id: str
    # None => per-layer ids via template; tuple => fixed layer set (crosscoder)
    layers: tuple[int, ...] | None = None
    id_template: str | None = None
    loader: str = "sae_lens"  # "sae_lens" | "gemma_npz" (raw Gemma Scope weights)


# Populated by resolve_decomp_specs() once sae_lens is importable and only if
# decompositions are actually requested -- left empty at import time so this
# module doesn't require sae_lens just to run --no-decomps.
DECOMP_SPECS: dict[str, DecompSpec] = {}


def _find_release_id(model_tag: str, site: str) -> str:
    """Look up the real SAELens release id for a Gemma Scope 2 site.

    Guessing the naming convention by hand got this wrong twice (first
    "gemma-scope-2-4b-it-resid_post_all", then confirmed the registry
    actually uses short site codes: res/mlp/att/transcoders, not the HF
    folder names resid_post/mlp_out/attn_out/transcoder). Instead, search
    the *installed* library's own registry for a release whose name
    contains both the model tag (e.g. "gemma-scope-2-4b-it") and the site.
    """
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError:
        # older SAELens versions used this path instead
        from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()
    candidates = [key for key in directory if model_tag in key and site in key]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        exact = f"{model_tag}-{site}"
        if exact in candidates:
            return exact
        raise RuntimeError(
            f"Multiple SAELens releases match model_tag={model_tag!r} "
            f"site={site!r}: {sorted(candidates)}. Pick the right one and "
            "hardcode it in SAE_SITE (or refine the match)."
        )

    # Nothing matched -- show what *is* registered for this model to help debug.
    near = sorted(key for key in directory if model_tag in key)
    raise RuntimeError(
        f"No SAELens release found for model_tag={model_tag!r} site={site!r}. "
        f"Releases registered for this model tag: {near or '(none -- check GEMMA_SCOPE_MODEL_TAG)'}"
    )


# ---------------------------------------------------------------------------
# Crosscoder: raw Gemma Scope JumpReLU loading (SAELens has no crosscoder class)
# ---------------------------------------------------------------------------

class GemmaScopeJumpReLUEncoder:
    """Minimal encode-only JumpReLU model over raw Gemma Scope weights.

    Mirrors Google's reference Gemma Scope loading code:
        pre  = x @ W_enc + b_enc
        f    = pre * (pre > threshold)
    For the crosscoder, x is the depth-ordered concat of resid_post at
    CROSSCODER_LAYERS, so W_enc has shape (4 * hidden_dim, d_sae).

    Exposes .encode(acts) and .cfg.d_sae to match the subset of the
    SAELens SAE interface that encode_sparse_batch()/run_decompositions()
    actually use, so it can sit in the same scope_model_cache.
    """

    # accepted key aliases in params files (Gemma Scope uses W_enc/b_enc/threshold;
    # tolerate case variants and log-threshold storage just in case)
    _ENC_KEYS = ("W_enc", "w_enc")
    _BIAS_KEYS = ("b_enc",)
    _THRESH_KEYS = ("threshold", "log_threshold")

    def __init__(self, params: dict[str, np.ndarray], device: str, source: str):
        import torch

        def pick(keys):
            for k in keys:
                if k in params:
                    return k, params[k]
            raise KeyError(
                f"None of {keys} found in crosscoder params from {source}; "
                f"available keys: {sorted(params.keys())}"
            )

        _, w_enc = pick(self._ENC_KEYS)
        _, b_enc = pick(self._BIAS_KEYS)
        thresh_key, thresh = pick(self._THRESH_KEYS)

        self.W_enc = torch.from_numpy(np.asarray(w_enc)).to(device=device, dtype=torch.float32)
        self.b_enc = torch.from_numpy(np.asarray(b_enc)).to(device=device, dtype=torch.float32)
        thresh_t = torch.from_numpy(np.asarray(thresh)).to(device=device, dtype=torch.float32)
        self.threshold = thresh_t.exp() if thresh_key == "log_threshold" else thresh_t

        if self.W_enc.ndim != 2:
            raise ValueError(f"Expected 2-D W_enc, got shape {tuple(self.W_enc.shape)} from {source}")
        self.d_in = int(self.W_enc.shape[0])
        self.d_sae = int(self.W_enc.shape[1])
        if self.b_enc.shape[-1] != self.d_sae or self.threshold.shape[-1] != self.d_sae:
            raise ValueError(
                f"Shape mismatch in crosscoder params from {source}: "
                f"W_enc {tuple(self.W_enc.shape)}, b_enc {tuple(self.b_enc.shape)}, "
                f"threshold {tuple(self.threshold.shape)}"
            )
        self.cfg = SimpleNamespace(d_sae=self.d_sae, d_in=self.d_in)
        self.source = source

    def encode(self, acts):
        import torch

        x = acts.to(device=self.W_enc.device, dtype=torch.float32)
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"Crosscoder expects d_in={self.d_in} "
                f"(= {len(CROSSCODER_LAYERS)} layers x hidden_dim), got {x.shape[-1]}. "
                "Check that CROSSCODER_LAYERS activations are concatenated in order."
            )
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def eval(self):  # interface parity with torch modules / SAELens SAEs
        return self


class GemmaScopeShardedCrosscoderEncoder:
    """Crosscoder with per-layer shards (params_layer_0..N.safetensors).

    Each shard encodes one Scope layer's resid_post (d_in=hidden_dim) into
    a slice of the 65k feature space; outputs are concatenated on the
    feature axis to match a single (4 * hidden_dim) -> 65536 encoder.
    """

    def __init__(self, shards: list[GemmaScopeJumpReLUEncoder]):
        if not shards:
            raise ValueError("crosscoder requires at least one params shard")
        self.shards = shards
        self.d_in = sum(s.d_in for s in shards)
        self.d_sae = sum(s.d_sae for s in shards)
        self.cfg = SimpleNamespace(d_sae=self.d_sae, d_in=self.d_in)
        self.source = "+".join(s.source for s in shards)

    def encode(self, acts):
        import torch

        x = acts.to(device=self.shards[0].W_enc.device, dtype=torch.float32)
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"Sharded crosscoder expects d_in={self.d_in} "
                f"(= {len(self.shards)} layers x hidden_dim), got {x.shape[-1]}. "
                "Check that CROSSCODER_LAYERS activations are concatenated in order."
            )
        parts = x.split([s.d_in for s in self.shards], dim=-1)
        outs = [shard.encode(part) for shard, part in zip(self.shards, parts)]
        return torch.cat(outs, dim=-1)

    def eval(self):
        return self


def _dir_has_crosscoder_params(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = os.listdir(path)
    if any(n in ("params.npz", "params.safetensors") for n in names):
        return True
    return any(
        n.startswith("params_layer_") and n.endswith((".npz", ".safetensors"))
        for n in names
    )


def _resolve_crosscoder_checkpoint_dir(crosscoder_path: str | None) -> str:
    """Return the directory containing crosscoder weight file(s)."""
    p = os.path.expanduser(crosscoder_path or DEFAULT_CROSSCODER_PATH)
    if os.path.isfile(p):
        return os.path.dirname(p)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"crosscoder path does not exist: {p}")

    if not _dir_has_crosscoder_params(p):
        nested = os.path.join(p, CROSSCODER_ID)
        if os.path.isdir(nested) and _dir_has_crosscoder_params(nested):
            return nested
        raise FileNotFoundError(
            f"No crosscoder params found under {p} or {nested}. "
            f"Expected params.npz, params.safetensors, or params_layer_*.safetensors."
        )
    return p


def _find_single_crosscoder_params_file(checkpoint_dir: str) -> str:
    candidates = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.endswith((".npz", ".safetensors")) and not f.startswith("params_layer_")
    )
    for preferred in ("params.npz", "params.safetensors"):
        if preferred in candidates:
            return os.path.join(checkpoint_dir, preferred)
    if len(candidates) == 1:
        return os.path.join(checkpoint_dir, candidates[0])
    raise FileNotFoundError(
        f"{checkpoint_dir}: expected exactly one unsharded params file "
        f"(.npz/.safetensors), found {candidates or 'none'}."
    )


def _find_sharded_crosscoder_params_files(checkpoint_dir: str) -> list[str]:
    shard_names = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("params_layer_") and f.endswith((".npz", ".safetensors"))
    )
    if not shard_names:
        return []
    return [os.path.join(checkpoint_dir, name) for name in shard_names]


def _load_params_file(path: str) -> dict[str, np.ndarray]:
    if path.endswith(".npz"):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        return load_file(path)
    raise ValueError(f"Unsupported crosscoder params format: {path}")


def _find_crosscoder_params_file(crosscoder_path: str | None) -> str:
    """Locate a single unsharded crosscoder params file (HF cache fallback)."""
    try:
        checkpoint_dir = _resolve_crosscoder_checkpoint_dir(crosscoder_path)
        return _find_single_crosscoder_params_file(checkpoint_dir)
    except FileNotFoundError:
        pass

    from huggingface_hub import hf_hub_download

    errors = []
    for fname in (f"{CROSSCODER_HF_SUBPATH}/params.npz",
                  f"{CROSSCODER_HF_SUBPATH}/params.safetensors"):
        for local_only in (True, False):
            try:
                return hf_hub_download(
                    repo_id=GEMMA_SCOPE_REPO,
                    filename=fname,
                    local_files_only=local_only,
                )
            except Exception as e:  # noqa: BLE001 -- collect and report all attempts
                errors.append(f"  {fname} (local_only={local_only}): {type(e).__name__}: {e}")
    raise FileNotFoundError(
        "Could not locate the crosscoder weights. Tried, in order:\n"
        f"  default path: {DEFAULT_CROSSCODER_PATH}\n"
        + "\n".join(errors)
        + f"\nDownload it with:\n"
        f"  hf download {GEMMA_SCOPE_REPO} "
        f"--include '{CROSSCODER_HF_SUBPATH}/*' "
        f"--local-dir {os.path.dirname(DEFAULT_CROSSCODER_PATH)}"
    )


def load_crosscoder(
    device: str,
    crosscoder_path: str | None,
) -> GemmaScopeJumpReLUEncoder | GemmaScopeShardedCrosscoderEncoder:
    checkpoint_dir = _resolve_crosscoder_checkpoint_dir(crosscoder_path)
    shard_files = _find_sharded_crosscoder_params_files(checkpoint_dir)
    if shard_files:
        log(f"Loading sharded crosscoder from {checkpoint_dir} ({len(shard_files)} shards)")
        shards = [
            GemmaScopeJumpReLUEncoder(_load_params_file(path), device=device, source=path)
            for path in shard_files
        ]
        return GemmaScopeShardedCrosscoderEncoder(shards)

    params_file = _find_single_crosscoder_params_file(checkpoint_dir)
    log(f"Loading crosscoder weights from {params_file}")
    params = _load_params_file(params_file)
    return GemmaScopeJumpReLUEncoder(params, device=device, source=params_file)


def resolve_decomp_specs(
    decomps: list[str],
    device: str,
    scope_model_cache: dict[tuple[str, str, str], object],
    crosscoder_path: str | None,
) -> list[str]:
    """Populate DECOMP_SPECS; return a possibly-filtered decomps list.

    Called once from main() before the base model download, so a bad
    release id / missing weight file fails fast.

    - SAE: release id resolved from SAELens's own registry (hand-guessing
      the naming convention failed twice; the registry is authoritative).
    - Crosscoder: not in SAELens's registry and not a SAELens-supported
      architecture, so it is loaded from the raw Gemma Scope params file
      (already in the local HF cache) via
      GemmaScopeJumpReLUEncoder. The loaded model is placed straight into
      scope_model_cache so the batch loop reuses this instance instead of
      loading twice. If loading fails, 'crosscoder' is dropped with a
      warning rather than crashing the run.
    """
    resolved = list(decomps)

    if "sae" in decomps:
        sae_release = _find_release_id(GEMMA_SCOPE_MODEL_TAG, SAE_SITE)
        log(f"Resolved SAE release: {sae_release}")
        DECOMP_SPECS["sae"] = DecompSpec(
            name="sae", release=sae_release, sae_id="", id_template=SAE_ID_TEMPLATE,
        )

    if "crosscoder" in decomps:
        spec = DecompSpec(
            name="crosscoder",
            release=GEMMA_SCOPE_REPO,          # informational: where the weights come from
            sae_id=CROSSCODER_HF_SUBPATH,      # informational: path within the repo
            layers=CROSSCODER_LAYERS,
            loader="gemma_npz",
        )
        try:
            model = load_crosscoder(device, crosscoder_path)
        except Exception as e:
            log(f"WARNING: crosscoder could not be loaded ({e}). Dropping "
                "'crosscoder' from --decomps for this run.")
            resolved.remove("crosscoder")
        else:
            log(f"Crosscoder loaded: d_in={model.d_in} d_sae={model.d_sae} "
                f"(expects concat of {len(CROSSCODER_LAYERS)} layers)")
            DECOMP_SPECS["crosscoder"] = spec
            # Pre-populate the cache so the batch loop reuses this instance.
            scope_model_cache[(spec.release, spec.sae_id, device)] = model

    if "transcoder" in decomps:
        tc_release = _find_release_id(GEMMA_SCOPE_MODEL_TAG, TRANSCODER_SITE)
        log(f"Resolved transcoder release: {tc_release}")
        DECOMP_SPECS["transcoder"] = DecompSpec(
            name="transcoder",
            release=tc_release,
            sae_id="",
            id_template=TRANSCODER_ID_TEMPLATE,
        )

    return resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Gemma-3-4B activations (XSTest).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF model id (gemma-3-4b-it has 34 layers).")
    p.add_argument("--layers", nargs="*", default=["all"],
                   help="HF hidden_states indices for raw activations: 'all' (default "
                        "→ 1,2,…,34), 'even', or explicit ints. NOT Gemma Scope layer ids.")
    p.add_argument("--pool", choices=["mean", "last", "both"], default="both",
                   help="Raw activation pooling: mean, last prompt token, or both "
                        "(default: both → {dataset}_mean_layer{L}.npz + {dataset}_last_layer{L}.npz).")
    p.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                   help=f"Dataset prefix for output filenames (default: {DEFAULT_DATASET_NAME}). "
                        "Files: {{dataset}}_{{feature_set}}_{{layer}}.npz")
    p.add_argument("--split", default="all", choices=["train", "all"],
                   help="Ignored for HF load (uses train or test split on hub); 'all' = full set.")
    p.add_argument("--hf-dataset", default=XSTEST_HF_PRIMARY,
                   help=f"HF dataset id (default: {XSTEST_HF_PRIMARY}; "
                        f"falls back to {XSTEST_HF_FALLBACK} if gated).")
    p.add_argument("--n", type=int, default=None,
                   help="Subsample this many prompts (default: use full split).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per forward pass (4B is lighter than 12B; increase if GPU allows).")
    p.add_argument("--max-len", type=int, default=256,
                   help="Max prompt tokens (truncate longer prompts). "
                        "Use 0 for no truncation (dynamic per-batch padding only).")
    p.add_argument("--gen-tokens", type=int, default=10,
                   help="Greedy-decode this many tokens after each prompt.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--cache", default=None,
                   help="Deprecated (ignored). Outputs are always per-layer files under --outdir.")
    p.add_argument("--outdir", default="activations",
                   help="Directory for per-layer activation files (default: activations/).")
    p.add_argument(
        "--decomps", nargs="*", default=["sae", "crosscoder"],
        choices=["sae", "transcoder", "crosscoder", "clt"],
        help="Gemma Scope 2 decompositions to extract (sparse COO). "
             "Default: sae + crosscoder. Add 'transcoder' for per-layer MLP-input "
             "transcoders (34 extra SAELens loads ≈ 11 GB of disk). "
             "'clt' is only available for 270M/1B.",
    )
    p.add_argument("--crosscoder-path", default=DEFAULT_CROSSCODER_PATH,
                   help="Local dir with crosscoder weights. Default: "
                        f"{DEFAULT_CROSSCODER_PATH} (auto-descends into "
                        f"{CROSSCODER_ID}/ if needed).")
    p.add_argument("--no-decomps", action="store_true",
                   help="Skip all Gemma Scope decomposition extraction.")
    p.add_argument("--no-sae", action="store_true",
                   help="Deprecated alias for --no-decomps.")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def scope_layer_to_hf_index(scope_layer: int) -> int:
    """Gemma Scope layer s (0-indexed) → HF hidden_states index.

    Scope SAE layer_{s} hooks model.layers[s].output.
    hidden_states[0] is embeddings, so block s appears at index s + 1.
    """
    return scope_layer + 1


def hf_index_to_scope_layer(hf_index: int) -> int:
    """HF hidden_states index k (k>=1) → Gemma Scope layer (0-indexed).

    Example: hidden_states[12] is Scope layer 11; Scope layer_12 needs hidden_states[13].
    """
    return hf_index - 1


def resolve_layers(arg_layers: list[str], num_hidden_layers: int) -> list[int]:
    """Return HF hidden_states indices to save as raw layer_{L} in the .npz.

    Default 'all' → 1, 2, …, num_hidden_layers (HF indices, not Scope indices).
    These are the k in hidden_states[k], not Gemma Scope checkpoint layer_{s}.
    """
    if not arg_layers:
        arg_layers = ["all"]
    if len(arg_layers) == 1 and arg_layers[0] == "even":
        layers = list(range(2, num_hidden_layers + 1, 2))
    elif len(arg_layers) == 1 and arg_layers[0] == "all":
        layers = list(range(1, num_hidden_layers + 1))
    else:
        layers = [int(x) for x in arg_layers]
    for L in layers:
        if not 1 <= L <= num_hidden_layers:
            raise ValueError(f"Layer {L} invalid; model has {num_hidden_layers} layers")
    return layers


def resolve_decomps(args: argparse.Namespace, model_id: str) -> list[str]:
    if args.no_decomps or args.no_sae:
        return []
    decomps = list(args.decomps) if args.decomps else ["sae", "crosscoder"]
    if "clt" in decomps and model_id not in CLT_AVAILABLE_MODELS:
        log(
            "WARNING: cross-layer transcoder (CLT) is not released for "
            f"{model_id}. Gemma Scope 2 CLTs exist only for 270M and 1B. "
            "Skipping 'clt'. Use --model google/gemma-3-1b-it to extract CLTs."
        )
        decomps = [d for d in decomps if d != "clt"]
    return decomps


def layer_filename_tag(hf_index: int) -> str:
    """HF layer index → filename segment, e.g. 12 → 'layer12'."""
    return f"layer{hf_index}"


def output_path(outdir: str, dataset_name: str, feature_set: str, layer_tag: str) -> str:
    """Build {dataset}_{feature_set}_{layer}.npz path."""
    return os.path.join(outdir, f"{dataset_name}_{feature_set}_{layer_tag}.npz")


def resolve_output_paths(
    outdir: str,
    dataset_name: str,
    raw_layers: list[int],
    pool_modes: list[str],
    decomps: list[str],
) -> list[tuple[str, str, str]]:
    """Return [(feature_set, layer_tag, path), ...] for every expected output file."""
    paths: list[tuple[str, str, str]] = []
    for mode in pool_modes:
        for L in raw_layers:
            tag = layer_filename_tag(L)
            paths.append((mode, tag, output_path(outdir, dataset_name, mode, tag)))
    for decomp in decomps:
        if decomp in ("sae", "transcoder"):
            for L in raw_layers:
                tag = layer_filename_tag(L)
                paths.append((decomp, tag, output_path(outdir, dataset_name, decomp, tag)))
        elif decomp == "crosscoder":
            tag = "layer" + "_".join(str(L) for L in CROSSCODER_LAYERS)
            paths.append((
                "crosscoder",
                tag,
                output_path(outdir, dataset_name, "crosscoder", tag),
            ))
    return paths


def build_shared_meta(
    args: argparse.Namespace,
    prompts: list[str],
    labels: np.ndarray,
    responses: list[str],
    response_ids: list[list[int]],
    hidden_dim: int,
    decomps: list[str],
    types: np.ndarray | None = None,
    hf_layer: int | None = None,
    feature_set: str | None = None,
) -> dict[str, np.ndarray]:
    meta: dict[str, np.ndarray] = {
        "prompts": np.array(prompts),
        "labels": labels,
        "responses": np.array(responses),
        "response_token_ids": np.array(response_ids, dtype=object),
        "meta_dataset": np.array(args.dataset_name),
        "meta_model": np.array(args.model),
        "meta_layer_index_convention": np.array(
            "Filenames use HF hidden_states index in layer{N}; "
            "Scope layer s = N - 1; Scope layer s -> hidden_states[s+1]"
        ),
        "meta_pool_arg": np.array(args.pool),
        "meta_gen_tokens": np.array(args.gen_tokens),
        "meta_hidden_dim": np.array(hidden_dim),
        "meta_decomps": np.array(decomps),
    }
    if types is not None:
        meta["types"] = types
        meta["meta_label_convention"] = np.array("True=unsafe, False=safe")
    if feature_set is not None:
        meta["meta_feature_set"] = np.array(feature_set)
    if hf_layer is not None:
        meta["meta_hf_layer"] = np.array(hf_layer)
        meta["meta_scope_layer"] = np.array(hf_index_to_scope_layer(hf_layer))
    return meta


def resolve_pool_modes(pool: str) -> list[str]:
    if pool == "both":
        return ["mean", "last"]
    return [pool]


def layers_needed(raw_layers: list[int], decomps: list[str]) -> list[int]:
    needed = set(raw_layers)
    if "crosscoder" in decomps:
        needed.update(scope_layer_to_hf_index(L) for L in CROSSCODER_LAYERS)
    return sorted(needed)


def _get_text_model(model):
    """Return the text-only transformer (Gemma3 wraps it under language_model)."""
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "language_model"):
        return inner.language_model
    return inner


class MlpInputCapture:
    """Capture pre_feedforward_layernorm outputs (MLP input) per Scope layer.

    Gemma Scope transcoders hook model.layers[s].pre_feedforward_layernorm.output,
    not resid_post / hidden_states. Hooks are registered once and only record
    activations from the prompt forward (before generate()).
    """

    def __init__(self) -> None:
        self.acts: dict[int, object] = {}
        self._handles: list = []
        self._active = False

    def register(self, model, scope_layers: list[int]) -> None:
        self.remove()
        text_model = _get_text_model(model)
        for scope_L in scope_layers:
            module = text_model.layers[scope_L].pre_feedforward_layernorm

            def hook_fn(module, inp, out, layer=scope_L):
                if self._active:
                    self.acts[layer] = out

            self._handles.append(module.register_forward_hook(hook_fn))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.acts.clear()

    def begin_batch(self) -> None:
        self._active = True
        self.acts.clear()

    def end_batch(self) -> None:
        self._active = False


def _load_xstest_dataset(hf_dataset: str):
    from datasets import get_dataset_split_names, load_dataset

    split_names = get_dataset_split_names(hf_dataset, trust_remote_code=True)
    if "train" in split_names:
        hf_split = "train"
    elif "test" in split_names:
        hf_split = "test"
    elif len(split_names) == 1:
        hf_split = split_names[0]
    else:
        raise ValueError(
            f"{hf_dataset}: expected a single split or train/test; got {split_names}"
        )
    log(f"Loading {hf_dataset} split={hf_split!r} ...")
    return load_dataset(hf_dataset, split=hf_split, trust_remote_code=True)


def load_prompts(split: str, n: int | None, seed: int, hf_dataset: str):
    log(f"Loading XSTest ({hf_dataset}) ...")
    try:
        ds = _load_xstest_dataset(hf_dataset)
    except Exception as e:
        err = str(e).lower()
        if hf_dataset != XSTEST_HF_FALLBACK and (
            "gated" in err or "datasetnotfound" in type(e).__name__.lower()
        ):
            log(
                f"WARNING: {hf_dataset} unavailable ({e}); "
                f"falling back to {XSTEST_HF_FALLBACK}"
            )
            ds = _load_xstest_dataset(XSTEST_HF_FALLBACK)
        else:
            raise

    if split not in ("train", "all"):
        raise ValueError(f"XSTest has no '{split}' split; use train or all.")

    prompts = [ex["prompt"].strip() for ex in ds]
    labels = np.array([ex["label"] == "unsafe" for ex in ds], dtype=bool)
    types = np.array([ex["type"] for ex in ds])

    n_use = len(prompts) if n is None else min(n, len(prompts))
    if n_use < len(prompts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(prompts), size=n_use, replace=False)
        prompts = [prompts[i] for i in idx]
        labels = labels[idx]
        types = types[idx]

    n_unsafe = int(labels.sum())
    log(
        f"Loaded {len(prompts)} prompts | unsafe={n_unsafe} safe={len(prompts) - n_unsafe}"
    )
    return prompts, labels, types


def encode_sparse_batch(model, acts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return COO (row, col, val) for nonzero features."""
    import torch

    with torch.no_grad():
        feats = model.encode(acts)
    feats = feats.detach().float().cpu().numpy()
    if feats.ndim == 3:
        feats = feats[:, 0, :]
    rows, cols = np.nonzero(feats)
    vals = feats[rows, cols]
    return rows.astype(np.int32), cols.astype(np.int32), vals.astype(np.float32)


def save_sparse_coo(
    path: str,
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray,
    n_samples: int,
    n_features: int,
    meta: dict[str, str] | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    payload = {
        "row": row,
        "col": col,
        "val": val,
        "shape": np.array([n_samples, n_features], dtype=np.int64),
        "format": np.array("coo"),
    }
    if meta:
        for k, v in meta.items():
            payload[f"meta_{k}"] = np.array(v)
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


class SparseAccumulator:
    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        self.n_features: int | None = None
        self.sample_offset = 0

    def append(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n_features: int):
        rows = rows + self.sample_offset
        self.rows.append(rows)
        self.cols.append(cols)
        self.vals.append(vals)
        self.n_features = n_features

    def bump_batch(self, batch_size: int) -> None:
        self.sample_offset += batch_size

    def concat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not self.rows:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                self.n_features or 0,
            )
        return (
            np.concatenate(self.rows),
            np.concatenate(self.cols),
            np.concatenate(self.vals),
            int(self.n_features),
        )


def load_scope_model(spec: DecompSpec, sae_id: str, device: str):
    from sae_lens import SAE

    try:
        model, _, _ = SAE.from_pretrained(
            release=spec.release,
            sae_id=sae_id,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Gemma Scope model release={spec.release!r} "
            f"sae_id={sae_id!r}: {e}\n"
            "The release id itself was resolved from SAELens's registry "
            "(see resolve_decomp_specs), so if this release loaded other "
            "sae_ids fine, the problem is likely this specific sae_id -- "
            "double check the layer/width/L0 combination actually exists "
            "under this release (e.g. print get_pretrained_saes_directory()"
            "[spec.release].saes_map to list what's available)."
        ) from e

    model.eval()
    return model


def get_scope_model(
    spec: DecompSpec,
    sae_id: str,
    device: str,
    model_cache: dict[tuple[str, str, str], object],
):
    """Load each Gemma Scope model once and reuse it across batches.

    Keying by (release, sae_id, device) lets the same cache handle ordinary
    per-layer SAEs and the fixed crosscoder without collisions. The
    crosscoder (loader="gemma_npz") is pre-loaded into the cache by
    resolve_decomp_specs(); reaching the miss path for it here means
    resolution was skipped, which is a bug worth failing loudly on.
    """
    key = (spec.release, sae_id, device)
    if key not in model_cache:
        if spec.loader == "gemma_npz":
            raise RuntimeError(
                f"{spec.name} uses raw Gemma Scope weights and should have been "
                "pre-loaded into the cache by resolve_decomp_specs()."
            )
        log(f"Loading Gemma Scope model once: release={spec.release} sae_id={sae_id}")
        model_cache[key] = load_scope_model(spec, sae_id, device)
    return model_cache[key]


def run_decompositions(
    decomps: list[str],
    raw_layers: list[int],
    last_acts: dict[int, object],
    last_mlp_acts: dict[int, object],
    device: str,
    accumulators: dict[str, SparseAccumulator | dict[int, SparseAccumulator]],
    model_cache: dict[tuple[str, str, str], object],
) -> None:
    import torch

    if "sae" in decomps:
        sae_spec = DECOMP_SPECS["sae"]
        sae_acc: dict[int, SparseAccumulator] = accumulators["sae"]  # type: ignore[assignment]
        for hf_idx in raw_layers:
            scope_L = hf_index_to_scope_layer(hf_idx)
            sae_id = sae_spec.id_template.format(layer=scope_L)
            sae = get_scope_model(sae_spec, sae_id, device, model_cache)
            acts = last_acts[hf_idx].unsqueeze(1).to(device=device, dtype=torch.bfloat16)
            rows, cols, vals = encode_sparse_batch(sae, acts)
            sae_acc[hf_idx].append(rows, cols, vals, int(sae.cfg.d_sae))

    if "transcoder" in decomps:
        tc_spec = DECOMP_SPECS["transcoder"]
        tc_acc: dict[int, SparseAccumulator] = accumulators["transcoder"]  # type: ignore[assignment]
        for hf_idx in raw_layers:
            scope_L = hf_index_to_scope_layer(hf_idx)
            if scope_L not in last_mlp_acts:
                raise RuntimeError(
                    f"Transcoder needs MLP input at Scope layer {scope_L} "
                    f"(HF block {hf_idx}); hook did not fire."
                )
            tc_id = tc_spec.id_template.format(layer=scope_L)
            transcoder = get_scope_model(tc_spec, tc_id, device, model_cache)
            acts = last_mlp_acts[scope_L].unsqueeze(1).to(device=device, dtype=torch.bfloat16)
            rows, cols, vals = encode_sparse_batch(transcoder, acts)
            tc_acc[hf_idx].append(rows, cols, vals, int(transcoder.cfg.d_sae))

    if "crosscoder" in decomps:
        cc_spec = DECOMP_SPECS["crosscoder"]
        cc_hf_indices = [scope_layer_to_hf_index(L) for L in CROSSCODER_LAYERS]
        missing = [i for i in cc_hf_indices if i not in last_acts]
        if missing:
            raise RuntimeError(
                f"Crosscoder needs scope layers {CROSSCODER_LAYERS} "
                f"(HF hidden_states {cc_hf_indices}); missing {missing}"
            )
        # Depth-ordered concat (layers 9,17,22,29) -> d_in = 4 * hidden_dim.
        concat = torch.cat([last_acts[i] for i in cc_hf_indices], dim=-1)
        crosscoder = get_scope_model(cc_spec, cc_spec.sae_id, device, model_cache)
        # GemmaScopeJumpReLUEncoder.encode casts to float32 internally.
        acts = concat.unsqueeze(1).to(device=device)
        rows, cols, vals = encode_sparse_batch(crosscoder, acts)
        cc_acc: SparseAccumulator = accumulators["crosscoder"]  # type: ignore[assignment]
        cc_acc.append(rows, cols, vals, int(crosscoder.cfg.d_sae))


def main() -> int:
    args = parse_args()
    if args.cache:
        log("WARNING: --cache is deprecated and ignored; outputs are per-layer files.")
    os.makedirs(args.outdir, exist_ok=True)
    pool_modes = resolve_pool_modes(args.pool)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    prompts, labels, types = load_prompts(args.split, args.n, args.seed, args.hf_dataset)
    decomps = resolve_decomps(args, args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()} | device: {device}")
    if device == "cpu":
        log("WARNING: no GPU detected — extraction will be very slow.")

    # Cache Gemma Scope models across batches. This avoids re-loading the same
    # per-layer SAEs plus the crosscoder on every batch. Created before
    # resolve_decomp_specs so the crosscoder loaded there is reused directly.
    scope_model_cache: dict[tuple[str, str, str], object] = {}

    if decomps:
        # Fail fast (or gracefully drop crosscoder) before the 8GB model download.
        decomps = resolve_decomp_specs(decomps, device, scope_model_cache, args.crosscoder_path)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    log(f"Loading {args.model} (first run downloads ~8 GB) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device).eval()

    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_layers = cfg.num_hidden_layers
    raw_layers = resolve_layers(args.layers, num_layers)
    capture_layers = layers_needed(raw_layers, decomps)
    hidden_dim = cfg.hidden_size

    output_paths = resolve_output_paths(
        args.outdir, args.dataset_name, raw_layers, pool_modes, decomps,
    )
    missing = [path for _, _, path in output_paths if not os.path.exists(path)]
    if not missing:
        log(f"All {len(output_paths)} output files already exist — nothing to do.")
        log(f"Naming: {args.dataset_name}_{{feature_set}}_{{layer}}.npz under {args.outdir}/")
        log("Delete files or change --outdir to re-extract.")
        return 0
    if len(missing) < len(output_paths):
        log(
            f"WARNING: {len(output_paths) - len(missing)}/{len(output_paths)} files exist; "
            f"will re-extract and overwrite all ({len(missing)} missing)."
        )

    log(f"Loaded {args.model} | {num_layers} layers | hidden_dim={hidden_dim}")
    log(f"Dataset prefix: {args.dataset_name}")
    log(f"Saving layers: {raw_layers} ({len(raw_layers)} total)")
    log(f"Output plan: {len(output_paths)} files ({pool_modes} + {decomps or 'no decomps'})")
    if args.max_len > 0:
        log(f"Prompt token cap: --max-len {args.max_len} (truncate longer prompts)")
    else:
        log("Prompt token cap: none (--max-len 0; dynamic per-batch padding only)")
    if decomps:
        log(f"Gemma Scope 2 decompositions: {decomps}")
        if "transcoder" in decomps:
            log(
                "  transcoder: MLP input via pre_feedforward_layernorm hooks "
                f"(Scope layers {[hf_index_to_scope_layer(L) for L in raw_layers]})"
            )
        if "crosscoder" in decomps:
            log(f"  crosscoder scope layers: {CROSSCODER_LAYERS} "
                f"-> HF hidden_states {list(scope_layer_to_hf_index(L) for L in CROSSCODER_LAYERS)}")
            cc_model = scope_model_cache[
                (DECOMP_SPECS["crosscoder"].release, DECOMP_SPECS["crosscoder"].sae_id, device)
            ]
            expected = len(CROSSCODER_LAYERS) * hidden_dim
            if cc_model.d_in != expected:  # type: ignore[union-attr]
                raise RuntimeError(
                    f"Crosscoder d_in={cc_model.d_in} != {len(CROSSCODER_LAYERS)} "  # type: ignore[union-attr]
                    f"* hidden_dim({hidden_dim}) = {expected}. Wrong weights for this model?"
                )

    accumulators: dict[str, SparseAccumulator | dict[int, SparseAccumulator]] = {}
    if "sae" in decomps:
        accumulators["sae"] = {L: SparseAccumulator() for L in raw_layers}
    if "transcoder" in decomps:
        accumulators["transcoder"] = {L: SparseAccumulator() for L in raw_layers}
    if "crosscoder" in decomps:
        accumulators["crosscoder"] = SparseAccumulator()

    mlp_capture = MlpInputCapture()
    if "transcoder" in decomps:
        tc_scope_layers = sorted(hf_index_to_scope_layer(L) for L in raw_layers)
        mlp_capture.register(model, tc_scope_layers)
        log(f"Registered MLP-input hooks on Scope layers {tc_scope_layers}")

    @torch.no_grad()
    def extract_batch(batch_prompts: list[str]):
        if args.max_len > 0:
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(device)
        else:
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(device)
        if "transcoder" in decomps:
            mlp_capture.begin_batch()
        out = model(**enc, output_hidden_states=True)
        if "transcoder" in decomps:
            mlp_capture.end_batch()

        mask = enc["attention_mask"].unsqueeze(-1)

        # Left-padding => every row's real content is right-aligned, so the
        # last real token is always at the final sequence position -- the
        # same index for every row. (attention_mask.sum(1)-1 would only be
        # correct for right-padding.)
        last_acts = {}
        last_mlp_acts = {}
        mean_acts: dict[int, object] = {}
        last_raw: dict[int, object] = {}
        for L in capture_layers:
            hs = out.hidden_states[L]
            last_acts[L] = hs[:, -1, :]
            if L in raw_layers:
                mean_acts[L] = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                last_raw[L] = last_acts[L]
        if "transcoder" in decomps:
            for scope_L, hs in mlp_capture.acts.items():
                last_mlp_acts[scope_L] = hs[:, -1, :]

        gen = model.generate(
            **enc,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        # Left-padding => every row's padded input has the same length, so the
        # generated continuation starts at this one fixed offset for every
        # row -- NOT at each row's real (unpadded) prompt length.
        prompt_len = enc["input_ids"].size(1)
        responses, response_ids = [], []
        for i in range(gen.size(0)):
            new_ids = gen[i, prompt_len:].tolist()
            response_ids.append(new_ids)
            responses.append(tok.decode(new_ids, skip_special_tokens=True))

        if decomps:
            run_decompositions(
                decomps,
                raw_layers,
                last_acts,
                last_mlp_acts,
                device,
                accumulators,
                scope_model_cache,
            )
            bs = len(batch_prompts)
            for acc in accumulators.values():
                if isinstance(acc, dict):
                    for sub in acc.values():
                        sub.bump_batch(bs)
                else:
                    acc.bump_batch(bs)

        raw_out: dict[str, dict[int, np.ndarray]] = {}
        if "mean" in pool_modes:
            raw_out["mean"] = {L: mean_acts[L].float().cpu().numpy() for L in raw_layers}
        if "last" in pool_modes:
            raw_out["last"] = {L: last_raw[L].float().cpu().numpy() for L in raw_layers}
        return raw_out, responses, response_ids

    t0 = time.time()
    feat_batches: dict[str, dict[int, list[np.ndarray]]] = {
        mode: {L: [] for L in raw_layers} for mode in pool_modes
    }
    all_responses: list[str] = []
    all_response_ids: list[list[int]] = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="extract"):
        batch = prompts[i:i + args.batch_size]
        raw_out, responses, response_ids = extract_batch(batch)
        for mode in pool_modes:
            for L in raw_layers:
                feat_batches[mode][L].append(raw_out[mode][L])
        all_responses.extend(responses)
        all_response_ids.extend(response_ids)

    all_activations = {
        mode: {L: np.concatenate(v, axis=0) for L, v in feat_batches[mode].items()}
        for mode in pool_modes
    }
    log(f"Extraction done in {time.time() - t0:.1f}s")

    for mode in pool_modes:
        for L in raw_layers:
            path = output_path(
                args.outdir, args.dataset_name, mode, layer_filename_tag(L),
            )
            meta = build_shared_meta(
                args, prompts, labels, all_responses, all_response_ids,
                hidden_dim, decomps, types=types, hf_layer=L, feature_set=mode,
            )
            np.savez_compressed(
                path,
                activations=all_activations[mode][L],
                meta_pool=np.array(mode),
                **meta,
            )
            log(f"Saved {path} ({all_activations[mode][L].shape})")

    if decomps:
        if "sae" in decomps:
            sae_acc = accumulators["sae"]
            assert isinstance(sae_acc, dict)
            sae_spec = DECOMP_SPECS["sae"]
            for L in raw_layers:
                row, col, val, n_feat = sae_acc[L].concat()
                path = output_path(
                    args.outdir, args.dataset_name, "sae", layer_filename_tag(L),
                )
                layer_meta = build_shared_meta(
                    args, prompts, labels, all_responses, all_response_ids,
                    hidden_dim, decomps, types=types, hf_layer=L, feature_set="sae",
                )
                save_sparse_coo(
                    path, row, col, val, len(prompts), n_feat,
                    meta={
                        "type": "sae",
                        "site": SAE_SITE,
                        "release": sae_spec.release,
                        "sae_id": sae_spec.id_template.format(layer=hf_index_to_scope_layer(L)),
                        "hf_hidden_states_index": str(L),
                        "token": "last_prompt",
                    },
                    extra=layer_meta,
                )
                log(f"Saved {path} ({row.size} nnz)")

        if "transcoder" in decomps:
            tc_acc = accumulators["transcoder"]
            assert isinstance(tc_acc, dict)
            tc_spec = DECOMP_SPECS["transcoder"]
            for L in raw_layers:
                row, col, val, n_feat = tc_acc[L].concat()
                scope_L = hf_index_to_scope_layer(L)
                path = output_path(
                    args.outdir, args.dataset_name, "transcoder", layer_filename_tag(L),
                )
                layer_meta = build_shared_meta(
                    args, prompts, labels, all_responses, all_response_ids,
                    hidden_dim, decomps, types=types, hf_layer=L, feature_set="transcoder",
                )
                save_sparse_coo(
                    path, row, col, val, len(prompts), n_feat,
                    meta={
                        "type": "transcoder",
                        "site": TRANSCODER_SITE,
                        "release": tc_spec.release,
                        "sae_id": tc_spec.id_template.format(layer=scope_L),
                        "hf_hidden_states_index": str(L),
                        "scope_layer": str(scope_L),
                        "hook": f"model.layers.{scope_L}.pre_feedforward_layernorm.output",
                        "token": "last_prompt",
                    },
                    extra=layer_meta,
                )
                log(f"Saved {path} ({row.size} nnz)")

        if "crosscoder" in decomps:
            cc_acc = accumulators["crosscoder"]
            assert isinstance(cc_acc, SparseAccumulator)
            cc_spec = DECOMP_SPECS["crosscoder"]
            row, col, val, n_feat = cc_acc.concat()
            layer_tag = "layer" + "_".join(str(L) for L in CROSSCODER_LAYERS)
            path = output_path(args.outdir, args.dataset_name, "crosscoder", layer_tag)
            cc_meta = build_shared_meta(
                args, prompts, labels, all_responses, all_response_ids,
                hidden_dim, decomps, types=types, feature_set="crosscoder",
            )
            save_sparse_coo(
                path, row, col, val, len(prompts), n_feat,
                meta={
                    "type": "crosscoder",
                    "site": CROSSCODER_SITE,
                    "release": cc_spec.release,
                    "sae_id": cc_spec.sae_id,
                    "loader": "gemma_npz (raw JumpReLU params; not SAELens)",
                    "scope_layers": ",".join(str(L) for L in CROSSCODER_LAYERS),
                    "hf_hidden_states_indices": ",".join(
                        str(scope_layer_to_hf_index(L)) for L in CROSSCODER_LAYERS
                    ),
                    "token": "last_prompt",
                },
                extra=cc_meta,
            )
            log(f"Saved {path} ({row.size} nnz)")

    log(f"Done — {len(output_paths)} files under {args.outdir}/")

    mlp_capture.remove()

    if scope_model_cache:
        log(f"Reused {len(scope_model_cache)} cached Gemma Scope model(s).")
        scope_model_cache.clear()
        if device == "cuda":
            torch.cuda.empty_cache()

    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())