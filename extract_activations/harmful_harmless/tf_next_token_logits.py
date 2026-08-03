"""Next-token concept logits (pooled over casing variants) at the last prompt position."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Gemma-3-4B-it (plain tokenizer, no chat template): include space-prefixed forms
# because wrapped prompts usually end with '.' not trailing whitespace.
_TRUE_WORDS = (
    "True", "TRUE", "true",
    " True", " TRUE", " true",
)
_FALSE_WORDS = (
    "False", "FALSE", "false",
    " False", " FALSE", " false",
)
_YES_WORDS = (
    "Yes", "YES", "yes",
    " Yes", " YES", " yes",
)
_NO_WORDS = (
    "No", "NO", "no",
    " No", " NO", " no",
)
_SAFE_WORDS = (
    "Safe", "SAFE", "safe",
    " Safe", " SAFE", " safe",
)
_UNSAFE_WORDS = (
    "Unsafe", "UNSAFE", "unsafe",
    " Unsafe", " UNSAFE", " unsafe",
)
_HARMLESS_WORDS = (
    "Harmless", "HARMLESS", "harmless",
    " Harmless", " HARMLESS", " harmless",
)
_HARMFUL_WORDS = (
    "Harmful", "HARMFUL", "harmful",
    " Harmful", " HARMFUL", " harmful",
)


@dataclass(frozen=True)
class ConceptPair:
    """Binary next-token concept: p_{positive} = softmax over pooled pos vs neg logits."""

    name: str  # used in p_{name}, e.g. true, yes, safe, harmless
    positive_words: tuple[str, ...]
    negative_words: tuple[str, ...]
    mass_key: str  # total vocab prob on either side's variant tokens


CONCEPT_PAIRS: tuple[ConceptPair, ...] = (
    ConceptPair("true", _TRUE_WORDS, _FALSE_WORDS, "tf_mass"),
    ConceptPair("yes", _YES_WORDS, _NO_WORDS, "yes_no_mass"),
    ConceptPair("safe", _SAFE_WORDS, _UNSAFE_WORDS, "safe_unsafe_mass"),
    ConceptPair("harmless", _HARMLESS_WORDS, _HARMFUL_WORDS, "harmful_harmless_mass"),
)

META_SCORING = (
    "last prompt token; each concept: logsumexp over casing variants, "
    "p_{positive}=softmax(pos,neg), mass=sum vocab prob on both sides"
)


def collect_single_token_ids(tokenizer, words: tuple[str, ...]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = int(ids[0])
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    if not out:
        raise ValueError(f"No single-token encodings for words: {words[:3]}...")
    return out


def get_tf_token_id_lists(tokenizer) -> tuple[list[int], list[int]]:
    return (
        collect_single_token_ids(tokenizer, _TRUE_WORDS),
        collect_single_token_ids(tokenizer, _FALSE_WORDS),
    )


def tf_token_id_tensors(tokenizer, device: torch.device | str):
    true_ids, false_ids = get_tf_token_id_lists(tokenizer)
    t = torch.tensor(true_ids, device=device, dtype=torch.long)
    f = torch.tensor(false_ids, device=device, dtype=torch.long)
    return t, f, true_ids, false_ids


def concept_token_id_lists(tokenizer) -> dict[str, list[int]]:
    """Map side name -> token ids, keys: true, false, yes, no, safe, unsafe, harmless, harmful."""
    out: dict[str, list[int]] = {}
    for pair in CONCEPT_PAIRS:
        out[pair.name] = collect_single_token_ids(tokenizer, pair.positive_words)
        neg_name = _negative_side_name(pair)
        out[neg_name] = collect_single_token_ids(tokenizer, pair.negative_words)
    return out


def _negative_side_name(pair: ConceptPair) -> str:
    if pair.name == "true":
        return "false"
    if pair.name == "yes":
        return "no"
    if pair.name == "safe":
        return "unsafe"
    if pair.name == "harmless":
        return "harmful"
    raise ValueError(pair.name)


def concept_token_tensors(
    tokenizer, device: torch.device | str
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    id_lists = concept_token_id_lists(tokenizer)
    tensors = {
        k: torch.tensor(v, device=device, dtype=torch.long) for k, v in id_lists.items()
    }
    return tensors, id_lists


def concept_logit_chunk_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for pair in CONCEPT_PAIRS:
        neg = _negative_side_name(pair)
        keys.extend([f"{pair.name}_logit", f"{neg}_logit", f"p_{pair.name}", pair.mass_key])
    keys.append("top_next_token_id")
    return tuple(keys)


def concept_meta_token_id_arrays(id_lists: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """NPZ metadata keys meta_{side}_token_ids for each pooled side."""
    meta: dict[str, np.ndarray] = {}
    for pair in CONCEPT_PAIRS:
        neg = _negative_side_name(pair)
        meta[f"meta_{pair.name}_token_ids"] = np.array(id_lists[pair.name], dtype=np.int64)
        meta[f"meta_{neg}_token_ids"] = np.array(id_lists[neg], dtype=np.int64)
    return meta


def _score_binary_pair(
    logits: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    pair: ConceptPair,
) -> dict[str, torch.Tensor]:
    neg_name = _negative_side_name(pair)
    pos_logit = torch.logsumexp(logits[:, pos_ids], dim=-1)
    neg_logit = torch.logsumexp(logits[:, neg_ids], dim=-1)
    stacked = torch.stack([pos_logit, neg_logit], dim=-1)
    p_pos = torch.softmax(stacked, dim=-1)[:, 0]
    probs = torch.softmax(logits, dim=-1)
    mass = probs[:, pos_ids].sum(-1) + probs[:, neg_ids].sum(-1)
    return {
        f"{pair.name}_logit": pos_logit,
        f"{neg_name}_logit": neg_logit,
        f"p_{pair.name}": p_pos,
        pair.mass_key: mass,
    }


def score_last_position_concept_logits(
    logits: torch.Tensor,
    concept_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Score T/F, Yes/No, Safe/Unsafe, Harmless/Harmful at the last sequence position."""
    if logits.dim() != 2:
        raise ValueError(f"expected (batch, vocab), got {tuple(logits.shape)}")
    out: dict[str, torch.Tensor] = {}
    for pair in CONCEPT_PAIRS:
        neg = _negative_side_name(pair)
        out.update(
            _score_binary_pair(
                logits,
                concept_tensors[pair.name],
                concept_tensors[neg],
                pair,
            )
        )
    out["top_next_token_id"] = logits.argmax(dim=-1)
    return out


def score_last_position_tf_logits(
    logits: torch.Tensor,
    true_ids: torch.Tensor,
    false_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Legacy T/F-only scorer (subset of :func:`score_last_position_concept_logits`)."""
    if logits.dim() != 2:
        raise ValueError(f"expected (batch, vocab), got {tuple(logits.shape)}")
    out = _score_binary_pair(logits, true_ids, false_ids, CONCEPT_PAIRS[0])
    out["top_next_token_id"] = logits.argmax(dim=-1)
    return out


def stacks_to_numpy(scored: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in scored.items()}
