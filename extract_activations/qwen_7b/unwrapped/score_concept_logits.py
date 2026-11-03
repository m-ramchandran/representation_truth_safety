#!/usr/bin/env python3
"""Thin wrapper around gemma_3_4b/unwrapped/score_concept_logits.py.

--model defaults to reading meta_model from the source activation file, so it
auto-resolves to Qwen/Qwen2.5-7B-Instruct for activations from this pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_GEMMA_UNWRAPPED = Path(__file__).resolve().parents[2] / "gemma_3_4b" / "unwrapped"
if str(_GEMMA_UNWRAPPED) not in sys.path:
    sys.path.insert(0, str(_GEMMA_UNWRAPPED))

from score_concept_logits import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
