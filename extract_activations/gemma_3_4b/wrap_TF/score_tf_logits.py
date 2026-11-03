#!/usr/bin/env python3
"""Thin wrapper around gemma_3_4b/unwrapped/score_concept_logits.py."""

from __future__ import annotations

import sys
from pathlib import Path

_UNWRAPPED = Path(__file__).resolve().parents[1] / "unwrapped"
if str(_UNWRAPPED) not in sys.path:
    sys.path.insert(0, str(_UNWRAPPED))

from score_concept_logits import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
