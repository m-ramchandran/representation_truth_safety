#!/usr/bin/env python
"""Extract Gemma-3-4B activations for PubMedQA: pqa_labeled (all yes/no rows)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pubmedqa_common import main

PUBMEDQA_CONFIG = "pqa_labeled"
DATASET_NAME = "pubmedqa_labeled"
DEFAULT_N = 0
DEFAULT_BALANCE = False

if __name__ == "__main__":
    sys.exit(main(
        PUBMEDQA_CONFIG,
        DATASET_NAME,
        default_n=DEFAULT_N,
        default_balance=DEFAULT_BALANCE,
    ))
