#!/usr/bin/env python
"""Extract Gemma-3-4B activations for PubMedQA: pqa_artificial (balanced subsample)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pubmedqa_common import main

PUBMEDQA_CONFIG = "pqa_artificial"
DATASET_NAME = "pubmedqa_artificial"
# Max balanced yes/no on pqa_artificial: 15,125 per class (minority cap).
DEFAULT_N = 30250
DEFAULT_BALANCE = True

if __name__ == "__main__":
    sys.exit(main(
        PUBMEDQA_CONFIG,
        DATASET_NAME,
        default_n=DEFAULT_N,
        default_balance=DEFAULT_BALANCE,
    ))
