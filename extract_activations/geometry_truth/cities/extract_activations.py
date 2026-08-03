#!/usr/bin/env python
"""Extract Gemma-3-4B activations for Geometry-of-Truth: got/cities."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from got_common import main

GOT_SHORT_NAME = "cities"
DATASET_NAME = "got_cities"

if __name__ == "__main__":
    sys.exit(main(GOT_SHORT_NAME, DATASET_NAME))
