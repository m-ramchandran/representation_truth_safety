"""Compatibility loader for the existing writeup analysis modules."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


WRITEUP_SCRIPTS = Path(__file__).resolve().parents[1] / "writeups" / "scripts"


def load(name: str):
    path = str(WRITEUP_SCRIPTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)
