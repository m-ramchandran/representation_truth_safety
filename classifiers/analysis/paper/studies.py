"""Import a study module from ``classifiers.analysis.studies`` by short name.

The paper build drives the study modules rather than duplicating their
computations, and rebinds their output paths first (see ``build.py``), so it
needs them as module objects rather than as imported names.
"""
from __future__ import annotations

import importlib
from types import ModuleType


PACKAGE = "classifiers.analysis.studies"


def load(name: str) -> ModuleType:
    return importlib.import_module(f"{PACKAGE}.{name}")
