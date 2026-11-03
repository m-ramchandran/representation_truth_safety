"""The one place the direction manifest's column names are written down.

steer.py reads columns that directions.py writes. On 2026-08-24 those two
drifted: `family` became `kind`, `regime` became `uses_target_data`, and
that column's values went from in_fit/out_of_fit/control to
yes/no/comparison. analyze.py was updated, steer.py was not, and nothing
caught it until twelve array tasks had each waited six hours in the GPU
queue and then died in ten seconds on an AttributeError.

So the names live here. The builder checks its output against them, the
sweep checks its input against them, and the preflight checks any manifest
already on disk. Neither end can move alone.
"""
from __future__ import annotations

import pandas as pd

# Every column the manifest must carry.
REQUIRED = ("name",              # steer.py, analyze.py
            "target",            # steer.py, to select a target's vectors
            "kind",              # steer.py, to find the dose base
            "uses_target_data",  # steer.py, to order real before comparison
            "source", "covariance", "gap", "flipped")

# The vocabulary of uses_target_data. steer.py tests against COMPARISON by
# name, so renaming a value here is a silent behaviour change rather than a
# crash — the sort key just stops firing. That is why the values are
# checked, not only the column names.
USES_VALUES = ("yes", "no", "comparison")
COMPARISON = "comparison"

# The kind that supplies each target's dose base.
OWN_DIFF = "own_diff"


def validate(manifest: pd.DataFrame, where: str) -> None:
    """Fail loudly, and with the fix in the message, if the manifest is not
    the shape both ends agreed on."""
    missing = [c for c in REQUIRED if c not in manifest.columns]
    if missing:
        raise SystemExit(
            f"{where}: manifest is missing {missing}; it has "
            f"{list(manifest.columns)}. steering/manifest_schema.py is the "
            "contract between directions.py and steer.py — if the builder "
            "renamed a column, update the readers in the same commit.")
    bad = sorted(set(manifest.uses_target_data.dropna()) - set(USES_VALUES))
    if bad:
        raise SystemExit(
            f"{where}: uses_target_data carries unexpected values {bad}; "
            f"the readers only know {list(USES_VALUES)}.")
    if OWN_DIFF not in set(manifest.kind):
        raise SystemExit(
            f"{where}: no row has kind == {OWN_DIFF!r}, so no target can be "
            "given a dose base.")
