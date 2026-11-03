"""Figure text sizes and matplotlib defaults — one home for the
large-text conventions (user preference; see Planning.md figure
rounds), import-safe anywhere like classifiers.data.

analysis_scripts must never import classifiers.scripts (cuML at load
time), but both packages may import this: scripts/runner_common.py and
analysis_scripts/plots.py read the same dicts, so the two figure
families cannot drift apart (they had — legend sizes diverged while
the sizes lived in two hand-mirrored copies).
"""

from __future__ import annotations

import matplotlib

# The full-size tier: heatmap grids, runner figures.
FS = {"cell": 11, "tick": 14, "label": 19, "subtitle": 14, "title": 22,
      "suptitle": 25, "legend": 13}

# A step smaller, for the single-axes coefficient plots (user
# 2026-07-24: the full sizes were a bit too big there).
FS_SMALL = {"cell": 10, "tick": 12, "label": 14, "subtitle": 13,
            "title": 16, "suptitle": 19, "legend": 12}


def install_rcparams() -> None:
    """FS as matplotlib-wide defaults, so every figure any module ever
    makes — line plots, histograms, future code — is large without
    opting in (the rule from the 2026-07-22 text-size round)."""
    matplotlib.rcParams.update({
        "font.size": FS["tick"],
        "axes.titlesize": FS["title"],
        "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "figure.titlesize": FS["suptitle"],
    })


def heatmap_size(n_rows: int, n_cols: int) -> tuple[float, float]:
    """Panel (width, height) inches that fit an annotated
    n_rows x n_cols heatmap — cells big enough for the 3-digit
    annotations, plus room for labels and the colorbar. The single
    source for a sizing rule that used to be pasted at five call
    sites."""
    return 0.8 * n_cols + 4.2, 0.7 * n_rows + 2.6
