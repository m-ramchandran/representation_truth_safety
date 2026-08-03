"""Shared paths and plotting helpers for the layer-17 boundary figures."""
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "classifiers" / "results" / "data_unwrap"
ARTIFACTS = RESULTS / "boundary_refits"
OUTPUT_ROOT = Path(__file__).resolve().parent
FIGURES = OUTPUT_ROOT / "figures"
TABLES = OUTPUT_ROOT / "tables"
SCOPES = ("common_claim", "harmful_harmless", "wildjailbreak", "merged")
SCOPE_LABELS = {
    "common_claim": "common claim",
    "harmful_harmless": "harmful harmless",
    "wildjailbreak": "wildjailbreak",
    "merged": "merged",
}


def configure_paths(*, results_root: Path | None = None,
                    output_root: Path | None = None) -> None:
    global RESULTS, ARTIFACTS, OUTPUT_ROOT, FIGURES, TABLES
    if results_root is not None:
        RESULTS = Path(results_root) / "data_unwrap"
        ARTIFACTS = RESULTS / "boundary_refits"
    if output_root is not None:
        OUTPUT_ROOT = Path(output_root)
        FIGURES = OUTPUT_ROOT / "figures"
        TABLES = OUTPUT_ROOT / "tables"
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def production_auc(methods: tuple[str, ...]) -> dict[tuple[str, str], float]:
    """Read production CV AUCs for the boundary-figure scopes and methods."""
    per_dataset = pd.read_csv(
        RESULTS / "per_dataset" / "tables" / "metrics_raw_last_layer17.csv")
    merged = pd.read_csv(
        RESULTS / "merged" / "tables" / "metrics_raw_last_layer17.csv")
    auc = {}
    for scope in SCOPES[:-1]:
        rows = per_dataset[
            (per_dataset["kind"] == "cv")
            & (per_dataset["train_dataset"] == scope)]
        for method in methods:
            auc[scope, method] = float(
                rows.loc[rows["method"] == method, "auc"].iloc[0])
    rows = merged[merged["kind"] == "overall"]
    for method in methods:
        auc["merged", method] = float(
            rows.loc[rows["method"] == method, "auc"].iloc[0])
    return auc


def point_style(n: int) -> tuple[float, float]:
    if n < 2_000:
        return 5.0, .40
    if n < 10_000:
        return 2.7, .27
    if n < 100_000:
        return .65, .13
    return .34, .075


def draw_surface(ax, xx: np.ndarray, yy: np.ndarray,
                 score: np.ndarray, threshold: float) -> None:
    centered = np.asarray(score).reshape(xx.shape) - threshold
    scale = max(float(np.quantile(np.abs(centered), .95)), 1e-6)
    levels = np.linspace(-scale, scale, 15)
    ax.contourf(
        xx, yy, np.clip(centered, -scale, scale), levels=levels,
        cmap="RdBu_r", extend="both", alpha=.68)
    if np.nanmin(centered) <= 0 <= np.nanmax(centered):
        ax.contour(xx, yy, centered, levels=[0], colors="#111827",
                   linewidths=1.15, zorder=4)
