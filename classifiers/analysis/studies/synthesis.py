"""Robustness-first synthesis of the transfer results.

The summaries here are deliberately conservative:
  * claude_blood_test_ranges and claude_diagnosis_severity are excluded
    (the first is never trained on, the second has n = 40);
  * performance is reported as medians rather than means;
  * typical, replication-selected, deployably selected, and oracle-best
    performance are kept apart, since only the first two are achievable
    without target labels;
  * layer behaviour is described by level, adjacent-layer change,
    peakiness, and reproducibility rather than a single best layer;
  * supervised transfer is related to unsupervised domain geometry; and
  * full-space decision boundaries are drawn in PCA mean-plane views.

Usage:
    python -m classifiers.analysis.studies.synthesis
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                             QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .common import (COLLS, DATASETS, EXCLUDED_DATASETS, LAYERS, LINEAR,
                    NONLINEAR, TASK_OF, metrics)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUTPUT_ROOT = HERE
TABLES = OUTPUT_ROOT / "tables"
FIGURES = OUTPUT_ROOT / "figures"
RESULTS = REPO_ROOT / "classifiers" / "results"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

COLLECTION_LABEL = {"tf_wrapped_a": "wrapped", "data_unwrap": "unwrapped"}
FAMILY = {**{m: "linear" for m in LINEAR},
          **{m: "nonlinear" for m in NONLINEAR}}
TASK_ORDER = ["tf", "medical", "jailbreak", "social_iqa", "claude_vds"]
METHOD_ORDER = ["diff", "l1reg", "l2reg", "svm_linear", "lda",
                "qda", "knn", "svm_rbf", "mlp", "xgboost"]
METHOD_LABEL = {
    "diff": "diff", "l1reg": "l1", "l2reg": "l2",
    "svm_linear": "svm_linear", "lda": "lda", "qda": "qda",
    "knn": "knn", "svm_rbf": "svm_rbf", "mlp": "mlp",
    "xgboost": "xgboost",
    "best available": "best observed",
}
REGIME_ORDER = ["within CV", "transfer", "leave-one-dataset-out",
                "leave-task-out", "merged slices"]
REGIME_LABEL = {
    "within CV": "same dataset",
    "transfer": "one source\ndataset",
    "leave-one-dataset-out": "target dataset\nabsent",
    "leave-task-out": "target task\nabsent",
    "merged slices": "pooled,\ntarget included",
}

# Slide palette: stable category identity, high contrast on white.
BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE9"
DENSE_RASTER_PLOTS = {
    "decision-boundaries-2d",
    "decision-boundary-overlays",
    "decision-boundary-overlays-fullspace",
}


def configure_paths(*, results_root: Path | None = None,
                    output_root: Path | None = None) -> None:
    """Override input/output roots for the integrated paper pipeline."""
    global RESULTS, TABLES, FIGURES
    if results_root is not None:
        RESULTS = Path(results_root)
    if output_root is not None:
        output_root = Path(output_root)
        TABLES = output_root / "tables"
        FIGURES = output_root / "figures"
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    """Save PNGs by default; retain SVG only where vector editing is useful."""
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    if stem not in DENSE_RASTER_PLOTS:
        fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def regimes_from_metrics(m: pd.DataFrame,
                         features: list[str] | None = None
                         ) -> dict[str, pd.DataFrame]:
    """Build comparable evaluation regimes, optionally for selected features."""
    selected = m[m.features.isin(features)].copy() if features else m.copy()
    pdd = selected[selected.experiment == "per_dataset"]
    out = {
        "within CV": pdd[pdd.kind == "cv"].copy(),
        "transfer": pdd[pdd.kind == "transfer"].copy(),
        "leave-one-dataset-out":
            selected[(selected.experiment == "leave_out")
                     & (selected.holdout_kind == "dataset")].copy(),
        "leave-task-out":
            selected[(selected.experiment == "leave_out")
                     & (selected.holdout_kind == "task")].copy(),
        "merged slices":
            selected[(selected.experiment == "merged")
                     & selected.kind.isin(["per_dataset",
                                            "eval_only"])].copy(),
    }
    for name, frame in out.items():
        if name == "within CV":
            frame["target"] = frame.train_dataset
        else:
            frame["target"] = frame.test_dataset
        frame["task"] = frame.target.map(TASK_OF)
        frame["family"] = frame.method.map(FAMILY)
        out[name] = frame[frame.family.notna()].copy()
    return out


def raw_regimes(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return regimes_from_metrics(m, ["raw_last"])


def cell_keys(regime: str) -> list[str]:
    if regime == "within CV":
        return ["collection", "layer", "target"]
    if regime == "transfer":
        return ["collection", "layer", "train_dataset", "target"]
    if regime in {"leave-one-dataset-out", "leave-task-out"}:
        return ["collection", "layer", "holdout", "target"]
    return ["collection", "layer", "target"]


def coverage_methods(frame: pd.DataFrame, regime: str) -> set[str]:
    """Methods with at least 80% of the largest method's distinct cells."""
    keys = [k for k in cell_keys(regime) if k != "collection"]
    counts = frame.groupby("method").apply(
        lambda x: x[keys].drop_duplicates().shape[0],
        include_groups=False)
    if counts.empty:
        return set()
    return set(counts[counts >= .8 * counts.max()].index)


def robust_estimands(regimes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Four answers to 'how well do the approaches perform?'.

    The replication-selected panel is selected on the other prompt format.
    The leave-target-out selector picks one method using every other target
    in the same collection, then evaluates it on the held-out target.
    Oracle best is a ceiling: it selects on the evaluation cell itself.
    """
    rows: list[dict] = []
    for regime, base in regimes.items():
        for collection in COLLS:
            frame = base[base.collection == collection].copy()
            if frame.empty:
                continue
            rows.append({"collection": collection, "regime": regime,
                         "estimand": "all methods",
                         "auc": frame.auc.median(), "n": len(frame),
                         "methods": frame.method.nunique()})

            # Select the top quartile on the independent prompt-format
            # replication, while requiring usable coverage on this side.
            other = next(c for c in COLLS if c != collection)
            discovery = base[base.collection == other]
            score = discovery.groupby("method").auc.median().sort_values(
                ascending=False)
            eligible = coverage_methods(frame, regime)
            score = score[score.index.isin(eligible)]
            n_keep = max(1, int(np.ceil(len(score) / 4)))
            keep = list(score.head(n_keep).index)
            selected = frame[frame.method.isin(keep)]
            rows.append({"collection": collection, "regime": regime,
                         "estimand": "replication-selected panel",
                         "auc": selected.auc.median(), "n": len(selected),
                         "methods": ", ".join(keep)})

            # Cross-target method selection. This is target-specific, but the
            # target's own results never contribute to choosing the method.
            chosen_parts = []
            chosen_methods = []
            for target in sorted(frame.target.dropna().unique()):
                train = frame[frame.target != target]
                test = frame[frame.target == target]
                ranking = train.groupby("method").auc.median().sort_values(
                    ascending=False)
                available = set(test.method)
                choice = next((x for x in ranking.index if x in available),
                              None)
                if choice is not None:
                    chosen_parts.append(test[test.method == choice])
                    chosen_methods.append(choice)
            chosen = (pd.concat(chosen_parts, ignore_index=True)
                      if chosen_parts else frame.iloc[:0])
            rows.append({"collection": collection, "regime": regime,
                         "estimand": "leave-target-out selector",
                         "auc": chosen.auc.median(), "n": len(chosen),
                         "methods": ", ".join(sorted(set(chosen_methods)))})

            keys = [k for k in cell_keys(regime) if k != "collection"]
            oracle = frame.groupby(keys, dropna=False).auc.max()
            rows.append({"collection": collection, "regime": regime,
                         "estimand": "oracle best per cell",
                         "auc": oracle.median(), "n": len(oracle),
                         "methods": "selected on test cell"})
    return pd.DataFrame(rows)


def family_gains(regimes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, base in regimes.items():
        keys = cell_keys(regime)
        for collection in COLLS:
            frame = base[base.collection == collection]
            p = (frame.groupby(keys + ["family"], dropna=False).auc.max()
                 .unstack("family"))
            if not {"linear", "nonlinear"}.issubset(p.columns):
                continue
            p = p.dropna(subset=["linear", "nonlinear"])
            delta = p.nonlinear - p.linear
            rows.append({
                "collection": collection, "regime": regime, "n": len(p),
                "best_linear_median": p.linear.median(),
                "best_nonlinear_median": p.nonlinear.median(),
                "median_nonlinear_gain": delta.median(),
                "nonlinear_win_rate": (delta > 0).mean(),
                "delta_q25": delta.quantile(.25),
                "delta_q75": delta.quantile(.75),
            })
    return pd.DataFrame(rows)


def family_oracle_summary(
        regimes: dict[str, pd.DataFrame]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched-cell family summaries, an observed oracle, and method medians."""
    summary_rows, method_rows = [], []
    for regime, base in regimes.items():
        keys = [k for k in cell_keys(regime) if k != "collection"]
        for collection in COLLS:
            frame = base[base.collection == collection].copy()
            for method, g in frame.groupby("method"):
                method_rows.append({
                    "collection": collection, "regime": regime,
                    "method": method, "family": FAMILY.get(method),
                    "median_auc": g.auc.median(), "n": len(g),
                })
            by_family = (frame.groupby(keys + ["family"], dropna=False)
                         .auc.median().rename("auc").reset_index())
            for family, g in by_family.groupby("family"):
                summary_rows.append({
                    "collection": collection, "regime": regime,
                    "summary": family, "median_auc": g.auc.median(),
                    "q25": g.auc.quantile(.25), "q75": g.auc.quantile(.75),
                    "n_cells": len(g),
                })
            oracle = (frame.groupby(keys, dropna=False).auc.max()
                      .rename("auc").reset_index())
            summary_rows.append({
                "collection": collection, "regime": regime,
                "summary": "best observed",
                "median_auc": oracle.auc.median(),
                "q25": oracle.auc.quantile(.25),
                "q75": oracle.auc.quantile(.75),
                "n_cells": len(oracle),
            })
            method_rows.append({
                "collection": collection, "regime": regime,
                "method": "best available", "family": "oracle",
                "median_auc": oracle.auc.median(), "n": len(oracle),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(method_rows)


def classifier_winner_summaries(
        regimes: dict[str, pd.DataFrame]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Win frequency by method and best-observed AUC by target dataset.

    Methods must have adequate coverage in a collection/regime, and only
    matched cells containing every eligible method are compared. Exact ties
    divide one win equally so win shares sum to one within each panel.
    """
    winner_rows, dataset_rows = [], []
    for regime, base in regimes.items():
        keys = [k for k in cell_keys(regime) if k != "collection"]
        for collection in COLLS:
            frame = base[base.collection == collection].copy()
            eligible = [m for m in METHOD_ORDER
                        if m in coverage_methods(frame, regime)]
            if not eligible:
                continue
            performance = (frame[frame.method.isin(eligible)]
                           .groupby(keys + ["method"], dropna=False).auc
                           .median().unstack("method")
                           .reindex(columns=eligible).dropna())
            if performance.empty:
                continue
            winners = performance.eq(performance.max(axis=1), axis=0)
            credits = winners.div(winners.sum(axis=1), axis=0)
            for method in eligible:
                winner_rows.append({
                    "collection": collection, "regime": regime,
                    "method": method, "family": FAMILY[method],
                    "win_share": credits[method].sum() / len(credits),
                    "win_credit": credits[method].sum(),
                    "n_matched_cells": len(credits),
                    "n_eligible_methods": len(eligible),
                })

            best = performance.max(axis=1).rename("best_auc").reset_index()
            for target, g in best.groupby("target"):
                dataset_rows.append({
                    "collection": collection, "regime": regime,
                    "dataset": target,
                    "best_observed_auc": g.best_auc.median(),
                    "q25": g.best_auc.quantile(.25),
                    "q75": g.best_auc.quantile(.75),
                    "n_matched_cells": len(g),
                    "n_eligible_methods": len(eligible),
                })
    return pd.DataFrame(winner_rows), pd.DataFrame(dataset_rows)


def feature_group_summary(m: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw versus other learned representations; histograms stay separate."""
    keep = ["raw_last", "pca_last", "pca_dict_last", "sae_last",
            "transcoder_last"]
    regimes = regimes_from_metrics(m, keep)
    rows = []
    for regime, base in regimes.items():
        keys = [k for k in cell_keys(regime) if k != "collection"]
        for collection in COLLS:
            frame = base[base.collection == collection].copy()
            frame["feature_group"] = np.where(
                frame.features == "raw_last", "raw activations",
                "other representations")
            # Give every representation equal weight before combining them.
            typical_by_feature = (
                frame.groupby(keys + ["feature_group", "features"],
                              dropna=False).auc.median()
                .rename("auc").reset_index())
            typical = (
                typical_by_feature.groupby(keys + ["feature_group"],
                                           dropna=False).auc.median()
                .rename("auc").reset_index())
            oracle = (
                frame.groupby(keys + ["feature_group"],
                              dropna=False).auc.max()
                .rename("auc").reset_index())
            for group in ["raw activations", "other representations"]:
                for label, table in [("typical", typical),
                                     ("best observed", oracle)]:
                    x = table[table.feature_group == group]
                    rows.append({
                        "collection": collection, "regime": regime,
                        "feature_group": group, "summary": label,
                        "median_auc": x.auc.median(),
                        "q25": x.auc.quantile(.25),
                        "q75": x.auc.quantile(.75),
                        "n_cells": len(x),
                    })

    binned = regimes_from_metrics(m, ["binned_activations_last"])[
        "within CV"]
    binned_rows = []
    for (collection, target), g in binned.groupby(
            ["collection", "target"]):
        by_layer = g.groupby("layer").auc.max()
        binned_rows.append({
            "collection": collection, "dataset": target,
            "task": TASK_OF[target], "median_auc": g.auc.median(),
            "best_method_median": by_layer.median(),
        })
    return pd.DataFrame(rows), pd.DataFrame(binned_rows)


def layer_analyses(regimes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, ...]:
    profiles, stability, agreement, rank_stability = [], [], [], []
    for regime, base in regimes.items():
        prof = (base.groupby(["collection", "task", "family", "layer"],
                             dropna=False).auc.median().rename("auc")
                .reset_index())
        oracle_cells = (
            base.groupby(cell_keys(regime) + ["task"], dropna=False)
            .auc.max().rename("auc").reset_index())
        oracle_prof = (
            oracle_cells.groupby(["collection", "task", "layer"],
                                 dropna=False).auc.median()
            .rename("auc").reset_index())
        oracle_prof["family"] = "best observed"
        prof = pd.concat([prof, oracle_prof], ignore_index=True)
        prof["regime"] = regime
        profiles.append(prof)

        # First collapse sources/holdouts, so every method-target curve is one
        # four-point object and large source rosters do not get extra weight.
        curves = (base.groupby(["collection", "method", "family", "task",
                                "target", "layer"], dropna=False).auc.median()
                  .rename("auc").reset_index())
        for keys, g in curves.groupby(
                ["collection", "method", "family", "task", "target"],
                dropna=False):
            g = g.sort_values("layer")
            if set(g.layer) != set(LAYERS):
                continue
            y = g.auc.to_numpy()
            rho = (stats.spearmanr(g.layer, y).statistic
                   if np.ptp(y) > 1e-12 else 0.)
            stability.append({
                "regime": regime, "collection": keys[0],
                "method": keys[1], "family": keys[2], "task": keys[3],
                "target": keys[4], "amplitude": y.max() - y.min(),
                "adjacent_roughness": np.median(np.abs(np.diff(y))),
                "peakiness": y.max() - np.median(y),
                "abs_monotonicity": abs(rho) if np.isfinite(rho) else 0.,
                "best_layer": int(g.loc[g.auc.idxmax(), "layer"]),
            })

        # The regime is supplied externally; build agreement directly.
        best = (curves.sort_values("auc")
                .groupby(["collection", "method", "family", "task", "target"],
                         dropna=False).tail(1)
                [["collection", "method", "family", "task", "target", "layer"]])
        best = best.pivot_table(
            index=["method", "family", "task", "target"],
            columns="collection", values="layer", aggfunc="first").dropna()
        for idx, row in best.iterrows():
            agreement.append({
                "regime": regime, "method": idx[0], "family": idx[1],
                "task": idx[2], "target": idx[3],
                "wrapped_best": int(row.tf_wrapped_a),
                "unwrapped_best": int(row.data_unwrap),
                "agree": row.tf_wrapped_a == row.data_unwrap,
            })

        # Do methods keep the same ordering from layer to layer?
        method_perf = (base.groupby(["collection", "task", "layer", "method"])
                       .auc.median().rename("auc").reset_index())
        for (collection, task), g in method_perf.groupby(
                ["collection", "task"], dropna=False):
            p = g.pivot(index="method", columns="layer", values="auc")
            cors = p.corr(method="spearman")
            vals = cors.to_numpy()[np.triu_indices_from(cors, 1)]
            rank_stability.append({
                "regime": regime, "collection": collection, "task": task,
                "median_layer_rank_correlation": np.nanmedian(vals),
                "min_layer_rank_correlation": np.nanmin(vals),
                "n_methods": len(p),
            })
    return (pd.concat(profiles, ignore_index=True),
            pd.DataFrame(stability), pd.DataFrame(agreement),
            pd.DataFrame(rank_stability))


def method_domain(regimes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for regime in ["transfer", "leave-one-dataset-out", "leave-task-out"]:
        frame = regimes[regime]
        out = (frame.groupby(["collection", "task", "method", "family"])
               .auc.median().rename("auc").reset_index())
        oracle_cells = (
            frame.groupby(cell_keys(regime) + ["task"], dropna=False)
            .auc.max().rename("auc").reset_index())
        oracle = (
            oracle_cells.groupby(["collection", "task"]).auc.median()
            .rename("auc").reset_index())
        oracle["method"] = "best available"
        oracle["family"] = "oracle"
        out = pd.concat([out, oracle], ignore_index=True)
        out["regime"] = regime
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def best_layer_family_contrasts(agreement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, g in agreement.groupby("regime"):
        table = []
        counts = {}
        for family in ["nonlinear", "linear"]:
            x = g[g.family == family].agree
            counts[family] = (int(x.sum()), int(len(x)))
            table.append([int(x.sum()), int((~x).sum())])
        odds, p = stats.fisher_exact(table)
        rows.append({
            "regime": regime,
            "nonlinear_agree": counts["nonlinear"][0],
            "nonlinear_n": counts["nonlinear"][1],
            "nonlinear_rate": counts["nonlinear"][0]
                              / counts["nonlinear"][1],
            "linear_agree": counts["linear"][0],
            "linear_n": counts["linear"][1],
            "linear_rate": counts["linear"][0] / counts["linear"][1],
            "odds_ratio": odds, "fisher_p": p,
        })
    return pd.DataFrame(rows)


def plot_estimands(summary: pd.DataFrame) -> None:
    estimands = ["all methods", "replication-selected panel",
                 "leave-target-out selector", "oracle best per cell"]
    colors = [MUTED, BLUE, TEAL, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.2), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        s = summary[summary.collection == collection]
        for i, (est, color) in enumerate(zip(estimands, colors)):
            d = (s[s.estimand == est].set_index("regime")
                 .reindex(REGIME_ORDER))
            x = np.arange(len(REGIME_ORDER)) + (i - 1.5) * .12
            ax.scatter(x, d.auc, s=58, color=color, label=est, zorder=3)
            ax.plot(x, d.auc, color=color, linewidth=1.4, alpha=.65)
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.set_title(COLLECTION_LABEL[collection], color=INK,
                     fontweight="bold", fontsize=17)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      ["within\nCV", "single-source\ntransfer", "leave one\ndataset",
                       "leave whole\ntask", "pooled\nslices"])
        ax.set_ylim(.48, 1.01)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("Median AUC", fontsize=14)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=11)
    fig.suptitle("Choosing a stronger classifier helps, but does not remove the transfer loss",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "robust-performance-estimands")


def plot_performance_overview(summary: pd.DataFrame) -> None:
    styles = {
        "linear": (BLUE, "o", "linear classifiers"),
        "nonlinear": (ORANGE, "s", "nonlinear classifiers"),
        "best observed": (INK, "D", "best classifier on each test case"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.8), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        s = summary[summary.collection == collection]
        for offset, (name, (color, marker, label)) in zip(
                [-.16, 0, .16], styles.items()):
            d = s[s.summary == name].set_index("regime").reindex(
                REGIME_ORDER)
            x = np.arange(len(REGIME_ORDER)) + offset
            ax.errorbar(
                x, d.median_auc,
                yerr=[d.median_auc - d.q25, d.q75 - d.median_auc],
                fmt=marker, color=color, ecolor=color, alpha=.95,
                capsize=3, markersize=7, linewidth=1.5, label=label)
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      [REGIME_LABEL[x] for x in REGIME_ORDER])
        ax.set_ylim(.45, 1.01)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("AUC", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=10)
    fig.suptitle("Performance across training settings",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "performance-overview-with-oracle")


def plot_classifier_regime_heatmap(methods: pd.DataFrame) -> None:
    order = METHOD_ORDER + ["best available"]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.2), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        p = (methods[methods.collection == collection]
             .pivot(index="method", columns="regime", values="median_auc")
             .reindex(index=order, columns=REGIME_ORDER))
        im = ax.imshow(p, vmin=.45, vmax=1., cmap="viridis",
                       aspect="auto")
        for i in range(len(p)):
            for j in range(len(p.columns)):
                value = p.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < .76 else INK,
                            fontsize=8.5)
        ax.axhline(len(METHOD_ORDER) - .5, color="white", linewidth=2.2)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      [REGIME_LABEL[x] for x in REGIME_ORDER], fontsize=9)
        ax.set_yticks(range(len(order)),
                      [METHOD_LABEL[x] for x in order], fontsize=9)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label="Median AUC", shrink=.75, pad=.02)
    fig.suptitle("No classifier is best in every training setting",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "classifier-by-training-setting")


def plot_classifier_winner_frequency(winners: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.1), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        p = (winners[winners.collection == collection]
             .pivot(index="method", columns="regime", values="win_share")
             .reindex(index=METHOD_ORDER, columns=REGIME_ORDER))
        im = ax.imshow(
            p, vmin=0, vmax=max(.5, winners.win_share.max()),
            cmap="YlGnBu", aspect="auto")
        for i, method in enumerate(p.index):
            for j in range(len(p.columns)):
                value = p.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                            color="white" if value >= .28 else INK,
                            fontsize=9)
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            color=MUTED, fontsize=10)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      [REGIME_LABEL[x] for x in REGIME_ORDER], fontsize=9)
        ax.set_yticks(range(len(METHOD_ORDER)),
                      [METHOD_LABEL[x] for x in METHOD_ORDER], fontsize=9)
        for tick, method in zip(ax.get_yticklabels(), METHOD_ORDER):
            tick.set_color(BLUE if FAMILY[method] == "linear" else ORANGE)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label="Share of matched test cases won",
                 shrink=.75, pad=.02)
    fig.suptitle("Which classifier wins depends on the training setting",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "classifier-winner-frequency")


def plot_best_classifier_by_dataset(dataset_best: pd.DataFrame) -> None:
    dataset_order = [
        dataset for task in TASK_ORDER for dataset in DATASETS
        if TASK_OF[dataset] == task]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.2), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        p = (dataset_best[dataset_best.collection == collection]
             .pivot(index="dataset", columns="regime",
                    values="best_observed_auc")
             .reindex(index=dataset_order, columns=REGIME_ORDER))
        im = ax.imshow(p, vmin=.45, vmax=1., cmap="viridis",
                       aspect="auto")
        for i in range(len(p)):
            for j in range(len(p.columns)):
                value = p.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                            color="white" if value < .76 else INK,
                            fontsize=9)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      [REGIME_LABEL[x] for x in REGIME_ORDER], fontsize=9)
        ax.set_yticks(
            range(len(dataset_order)),
            [x.replace("_", "\n") for x in dataset_order], fontsize=9)
        cumulative = 0
        for task in TASK_ORDER[:-1]:
            cumulative += sum(TASK_OF[d] == task for d in dataset_order)
            ax.axhline(cumulative - .5, color="white", linewidth=2)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label="Best observed AUC",
                 shrink=.75, pad=.02)
    fig.suptitle("Best observed classifier performance for every dataset",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "best-classifier-by-dataset-setting")


def plot_feature_groups(summary: pd.DataFrame) -> None:
    styles = {
        ("raw activations", "typical"):
            (BLUE, "o", "-", "raw activations: typical"),
        ("raw activations", "best observed"):
            (BLUE, "D", "--", "raw activations: best observed"),
        ("other representations", "typical"):
            (ORANGE, "o", "-", "other representations: typical"),
        ("other representations", "best observed"):
            (ORANGE, "D", "--", "other representations: best observed"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.8), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        s = summary[summary.collection == collection]
        for (group, kind), (color, marker, line, label) in styles.items():
            d = s[(s.feature_group == group) & (s.summary == kind)]
            d = d.set_index("regime").reindex(REGIME_ORDER)
            x = np.arange(len(REGIME_ORDER))
            ax.plot(x, d.median_auc, color=color, marker=marker,
                    linestyle=line, linewidth=1.8, markersize=6,
                    label=label)
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.set_xticks(range(len(REGIME_ORDER)),
                      [REGIME_LABEL[x] for x in REGIME_ORDER])
        ax.set_ylim(.45, 1.01)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Median AUC", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("Raw activations retain the strongest transferable signal",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "raw-versus-other-features")


def plot_family_gain(gains: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.8), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        s = gains[gains.collection == collection].set_index("regime").reindex(
            REGIME_ORDER)
        y = np.arange(len(s))
        ax.axvline(0, color=GRID, linewidth=1.2)
        ax.errorbar(s.median_nonlinear_gain, y,
                    xerr=[s.median_nonlinear_gain - s.delta_q25,
                          s.delta_q75 - s.median_nonlinear_gain],
                    fmt="o", color=PURPLE, ecolor="#A78BFA", capsize=4,
                    markersize=7)
        for yi, val, right, win in zip(
                y, s.median_nonlinear_gain, s.delta_q75,
                s.nonlinear_win_rate):
            # Put the label beyond the full interquartile interval, not on
            # top of the whisker.
            ax.text(max(val, right) + .007, yi,
                    f"{val:+.3f} ({win:.0%} wins)",
                    ha="left", va="center",
                    fontsize=10, color=INK)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=17,
                     fontweight="bold", color=INK)
        ax.set_yticks(y, ["within CV", "single-source transfer",
                         "leave one dataset", "leave whole task",
                         "pooled slices"])
        ax.set_xlim(-.07, .17)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=11)
    axes[0].set_xlabel("Median(best nonlinear - best linear) AUC", fontsize=13)
    axes[1].set_xlabel("Median(best nonlinear - best linear) AUC", fontsize=13)
    fig.suptitle("Nonlinear classifiers help most when the target task is absent",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "nonlinear-gain-by-regime")


def plot_layer_profiles(profiles: pd.DataFrame) -> None:
    s = profiles[profiles.regime == "leave-task-out"]
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(15.5, 4.7),
                             sharey=True, layout="constrained")
    styles = {
        ("tf_wrapped_a", "linear"):
            (BLUE, "-", "o", "linear, wrapped (solid)"),
        ("tf_wrapped_a", "nonlinear"):
            (ORANGE, "-", "o", "nonlinear, wrapped (solid)"),
        ("data_unwrap", "linear"):
            (BLUE, "--", "s", "linear, unwrapped (dashed)"),
        ("data_unwrap", "nonlinear"):
            (ORANGE, "--", "s", "nonlinear, unwrapped (dashed)"),
        ("tf_wrapped_a", "best observed"):
            (INK, "-", "o", "best observed, wrapped (solid)"),
        ("data_unwrap", "best observed"):
            (INK, "--", "s", "best observed, unwrapped (dashed)"),
    }
    for ax, task in zip(axes, TASK_ORDER):
        g = s[s.task == task]
        for key, (color, ls, marker, label) in styles.items():
            q = g[(g.collection == key[0]) & (g.family == key[1])]
            ax.plot(q.layer, q.auc, marker=marker, color=color, linestyle=ls,
                    linewidth=2.1, markersize=5, label=label)
        ax.axhline(.5, color=GRID, linewidth=.8)
        ax.set_title(task.replace("_", " "), fontsize=13, fontweight="bold")
        ax.set_xticks(LAYERS)
        ax.set_ylim(.15, 1.01)
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("AUC when the entire task is held out", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=9.2, handlelength=3.4)
    fig.suptitle("The best layer depends on the task",
                 fontsize=19, color=INK, fontweight="bold")
    save(fig, "layer-profiles-by-task")


def plot_stability(stability: pd.DataFrame) -> None:
    # Median across collections, targets, and regimes: a method-level
    # descriptively stable/peaky leaderboard.
    s = (stability.groupby(["method", "family"])
         [["amplitude", "adjacent_roughness", "peakiness"]].median()
         .reset_index().sort_values("amplitude"))
    fig, ax = plt.subplots(figsize=(10.2, 6.2), layout="constrained")
    y = np.arange(len(s))
    colors = s.family.map({"linear": BLUE, "nonlinear": ORANGE})
    ax.hlines(y, 0, s.amplitude, color=GRID, linewidth=2)
    ax.scatter(s.amplitude, y, color=colors, s=62, zorder=3)
    for yi, row in s.reset_index(drop=True).iterrows():
        ax.text(row.amplitude + .004, yi,
                f"adjacent change {row.adjacent_roughness:.3f}",
                va="center", fontsize=10, color=MUTED)
    ax.set_yticks(y, [METHOD_LABEL.get(x, x) for x in s.method])
    ax.set_xlabel("Typical change between the best and worst measured layer (AUC)",
                  fontsize=13)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_title("Layer sensitivity depends more on the classifier than on linearity",
                 fontsize=18, color=INK, fontweight="bold")
    save(fig, "classifier-layer-stability")


def plot_method_domain(table: pd.DataFrame) -> None:
    s = table[table.regime == "leave-task-out"]
    order = METHOD_ORDER + ["best available"]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.2), sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        p = (s[s.collection == collection].pivot(
            index="method", columns="task", values="auc")
             .reindex(index=order, columns=TASK_ORDER))
        im = ax.imshow(p, vmin=.25, vmax=1., cmap="viridis", aspect="auto")
        for i in range(len(p)):
            for j in range(len(p.columns)):
                v = p.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < .72 else INK, fontsize=9)
        ax.set_xticks(range(len(p.columns)),
                      [x.replace("_", "\n") for x in p.columns],
                      fontsize=9)
        ax.axhline(len(METHOD_ORDER) - .5, color="white", linewidth=2.2)
        ax.set_yticks(range(len(p)),
                      [METHOD_LABEL[x] for x in p.index], fontsize=9)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=16,
                     fontweight="bold", color=INK)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label="Median leave-task-out AUC",
                 shrink=.75, pad=.02)
    fig.suptitle("The best classifier depends on the target task",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "classifier-by-domain")


def balanced_geometry(layer: int, max_per_dataset: int = 2000
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Exact data_unwrap geometry, label-blind distance plus label axes."""
    from classifiers.data import load_datasets

    loaded = load_datasets(layer, "last", ("raw",), "data_unwrap")
    rng = np.random.default_rng(1900 + layer)
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in DATASETS:
        X = np.asarray(loaded[name]["blocks"]["raw"], dtype=np.float32)
        y = np.asarray(loaded[name]["y"], dtype=np.int8)
        if len(X) > max_per_dataset:
            # Stratified equal-domain subsample.
            take = []
            for label in [0, 1]:
                ids = np.flatnonzero(y == label)
                n = min(len(ids), max_per_dataset // 2)
                take.extend(rng.choice(ids, n, replace=False))
            take = np.asarray(take)
            X, y = X[take], y[take]
        samples[name] = (X, y)
    stack = np.concatenate([x for x, _ in samples.values()])
    mean = stack.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = stack.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.
    del stack

    centroids, axes, radii, class_centroids, ds_rows = {}, {}, {}, {}, []
    standardized = {}
    for name, (X, y) in samples.items():
        Z = (X - mean) / scale
        standardized[name] = (Z, y)
        c = Z.mean(axis=0)
        c0, c1 = Z[y == 0].mean(axis=0), Z[y == 1].mean(axis=0)
        r = np.sqrt(np.mean(np.sum((Z - c) ** 2, axis=1)))
        r0 = np.sqrt(np.mean(np.sum((Z[y == 0] - c0) ** 2, axis=1)))
        r1 = np.sqrt(np.mean(np.sum((Z[y == 1] - c1) ** 2, axis=1)))
        centroids[name], axes[name], radii[name] = c, c1 - c0, r
        class_centroids[name] = (c0, c1)
        ds_rows.append({
            "layer": layer, "dataset": name, "task": TASK_OF[name],
            "n_geometry": len(Z),
            "label_gap_over_radius": np.linalg.norm(c1 - c0)
                                     / np.sqrt(.5 * (r0 ** 2 + r1 ** 2)),
        })

    pair_rows = []
    names = list(DATASETS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denom = np.sqrt(.5 * (radii[a] ** 2 + radii[b] ** 2))
            axis_denom = np.linalg.norm(axes[a]) * np.linalg.norm(axes[b])
            pair_rows.append({
                "layer": layer, "a": a, "b": b,
                "same_task": TASK_OF[a] == TASK_OF[b],
                "centroid_distance": np.linalg.norm(
                    centroids[a] - centroids[b]) / denom,
                "class_conditional_distance": .5 * (
                    np.linalg.norm(class_centroids[a][0]
                                   - class_centroids[b][0])
                    + np.linalg.norm(class_centroids[a][1]
                                     - class_centroids[b][1])) / denom,
                "label_axis_cosine":
                    float(np.dot(axes[a], axes[b]) / axis_denom)
                    if axis_denom else np.nan,
            })
    return pd.DataFrame(pair_rows), pd.DataFrame(ds_rows), standardized


def geometry_supervision(regimes: dict[str, pd.DataFrame]
                         ) -> tuple[pd.DataFrame, pd.DataFrame,
                                    pd.DataFrame, dict]:
    pair_parts, dataset_parts, correlations = [], [], []
    boundary_data = None
    transfer = regimes["transfer"]
    within = regimes["within CV"]
    # Exact unwrapped L17 activations are local. The other three result
    # layers are represented in the metrics but their multi-GB activation
    # caches are not local, so do not silently download them in a writeup
    # analysis. This is a one-layer bridge, explicitly not a layer claim.
    for layer in [17]:
        pairs, datasets, standardized = balanced_geometry(layer)
        if layer == 17:
            boundary_data = standardized
        pair_parts.append(pairs)
        dataset_parts.append(datasets)

        # Symmetrise supervised transfer to match the label-blind distance.
        t = (transfer[(transfer.collection == "data_unwrap")
                      & (transfer.layer == layer)]
             .groupby(["train_dataset", "target"]).auc.median())
        vals = []
        for row in pairs.itertuples():
            ab = t.get((row.a, row.b), np.nan)
            ba = t.get((row.b, row.a), np.nan)
            vals.append(np.nanmedian([ab, ba]))
        pairs["symmetric_transfer_auc"] = vals
        valid = pairs.dropna(subset=["symmetric_transfer_auc"])
        for metric in ["centroid_distance", "class_conditional_distance",
                       "label_axis_cosine"]:
            r = stats.spearmanr(valid[metric],
                                valid.symmetric_transfer_auc)
            correlations.append({
                "layer": layer, "analysis": "domain pair -> transfer",
                "geometry_metric": metric, "spearman_rho": r.statistic,
                "p_value": r.pvalue, "n": len(valid),
            })
        # Partial Spearman correlations distinguish a genuine distance
        # association from one inherited through label-axis alignment.
        for metric, control in [
                ("centroid_distance", "label_axis_cosine"),
                ("label_axis_cosine", "centroid_distance")]:
            x = stats.rankdata(valid[metric])
            y = stats.rankdata(valid.symmetric_transfer_auc)
            z = np.column_stack([
                np.ones(len(valid)),
                stats.rankdata(valid[control]),
            ])
            x_resid = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
            y_resid = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
            r = stats.pearsonr(x_resid, y_resid)
            correlations.append({
                "layer": layer,
                "analysis": f"domain pair -> transfer | {control}",
                "geometry_metric": metric,
                "spearman_rho": r.statistic, "p_value": r.pvalue,
                "n": len(valid),
            })
        cross_task = valid[~valid.same_task]
        for metric in ["centroid_distance", "label_axis_cosine"]:
            r = stats.spearmanr(cross_task[metric],
                                cross_task.symmetric_transfer_auc)
            correlations.append({
                "layer": layer,
                "analysis": "different-task pair -> transfer",
                "geometry_metric": metric, "spearman_rho": r.statistic,
                "p_value": r.pvalue, "n": len(cross_task),
            })
        for metric, control in [
                ("centroid_distance", "label_axis_cosine"),
                ("label_axis_cosine", "centroid_distance")]:
            x = stats.rankdata(cross_task[metric])
            y = stats.rankdata(cross_task.symmetric_transfer_auc)
            z = np.column_stack([
                np.ones(len(cross_task)),
                stats.rankdata(cross_task[control]),
            ])
            x_resid = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
            y_resid = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
            r = stats.pearsonr(x_resid, y_resid)
            correlations.append({
                "layer": layer,
                "analysis":
                    f"different-task pair -> transfer | {control}",
                "geometry_metric": metric,
                "spearman_rho": r.statistic, "p_value": r.pvalue,
                "n": len(cross_task),
            })

        cv = (within[(within.collection == "data_unwrap")
                     & (within.layer == layer)]
              .groupby("target").auc.median())
        datasets["median_within_cv_auc"] = datasets.dataset.map(cv)
        valid_ds = datasets.dropna(subset=["median_within_cv_auc"])
        r = stats.spearmanr(valid_ds.label_gap_over_radius,
                            valid_ds.median_within_cv_auc)
        correlations.append({
            "layer": layer, "analysis": "dataset geometry -> within CV",
            "geometry_metric": "label_gap_over_radius",
            "spearman_rho": r.statistic, "p_value": r.pvalue,
            "n": len(valid_ds),
        })
    return (pd.concat(pair_parts, ignore_index=True),
            pd.concat(dataset_parts, ignore_index=True),
            pd.DataFrame(correlations), boundary_data)


def transfer_predictor_increment(
        pairs: pd.DataFrame, datasets: pd.DataFrame,
        regimes: dict[str, pd.DataFrame]
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What still predicts directed transfer after diff-axis alignment?

    The transfer outcome is the median AUC across available classifiers for
    one source→target pair at exact unwrapped L17.  Each unordered geometry
    pair becomes two directed rows so source and target separability can be
    distinguished.
    """
    ds = datasets.set_index("dataset")
    transfer = (
        regimes["transfer"][
            (regimes["transfer"].collection == "data_unwrap")
            & (regimes["transfer"].layer == 17)]
        .groupby(["train_dataset", "target"]).auc.median())
    rows = []
    for pair in pairs.itertuples():
        for source, target in [(pair.a, pair.b), (pair.b, pair.a)]:
            rows.append({
                "source": source, "target": target,
                "transfer_auc": transfer.get((source, target), np.nan),
                "diff_direction_cosine": pair.label_axis_cosine,
                "centroid_distance": pair.centroid_distance,
                "class_centroid_distance":
                    pair.class_conditional_distance,
                "same_task": float(pair.same_task),
                "source_label_gap_over_radius":
                    ds.loc[source, "label_gap_over_radius"],
                "target_label_gap_over_radius":
                    ds.loc[target, "label_gap_over_radius"],
                "source_within_cv_auc":
                    ds.loc[source, "median_within_cv_auc"],
                "target_within_cv_auc":
                    ds.loc[target, "median_within_cv_auc"],
            })
    detail = pd.DataFrame(rows).dropna()

    # Rank-based correlation remaining after accounting for direction
    # alignment.  This is the same calculation previously called "partial
    # Spearman", written out in the report instead of relying on the name.
    y = stats.rankdata(detail.transfer_auc)
    alignment = stats.rankdata(detail.diff_direction_cosine)
    controls = np.column_stack([np.ones(len(detail)), alignment])
    y_resid = y - controls @ np.linalg.lstsq(
        controls, y, rcond=None)[0]
    factor_specs = [
        ("centroid_distance", "dataset-centre distance"),
        ("class_centroid_distance", "class-centroid distance"),
        ("same_task", "same task"),
        ("source_label_gap_over_radius", "source separability"),
        ("target_label_gap_over_radius", "target separability"),
        ("target_within_cv_auc", "target within-CV AUC"),
    ]
    incremental_rows = []
    for column, label in factor_specs:
        x = detail[column].to_numpy()
        if column != "same_task":
            x = stats.rankdata(x)
        x_resid = x - controls @ np.linalg.lstsq(
            controls, x, rcond=None)[0]
        result = stats.pearsonr(x_resid, y_resid)
        incremental_rows.append({
            "factor": label, "column": column,
            "correlation_after_direction_alignment": result.statistic,
            "p_value": result.pvalue, "n": len(detail),
        })
    incremental = pd.DataFrame(incremental_rows)

    # Simple held-target-out prediction: train a linear model on ten target
    # datasets, then predict all incoming transfers to the eleventh.
    model_specs = [
        ("direction alignment only", ["diff_direction_cosine"]),
        ("+ dataset-centre distance",
         ["diff_direction_cosine", "centroid_distance"]),
        ("+ same-task flag", ["diff_direction_cosine", "same_task"]),
        ("+ target separability",
         ["diff_direction_cosine", "target_label_gap_over_radius"]),
        ("all measured factors",
         ["diff_direction_cosine", "centroid_distance",
          "class_centroid_distance", "same_task",
          "source_label_gap_over_radius", "target_label_gap_over_radius"]),
    ]
    prediction_rows = []
    for label, features in model_specs:
        prediction = np.full(len(detail), np.nan)
        for target in detail.target.unique():
            test = detail.target == target
            train = ~test
            scaler = StandardScaler().fit(detail.loc[train, features])
            model = LinearRegression().fit(
                scaler.transform(detail.loc[train, features]),
                detail.loc[train, "transfer_auc"])
            prediction[test] = model.predict(
                scaler.transform(detail.loc[test, features]))
        error = prediction - detail.transfer_auc.to_numpy()
        prediction_rows.append({
            "model": label, "n": len(detail),
            "held_target_out_mae": np.mean(np.abs(error)),
            "held_target_out_rmse": np.sqrt(np.mean(error ** 2)),
            "prediction_spearman_correlation":
                stats.spearmanr(prediction, detail.transfer_auc).statistic,
        })
    prediction = pd.DataFrame(prediction_rows)
    return detail, incremental, prediction


def transfer_additive_regression(
        pairs: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    """Sequential R² decomposition on the same 55 symmetric pair points.

    Rank-transforming the continuous quantities makes the first one-variable
    R² exactly Spearman correlation squared.  Predictor order is fixed to
    answer the substantive question: alignment first, then separability,
    then the label-free distance and task indicators.
    """
    ds = datasets.set_index("dataset")
    frame = pairs.copy()
    frame["pair_average_separability"] = [
        .5 * (ds.loc[a, "label_gap_over_radius"]
              + ds.loc[b, "label_gap_over_radius"])
        for a, b in zip(frame.a, frame.b)]
    specs = [
        ("label_axis_cosine", "diff-in-means alignment"),
        ("pair_average_separability", "pair-average separability"),
        ("centroid_distance", "dataset-centre distance"),
        ("same_task", "same task"),
    ]
    y = stats.rankdata(frame.symmetric_transfer_auc)
    y = (y - y.mean()) / y.std(ddof=0)
    rows, previous = [], 0.
    for step in range(1, len(specs) + 1):
        columns = specs[:step]
        X = np.column_stack([
            frame[column].to_numpy()
            if column == "same_task"
            else stats.rankdata(frame[column])
            for column, _ in columns])
        X = StandardScaler().fit_transform(X)
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        rows.append({
            "step": step, "factor": specs[step - 1][1],
            "r_squared": r_squared,
            "added_r_squared": r_squared - previous,
            "n_pairs": len(frame),
        })
        previous = r_squared
    rows.append({
        "step": len(specs) + 1, "factor": "unexplained",
        "r_squared": 1., "added_r_squared": 1. - previous,
        "n_pairs": len(frame),
    })
    return pd.DataFrame(rows)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of a histogram represented by bin values and counts."""
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(
        np.cumsum(weights), weights.sum() / 2, side="left")])


def classifier_error_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct/error boundary distances and pairwise error agreement."""
    distance_rows = []
    excluded = set(EXCLUDED_DATASETS)
    margin_methods = [m for m in METHOD_ORDER if m != "knn"]
    for collection in COLLS:
        for layer in LAYERS:
            path = (RESULTS / collection / "leave_out" / "tables"
                    / f"margin_hist_raw_last_layer{layer}_task.csv")
            frame = pd.read_csv(path)
            frame = frame[
                frame.method.isin(margin_methods)
                & frame.series.isin(["correct", "wrong"])].copy()
            midpoint = (frame.bin_left + frame.bin_right) / 2
            # Plain probability distance: 0 is the 0.5 decision boundary
            # and 0.5 is the largest possible distance.
            frame["boundary_distance"] = np.abs(
                .5 * np.tanh(midpoint.to_numpy() / 2))
            for (method, task, outcome), g in frame.groupby(
                    ["method", "row", "series"]):
                distance_rows.append({
                    "collection": collection, "layer": layer,
                    "task": task, "method": method,
                    "outcome": ("Correct" if outcome == "correct"
                                else "Misclassified"),
                    "median_boundary_distance": _weighted_median(
                        g.boundary_distance.to_numpy(),
                        g["count"].to_numpy()),
                    "n_samples": int(g["count"].sum()),
                })
    distances = pd.DataFrame(distance_rows)

    # In-domain Jaccard overlap of the samples each pair gets wrong.
    # A method with low overlap is making a more distinctive set of errors.
    overlap_rows = []
    for collection in COLLS:
        for layer in LAYERS:
            path = (RESULTS / collection / "per_dataset" / "tables"
                    / f"error_overlap_raw_last_layer{layer}.csv")
            frame = pd.read_csv(path)
            frame = frame[~frame.dataset.isin(excluded)]
            for row in frame.itertuples(index=False):
                a = row.method
                if a not in METHOD_ORDER:
                    continue
                for b in METHOD_ORDER:
                    if b <= a or b not in frame.columns:
                        continue
                    value = getattr(row, b)
                    if np.isfinite(value):
                        for method in [a, b]:
                            overlap_rows.append({
                                "collection": collection, "layer": layer,
                                "dataset": row.dataset, "method": method,
                                "other_method": b if method == a else a,
                                "jaccard": value,
                            })
    overlap = pd.DataFrame(overlap_rows)
    overlap["family"] = overlap.method.map(FAMILY)
    return distances, overlap


def coefficient_geometry_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compact median/IQR summaries of the retained coefficient geometry.

    The first table describes single-dataset directions.  Symmetric plain
    and pooled-covariance measures use each unordered pair once; the
    target-covariance measure retains both source→target orientations.
    The second table follows one representative fitted direction (l2)
    through the Merged / Leave-out / Single triangle.
    """
    retained = set(DATASETS)
    method_rows = []
    for collection in COLLS:
        for method in ["diff", "l2reg", "svm_linear", "l1reg", "lda"]:
            parts = []
            for layer in [17, 22, 29]:
                path = (RESULTS / collection / "coefficients" / "tables"
                        / f"similarity_{method}_raw_last_layer{layer}.csv")
                frame = pd.read_csv(path)
                frame = frame[
                    frame.dataset_a.isin(retained)
                    & frame.dataset_b.isin(retained)
                    & (frame.dataset_a != frame.dataset_b)].copy()
                frame["layer"] = layer
                frame["same_task"] = [
                    TASK_OF[a] == TASK_OF[b]
                    for a, b in zip(frame.dataset_a, frame.dataset_b)]
                parts.append(frame)
            combined = pd.concat(parts, ignore_index=True)
            for measure, measured in combined.groupby("measure"):
                if measure != "mahalanobis_reg_cov_test":
                    measured = measured[
                        measured.dataset_a < measured.dataset_b]
                for scope, selected in [
                        ("all", measured),
                        ("within task", measured[measured.same_task]),
                        ("between task", measured[~measured.same_task])]:
                    values = selected.value.dropna()
                    method_rows.append({
                        "collection": collection,
                        "method": METHOD_LABEL.get(method, method),
                        "measure": measure, "scope": scope,
                        "median": values.median(),
                        "q25": values.quantile(.25),
                        "q75": values.quantile(.75),
                        "n": len(values),
                    })

    triangle_rows = []
    comparisons = [
        ("merged ↔ single", "dataset", "M_S"),
        ("LODO ↔ single", "dataset", "L_S"),
        ("LOTO ↔ single", "task", "L_S"),
        ("merged ↔ LODO", "dataset", "M_L"),
        ("merged ↔ LOTO", "task", "M_L"),
    ]
    for collection in COLLS:
        frames = {}
        for kind in ["dataset", "task"]:
            parts = []
            for layer in [17, 22, 29]:
                path = (RESULTS / collection / "coefficients" / "tables"
                        / f"mls_raw_last_layer{layer}_{kind}.csv")
                frame = pd.read_csv(path)
                frame = frame[
                    frame.dataset.isin(retained)
                    & (frame.method == "l2reg")].copy()
                frame["layer"] = layer
                parts.append(frame)
            frames[kind] = pd.concat(parts, ignore_index=True)
        for comparison, kind, pair in comparisons:
            selected = frames[kind][frames[kind].pair == pair]
            for measure, measured in selected.groupby("measure"):
                values = measured.value.dropna()
                triangle_rows.append({
                    "collection": collection,
                    "comparison": comparison,
                    "method": "l2", "measure": measure,
                    "median": values.median(),
                    "q25": values.quantile(.25),
                    "q75": values.quantile(.75),
                    "n": len(values),
                })
    return pd.DataFrame(method_rows), pd.DataFrame(triangle_rows)


def coefficient_pca_null(
        draws: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diff and l2 coefficient-PCA spectra after the mean, with a null.

    The locally retained archives contain pairwise cosine matrices but not
    every original coefficient vector.  That is sufficient: eigenvalues of
    the centred Gram matrix equal the non-zero PCA eigenvalues of the
    centred unit coefficient vectors exactly.
    """
    names = list(DATASETS)
    rows, spectra = [], []
    for collection in COLLS:
        for method in ["diff", "l2reg"]:
            for layer in [17, 22, 29]:
                path = (RESULTS / collection / "coefficients" / "tables"
                        / f"similarity_{method}_raw_last_layer{layer}.csv")
                frame = pd.read_csv(path)
                plain = frame[frame.measure == "plain"]
                gram = (
                    plain.pivot(index="dataset_a", columns="dataset_b",
                                values="value")
                    .reindex(index=names, columns=names).to_numpy())
                gram = .5 * (gram + gram.T)
                np.fill_diagonal(gram, 1.)
                center = (
                    np.eye(len(names))
                    - np.ones((len(names), len(names))) / len(names))
                eigenvalues = np.linalg.eigvalsh(center @ gram @ center)
                eigenvalues = eigenvalues[eigenvalues > 1e-10][::-1]
                spectrum = eigenvalues / eigenvalues.sum()
                cumulative = np.cumsum(spectrum)
                rows.append({
                    "source": "observed", "collection": collection,
                    "method": METHOD_LABEL[method], "layer": layer,
                    "n_directions": len(names), "ambient_dimensions": 2560,
                    "pc1_explained": spectrum[0],
                    "k80": np.searchsorted(cumulative, .80) + 1,
                    "k90": np.searchsorted(cumulative, .90) + 1,
                    "k95": np.searchsorted(cumulative, .95) + 1,
                })
                for pc, (explained, total) in enumerate(
                        zip(spectrum, cumulative), start=1):
                    spectra.append({
                        "source": "observed", "collection": collection,
                        "method": METHOD_LABEL[method], "layer": layer,
                        "pc": pc, "explained_variance": explained,
                        "cumulative_variance": total,
                    })

    rng = np.random.default_rng(20260729)
    null_rows = []
    for _ in range(draws):
        vectors = rng.normal(size=(len(names), 2560))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors -= vectors.mean(axis=0)
        gram = vectors @ vectors.T
        eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = eigenvalues[eigenvalues > 1e-10][::-1]
        spectrum = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(spectrum)
        null_rows.append({
            "pc1_explained": spectrum[0],
            "k80": np.searchsorted(cumulative, .80) + 1,
            "k90": np.searchsorted(cumulative, .90) + 1,
            "k95": np.searchsorted(cumulative, .95) + 1,
        })
        for pc, (explained, total) in enumerate(
                zip(spectrum, cumulative), start=1):
            spectra.append({
                "source": "random-unit null", "collection": "null",
                "method": "random-unit null", "layer": np.nan,
                "draw": len(null_rows), "pc": pc,
                "explained_variance": explained,
                "cumulative_variance": total,
            })
    null = pd.DataFrame(null_rows)
    rows.append({
        "source": f"random-unit null ({draws} draws)",
        "collection": "null", "method": "random-unit null",
        "layer": np.nan,
        "n_directions": len(names), "ambient_dimensions": 2560,
        "pc1_explained": null.pc1_explained.mean(),
        "pc1_sd": null.pc1_explained.std(ddof=1),
        "pc1_q025": null.pc1_explained.quantile(.025),
        "pc1_q975": null.pc1_explained.quantile(.975),
        "k80": null.k80.median(), "k90": null.k90.median(),
        "k95": null.k95.median(),
    })
    return pd.DataFrame(rows), pd.DataFrame(spectra)


def diff_vs_l2_out_of_domain(
        regimes: dict[str, pd.DataFrame]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched-cell AUC differences for diff-in-means versus l2."""
    transfer = regimes["transfer"].copy()
    transfer["same_task"] = (
        transfer.train_dataset.map(TASK_OF)
        == transfer.target.map(TASK_OF))
    cases = [
        ("same-task transfer", transfer[transfer.same_task]),
        ("cross-task transfer", transfer[~transfer.same_task]),
        ("target dataset absent", regimes["leave-one-dataset-out"]),
        ("target task absent", regimes["leave-task-out"]),
    ]
    detail = []
    for comparison, frame in cases:
        keys = cell_keys(
            "transfer" if "transfer" in comparison
            else ("leave-one-dataset-out"
                  if comparison == "target dataset absent"
                  else "leave-task-out"))
        paired = (
            frame[frame.method.isin(["diff", "l2reg"])]
            .pivot_table(index=keys, columns="method", values="auc",
                         aggfunc="median")
            .dropna(subset=["diff", "l2reg"]).reset_index())
        paired["comparison"] = comparison
        paired["delta_auc"] = paired["diff"] - paired["l2reg"]
        detail.append(paired)
    detail = pd.concat(detail, ignore_index=True)

    rows = []
    for (collection, comparison), group in detail.groupby(
            ["collection", "comparison"], sort=False):
        delta = group.delta_auc
        try:
            p_value = stats.wilcoxon(delta).pvalue
        except ValueError:
            p_value = np.nan
        rows.append({
            "collection": collection, "comparison": comparison,
            "n": len(delta), "diff_median_auc": group["diff"].median(),
            "l2_median_auc": group["l2reg"].median(),
            "median_delta_auc": delta.median(),
            "q25_delta_auc": delta.quantile(.25),
            "q75_delta_auc": delta.quantile(.75),
            "mean_delta_auc": delta.mean(),
            "diff_win_rate": (delta > 0).mean(),
            "wilcoxon_p": p_value,
        })
    return detail, pd.DataFrame(rows)


def wildjailbreak_row_types() -> dict[str, np.ndarray]:
    """Recover the vanilla/adversarial label for each saved row.

    The unwrapped activation archive stores the original four-way
    ``data_types`` field. The wrapped logit archive stores the exact wrapped
    prompts but not that field. Most wrapped prompts join exactly after
    removing the wrapper. The remaining rows are the rewritten
    adversarial-benign examples; the balanced 12,500-per-category design and
    labels provide an exact, checked recovery.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    def cached_file(filename: str) -> Path:
        try:
            return Path(hf_hub_download(
                "anqiwang02/Activations", filename, repo_type="dataset",
                local_files_only=True))
        except Exception:
            # A newer cached repository revision can move the ``main`` ref
            # even though the required file remains in an older snapshot.
            root = (Path(HF_HUB_CACHE)
                    / "datasets--anqiwang02--Activations" / "snapshots")
            matches = list(root.glob(f"*/{filename}"))
            if not matches:
                raise FileNotFoundError(
                    f"{filename} is not in the local Hugging Face cache")
            return max(matches, key=lambda path: path.stat().st_mtime)

    unwrapped_path = cached_file(
        "data_unwrap/wildjailbreak/wildjailbreak_last_layer18.npz")
    wrapped_path = cached_file(
        "tf_wrapped_update/A/wildjailbreak/"
        "wildjailbreak_tf_a_tf_logits.npz")

    with np.load(unwrapped_path, allow_pickle=True) as data:
        unwrapped_prompts = data["prompts"].astype(str)
        unwrapped_types = data["data_types"].astype(str)
    with np.load(wrapped_path, allow_pickle=True) as data:
        wrapped_prompts = data["prompts"].astype(str)
        wrapped_labels = data["labels"].astype(bool)

    prefix = (
        "Determine whether the following is True or False.\n\n"
        'It is safe for an AI assistant to answer the following request: "')
    if not np.all(np.char.startswith(wrapped_prompts, prefix)):
        raise ValueError("Unexpected WildJailbreak wrapper text")
    inner = np.array([
        text[len(prefix):-1] if text.endswith('"') else text[len(prefix):]
        for text in wrapped_prompts
    ])
    prompt_to_type = dict(zip(unwrapped_prompts, unwrapped_types))
    wrapped_types = np.array(
        [prompt_to_type.get(text, "") for text in inner], dtype="<U19")
    missing = wrapped_types == ""
    if not np.all(wrapped_labels[missing]):
        raise ValueError(
            "Unmatched wrapped WildJailbreak rows are not all benign")
    wrapped_types[missing] = "adversarial_benign"

    expected = {
        "vanilla_harmful": 12500, "vanilla_benign": 12500,
        "adversarial_harmful": 12500, "adversarial_benign": 12500,
    }
    observed = dict(zip(*np.unique(wrapped_types, return_counts=True)))
    if observed != expected:
        raise ValueError(
            f"Wrapped WildJailbreak subtype recovery failed: {observed}")
    return {
        "data_unwrap": unwrapped_types,
        "tf_wrapped_a": wrapped_types,
    }


def wildjailbreak_split_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-score saved held-out predictions by prompt type.

    Prediction arrays are currently available for leave-task-out in both
    prompt formats and leave-one-dataset-out in the unwrapped format.
    """
    row_types = wildjailbreak_row_types()
    rows = []
    for collection in COLLS:
        pred_dir = RESULTS / collection / "leave_out" / "predictions"
        for path in pred_dir.glob("raw_last_layer*.npz"):
            name = path.stem
            if "_dataset_wildjailbreak" in name:
                regime = "target dataset absent"
            elif "_task_jailbreak" in name:
                regime = "target task absent"
            else:
                continue
            method = next(
                (m for m in sorted(METHOD_ORDER, key=len, reverse=True)
                 if f"_{m}_" in name), None)
            match = re.search(r"_layer(\d+)_", name)
            if method is None or match is None:
                continue
            layer = int(match.group(1))
            with np.load(path, allow_pickle=True) as data:
                keep = data["holdout_dataset"].astype(str) == "wildjailbreak"
                sample_ids = data["holdout_sample_id"].astype(str)[keep]
                y = data["holdout_y"][keep].astype(int)
                score = data["holdout_proba"][keep].astype(float)
            row_index = np.array(
                [int(sample_id.rsplit(":", 1)[1])
                 for sample_id in sample_ids])
            types = row_types[collection][row_index]
            for subtype in ["vanilla", "adversarial"]:
                selected = np.char.startswith(types, subtype)
                if len(np.unique(y[selected])) != 2:
                    continue
                rows.append({
                    "collection": collection, "regime": regime,
                    "layer": layer, "method": method,
                    "family": FAMILY[method], "prompt_type": subtype,
                    "auc": roc_auc_score(y[selected], score[selected]),
                    "n": selected.sum(), "positive_n": y[selected].sum(),
                })
    detailed = pd.DataFrame(rows).drop_duplicates(
        ["collection", "regime", "layer", "method", "prompt_type"])

    summary_rows = []
    for (collection, regime, prompt_type), group in detailed.groupby(
            ["collection", "regime", "prompt_type"]):
        for family in ["linear", "nonlinear"]:
            values = group[group.family == family].auc
            summary_rows.append({
                "collection": collection, "regime": regime,
                "prompt_type": prompt_type, "summary": family,
                "median_auc": values.median(), "q25": values.quantile(.25),
                "q75": values.quantile(.75), "n_fits": len(values),
            })
        oracle = group.groupby("layer").auc.max()
        summary_rows.append({
            "collection": collection, "regime": regime,
            "prompt_type": prompt_type, "summary": "best observed",
            "median_auc": oracle.median(), "q25": oracle.quantile(.25),
            "q75": oracle.quantile(.75), "n_fits": len(oracle),
        })
    return detailed, pd.DataFrame(summary_rows)


def wildjailbreak_output_logits() -> pd.DataFrame:
    """AUC of the model's own paired-token probabilities by prompt type."""
    from huggingface_hub.constants import HF_HUB_CACHE

    filename = (
        "tf_wrapped_update/A/wildjailbreak/"
        "wildjailbreak_tf_a_tf_logits.npz")
    root = (Path(HF_HUB_CACHE)
            / "datasets--anqiwang02--Activations" / "snapshots")
    matches = list(root.glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"{filename} is not in the local cache")
    path = max(matches, key=lambda item: item.stat().st_mtime)
    types = wildjailbreak_row_types()["tf_wrapped_a"]
    rows = []
    with np.load(path, allow_pickle=True) as data:
        y = data["labels"].astype(int)
        for prompt_type in ["vanilla", "adversarial"]:
            keep = np.char.startswith(types, prompt_type)
            for score, label in [
                    ("p_true", "True versus False"),
                    ("p_safe", "safe versus unsafe"),
                    ("p_harmless", "harmless versus harmful")]:
                rows.append({
                    "prompt_type": prompt_type, "score": score,
                    "score_label": label,
                    "auc": roc_auc_score(y[keep], data[score][keep]),
                    "n": keep.sum(),
                })
    return pd.DataFrame(rows)


def plot_geometry_transfer(pairs: pd.DataFrame,
                           correlations: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.7),
                             layout="constrained")
    colors = pairs.same_task.map({True: TEAL, False: MUTED})
    axes[0].scatter(pairs.centroid_distance, pairs.symmetric_transfer_auc,
                    c=colors, s=36, alpha=.72, edgecolor="none")
    direction_distance = 1 - pairs.label_axis_cosine
    axes[1].scatter(direction_distance, pairs.symmetric_transfer_auc,
                    c=colors, s=36, alpha=.72, edgecolor="none")
    for ax in axes:
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.grid(color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_ylabel("Median bidirectional transfer AUC", fontsize=12)
    axes[0].set_xlabel("Distance between dataset centres (in within-dataset radii)",
                       fontsize=11)
    axes[1].set_xlabel(
        "Distance between diff-in-means directions per dataset (1 − cosine)",
        fontsize=11)
    for ax, metric in zip(axes, ["centroid_distance", "label_axis_cosine"]):
        r = correlations[(correlations.analysis == "domain pair -> transfer")
                         & (correlations.geometry_metric == metric)]
        control = ("label_axis_cosine" if metric == "centroid_distance"
                   else "centroid_distance")
        partial = correlations[
            (correlations.analysis
             == f"domain pair -> transfer | {control}")
            & (correlations.geometry_metric == metric)]
        signs = -1 if metric == "label_axis_cosine" else 1
        ax.text(
            .02, .04,
            "Spearman corr.: "
            + ", ".join(
                f"{signs * x:+.2f}" for x in r.spearman_rho)
            + "\nAfter accounting for the other geometric quantity: "
            + ", ".join(
                f"{signs * x:+.2f}" for x in partial.spearman_rho),
            transform=ax.transAxes, fontsize=9.5, color=INK)
    fig.suptitle("Transfer is highest when the diff-in-means directions align",
                 fontsize=19, color=INK, fontweight="bold")
    save(fig, "geometry-predicts-transfer")


def plot_transfer_predictor_increment(
        incremental: pd.DataFrame, prediction: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.0),
                             layout="constrained")

    order = [
        "dataset-centre distance", "class-centroid distance", "same task",
        "source separability", "target separability",
        "target within-CV AUC",
    ]
    data = incremental.set_index("factor").reindex(order)
    y = np.arange(len(data))
    colors = [TEAL if name in
              {"target separability", "target within-CV AUC"} else MUTED
              for name in data.index]
    axes[0].barh(
        y, data.correlation_after_direction_alignment,
        color=colors, alpha=.9)
    for yi, value in enumerate(
            data.correlation_after_direction_alignment):
        axes[0].text(
            value + (.012 if value >= 0 else -.012), yi, f"{value:+.2f}",
            ha="left" if value >= 0 else "right", va="center",
            fontsize=10, color=INK)
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].set_yticks(y, data.index)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-.25, .48)
    axes[0].set_xlabel(
        "Remaining rank correlation after accounting for direction alignment")
    axes[0].set_title("What information remains?", fontsize=14, color=INK,
                      fontweight="bold")

    model_order = [
        "direction alignment only", "+ dataset-centre distance",
        "+ same-task flag", "+ target separability",
        "all measured factors",
    ]
    models = prediction.set_index("model").reindex(model_order)
    labels = [
        "alignment only", "+ distance", "+ same task",
        "+ target separability", "all measured",
    ]
    x = np.arange(len(models))
    bar_colors = [TEAL if i == 3 else MUTED for i in range(len(models))]
    axes[1].bar(x, models.held_target_out_mae,
                color=bar_colors, alpha=.9)
    for xi, value in enumerate(models.held_target_out_mae):
        axes[1].text(xi, value + .0012, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=10, color=INK)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylim(0, .067)
    axes[1].set_ylabel("Mean absolute AUC error (lower is better)")
    axes[1].set_title("Target left out of model fitting",
                      fontsize=14, color=INK, fontweight="bold")

    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y",
                color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Direction alignment explains task and distance—but not target difficulty",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "transfer-predictor-increment")


def plot_transfer_additive_regression(summary: pd.DataFrame) -> None:
    colors = [BLUE, TEAL, MUTED, ORANGE, "#E5E7EB"]
    fig, ax = plt.subplots(figsize=(13.33, 3.9), layout="constrained")
    left = 0.
    for color, row in zip(colors, summary.itertuples()):
        width = row.added_r_squared
        ax.barh([0], [width], left=left, color=color, height=.48,
                edgecolor="white", linewidth=1.5,
                label=f"{row.factor}: {width:.1%}")
        if width >= .045:
            ax.text(left + width / 2, 0, f"{width:.1%}",
                    ha="center", va="center", fontsize=13,
                    color="white" if row.factor != "unexplained" else INK,
                    fontweight="bold")
        left += width
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Share of ranked transfer variation")
    ax.grid(axis="x", color=GRID, linewidth=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.legend(frameon=False, ncol=3, loc="outside lower center",
               fontsize=10)
    ax.set_title(
        "Additive rank regression: alignment explains 67.6%; separability adds 4.9%",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "transfer-additive-regression")


def plot_coefficient_similarity(summary: pd.DataFrame) -> None:
    """Dot-and-IQR replacement for the dense coefficient table."""
    methods = ["diff", "l2", "svm_linear", "l1", "lda"]
    scopes = [
        ("within task", TEAL, "within task"),
        ("between task", ORANGE, "between tasks"),
    ]
    measures = [
        ("plain", "Plain cosine"),
        ("mahalanobis_reg_cov_pool", "Pooled-covariance cosine"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 9.2),
                             sharex=True, sharey=True,
                             layout="constrained")
    for row, collection in enumerate(COLLS):
        for col, (measure, measure_label) in enumerate(measures):
            ax = axes[row, col]
            selected = summary[
                (summary.collection == collection)
                & (summary.measure == measure)]
            for offset, (scope, color, label) in zip(
                    [-.13, .13], scopes):
                data = (
                    selected[selected.scope == scope]
                    .set_index("method").reindex(methods))
                y = np.arange(len(methods)) + offset
                ax.errorbar(
                    data["median"], y,
                    xerr=[data["median"] - data.q25,
                          data.q75 - data["median"]],
                    fmt="o", color=color, ecolor=color, capsize=3,
                    markersize=6, linewidth=1.5,
                    label=label if row == 0 and col == 0 else None)
            ax.axvline(0, color=GRID, linewidth=1)
            ax.grid(axis="x", color=GRID, linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-1, 1)
            ax.set_yticks(np.arange(len(methods)), methods)
            ax.invert_yaxis()
            ax.set_title(
                f"{COLLECTION_LABEL[collection]} · {measure_label}",
                fontsize=13, color=INK, fontweight="bold")
            ax.set_xlabel("Cosine similarity (dot = median; line = middle 50%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2,
               frameon=False)
    fig.suptitle(
        "Diff-in-means directions align within tasks; fitted probes mostly do not",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "coefficient-similarity-summary")


def plot_coefficient_training_states(
        pairs: pd.DataFrame, training: pd.DataFrame) -> None:
    """Show how l2 direction similarity changes across training regimes."""
    single = pairs[
        (pairs.method == "l2") & (pairs.scope == "all")
        & pairs.measure.isin(["plain", "mahalanobis_reg_cov_pool"])
    ].copy()
    single["comparison"] = "single ↔ single"
    selected = pd.concat(
        [single[["collection", "comparison", "measure", "median",
                 "q25", "q75"]],
         training[
             training.measure.isin(
                 ["plain", "mahalanobis_reg_cov_pool"])
         ][["collection", "comparison", "measure", "median", "q25", "q75"]]],
        ignore_index=True)
    comparisons = [
        "single ↔ single", "merged ↔ single", "LODO ↔ single",
        "LOTO ↔ single", "merged ↔ LODO", "merged ↔ LOTO",
    ]
    styles = [
        ("plain", BLUE, "plain cosine"),
        ("mahalanobis_reg_cov_pool", PURPLE,
         "pooled-covariance cosine"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.7),
                             sharey=True, layout="constrained")
    for ax, collection in zip(axes, COLLS):
        frame = selected[selected.collection == collection]
        for offset, (measure, color, label) in zip([-.13, .13], styles):
            data = (
                frame[frame.measure == measure]
                .set_index("comparison").reindex(comparisons))
            y = np.arange(len(comparisons)) + offset
            ax.errorbar(
                data["median"], y,
                xerr=[data["median"] - data.q25,
                      data.q75 - data["median"]],
                fmt="o", color=color, ecolor=color, capsize=3,
                markersize=6, linewidth=1.5, label=label)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-.1, 1.02)
        ax.set_yticks(np.arange(len(comparisons)), comparisons)
        ax.invert_yaxis()
        ax.set_xlabel("Cosine similarity (dot = median; line = middle 50%)")
        ax.set_title(COLLECTION_LABEL[collection], fontsize=14, color=INK,
                     fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Removing a whole task changes the pooled l2 direction much more",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "coefficient-training-state-summary")


def plot_coefficient_pca(spectra: pd.DataFrame) -> None:
    """Cumulative residual coefficient-PCA spectra for diff and l2."""
    null = spectra[spectra.source == "random-unit null"]
    null_summary = (
        null.groupby("pc").cumulative_variance
        .agg(median="median",
             q025=lambda x: x.quantile(.025),
             q975=lambda x: x.quantile(.975)).reset_index())
    styles = [("diff", ORANGE, "diff-in-means"),
              ("l2", BLUE, "l2")]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 5.8),
                             sharex=True, sharey=True,
                             layout="constrained")
    for ax, collection in zip(axes, COLLS):
        observed = spectra[
            (spectra.source == "observed")
            & (spectra.collection == collection)]
        for method, color, label in styles:
            method_data = observed[observed.method == method]
            grouped = (
                method_data.groupby("pc").cumulative_variance
                .agg(median="median", low="min", high="max").reset_index())
            ax.fill_between(grouped.pc, grouped.low, grouped.high,
                            color=color, alpha=.13)
            ax.plot(grouped.pc, grouped["median"], color=color,
                    marker="o", markersize=4, linewidth=2, label=label)
        ax.fill_between(null_summary.pc, null_summary.q025,
                        null_summary.q975, color=MUTED, alpha=.12)
        ax.plot(null_summary.pc, null_summary["median"], color=MUTED,
                linestyle="--", linewidth=1.8,
                label="random directions")
        ax.axhline(.8, color=GRID, linewidth=1)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1.02)
        ax.grid(color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of residual coefficient PCs")
        ax.set_title(COLLECTION_LABEL[collection], fontsize=14, color=INK,
                     fontweight="bold")
    axes[0].set_ylabel("Cumulative residual variance explained")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Diff-in-means residuals have structure; l2 residuals look noise-like",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "coefficient-pca-diff-vs-l2")


def plot_diff_l2_out_of_domain(summary: pd.DataFrame) -> None:
    order = [
        "same-task transfer", "cross-task transfer",
        "target dataset absent", "target task absent",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.2),
                             sharey=True, layout="constrained")
    for ax, collection in zip(axes, COLLS):
        data = (
            summary[summary.collection == collection]
            .set_index("comparison").reindex(order))
        x = np.arange(len(order))
        ax.errorbar(
            x, data.median_delta_auc,
            yerr=[data.median_delta_auc - data.q25_delta_auc,
                  data.q75_delta_auc - data.median_delta_auc],
            fmt="o", color=TEAL, ecolor=TEAL, capsize=4,
            markersize=7, linewidth=1.7)
        for xi, row in zip(x, data.itertuples()):
            ax.text(xi, row.q75_delta_auc + .018,
                    f"{row.median_delta_auc:+.3f}\n"
                    f"{row.diff_win_rate:.0%} wins",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(0, color=INK, linewidth=1)
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(
            x, ["same-task\ntransfer", "cross-task\ntransfer",
                "target dataset\nabsent", "target task\nabsent"])
        ax.set_ylim(-.24, .24)
        ax.set_title(COLLECTION_LABEL[collection], fontsize=14, color=INK,
                     fontweight="bold")
    axes[0].set_ylabel("Paired AUC difference (diff-in-means − l2)")
    fig.suptitle(
        "Diff-in-means does not generally beat l2 out of domain",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "diff-vs-l2-out-of-domain")


def plot_classifier_errors(distances: pd.DataFrame,
                           overlap: pd.DataFrame) -> None:
    order = [m for m in METHOD_ORDER
             if m in set(distances.method) | set(overlap.method)]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), sharey=True,
                             layout="constrained")

    values = [overlap.loc[overlap.method == method, "jaccard"].dropna()
              for method in order]
    y = np.arange(len(order))
    axes[0].boxplot(
        values, positions=y, vert=False, widths=.58, patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": TEAL}, capprops={"color": TEAL},
        boxprops={"facecolor": "#CCFBF1", "edgecolor": TEAL})
    axes[0].set_xlabel("Agreement on which samples are wrong (Jaccard)")
    axes[0].set_xlim(0, 1)

    summary = (distances.groupby(["method", "outcome"])
               .median_boundary_distance
               .agg(median="median",
                    q25=lambda x: x.quantile(.25),
                    q75=lambda x: x.quantile(.75))
               .reset_index())
    height = .34
    for outcome, color, offset in [
            ("Correct", BLUE, -height / 2),
            ("Misclassified", ORANGE, height / 2)]:
        s = (summary[summary.outcome == outcome]
             .set_index("method").reindex(order))
        axes[1].barh(
            y + offset, s["median"], height=height, color=color,
            alpha=.9, label=outcome,
            xerr=np.vstack([s["median"] - s.q25,
                            s.q75 - s["median"]]),
            error_kw={"ecolor": INK, "elinewidth": .9, "capsize": 2})
    axes[1].set_xlabel(
        "Median distance from the decision boundary  |probability − 0.5|")
    axes[1].set_xlim(0, .5)
    axes[1].legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9, length=0)
    axes[0].set_yticks(range(len(order)),
                       [METHOD_LABEL[x] for x in order], fontsize=10)
    fig.suptitle("Correct predictions tend to be farther from the boundary",
                 fontsize=20, color=INK, fontweight="bold")
    save(fig, "classifier-error-profiles")


def plot_wildjailbreak_split(summary: pd.DataFrame) -> None:
    panels = [
        ("tf_wrapped_a", "target task absent"),
        ("data_unwrap", "target dataset absent"),
        ("data_unwrap", "target task absent"),
    ]
    styles = {
        "linear": (BLUE, "o", "linear classifiers"),
        "nonlinear": (ORANGE, "s", "nonlinear classifiers"),
        "best observed": (INK, "D", "best classifier at each layer"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.5), sharey=True,
                             layout="constrained")
    for ax, (collection, regime) in zip(axes, panels):
        panel = summary[
            (summary.collection == collection) & (summary.regime == regime)]
        for offset, (kind, (color, marker, label)) in zip(
                [-.15, 0, .15], styles.items()):
            data = (panel[panel.summary == kind]
                    .set_index("prompt_type")
                    .reindex(["vanilla", "adversarial"]))
            x = np.arange(2) + offset
            ax.errorbar(
                x, data.median_auc,
                yerr=[data.median_auc - data.q25,
                      data.q75 - data.median_auc],
                fmt=marker, color=color, ecolor=color, capsize=3,
                markersize=7, linewidth=1.5, label=label)
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.set_xticks([0, 1], ["vanilla", "adversarial"])
        ax.set_ylim(.45, 1.01)
        ax.set_title(
            f"{COLLECTION_LABEL[collection]}\n{regime}",
            fontsize=13, fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("AUC", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=10)
    fig.suptitle("WildJailbreak: vanilla and adversarial prompts behave differently",
                 fontsize=19, color=INK, fontweight="bold")
    save(fig, "wildjailbreak-by-prompt-type")


def plot_wildjailbreak_logits(summary: pd.DataFrame,
                              logits: pd.DataFrame) -> None:
    activation = summary[
        (summary.collection == "tf_wrapped_a")
        & (summary.regime == "target task absent")]
    fig, ax = plt.subplots(figsize=(10.2, 5.5), layout="constrained")
    x = np.arange(2)
    for offset, kind, color, marker, label in [
            (-.24, "nonlinear", ORANGE, "s",
             "nonlinear activation classifiers"),
            (-.08, "best observed", INK, "D",
             "best activation classifier at each layer")]:
        data = (activation[activation.summary == kind]
                .set_index("prompt_type")
                .reindex(["vanilla", "adversarial"]))
        ax.errorbar(
            x + offset, data.median_auc,
            yerr=[data.median_auc - data.q25,
                  data.q75 - data.median_auc],
            fmt=marker, color=color, ecolor=color, capsize=3,
            markersize=7, linewidth=1.5, label=label)
    for offset, score, color, marker, label in [
            (.08, "p_true", PURPLE, "o", "model True/False probability"),
            (.24, "p_safe", TEAL, "^", "model safe/unsafe probability")]:
        data = (logits[logits.score == score].set_index("prompt_type")
                .reindex(["vanilla", "adversarial"]))
        ax.scatter(x + offset, data.auc, color=color, marker=marker, s=68,
                   label=label, zorder=3)
    ax.axhline(.5, color=GRID, linewidth=1)
    ax.set_xticks(x, ["vanilla", "adversarial"])
    ax.set_ylim(.45, .9)
    ax.set_ylabel("AUC", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper right")
    ax.set_title(
        "Jailbreak task absent: activation probes vs the model's output logits",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "wildjailbreak-activation-and-output")


def stratified_sample(Z: np.ndarray, y: np.ndarray, n: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    take = []
    for label in [0, 1]:
        ids = np.flatnonzero(y == label)
        k = min(len(ids), n // 2)
        take.extend(rng.choice(ids, k, replace=False))
    take = np.asarray(take)
    return Z[take], y[take]


def boundary_scope_sample(boundary_data: dict, scope: str
                          ) -> tuple[np.ndarray, np.ndarray, str]:
    """Use the same deterministic sample in every boundary figure."""
    seed = {"common_claim": 991, "harmful_harmless": 992,
            "wildjailbreak": 993, "merged": 994}[scope]
    rng = np.random.default_rng(seed)
    if scope == "merged":
        xs, ys = [], []
        for d in DATASETS:
            X_s, y_s = stratified_sample(*boundary_data[d], 220, rng)
            xs.append(X_s)
            ys.append(y_s)
        return np.concatenate(xs), np.concatenate(ys), "merged"
    X, y = stratified_sample(*boundary_data[scope], 1600, rng)
    return X, y, scope.replace("_", " ")


def model_scores(model, X: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Return a continuous label-1 score and its decision threshold."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).reshape(-1), \
            "decision_function", 0.
    return np.asarray(model.predict_proba(X))[:, 1], "predict_proba", .5


def cv_model_scores(model, X: np.ndarray, y: np.ndarray,
                    cv: StratifiedKFold) -> np.ndarray:
    method = ("decision_function" if hasattr(model, "decision_function")
              else "predict_proba")
    score = cross_val_predict(model, X, y, cv=cv, method=method, n_jobs=None)
    return score if score.ndim == 1 else score[:, 1]


def plot_decision_boundaries(boundary_data: dict) -> pd.DataFrame:
    """Compare full-space slices with an RBF fit using only the plotted PCs."""
    rng = np.random.default_rng(991)
    scopes = ["common_claim", "harmful_harmless", "wildjailbreak", "merged"]
    models = [
        ("l2", "l2, all dimensions",
         LogisticRegression(C=.1, max_iter=2000, solver="liblinear"),
         "full"),
        ("svm_rbf", "svm_rbf, all dimensions",
         SVC(C=2., gamma="scale"), "full"),
        ("svm_rbf", "svm_rbf, PC1 and PC2 only",
         SVC(C=2., gamma="scale"), "pc"),
    ]
    fig, axes = plt.subplots(len(scopes), 3, figsize=(15.5, 13.3),
                             layout="constrained")
    rows = []
    for i, scope in enumerate(scopes):
        X, y, title = boundary_scope_sample(boundary_data, scope)

        # PCA supplies display coordinates only. The classifiers below see
        # every activation dimension.
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        X2 = pca.fit_transform(X)
        xlo, xhi = np.quantile(X2[:, 0], [.01, .99])
        ylo, yhi = np.quantile(X2[:, 1], [.01, .99])
        padx, pady = .12 * (xhi - xlo), .12 * (yhi - ylo)
        xx, yy = np.meshgrid(np.linspace(xlo - padx, xhi + padx, 140),
                             np.linspace(ylo - pady, yhi + pady, 140))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for j, (method, model_name, model, fit_space) in enumerate(models):
            X_fit = X if fit_space == "full" else X2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_cv = cv_model_scores(model, X_fit, y, cv)
                auc = roc_auc_score(y, score_cv)
                model.fit(X_fit, y)
                point_score, _, threshold = model_scores(model, X_fit)
                if fit_space == "full":
                    # Exact slice of the full-space fit through the PCA
                    # mean: omitted components are fixed at their mean.
                    plane_scores = []
                    for start in range(0, len(grid), 350):
                        plane = pca.inverse_transform(
                            grid[start:start + 350])
                        plane_scores.append(model_scores(model, plane)[0])
                    plane_score = np.concatenate(plane_scores)
                else:
                    plane_score = model_scores(model, grid)[0]
                if threshold == 0:
                    surface = 1 / (
                        1 + np.exp(-np.clip(plane_score, -8, 8)))
                else:
                    surface = plane_score
                zz = surface.reshape(xx.shape)
            ax = axes[i, j]
            ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 17),
                        cmap="RdBu_r", vmin=0, vmax=1, alpha=.68)
            ax.contour(xx, yy, zz, levels=[.5], colors=INK,
                       linewidths=1.7)
            show = np.concatenate([
                rng.choice(np.flatnonzero(y == label),
                           min(260, (y == label).sum()), replace=False)
                for label in [0, 1]])
            ax.scatter(X2[show, 0], X2[show, 1], c=y[show],
                       cmap="coolwarm", vmin=0, vmax=1, s=9, alpha=.56,
                       edgecolor="none")
            wrong = show[(point_score[show] >= threshold) != y[show]]
            if len(wrong):
                ax.scatter(X2[wrong, 0], X2[wrong, 1],
                           facecolors="none", edgecolors=INK, s=24,
                           linewidths=.7)
            space_label = ("all-dimensional CV AUC" if fit_space == "full"
                           else "two-PC CV AUC")
            ax.set_title(f"{title} — {model_name}\n{space_label} {auc:.3f}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                          fontsize=9)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                          fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            rows.append({
                "scope": scope, "method": method, "fit_space": fit_space,
                "layer": 17,
                "n": len(y), "pca_variance_2d":
                    pca.explained_variance_ratio_.sum(),
                "cv_auc": auc,
                "cv_auc_full_space": auc if fit_space == "full" else np.nan,
                "full_fit_error_rate":
                    np.mean((point_score >= threshold) != y),
                "positive_share_on_mean_plane":
                    np.mean(plane_score >= threshold),
            })
    fig.suptitle("All-dimensional boundaries and a two-component comparison",
                 fontsize=19, color=INK, fontweight="bold")
    fig.text(.5, -.005,
             "Blue = label 0; red = label 1; black rings = fit errors. In the first two columns, the line is the all-dimensional boundary sliced through the PCA mean.",
             ha="center", fontsize=10, color=MUTED)
    save(fig, "decision-boundaries-2d")
    return pd.DataFrame(rows)


def plot_boundary_overlays(boundary_data: dict) -> pd.DataFrame:
    """Overlay several two-PC fits so their boundary shapes can be compared."""
    rng = np.random.default_rng(772)
    scopes = ["common_claim", "harmful_harmless", "wildjailbreak", "merged"]
    methods = [
        ("l2", "l2 logistic", LogisticRegression(
            C=.1, max_iter=2000, solver="liblinear"), BLUE, "-", 2.3),
        ("lda", "LDA", LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"), TEAL, "--", 2.1),
        ("qda", "QDA", QuadraticDiscriminantAnalysis(
            reg_param=.1), ORANGE, "-.", 2.2),
        ("svm_rbf", "RBF SVM", SVC(
            C=2., gamma="scale"), PURPLE, "-", 2.5),
        ("knn", "25-NN", KNeighborsClassifier(
            n_neighbors=25, weights="distance"), MUTED, ":", 2.2),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 10.3))
    fig.subplots_adjust(left=.07, right=.98, bottom=.07, top=.83,
                        hspace=.34, wspace=.18)
    rows = []
    for ax, scope in zip(axes.ravel(), scopes):
        X, y, title = boundary_scope_sample(boundary_data, scope)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        X2 = pca.fit_transform(X)
        xlo, xhi = np.quantile(X2[:, 0], [.01, .99])
        ylo, yhi = np.quantile(X2[:, 1], [.01, .99])
        padx, pady = .12 * (xhi - xlo), .12 * (yhi - ylo)
        xx, yy = np.meshgrid(np.linspace(xlo - padx, xhi + padx, 180),
                             np.linspace(ylo - pady, yhi + pady, 180))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        show = np.concatenate([
            rng.choice(np.flatnonzero(y == label),
                       min(330, (y == label).sum()), replace=False)
            for label in [0, 1]])
        ax.scatter(X2[show, 0], X2[show, 1], c=y[show], cmap="coolwarm",
                   vmin=0, vmax=1, s=11, alpha=.35, edgecolor="none")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        auc_text = []
        for method, display, model, color, ls, width in methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score_cv = cv_model_scores(model, X2, y, cv)
                auc = roc_auc_score(y, score_cv)
                model.fit(X2, y)
                grid_score, _, threshold = model_scores(model, grid)
            ax.contour(xx, yy, grid_score.reshape(xx.shape),
                       levels=[threshold], colors=[color],
                       linestyles=[ls], linewidths=[width])
            auc_text.append(f"{method} {auc:.3f}")
            rows.append({
                "scope": scope, "method": method, "fit_space": "pc",
                "layer": 17, "n": len(y), "cv_auc": auc,
                "pca_variance_2d": pca.explained_variance_ratio_.sum(),
            })
        ax.text(.02, .02, "\n".join(auc_text), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.2, color=INK,
                bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
                      "edgecolor": GRID, "alpha": .90})
        ax.set_title(title, fontsize=15, color=INK, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color=color, linestyle=ls, linewidth=width,
               label=display)
        for _, display, _, color, ls, width in methods
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(.5, .91), ncol=5, frameon=False,
               handlelength=3.4)
    fig.suptitle(
        "Different classifiers draw different boundaries using PC1 and PC2",
        y=.985, fontsize=19, color=INK, fontweight="bold")
    save(fig, "decision-boundary-overlays")
    return pd.DataFrame(rows)


def full_mean_plane_scores(model, method: str, pca: PCA,
                           grid: np.ndarray) -> np.ndarray:
    """Evaluate a full raw-space fit on x=mean+PC1*z1+PC2*z2.

    The generic path materialises the 2,560-dimensional plane in small
    batches.  kNN gets an algebraically identical shortcut: its squared
    raw-space distance is the displayed-plane distance plus each training
    point's fixed squared residual outside that plane.
    """
    if method == "knn":
        train = np.asarray(model._fit_X)
        train_pc = pca.transform(train)
        reconstructed = pca.inverse_transform(train_pc)
        residual_sq = np.sum((train - reconstructed) ** 2, axis=1)
        scores = []
        for start in range(0, len(grid), 300):
            block = grid[start:start + 300]
            distance_sq = (
                np.sum((block[:, None, :] - train_pc[None, :, :]) ** 2,
                       axis=2)
                + residual_sq[None, :])
            k = model.n_neighbors
            ids = np.argpartition(distance_sq, k - 1, axis=1)[:, :k]
            selected_sq = np.take_along_axis(distance_sq, ids, axis=1)
            selected_y = model._y[ids]
            distance = np.sqrt(np.maximum(selected_sq, 0))
            weight = 1 / np.maximum(distance, 1e-12)
            zero = distance <= 1e-12
            any_zero = zero.any(axis=1)
            weight[any_zero] = zero[any_zero]
            scores.append(
                np.sum(weight * (selected_y == model.classes_[1]), axis=1)
                / np.sum(weight, axis=1))
        return np.concatenate(scores)

    scores = []
    for start in range(0, len(grid), 300):
        plane = pca.inverse_transform(grid[start:start + 300])
        scores.append(model_scores(model, plane)[0])
    return np.concatenate(scores)


def plot_full_boundary_overlays(boundary_data: dict) -> pd.DataFrame:
    """Overlay full raw-space fits as exact slices through the PCA mean."""
    rng = np.random.default_rng(772)
    scopes = ["common_claim", "harmful_harmless", "wildjailbreak", "merged"]
    method_specs = [
        ("l2", "l2 logistic", lambda: LogisticRegression(
            C=.1, max_iter=2000, solver="liblinear"), BLUE, "-", 2.3),
        ("lda", "LDA", lambda: LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"), TEAL, "--", 2.1),
        ("qda", "QDA", lambda: QuadraticDiscriminantAnalysis(
            solver="eigen", shrinkage=.15), ORANGE, "-.", 2.2),
        ("svm_rbf", "RBF SVM", lambda: SVC(
            C=2., gamma="scale"), PURPLE, "-", 2.5),
        ("knn", "25-NN", lambda: KNeighborsClassifier(
            n_neighbors=25, weights="distance"), MUTED, ":", 2.2),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 10.3),
                             layout="constrained")
    rows = []
    for ax, scope in zip(axes.ravel(), scopes):
        X, y, title = boundary_scope_sample(boundary_data, scope)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
        X2 = pca.fit_transform(X)
        xlo, xhi = np.quantile(X2[:, 0], [.01, .99])
        ylo, yhi = np.quantile(X2[:, 1], [.01, .99])
        padx, pady = .12 * (xhi - xlo), .12 * (yhi - ylo)
        xx, yy = np.meshgrid(np.linspace(xlo - padx, xhi + padx, 110),
                             np.linspace(ylo - pady, yhi + pady, 110))
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        show = np.concatenate([
            rng.choice(np.flatnonzero(y == label),
                       min(330, (y == label).sum()), replace=False)
            for label in [0, 1]])
        ax.scatter(X2[show, 0], X2[show, 1], c=y[show], cmap="coolwarm",
                   vmin=0, vmax=1, s=11, alpha=.35, edgecolor="none")
        for method, display, factory, color, ls, width in method_specs:
            model = factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
                grid_score = full_mean_plane_scores(
                    model, method, pca, grid)
                _, _, threshold = model_scores(model, X[:2])
            crosses = (grid_score.min() <= threshold
                       <= grid_score.max())
            if crosses:
                ax.contour(
                    xx, yy, grid_score.reshape(xx.shape),
                    levels=[threshold], colors=[color],
                    linestyles=[ls], linewidths=[width])
            rows.append({
                "scope": scope, "method": method,
                "fit_space": "full_raw_mean_plane_slice",
                "layer": 17, "n": len(y),
                "pca_variance_2d": pca.explained_variance_ratio_.sum(),
                "boundary_crosses_mean_plane": crosses,
                "positive_share_on_mean_plane":
                    np.mean(grid_score >= threshold),
            })
        ax.set_title(title, fontsize=15, color=INK, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color=color, linestyle=ls, linewidth=width,
               label=display)
        for _, display, _, color, ls, width in method_specs
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5,
               frameon=False, handlelength=3.4)
    fig.suptitle(
        "The same PCA view, now slicing classifiers fitted in all 2,560 dimensions",
        fontsize=18, color=INK, fontweight="bold")
    save(fig, "decision-boundary-overlays-fullspace")
    return pd.DataFrame(rows)


def main(*, results_root: Path | None = None,
         output_root: Path | None = None) -> None:
    configure_paths(results_root=results_root, output_root=output_root)
    m = metrics(RESULTS)
    assert not (set(m.train_dataset.dropna()) | set(m.test_dataset.dropna())
                | set(m.holdout.dropna())) & set(EXCLUDED_DATASETS)
    regimes = raw_regimes(m)

    estimands = robust_estimands(regimes)
    gains = family_gains(regimes)
    family_summary, method_regimes = family_oracle_summary(regimes)
    winner_frequency, dataset_best = classifier_winner_summaries(regimes)
    feature_summary, binned_summary = feature_group_summary(m)
    profiles, stability, agreement, rank_stability = layer_analyses(regimes)
    domains = method_domain(regimes)
    layer_contrasts = best_layer_family_contrasts(agreement)
    boundary_distances, error_overlap = classifier_error_profiles()
    coefficient_pairs, coefficient_triangle = \
        coefficient_geometry_summaries()
    coefficient_spectrum, coefficient_spectrum_detail = \
        coefficient_pca_null()
    diff_l2_detail, diff_l2_summary = diff_vs_l2_out_of_domain(regimes)
    wild_detail, wild_summary = wildjailbreak_split_results()
    wild_logits = wildjailbreak_output_logits()

    outputs = {
        "robust_estimands.csv": estimands,
        "nonlinear_gain_median.csv": gains,
        "performance_family_oracle.csv": family_summary,
        "classifier_by_regime_median.csv": method_regimes,
        "classifier_winner_frequency.csv": winner_frequency,
        "best_classifier_by_dataset_setting.csv": dataset_best,
        "feature_group_performance.csv": feature_summary,
        "binned_within_dataset.csv": binned_summary,
        "layer_profiles_median.csv": profiles,
        "layer_stability.csv": stability,
        "best_layer_agreement.csv": agreement,
        "best_layer_family_contrasts.csv": layer_contrasts,
        "layer_method_rank_stability.csv": rank_stability,
        "classifier_by_domain_median.csv": domains,
        "classifier_error_profiles.csv": boundary_distances,
        "error_overlap_by_classifier.csv": error_overlap,
        "coefficient_pair_geometry_summary.csv": coefficient_pairs,
        "coefficient_training_geometry_summary.csv": coefficient_triangle,
        "coefficient_pca_random_null.csv": coefficient_spectrum,
        "coefficient_pca_spectra.csv": coefficient_spectrum_detail,
        "diff_vs_l2_out_of_domain_cells.csv": diff_l2_detail,
        "diff_vs_l2_out_of_domain_summary.csv": diff_l2_summary,
        "wildjailbreak_by_prompt_type.csv": wild_detail,
        "wildjailbreak_by_prompt_type_summary.csv": wild_summary,
        "wildjailbreak_output_logits_by_prompt_type.csv": wild_logits,
    }
    for filename, frame in outputs.items():
        frame.to_csv(TABLES / filename, index=False)

    plot_estimands(estimands)
    plot_performance_overview(family_summary)
    plot_classifier_regime_heatmap(method_regimes)
    plot_classifier_winner_frequency(winner_frequency)
    plot_best_classifier_by_dataset(dataset_best)
    plot_feature_groups(feature_summary)
    plot_family_gain(gains)
    plot_layer_profiles(profiles)
    plot_stability(stability)
    plot_method_domain(domains)
    plot_classifier_errors(boundary_distances, error_overlap)
    plot_coefficient_similarity(coefficient_pairs)
    plot_coefficient_training_states(
        coefficient_pairs, coefficient_triangle)
    plot_coefficient_pca(coefficient_spectrum_detail)
    plot_diff_l2_out_of_domain(diff_l2_summary)
    plot_wildjailbreak_split(wild_summary)
    plot_wildjailbreak_logits(wild_summary, wild_logits)

    pairs, datasets, geometry_corr, boundary_data = geometry_supervision(
        regimes)
    pairs.to_csv(TABLES / "domain_geometry_transfer.csv", index=False)
    datasets.to_csv(TABLES / "dataset_label_geometry.csv", index=False)
    geometry_corr.to_csv(TABLES / "geometry_supervision_correlations.csv",
                         index=False)
    plot_geometry_transfer(pairs, geometry_corr)
    predictor_detail, predictor_increment, predictor_cv = \
        transfer_predictor_increment(pairs, datasets, regimes)
    predictor_detail.to_csv(
        TABLES / "transfer_predictor_directed_pairs.csv", index=False)
    predictor_increment.to_csv(
        TABLES / "transfer_predictor_increment.csv", index=False)
    predictor_cv.to_csv(
        TABLES / "transfer_predictor_held_target_out.csv", index=False)
    plot_transfer_predictor_increment(predictor_increment, predictor_cv)
    additive = transfer_additive_regression(pairs, datasets)
    additive.to_csv(
        TABLES / "transfer_additive_regression.csv", index=False)
    plot_transfer_additive_regression(additive)
    boundary = plot_decision_boundaries(boundary_data)
    boundary.to_csv(TABLES / "decision_boundaries_2d.csv", index=False)
    boundary_overlays = plot_boundary_overlays(boundary_data)
    boundary_overlays.to_csv(
        TABLES / "decision_boundary_overlays.csv", index=False)
    full_boundary_overlays = plot_full_boundary_overlays(boundary_data)
    full_boundary_overlays.to_csv(
        TABLES / "decision_boundary_overlays_fullspace.csv", index=False)

    print("\nRobust estimands")
    print(estimands.pivot(index="regime", columns=["collection", "estimand"],
                          values="auc").round(3).to_string())
    print("\nNonlinear gain")
    print(gains[["collection", "regime", "median_nonlinear_gain",
                 "nonlinear_win_rate"]].round(3).to_string(index=False))
    print("\nLayer stability by family")
    print(stability.groupby("family")[
        ["amplitude", "adjacent_roughness", "peakiness"]].median()
          .round(3).to_string())
    print("\nBest-layer agreement")
    print(agreement.groupby(["regime", "family"]).agree.mean()
          .round(3).to_string())
    print("\nGeometry correlations")
    print(geometry_corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
