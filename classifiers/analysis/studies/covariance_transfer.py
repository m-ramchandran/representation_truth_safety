"""Does label-free covariance similarity predict cross-dataset transfer?

The analysis uses exact unwrapped layer-17 activations for the retained eleven
datasets. Sampling, common standardisation, PCA fitting, covariance estimation,
and covariance comparison never use labels. Labels enter only in the transfer
outcome and in the diagnostic diff-in-means alignment control.

Outputs:
  tables/covariance_transfer_pairs.csv
  tables/covariance_transfer_summary.csv
  tables/covariance_transfer_dimension_robustness.csv
  tables/covariance_transfer_prediction.csv
  figures/covariance-similarity-transfer.{png,svg}
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import synthesis as g
from .common import load_standardized_raw


LAYER = 17
MAX_PER_DATASET = 2000
PCA_FIT_PER_DATASET = 500
PCA_DIMS = (16, 32, 64)
LEADING_DIRECTIONS = 10
RANDOM_SEED = 2817

METRIC_LABELS = {
    "covariance_shape_similarity": "Full covariance shape",
    "variance_profile_similarity": "Variance by coordinate",
    "leading_variation_overlap": "Main variation directions",
}
CONTROL_LABELS = {
    "unadjusted": "By itself",
    "label_free_controls": "After dataset distance + same task",
    "alignment_control": "After diff-in-means alignment",
}
CONTROLS = {
    "unadjusted": [],
    "label_free_controls": ["centroid_distance", "same_task"],
    "alignment_control": ["label_axis_cosine"],
}


def label_free_geometry() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Uniformly sample and standardise activations without reading labels."""
    standardized, labels = load_standardized_raw(
        layer=LAYER, max_per_dataset=MAX_PER_DATASET, seed=RANDOM_SEED)

    centroids, radii, label_axes = {}, {}, {}
    for name, Z in standardized.items():
        centre = Z.mean(axis=0)
        centroids[name] = centre
        radii[name] = np.sqrt(np.mean(np.sum((Z - centre) ** 2, axis=1)))
        y = labels[name]
        label_axes[name] = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)

    pair_rows = []
    for i, a in enumerate(g.DATASETS):
        for b in g.DATASETS[i + 1:]:
            radius = np.sqrt(.5 * (radii[a] ** 2 + radii[b] ** 2))
            axis_norm = (np.linalg.norm(label_axes[a])
                         * np.linalg.norm(label_axes[b]))
            pair_rows.append({
                "a": a, "b": b,
                "same_task": g.TASK_OF[a] == g.TASK_OF[b],
                "centroid_distance":
                    np.linalg.norm(centroids[a] - centroids[b]) / radius,
                "label_axis_cosine":
                    float(label_axes[a] @ label_axes[b] / axis_norm),
            })
    return pd.DataFrame(pair_rows), standardized


def covariance_metrics(
        pairs: pd.DataFrame, standardized: dict[str, np.ndarray]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare shrinkage covariance estimates in one shared unlabeled PCA basis."""
    rng = np.random.default_rng(RANDOM_SEED)
    fit = []
    for Z in standardized.values():
        take = rng.choice(
            len(Z), min(PCA_FIT_PER_DATASET, len(Z)), replace=False)
        fit.append(Z[take])
    basis = PCA(
        n_components=max(PCA_DIMS), svd_solver="randomized",
        iterated_power=4, random_state=RANDOM_SEED,
    ).fit(np.concatenate(fit))
    coordinates = {
        name: basis.transform(Z) for name, Z in standardized.items()}

    detail = pairs.copy()
    robustness_rows = []
    for dimension in PCA_DIMS:
        covariances, variances, directions = {}, {}, {}
        for name, all_coordinates in coordinates.items():
            coordinates_d = all_coordinates[:, :dimension]
            covariance = LedoitWolf().fit(coordinates_d).covariance_
            covariances[name] = covariance
            variances[name] = np.diag(covariance)
            _, vectors = np.linalg.eigh(covariance)
            top = min(LEADING_DIRECTIONS, dimension)
            directions[name] = vectors[:, -top:]

        suffix = f"_{dimension}d"
        detail[f"covariance_shape_similarity{suffix}"] = [
            np.sum(covariances[a] * covariances[b])
            / (np.linalg.norm(covariances[a])
               * np.linalg.norm(covariances[b]))
            for a, b in zip(detail.a, detail.b)]
        detail[f"variance_profile_similarity{suffix}"] = [
            variances[a] @ variances[b]
            / (np.linalg.norm(variances[a]) * np.linalg.norm(variances[b]))
            for a, b in zip(detail.a, detail.b)]
        detail[f"leading_variation_overlap{suffix}"] = [
            np.sum((directions[a].T @ directions[b]) ** 2)
            / directions[a].shape[1]
            for a, b in zip(detail.a, detail.b)]

    for base in METRIC_LABELS:
        detail[base] = detail[f"{base}_{max(PCA_DIMS)}d"]
        for dimension in PCA_DIMS:
            column = f"{base}_{dimension}d"
            for control_name, controls in CONTROLS.items():
                result = partial_rank_correlation(
                    detail, column, "symmetric_transfer_auc", controls)
                robustness_rows.append({
                    "dimension": dimension, "metric": base,
                    "metric_label": METRIC_LABELS[base],
                    "control": control_name,
                    "control_label": CONTROL_LABELS[control_name],
                    "correlation_with_transfer": result.statistic,
                    "ordinary_p_value": result.pvalue,
                    "n_pairs": len(detail),
                })
    return detail, pd.DataFrame(robustness_rows)


def residualized_ranks(
        frame: pd.DataFrame, column: str, controls: list[str]) -> np.ndarray:
    values = stats.rankdata(frame[column])
    if not controls:
        return values - values.mean()
    design = np.column_stack([
        np.ones(len(frame)),
        *[frame[c].astype(float).to_numpy()
          if c == "same_task" else stats.rankdata(frame[c])
          for c in controls],
    ])
    return values - design @ np.linalg.lstsq(
        design, values, rcond=None)[0]


def partial_rank_correlation(
        frame: pd.DataFrame, x: str, y: str, controls: list[str]):
    return stats.pearsonr(
        residualized_ranks(frame, x, controls),
        residualized_ranks(frame, y, controls),
    )


def add_transfer_outcome(
        pairs: pd.DataFrame, regimes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    transfer = (
        regimes["transfer"][
            (regimes["transfer"].collection == "data_unwrap")
            & (regimes["transfer"].layer == LAYER)]
        .groupby(["train_dataset", "target"]).auc.median())
    pairs["symmetric_transfer_auc"] = [
        np.nanmedian([
            transfer.get((a, b), np.nan),
            transfer.get((b, a), np.nan),
        ])
        for a, b in zip(pairs.a, pairs.b)]
    return pairs.dropna(subset=["symmetric_transfer_auc"]).reset_index(
        drop=True)


def shuffled_dataset_p_value(
        pairs: pd.DataFrame, metric: str, controls: list[str],
        permutations: int = 10000) -> float:
    """Shuffle whole dataset identities rather than treating 55 pairs as independent."""
    names = list(g.DATASETS)
    name_index = {name: i for i, name in enumerate(names)}
    matrix = np.eye(len(names))
    for row in pairs.itertuples():
        i, j = name_index[row.a], name_index[row.b]
        matrix[i, j] = matrix[j, i] = getattr(row, metric)
    upper = np.triu_indices(len(names), 1)
    y_residual = residualized_ranks(
        pairs, "symmetric_transfer_auc", controls)
    observed = partial_rank_correlation(
        pairs, metric, "symmetric_transfer_auc", controls).statistic
    rng = np.random.default_rng(RANDOM_SEED + len(controls))
    extreme = 0
    shuffled = pairs.copy()
    for _ in range(permutations):
        order = rng.permutation(len(names))
        shuffled[metric] = matrix[order][:, order][upper]
        x_residual = residualized_ranks(shuffled, metric, controls)
        correlation = np.corrcoef(x_residual, y_residual)[0, 1]
        extreme += abs(correlation) >= abs(observed)
    return (extreme + 1) / (permutations + 1)


def summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, metric_label in METRIC_LABELS.items():
        for control_name, controls in CONTROLS.items():
            result = partial_rank_correlation(
                detail, metric, "symmetric_transfer_auc", controls)
            rows.append({
                "metric": metric, "metric_label": metric_label,
                "control": control_name,
                "control_label": CONTROL_LABELS[control_name],
                "correlation_with_transfer": result.statistic,
                "ordinary_p_value": result.pvalue,
                "dataset_identity_shuffle_p_value":
                    shuffled_dataset_p_value(detail, metric, controls)
                    if metric == "leading_variation_overlap" else np.nan,
                "n_pairs": len(detail),
            })
    return pd.DataFrame(rows)


def held_dataset_out_prediction(detail: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("dataset-centre distance", ["centroid_distance"]),
        ("covariance shape", ["covariance_shape_similarity"]),
        ("main variation directions", ["leading_variation_overlap"]),
        ("distance + same task", ["centroid_distance", "same_task"]),
        ("same task + main variation directions",
         ["same_task", "leading_variation_overlap"]),
        ("diff-in-means alignment", ["label_axis_cosine"]),
        ("alignment + main variation directions",
         ["label_axis_cosine", "leading_variation_overlap"]),
    ]
    rows = []
    for label, features in specs:
        prediction = np.full(len(detail), np.nan)
        for held in g.DATASETS:
            test = (detail.a == held) | (detail.b == held)
            train = ~test
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(detail.loc[train, features],
                      detail.loc[train, "symmetric_transfer_auc"])
            prediction[test] = model.predict(detail.loc[test, features])
        error = prediction - detail.symmetric_transfer_auc.to_numpy()
        rows.append({
            "model": label, "features": ", ".join(features),
            "held_dataset_out_mae": np.mean(np.abs(error)),
            "held_dataset_out_rmse": np.sqrt(np.mean(error ** 2)),
            "prediction_rank_correlation": stats.spearmanr(
                prediction, detail.symmetric_transfer_auc).statistic,
            "n_pairs": len(detail),
        })
    return pd.DataFrame(rows)


def plot_result(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.8), layout="constrained")
    for same_task, color, label in [
            (False, g.MUTED, "different task"),
            (True, g.TEAL, "same task")]:
        part = detail[detail.same_task == same_task]
        axes[0].scatter(
            part.leading_variation_overlap, part.symmetric_transfer_auc,
            color=color, alpha=.82, s=46, label=label,
            edgecolor="white", linewidth=.4)
    axes[0].set_xlabel(
        "Overlap in the datasets' main variation directions")
    axes[0].set_ylabel("Median transfer AUC")
    axes[0].set_xlim(-.02, 1.02)
    axes[0].set_ylim(.35, .9)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(color=g.GRID, linewidth=.7)
    axes[0].spines[["top", "right"]].set_visible(False)

    metric_order = list(METRIC_LABELS)
    controls = ["unadjusted", "label_free_controls", "alignment_control"]
    colors = [g.TEAL, g.BLUE, g.ORANGE]
    y = np.arange(len(metric_order))
    height = .22
    for offset, control, color in zip([-.24, 0, .24], controls, colors):
        part = (summary[summary.control == control]
                .set_index("metric").reindex(metric_order))
        axes[1].barh(
            y + offset, part.correlation_with_transfer, height=height,
            color=color, label=CONTROL_LABELS[control])
        for yi, value in zip(y + offset, part.correlation_with_transfer):
            axes[1].text(
                value + (.015 if value >= 0 else -.015), yi,
                f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)
    axes[1].axvline(0, color=g.INK, linewidth=1)
    axes[1].set_yticks(y, [METRIC_LABELS[m] for m in metric_order])
    axes[1].set_xlim(-.32, .66)
    axes[1].set_xlabel("Correlation with transfer AUC")
    axes[1].grid(axis="x", color=g.GRID, linewidth=.7)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="outside lower center",
        ncol=3, fontsize=9)
    fig.suptitle(
        "Covariance similarity predicts transfer, but not beyond label-axis alignment",
        fontsize=19, color=g.INK, fontweight="bold")
    g.save(fig, "covariance-similarity-transfer")


def main(*, results_root=None, output_root=None) -> None:
    g.configure_paths(results_root=results_root, output_root=output_root)
    regimes = g.raw_regimes(g.metrics(g.RESULTS))
    pairs, standardized = label_free_geometry()
    pairs = add_transfer_outcome(pairs, regimes)
    detail, robustness = covariance_metrics(pairs, standardized)
    summary = summary_table(detail)
    prediction = held_dataset_out_prediction(detail)

    detail.to_csv(
        g.TABLES / "covariance_transfer_pairs.csv", index=False)
    summary.to_csv(
        g.TABLES / "covariance_transfer_summary.csv", index=False)
    robustness.to_csv(
        g.TABLES / "covariance_transfer_dimension_robustness.csv",
        index=False)
    prediction.to_csv(
        g.TABLES / "covariance_transfer_prediction.csv", index=False)
    plot_result(detail, summary)

    print(summary.round(4).to_string(index=False))
    print("\nHeld-dataset-out prediction")
    print(prediction.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
