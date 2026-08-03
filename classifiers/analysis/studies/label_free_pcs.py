"""Can unlabeled PC properties identify directions likely to carry labels?

For each of the retained eleven datasets in turn, its labels are hidden while
a simple ridge model learns from the other ten datasets which *unlabeled*
one-dimensional PC properties predict actual label separation. The selector is
then frozen, ten PCs are chosen for the held-out dataset, and its labels are
revealed only for evaluation.

The experiment compares ordinary pooled PCA with a common PCA fitted after
subtracting each dataset's own unlabeled mean. It reports how much of the
target diff-in-means direction lies in the selected PCs and the AUC of a pooled
source logistic classifier restricted to those PCs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import synthesis as g
from .common import load_standardized_raw


LAYER = 17
N_COMPONENTS = 64
N_SELECTED = 10
MAX_PER_DATASET = 2000
PCA_FIT_PER_DATASET = 500
RANDOM_REPEATS = 10
RANDOM_SEED = 2917

FEATURE_COLUMNS = [
    "log_variance",
    "absolute_skew",
    "excess_kurtosis",
    "bimodality_score",
    "middle_split_separation",
    "dataset_mean_fraction",
    "variance_heterogeneity",
    "log_explained_variance",
    "pc_rank",
]

METHOD_LABELS = {
    "ordinary_top_variance": "Ordinary top PCs",
    "within_top_variance": "Within-dataset top PCs",
    "within_low_dataset_identity": "Least dataset identity",
    "within_predicted_label_like": "Predicted label-like PCs",
    "within_random": "Random PCs",
    "within_oracle": "Target-label oracle",
    "within_all": "All 64 within-dataset PCs",
}
METHOD_ORDER = list(METHOD_LABELS)


def load_standardized() -> tuple[dict[str, np.ndarray],
                                 dict[str, np.ndarray]]:
    """Uniform sampling and common scaling; target labels do not affect either."""
    return load_standardized_raw(
        layer=LAYER, max_per_dataset=MAX_PER_DATASET, seed=RANDOM_SEED)


def fit_bases(
        standardized: dict[str, np.ndarray]
        ) -> tuple[dict[str, PCA], dict[str, dict[str, np.ndarray]]]:
    """Ordinary pooled PCA and PCA of within-dataset deviations."""
    rng = np.random.default_rng(RANDOM_SEED)
    ordinary_fit, within_fit = [], []
    for Z in standardized.values():
        take = rng.choice(
            len(Z), min(PCA_FIT_PER_DATASET, len(Z)), replace=False)
        ordinary_fit.append(Z[take])
        within_fit.append((Z - Z.mean(axis=0))[take])
    bases = {
        "ordinary": PCA(
            n_components=N_COMPONENTS, svd_solver="randomized",
            iterated_power=4, random_state=RANDOM_SEED,
        ).fit(np.concatenate(ordinary_fit)),
        "within": PCA(
            n_components=N_COMPONENTS, svd_solver="randomized",
            iterated_power=4, random_state=RANDOM_SEED + 1,
        ).fit(np.concatenate(within_fit)),
    }
    scores = {
        "ordinary": {
            name: bases["ordinary"].transform(Z)
            for name, Z in standardized.items()},
        "within": {
            name: bases["within"].transform(Z - Z.mean(axis=0))
            for name, Z in standardized.items()},
    }
    return bases, scores


def label_association(score: np.ndarray, y: np.ndarray) -> float:
    """Sign-free one-PC label separation: 0 is chance, 1 is perfect."""
    return 2 * abs(roc_auc_score(y, score) - .5)


def pc_feature_table(
        bases: dict[str, PCA], scores: dict[str, dict[str, np.ndarray]],
        labels: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for basis_name, dataset_scores in scores.items():
        means = np.stack([
            dataset_scores[name].mean(axis=0) for name in g.DATASETS])
        variances = np.stack([
            dataset_scores[name].var(axis=0, ddof=1)
            for name in g.DATASETS])
        between = means.var(axis=0, ddof=1)
        within = variances.mean(axis=0)
        dataset_mean_fraction = between / (between + within + 1e-12)
        variance_heterogeneity = np.std(
            np.log(variances + 1e-12), axis=0)
        explained = bases[basis_name].explained_variance_

        for dataset in g.DATASETS:
            y = labels[dataset]
            for pc in range(N_COMPONENTS):
                values = dataset_scores[dataset][:, pc]
                variance = np.var(values, ddof=1)
                skew = stats.skew(values, bias=False)
                pearson_kurtosis = stats.kurtosis(
                    values, fisher=False, bias=False)
                excess_kurtosis = pearson_kurtosis - 3
                bimodality = (
                    (skew ** 2 + 1) / pearson_kurtosis
                    if pearson_kurtosis > 1e-8 else 0)
                median = np.median(values)
                low, high = values[values <= median], values[values > median]
                split = (
                    (high.mean() - low.mean()) / np.sqrt(variance)
                    if len(low) and len(high) and variance > 1e-12 else 0)
                rows.append({
                    "basis": basis_name, "dataset": dataset, "pc": pc,
                    "label_association": label_association(values, y),
                    "log_variance": np.log(variance + 1e-12),
                    "absolute_skew": abs(skew),
                    "excess_kurtosis": excess_kurtosis,
                    "bimodality_score": bimodality,
                    "middle_split_separation": split,
                    "dataset_mean_fraction": dataset_mean_fraction[pc],
                    "variance_heterogeneity":
                        variance_heterogeneity[pc],
                    "log_explained_variance":
                        np.log(explained[pc] + 1e-12),
                    "pc_rank": pc / (N_COMPONENTS - 1),
                })
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)


def predicted_label_likelihood(
        features: pd.DataFrame, basis: str, target: str
        ) -> tuple[np.ndarray, float]:
    """Fit on source dataset-PC rows, then rank the target's PCs."""
    table = features[features.basis == basis]
    train = table[table.dataset != target]
    test = table[table.dataset == target].sort_values("pc")
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.),
    ).fit(train[FEATURE_COLUMNS], train.label_association)
    prediction = model.predict(test[FEATURE_COLUMNS])
    rank_correlation = stats.spearmanr(
        prediction, test.label_association).statistic
    return prediction, rank_correlation


def low_dataset_identity_indices(
        features: pd.DataFrame, basis: str) -> np.ndarray:
    """PCs with small centroid separation and stable variance across datasets."""
    table = (features[features.basis == basis]
             .drop_duplicates("pc").set_index("pc"))
    score = (
        table.dataset_mean_fraction.rank(pct=True)
        + table.variance_heterogeneity.rank(pct=True)
    )
    return score.nsmallest(N_SELECTED).index.to_numpy()


def source_classifier_auc(
        scores: dict[str, np.ndarray], labels: dict[str, np.ndarray],
        target: str, selected: np.ndarray) -> float:
    sources = [name for name in g.DATASETS if name != target]
    X = np.concatenate([scores[name][:, selected] for name in sources])
    y = np.concatenate([labels[name] for name in sources])
    weights = np.concatenate([
        np.where(labels[name] == 1,
                 .5 / labels[name].sum(),
                 .5 / (labels[name] == 0).sum())
        for name in sources
    ])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1., solver="liblinear", max_iter=2000,
            random_state=RANDOM_SEED),
    )
    model.fit(X, y, logisticregression__sample_weight=weights)
    probability = model.predict_proba(scores[target][:, selected])[:, 1]
    return roc_auc_score(labels[target], probability)


def captured_direction_energy(
        components: np.ndarray, selected: np.ndarray,
        Z: np.ndarray, y: np.ndarray) -> float:
    direction = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
    numerator = np.sum((components[selected] @ direction) ** 2)
    return numerator / np.sum(direction ** 2)


def evaluate(
        standardized: dict[str, np.ndarray], labels: dict[str, np.ndarray],
        bases: dict[str, PCA], scores: dict[str, dict[str, np.ndarray]],
        features: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    low_identity = low_dataset_identity_indices(features, "within")
    for target in g.DATASETS:
        prediction, selector_correlation = predicted_label_likelihood(
            features, "within", target)
        target_features = (features[
            (features.basis == "within")
            & (features.dataset == target)]
            .sort_values("pc"))
        selections = [
            ("ordinary_top_variance", "ordinary",
             np.arange(N_SELECTED), np.nan),
            ("within_top_variance", "within",
             np.arange(N_SELECTED), np.nan),
            ("within_low_dataset_identity", "within",
             low_identity, np.nan),
            ("within_predicted_label_like", "within",
             np.argsort(prediction)[-N_SELECTED:], selector_correlation),
            ("within_oracle", "within",
             target_features.nlargest(
                 N_SELECTED, "label_association").pc.to_numpy(),
             1.),
            ("within_all", "within",
             np.arange(N_COMPONENTS), np.nan),
        ]
        for method, basis, selected, selector_correlation in selections:
            rows.append({
                "dataset": target, "method": method, "basis": basis,
                "n_selected": len(selected),
                "captured_direction_energy": captured_direction_energy(
                    bases[basis].components_, selected,
                    standardized[target], labels[target]),
                "held_dataset_out_auc": source_classifier_auc(
                    scores[basis], labels, target, selected),
                "selector_rank_correlation": selector_correlation,
            })

        random_energy, random_auc = [], []
        for _ in range(RANDOM_REPEATS):
            selected = rng.choice(
                N_COMPONENTS, N_SELECTED, replace=False)
            random_energy.append(captured_direction_energy(
                bases["within"].components_, selected,
                standardized[target], labels[target]))
            random_auc.append(source_classifier_auc(
                scores["within"], labels, target, selected))
        rows.append({
            "dataset": target, "method": "within_random",
            "basis": "within", "n_selected": N_SELECTED,
            "captured_direction_energy": np.median(random_energy),
            "held_dataset_out_auc": np.median(random_auc),
            "selector_rank_correlation": np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_raw_centering(
        standardized: dict[str, np.ndarray],
        labels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Isolate per-dataset mean removal without PCA or PC selection."""
    raw_versions = {
        "raw_uncentered": standardized,
        "raw_dataset_centered": {
            name: Z - Z.mean(axis=0)
            for name, Z in standardized.items()
        },
    }
    all_dimensions = np.arange(
        next(iter(standardized.values())).shape[1])
    rows = []
    for target in g.DATASETS:
        for method, scores in raw_versions.items():
            rows.append({
                "dataset": target,
                "method": method,
                "held_dataset_out_auc": source_classifier_auc(
                    scores, labels, target, all_dimensions),
            })
    results = pd.DataFrame(rows)
    pivot = results.pivot(
        index="dataset", columns="method",
        values="held_dataset_out_auc")
    results = results.merge(
        (pivot.raw_dataset_centered - pivot.raw_uncentered)
        .rename("centered_minus_uncentered"),
        left_on="dataset", right_index=True,
    )
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, part in results.groupby("method"):
        row = {
            "method": method, "method_label": METHOD_LABELS[method],
            "n_datasets": len(part),
        }
        for metric in [
                "captured_direction_energy", "held_dataset_out_auc",
                "selector_rank_correlation"]:
            row[f"{metric}_median"] = part[metric].median()
            row[f"{metric}_q25"] = part[metric].quantile(.25)
            row[f"{metric}_q75"] = part[metric].quantile(.75)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = METHOD_ORDER
    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, 6.8), sharey=True, layout="constrained")
    plot_specs = [
        ("captured_direction_energy",
         "Fraction of target label direction inside selected PCs", (0, 1.)),
        ("held_dataset_out_auc",
         "AUC on the dataset whose labels were hidden", (.35, 1.)),
    ]
    colors = {
        "ordinary_top_variance": g.MUTED,
        "within_top_variance": g.BLUE,
        "within_low_dataset_identity": g.TEAL,
        "within_predicted_label_like": g.PURPLE,
        "within_random": "#98A2B3",
        "within_oracle": g.ORANGE,
        "within_all": g.INK,
    }
    summary_by_method = summary.set_index("method")
    for ax, (metric, label, limits) in zip(axes, plot_specs):
        values = [
            results.loc[results.method == method, metric].dropna()
            for method in order]
        boxes = ax.boxplot(
            values, positions=np.arange(len(order)), vert=False,
            widths=.55, patch_artist=True, showfliers=False,
            medianprops={"color": "white", "linewidth": 1.5})
        for box, method in zip(boxes["boxes"], order):
            box.set_facecolor(colors[method])
            box.set_edgecolor(colors[method])
        for key in ["whiskers", "caps"]:
            for artist in boxes[key]:
                artist.set_color(g.MUTED)
        for i, method in enumerate(order):
            method_summary = summary_by_method.loc[method]
            median = method_summary[f"{metric}_median"]
            box_right = method_summary[f"{metric}_q75"]
            ax.text(
                min(box_right + .012, limits[1] - .035), i,
                f"{median:.2f}", va="center", ha="left",
                fontsize=9, color=g.INK)
        ax.set_xlabel(label)
        ax.set_xlim(*limits)
        ax.grid(axis="x", color=g.GRID, linewidth=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(
        range(len(order)), [METHOD_LABELS[m] for m in order])

    title = "Centering helps PCA choose better PCs; extra scoring adds little"
    fig.suptitle(title, fontsize=20, color=g.INK, fontweight="bold")
    g.save(fig, "label-free-pc-selection")


def main(*, results_root=None, output_root=None) -> None:
    g.configure_paths(results_root=results_root, output_root=output_root)
    standardized, labels = load_standardized()
    bases, scores = fit_bases(standardized)
    features = pc_feature_table(bases, scores, labels)
    results = evaluate(
        standardized, labels, bases, scores, features)
    summary = summarize(results)

    features.to_csv(
        g.TABLES / "label_free_pc_features.csv", index=False)
    results.to_csv(
        g.TABLES / "label_free_pc_selection_by_dataset.csv", index=False)
    summary.to_csv(
        g.TABLES / "label_free_pc_selection_summary.csv", index=False)
    raw_centering = evaluate_raw_centering(standardized, labels)
    raw_centering.to_csv(
        g.TABLES / "label_free_raw_centering_by_dataset.csv", index=False)
    raw_centering_summary = (
        raw_centering.groupby("method").held_dataset_out_auc
        .agg(
            median="median",
            q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75),
        )
        .reset_index()
    )
    raw_centering_summary.to_csv(
        g.TABLES / "label_free_raw_centering_summary.csv", index=False)
    plot_results(results, summary)
    print(summary.round(4).to_string(index=False))
    print("\nRaw activation centering control")
    print(raw_centering_summary.round(4).to_string(index=False))
    print(
        "median centered - uncentered:",
        raw_centering.drop_duplicates("dataset")
        .centered_minus_uncentered.median().round(4),
    )


if __name__ == "__main__":
    main()
