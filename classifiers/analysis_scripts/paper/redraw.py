"""Redraw curated paper figures from their saved tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._writeup import load


def read(tables: Path, name: str) -> pd.DataFrame:
    path = tables / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def redraw(output_root: Path) -> int:
    tables = output_root / "tables"
    g = load("g_robust_synthesis")
    g.configure_paths(output_root=output_root)

    calls = [
        (g.plot_estimands, (read(tables, "robust_estimands.csv"),)),
        (g.plot_performance_overview,
         (read(tables, "performance_family_oracle.csv"),)),
        (g.plot_classifier_regime_heatmap,
         (read(tables, "classifier_by_regime_median.csv"),)),
        (g.plot_classifier_winner_frequency,
         (read(tables, "classifier_winner_frequency.csv"),)),
        (g.plot_best_classifier_by_dataset,
         (read(tables, "best_classifier_by_dataset_setting.csv"),)),
        (g.plot_feature_groups,
         (read(tables, "feature_group_performance.csv"),)),
        (g.plot_family_gain, (read(tables, "nonlinear_gain_median.csv"),)),
        (g.plot_layer_profiles,
         (read(tables, "layer_profiles_median.csv"),)),
        (g.plot_stability, (read(tables, "layer_stability.csv"),)),
        (g.plot_method_domain,
         (read(tables, "classifier_by_domain_median.csv"),)),
        (g.plot_classifier_errors,
         (read(tables, "classifier_error_profiles.csv"),
          read(tables, "error_overlap_by_classifier.csv"))),
        (g.plot_coefficient_similarity,
         (read(tables, "coefficient_pair_geometry_summary.csv"),)),
        (g.plot_coefficient_training_states,
         (read(tables, "coefficient_pair_geometry_summary.csv"),
          read(tables, "coefficient_training_geometry_summary.csv"))),
        (g.plot_coefficient_pca,
         (read(tables, "coefficient_pca_spectra.csv"),)),
        (g.plot_diff_l2_out_of_domain,
         (read(tables, "diff_vs_l2_out_of_domain_summary.csv"),)),
        (g.plot_wildjailbreak_split,
         (read(tables, "wildjailbreak_by_prompt_type_summary.csv"),)),
        (g.plot_wildjailbreak_logits,
         (read(tables, "wildjailbreak_by_prompt_type_summary.csv"),
          read(tables, "wildjailbreak_output_logits_by_prompt_type.csv"))),
        (g.plot_geometry_transfer,
         (read(tables, "domain_geometry_transfer.csv"),
          read(tables, "geometry_supervision_correlations.csv"))),
        (g.plot_transfer_predictor_increment,
         (read(tables, "transfer_predictor_increment.csv"),
          read(tables, "transfer_predictor_held_target_out.csv"))),
        (g.plot_transfer_additive_regression,
         (read(tables, "transfer_additive_regression.csv"),)),
    ]
    for function, args in calls:
        function(*args)

    covariance = load("h_covariance_transfer")
    covariance.plot_result(
        read(tables, "covariance_transfer_pairs.csv"),
        read(tables, "covariance_transfer_summary.csv"))
    pc_selection = load("i_label_like_pcs")
    pc_selection.plot_results(
        read(tables, "label_free_pc_selection_by_dataset.csv"),
        read(tables, "label_free_pc_selection_summary.csv"))
    return len(calls) + 2
