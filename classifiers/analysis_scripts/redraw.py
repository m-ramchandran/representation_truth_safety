"""Redraw every figure from the saved csvs — no npz, no activations.

Every figure in results/ is a pure function of a csv sitting in the
same experiment's tables/ (the npz-crunching and activation loading
exist only to produce those tables). So a styling / formatting change
does not need the cluster: edit the figure code, run this locally,
and every figure regenerates in seconds from the local tables.

    python -m classifiers.analysis_scripts.redraw --data gemma_tf_a \
        [--only auc complexity mls ...] [--root ...]

Families (--only, default all): auc (the runners' per_dataset /
leave_out / merged heatmaps), prediction_correlation (per_dataset /
merged CV panels, and leave_out's held-out ones),
margins (the score-distribution grid, from its bin-count csv),
got_datasets, gates_routing, complexity, support_sizes, similarity,
mls, pca (reconstruction / individual / heldout / merged / leave_out),
cross_methods (the regime grids, per method and best-of),
missed_by_all (the samples no method gets right).
Families whose csvs are absent are skipped with a note. The
score-distribution grid joined on 2026-07-25, when it started writing
the bin counts behind it — before that it read per-sample log-odds
from the npz and could only be drawn on the cluster — and
cross_methods on 2026-07-28, which is every figure results/ holds.

This module re-reads what the writers wrote, so it knows their csv
schemas; a module that changes its output columns updates its section
here (each section names its writer).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..auc_figures import (METHOD_ORDER, plot_heatmaps,
                           plot_method_by_dataset)
from ..data import COLLECTIONS
from . import complexity as complexity_mod
from . import coef_pca, coef_sparsity
from .coef_similarity import mls_figure
from .directions import LINEAR_METHODS
from .load_results import analysis_dirs, load_metrics, add_data_args, \
    resolve_root
from .plots import layer_panels, method_panels, save

FAMILIES = ("auc", "prediction_correlation",
            "margins", "got_datasets",
            "gates_routing", "complexity", "support_sizes", "similarity",
            "mls", "pca", "rs_vs_single", "rs_holdout", "cross_methods",
            "missed_by_all")


def tags_and_layers(root: Path, experiment: str) -> dict:
    """tag -> sorted layers present in an experiment's metrics csvs."""
    m = load_metrics(experiment, root)
    return {tag: sorted(g["layer"].unique())
            for tag, g in m.groupby("features")}


def redraw_auc(root: Path, c) -> int:
    """The runners' AUC figures (writers: run_per_dataset,
    run_leave_out, run_merged — via classifiers.auc_figures)."""
    n = 0
    exp = root / "per_dataset"
    if (exp / "tables").exists():
        for tag, layers in tags_and_layers(root, "per_dataset").items():
            for layer in layers:
                csv_path = exp / "tables" / f"metrics_{tag}_layer{layer}.csv"
                rows = pd.read_csv(csv_path)
                train = [d for d in c.datasets
                         if d in set(rows["train_dataset"])]
                tests = [d for d in c.datasets
                         if d in set(rows["test_dataset"]) | set(train)]
                plot_heatmaps(csv_path, tag, layer,
                              [m for m in METHOD_ORDER
                               if m in set(rows["method"])],
                              train, tests, exp)
                n += 1
    exp = root / "leave_out"
    if (exp / "tables").exists():
        for tag, layers in tags_and_layers(root, "leave_out").items():
            for kind in ("dataset", "task"):
                plot_method_by_dataset(
                    exp / "tables",
                    lambda L: f"metrics_{tag}_layer{L}_{kind}.csv",
                    tag, layers, METHOD_ORDER, c.datasets,
                    xlabel="held-out test dataset",
                    title=f"held-out AUC, {tag}, leave one {kind} out",
                    out_path=exp / "figures" / f"auc_{tag}_{kind}.png")
                n += 1
    exp = root / "merged"
    if (exp / "tables").exists():
        for tag, layers in tags_and_layers(root, "merged").items():
            plot_method_by_dataset(
                exp / "tables", lambda L: f"metrics_{tag}_layer{L}.csv",
                tag, layers, METHOD_ORDER, c.datasets,
                xlabel="evaluated on",
                title=f"pooled nested-CV AUC, {tag}",
                out_path=exp / "figures" / f"auc_{tag}.png")
            n += 1
    return n


def each_csv(tables: Path, pattern: str):
    """(match, DataFrame) per csv whose name matches the regex."""
    for path in sorted(tables.glob("*.csv")):
        m = re.fullmatch(pattern, path.name)
        if m:
            yield m, path


def redraw_prediction_correlation(root: Path) -> int:
    """Writer: errors.py (prediction_correlation_<stem>.csv, a
    method x method matrix per dataset). AUCs for the tick labels come
    from the runner metrics."""
    from ..fig_style import heatmap_size
    from .plots import heatmap, panel_grid, subtitle_corner

    n = 0
    for experiment in ("per_dataset", "merged"):
        tables, figures = analysis_dirs(root, experiment)
        for m, path in each_csv(tables,
                                r"prediction_correlation_(.+)\.csv"):
            stem = m.group(1)
            tag, layer = stem.rsplit("_layer", 1)
            df = pd.read_csv(path, index_col=0)
            # written since 2026-07-25; older csvs predate the column
            floor = df["min_auc"].iloc[0] if "min_auc" in df else 0.7
            meta = [c for c in ("dataset", "min_auc") if c in df]
            per_ds = {ds: g.drop(columns=meta).dropna(axis=1, how="all")
                      for ds, g in df.groupby("dataset")}
            met = load_metrics(experiment, root)
            met = met[(met["features"] == tag)
                      & (met["layer"] == int(layer))]
            # the CV rows only: a metrics csv holds one cv row per fit
            # next to a dozen transfer / eval_only ones, and keying the
            # whole frame let a transfer AUC win the (dataset, method)
            # slot — labels then showed some other dataset's score
            # (e.g. l1reg on claude_anatomy read 0.42, its claude_vds
            # transfer, in place of the 0.97 CV AUC)
            key, kind = ("train_dataset", "cv") if experiment == "per_dataset" \
                else ("test_dataset", "per_dataset")
            aucs = {(r[key], r["method"]): r["auc"]
                    for _, r in met[met["kind"] == kind].iterrows()}
            k = max(len(g.columns) for g in per_ds.values())
            fig, axes = panel_grid(len(per_ds), *heatmap_size(k, k))
            for ax, (ds, g) in zip(axes, sorted(per_ds.items())):
                labels = [f"{c} (AUC {aucs[(ds, c)]:.2f})"
                          if (ds, c) in aucs else c for c in g.columns]
                g = g.copy()
                g.index, g.columns = labels, labels
                heatmap(g, vmin=-1, vmax=1, cmap="coolwarm", ax=ax,
                        title=ds)
            fig.suptitle("Rank correlation between methods' CV "
                         "predictions\n")
            subtitle_corner(fig, f"{stem} — only methods clearing AUC "
                                 f"{floor:g} on the dataset are shown")
            save(fig, figures / f"prediction_correlation_{stem}.png")
            n += 1
    return n + redraw_holdout_correlation(root)


def redraw_holdout_correlation(root: Path) -> int:
    """Writer: errors.py (prediction_correlation_<stem>_<kind>.csv, a
    method x method matrix per (held-out unit, held-out dataset)). This
    one needs no metrics lookup: the writer stores each row method's two
    AUCs — pool CV and held-out slice — beside its matrix row."""
    from ..fig_style import heatmap_size
    from .plots import heatmap, panel_grid, subtitle_corner
    from .errors import HOLDOUT_LABEL

    n = 0
    tables, figures = analysis_dirs(root, "leave_out")
    for m, path in each_csv(
            tables, r"prediction_correlation_(.+)_(dataset|task)\.csv"):
        stem, kind = m.group(1), m.group(2)
        df = pd.read_csv(path, index_col=0)
        meta = ["holdout", "dataset", "cv_auc", "eval_auc", "min_auc"]
        floor = df["min_auc"].iloc[0]
        panels = [(h, ds, g) for (h, ds), g
                  in df.groupby(["holdout", "dataset"], sort=True)]
        # concat unioned the columns across panels, so a panel's matrix
        # is its own rows' square block — reindex rather than dropna,
        # which would also eat a column that is legitimately all-NaN
        k = max(len(g.index) for _, _, g in panels)
        fig, axes = panel_grid(len(panels), *heatmap_size(k, k))
        for ax, (holdout, dataset, g) in zip(axes, panels):
            labels = [f"{c} (AUC {r.cv_auc:.2f}→{r.eval_auc:.2f})"
                      for c, r in zip(g.index, g.itertuples())]
            g = g.reindex(columns=list(g.index))
            g.index, g.columns = labels, labels
            heatmap(g, vmin=-1, vmax=1, cmap="coolwarm", ax=ax,
                    title=dataset if holdout == dataset
                    else f"{holdout} → {dataset}")
        fig.suptitle("Rank correlation between methods' held-out "
                     "predictions\n")
        subtitle_corner(fig, f"{stem} — leave one {kind} out; labels "
                             f"{HOLDOUT_LABEL}, and only methods clearing "
                             f"AUC {floor:g} on both are shown")
        save(fig, figures / f"prediction_correlation_{stem}_{kind}.png")
        n += 1
    return n


def redraw_margins(root: Path) -> int:
    """Writer: margins.py (margin_hist_<stem>[_<kind>].csv — bin counts
    per method x row unit x series, with each panel's AUC and confidence
    threshold). The score-distribution grid was the one figure here that
    needed the cluster: it read per-sample log-odds from the npz, which
    never leaves it. margins.py now writes the counts a histogram
    actually needs, so this redraws like everything else. leave_out has
    one csv per holdout kind."""
    from .margins import margin_grid

    n = 0
    for experiment in ("per_dataset", "leave_out", "merged"):
        tables, figures = analysis_dirs(root, experiment)
        pattern = r"margin_hist_(.+)_(dataset|task)\.csv" \
            if experiment == "leave_out" else r"margin_hist_(.+)\.csv"
        for m, path in each_csv(tables, pattern):
            stem = m.group(1)
            kind = m.group(2) if experiment == "leave_out" else None
            table = pd.read_csv(path)
            save(margin_grid(table, stem, experiment, kind),
                 figures / (f"margins_{stem}"
                            + (f"_{kind}" if kind else "") + ".png"))
            n += 1
    return n


def redraw_got_datasets(root: Path) -> int:
    """Writer: per_slice.py (got_datasets_<tag>_layer<L>.csv, index
    method x part columns)."""
    n = 0
    for experiment in ("per_dataset", "merged"):
        tables, figures = analysis_dirs(root, experiment)
        by_tag = defaultdict(dict)
        for m, path in each_csv(tables,
                                r"got_datasets_(.+)_layer(\d+)\.csv"):
            by_tag[m.group(1)][int(m.group(2))] = pd.read_csv(path,
                                                              index_col=0)
        for tag, by_layer in by_tag.items():
            save(layer_panels(by_layer,
                              f"AUC by got dataset ({experiment})", tag,
                              vmin=0.5, vmax=1.0),
                 figures / f"got_datasets_{tag}.png")
            n += 1
    return n


def redraw_gates_routing(root: Path) -> int:
    """Writer: gates.py (gates_routing_<tag>_layer<L>.csv; the
    grouping == dataset rows are the figure)."""
    tables, figures = analysis_dirs(root, "merged")
    by_tag = defaultdict(dict)
    for m, path in each_csv(tables, r"gates_routing_(.+)_layer(\d+)\.csv"):
        df = pd.read_csv(path, index_col=0)
        df = df[df["grouping"] == "dataset"].drop(columns="grouping")
        by_tag[m.group(1)][int(m.group(2))] = df
    n = 0
    for tag, by_layer in by_tag.items():
        save(layer_panels(by_layer,
                          "Share of each dataset routed to each expert",
                          tag, vmin=0, vmax=1, cmap="magma"),
             figures / f"gates_routing_{tag}.png")
        n += 1
    return n


def redraw_complexity(root: Path) -> int:
    """Writer: complexity.py (complexity.csv per experiment)."""
    n = 0
    for experiment in ("per_dataset", "leave_out", "merged"):
        tables, figures = analysis_dirs(root, experiment)
        path = tables / "complexity.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for features, g in df.groupby("features"):
            save(complexity_mod.complexity_figure(g, experiment, features),
                 figures / f"complexity_{features}.png")
            n += 1
    return n


def redraw_support_sizes(root: Path) -> int:
    """Writer: coef_sparsity.py (support_sizes.csv)."""
    tables, figures = analysis_dirs(root)
    path = tables / "support_sizes.csv"
    if not path.exists():
        return 0
    save(coef_sparsity.size_panels(pd.read_csv(path)),
         figures / "support_sizes.png")
    return 1


def redraw_similarity(root: Path) -> int:
    """Writer: coef_similarity.py (similarity_<method>_<stem>.csv, long
    form; the method comes from the filename)."""
    tables, figures = analysis_dirs(root)
    by_stem = defaultdict(dict)   # stem -> {(method, measure): Series}
    method_re = "|".join(LINEAR_METHODS)
    for m, path in each_csv(tables,
                            rf"similarity_({method_re})_(.+)\.csv"):
        long = pd.read_csv(path)
        for measure, g in long.groupby("measure"):
            by_stem[m.group(2)][(m.group(1), measure)] = g.set_index(
                ["dataset_a", "dataset_b"])["value"] \
                .rename_axis(["fit", "dataset"])
    n = 0
    for stem, panels in by_stem.items():
        for measure in {mm for _, mm in panels}:
            s = pd.concat({meth: series
                           for (meth, mm), series in panels.items()
                           if mm == measure}, names=["method"])
            save(method_panels(s, vmin=-1, vmax=1, cmap="coolwarm",
                               title=f"Direction similarity ({measure})",
                               subtitle=stem),
                 figures / f"similarity_{measure}_{stem}.png")
            n += 1
    return n


def redraw_mls(root: Path) -> int:
    """Writer: coef_similarity.py (mls_<stem>_<kind>.csv)."""
    tables, figures = analysis_dirs(root)
    n = 0
    for m, path in each_csv(tables, r"mls_(.+)_(dataset|task)\.csv"):
        stem, kind = m.group(1), m.group(2)
        save(mls_figure(pd.read_csv(path), stem, kind),
             figures / f"mls_{stem}_{kind}.png")
        n += 1
    return n


def redraw_pca(root: Path) -> int:
    """Writer: coef_pca.py (pca_* csvs per layer; the cross-layer
    figures reassemble here exactly as coef_pca.main does)."""
    tables, figures = analysis_dirs(root)
    n = 0
    line = {"reconstruction": "Rank-k reconstruction AUC",
            "heldout": "Held-out subspace AUC"}
    sub = {"reconstruction": "PCA fit on single-dataset coefficients",
           "heldout": "PCA fit on the other datasets' single coefficients"}
    for kind_name, title in line.items():
        by_fstem = defaultdict(dict)
        for m, path in each_csv(
                tables, rf"pca_{kind_name}_(.+)_layer(\d+)\.csv"):
            by_fstem[m.group(1)][int(m.group(2))] = pd.read_csv(
                path, index_col=0)
        for fstem, by_layer in by_fstem.items():
            save(coef_pca.layer_line_figure(
                     by_layer, title, f"{sub[kind_name]} — {fstem}"),
                 figures / f"pca_{kind_name}_{fstem}.png")
            n += 1
    by_fstem = defaultdict(dict)
    for m, path in each_csv(tables, r"pca_individual_(.+)_layer(\d+)\.csv"):
        by_fstem[m.group(1)][int(m.group(2))] = pd.read_csv(path,
                                                            index_col=0)
    for fstem, by_layer in by_fstem.items():
        vmin = min(.5, min(float(v.min().min()) for v in by_layer.values()))
        save(layer_panels(by_layer, "Single-PC reconstruction AUC",
                          f"PCA fit on single-dataset coefficients — "
                          f"{fstem}", vmin=vmin, vmax=1),
             figures / f"pca_individual_{fstem}.png")
        n += 1
    for m, path in each_csv(
            tables, r"pca_merged_(.+)_layer(\d+)_(dataset|task)\.csv"):
        stem = f"{m.group(1)}_layer{m.group(2)}"
        save(coef_pca.merged_figure(pd.read_csv(path).fillna({"dataset": ""}),
                                    stem),
             figures / f"pca_merged_{stem}_{m.group(3)}.png")
        n += 1
    by_key = defaultdict(dict)
    for m, path in each_csv(
            tables, r"pca_leave_out_(.+)_layer(\d+)_(dataset|task)\.csv"):
        by_key[(m.group(1), m.group(3))][int(m.group(2))] = pd.read_csv(
            path, index_col=0)
    for (fstem, kind), by_layer in by_key.items():
        save(layer_panels(by_layer,
                          "Leave-out direction vs the pool's singles",
                          f"Leave-out pool directions vs single-dataset "
                          f"coefficients — {fstem}, held-out {kind}",
                          vmin=0, vmax=1),
             figures / f"pca_leave_out_{fstem}_{kind}.png")
        n += 1
    return n


def redraw_rs_vs_single(root: Path) -> int:
    """Writer: rs_directions.py (rs_vs_single_<stem>.csv, long form; one
    figure per random-slopes variant in the csv)."""
    from .rs_directions import rs_figure
    tables, figures = analysis_dirs(root)
    n = 0
    for m, path in each_csv(tables, r"rs_vs_single_(.+)\.csv"):
        df = pd.read_csv(path)
        for variant in sorted(df["variant"].unique()):
            save(rs_figure(df, m.group(1), variant),
                 figures / f"rs_vs_single_{m.group(1)}_{variant}.png")
            n += 1
    return n


def redraw_missed_by_all(root: Path) -> int:
    """Writer: errors.py (missed_by_all_<stem>.csv, one row per sample
    nobody gets right). The bar denominators are dataset sizes, which
    that csv does not carry — errors.dataset_sizes reads them off the
    metrics, the same call the writer makes."""
    from .errors import dataset_sizes, missed_figure

    n = 0
    for experiment in ("per_dataset", "merged"):
        tables, figures = analysis_dirs(root, experiment)
        for m, path in each_csv(tables, r"missed_by_all_(.+)\.csv"):
            stem = m.group(1)
            rates_path = tables / f"error_rates_{stem}.csv"
            corr_path = tables / f"prediction_correlation_{stem}.csv"
            if not rates_path.exists():
                continue
            # the floor is only recorded on the correlation csv
            floor = 0.7
            if corr_path.exists():
                corr = pd.read_csv(corr_path)
                if "min_auc" in corr:
                    floor = corr["min_auc"].iloc[0]
            fig = missed_figure(pd.read_csv(path),
                                pd.read_csv(rates_path, index_col=0),
                                dataset_sizes(root, experiment), stem,
                                floor)
            if fig is None:
                continue
            save(fig, figures / f"missed_by_all_{stem}.png")
            n += 1
    return n


def redraw_cross_methods(root: Path, c) -> int:
    """Writer: cross_methods.py (regimes_<tag>.csv under
    cross_methods/, long form — method x dataset x regime x layer, the
    figures' whole input including the recomputed in_domain_fit rows).
    Row order is the only thing not in the csv: it comes from the
    collection, as it does in the writer."""
    from .cross_methods import figures_from_long
    tables, figures = analysis_dirs(root, "cross_methods")
    n = 0
    for m, path in each_csv(tables, r"regimes_(.+)\.csv"):
        long = pd.read_csv(path)
        order = [d for d in c.datasets if d in set(long["dataset"])]
        n += figures_from_long(long, order, m.group(1), figures)
    return n


def redraw_rs_holdout(root: Path) -> int:
    """Writer: rs_holdout.py (rs_holdout_<tag>.csv under leave_out/)."""
    from .rs_holdout import rs_figure
    tables, figures = analysis_dirs(root, "leave_out")
    n = 0
    for m, path in each_csv(tables, r"rs_holdout_(.+)\.csv"):
        save(rs_figure(pd.read_csv(path), m.group(1)),
             figures / f"rs_holdout_{m.group(1)}.png")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", default=list(FAMILIES),
                    choices=FAMILIES)
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    c = COLLECTIONS[args.data]

    draw = {"auc": lambda: redraw_auc(args.root, c),
            "prediction_correlation":
                lambda: redraw_prediction_correlation(args.root),
            "margins": lambda: redraw_margins(args.root),
            "got_datasets": lambda: redraw_got_datasets(args.root),
            "gates_routing": lambda: redraw_gates_routing(args.root),
            "complexity": lambda: redraw_complexity(args.root),
            "support_sizes": lambda: redraw_support_sizes(args.root),
            "similarity": lambda: redraw_similarity(args.root),
            "mls": lambda: redraw_mls(args.root),
            "pca": lambda: redraw_pca(args.root),
            "rs_vs_single": lambda: redraw_rs_vs_single(args.root),
            "rs_holdout": lambda: redraw_rs_holdout(args.root),
            "cross_methods": lambda: redraw_cross_methods(args.root, c),
            "missed_by_all": lambda: redraw_missed_by_all(args.root)}
    for family in args.only:
        try:
            n = draw[family]()
        except FileNotFoundError as e:
            print(f"{family}: skipped ({e})")
            continue
        print(f"{family}: {n} figure(s)")


if __name__ == "__main__":
    main()
