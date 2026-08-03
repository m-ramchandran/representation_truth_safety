"""Do the methods miss the same samples?

For one (tag, layer) of per_dataset or merged, per dataset: each
method's CV error rate (error_rates csv — kept off the figure, where
it crowded the panel titles); how much the methods' error sets overlap
(|both wrong| / |either wrong| for each pair); the samples every method
gets wrong (candidate hard or mislabeled prompts — the second figure,
one bar per dataset splitting the best single method's error rate into
the shared misses it made confidently, the shared ones it nearly got,
and the errors another method already avoids); and whether averaging
the methods' probabilities beats the best single method.

leave_out is the off-domain counterpart and produces only the
correlation figure — its training sets overlap across holdouts, so no
sample has one CV prediction for the error-set analyses to key on. One
figure per holdout kind, one panel per (held-out unit, held-out
dataset): a dataset holdout gives one panel, a task holdout one per
member dataset. The question there is whether methods that all transfer
rank the held-out samples the same way.

Agreement is only meaningful among methods that can actually classify
the dataset, so each (dataset, method) must clear a CV-AUC floor to
enter any of the analyses (--min-auc, default 0.7). A dataset where fewer
than two methods qualify is dropped; what was dropped is printed. For
leave_out the floor applies twice, to the training datasets' CV and
the held-out slice alike.

    python -m classifiers.analysis.errors --experiment per_dataset \
        --tag raw_last --layer 9 [--min-auc 0.7] [--root ...]

Outputs join the runner's own tables/ and figures/ under
<root>/<experiment>/.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..fig_style import heatmap_size
from .load_results import (add_data_args, analysis_dirs, auc_or_nan,
                           cv_predictions, eval_predictions, is_wrong,
                           load_metrics, resolve_root)
from .plots import FS, heatmap, line_titles, panel_grid, save, \
    subtitle_corner


def wrong(preds: pd.DataFrame) -> pd.DataFrame:
    """sample x method matrix of errors (prediction = proba >= 0.5)."""
    p = preds.assign(wrong=is_wrong(preds["proba"], preds["y"]))
    return p.pivot(index="sample_id", columns="method", values="wrong")


def apply_floor(preds: pd.DataFrame, min_auc: float) -> pd.DataFrame:
    """Keep only (dataset, method) pairs whose CV AUC clears the floor,
    and datasets where at least two methods survive. Prints what was
    dropped."""
    auc = preds.groupby(["dataset", "method"]).apply(
        lambda d: auc_or_nan(d["y"], d["proba"]))
    bad = auc[~(auc >= min_auc)]
    if len(bad):
        by_ds = {ds: [f"{m} ({auc[(ds, m)]:.2f})" for _, m in g.index]
                 for ds, g in bad.groupby(level="dataset")}
        for ds, methods in by_ds.items():
            print(f"  {ds}: below AUC {min_auc:g} — dropped "
                  + ", ".join(methods))
    kept = preds.set_index(["dataset", "method"]).loc[
        auc[auc >= min_auc].index].reset_index()
    counts = kept.groupby("dataset")["method"].nunique()
    for ds in counts[counts < 2].index:
        print(f"  {ds}: fewer than 2 qualifying methods, dataset dropped")
    return kept[kept["dataset"].isin(counts[counts >= 2].index)]


def prediction_correlation(preds: pd.DataFrame) -> pd.DataFrame:
    """method x method Spearman correlation of the per-sample predicted
    probabilities. This replaced a 0.5-thresholded error overlap: a hard
    error set throws the scores away and moves with calibration, while a
    rank correlation compares the full ordering.
    Spearman is invariant to any monotone rescaling, so probabilities
    and log-odds give the same answer and knn — which has no margin —
    is included like every other method."""
    wide = preds.pivot(index="sample_id", columns="method", values="proba")
    return wide.corr(method="spearman")


def method_aucs(preds: pd.DataFrame) -> pd.Series:
    """Each method's CV AUC on this dataset — shown in the correlation
    figure's tick labels so agreement and quality read together."""
    return preds.groupby("method").apply(
        lambda d: auc_or_nan(d["y"], d["proba"]))


def error_overlap(w: pd.DataFrame) -> pd.DataFrame:
    """method x method: |both wrong| / |either wrong| (diagonal = 1)."""
    m = w.to_numpy(dtype=float)
    both = m.T @ m
    either = m.sum(0)[:, None] + m.sum(0)[None, :] - both
    return pd.DataFrame(np.where(either > 0, both / np.maximum(either, 1),
                                 np.nan),
                        index=w.columns, columns=w.columns)


def missed_by_all(preds: pd.DataFrame) -> pd.DataFrame:
    """Samples every method gets wrong, with the mean probability —
    far-from-0.5 rows are confidently wrong everywhere."""
    w = wrong(preds)
    hard = w.index[w.all(axis=1)]
    out = (preds[preds["sample_id"].isin(hard)]
           .groupby(["dataset", "sample_id"])
           .agg(y=("y", "first"), mean_proba=("proba", "mean"))
           .reset_index())
    return out.sort_values("mean_proba")


# |mean proba - 0.5| above this is "confident": p < 0.2 or p > 0.8
CONFIDENT = 0.3


def dataset_sizes(root, experiment: str) -> dict:
    """dataset -> how many samples it has, from the metrics rows that
    score a whole dataset once (per_dataset's cv rows, merged's
    per-dataset slices). missed_by_all holds only the missed samples,
    so the denominator has to come from here — and both this module and
    redraw.py read it the same way, so their figures agree."""
    m = load_metrics(experiment, root)
    kind = "cv" if experiment == "per_dataset" else "per_dataset"
    return m[m["kind"] == kind].groupby("test_dataset")["n_test"] \
        .max().to_dict()


def missed_figure(missed: pd.DataFrame, rates: pd.DataFrame, sizes: dict,
                  stem: str, min_auc: float):
    """Three bars per dataset, all as a share of the dataset: the
    samples every method gets wrong with a mean probability outside
    0.2-0.8, the ones every method gets wrong from inside it, and the
    best single method's error rate for scale. Colored bars far short
    of the grey one (got) mean the failures are idiosyncratic and the
    shared few look like mislabels; colored bars close to it
    (social_iqa) mean the whole zoo is stuck on the same samples."""
    m = missed.assign(
        confident=(missed["mean_proba"] - 0.5).abs() > CONFIDENT)
    g = m.groupby("dataset").agg(missed=("sample_id", "size"),
                                 confident=("confident", "sum"))
    g["n"] = pd.Series(sizes).reindex(g.index)
    g = g.dropna(subset=["n"])
    if g.empty:
        return None
    g["conf_pct"] = 100 * g["confident"] / g["n"]
    g["close_pct"] = 100 * (g["missed"] - g["confident"]) / g["n"]
    g["best_pct"] = 100 * rates.min(axis=1).reindex(g.index)
    g["methods"] = rates.notna().sum(axis=1).reindex(g.index)
    g["total"] = g["conf_pct"] + g["close_pct"]
    # every sample nobody gets right is one the best method missed too,
    # so the backdrop never ends left of the colored bar inside it
    g = g.sort_values("total")                     # longest bar on top

    y = np.arange(len(g), dtype=float)
    fig, ax = plt.subplots(figsize=(13.5, 0.95 * len(g) + 3.4),
                           layout="constrained")
    # three plain bars side by side per dataset — stacking them buried
    # the two colored quantities inside the grey one, which is the
    # least interesting of the three
    series = [(0.28, "conf_pct", "confident", "#265fa8",
               "every method wrong — p<0.2 or p>0.8"),
              (0.0, "close_pct", "close", "#9ec2e8",
               "every method wrong — p between 0.2 and 0.8"),
              (-0.28, "best_pct", None, "#b6b6b6",
               "the best single method's error rate")]
    bars, lim = [], float(g["best_pct"].max()) * 1.18
    for off, col, count, color, label in series:
        bars.append(ax.barh(y + off, g[col], 0.26, color=color, zorder=2,
                            label=label))
        if count is None:
            continue
        n = g["confident"] if count == "confident" \
            else g["missed"] - g["confident"]
        for yi, x, k in zip(y, g[col], n):
            ax.annotate(f"{int(k):,}", (x, yi + off),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=FS["cell"])
    ax.set_yticks(y, [f"{d} · {int(k)} methods · {int(nn):,} samples"
                      for d, k, nn in zip(g.index, g["methods"], g["n"])],
                  fontsize=FS["tick"])
    ax.tick_params(axis="x", labelsize=FS["tick"])
    ax.set_xlabel("% of the dataset's samples", fontsize=FS["label"])
    ax.set_xlim(0, lim)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9")
    line_titles(ax, "Samples every method gets wrong",
                f"{stem} — p is the mean probability across methods; "
                f"only methods clearing AUC {min_auc:g} on the dataset "
                f"count as a method here", fs=FS, wrap_in=8)
    ax.legend(bars, [b.get_label() for b in bars],
              fontsize=FS["legend"], loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    return fig


def ensemble_auc(preds: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: each method's AUC and the AUC of the mean probability
    across methods — does averaging beat the best single method?"""
    rows = {}
    for method, g in preds.groupby("method"):
        rows[method] = g.groupby("dataset").apply(
            lambda d: auc_or_nan(d["y"], d["proba"]))
    mean = preds.groupby(["dataset", "sample_id"]).agg(
        y=("y", "first"), proba=("proba", "mean")).reset_index()
    rows["mean_proba"] = mean.groupby("dataset").apply(
        lambda d: auc_or_nan(d["y"], d["proba"]))
    return pd.DataFrame(rows).round(3)


# "pool" is reserved for merged, where it means the whole corpus — on a
# leave-one-out figure it reads as "trained on everything", the opposite
# of the truth
HOLDOUT_LABEL = "AUC on the training datasets → on the held-out one"


def training_aucs(cv: pd.DataFrame) -> pd.Series:
    """(fit, method) -> that fit's AUC on the CV predictions over its own
    training datasets — the in-domain half of leave_out's two-AUC
    floor."""
    return cv.groupby(["fit", "method"]).apply(
        lambda d: auc_or_nan(d["y"], d["proba"]))


def holdout_panels(cv: pd.DataFrame, held: pd.DataFrame, min_auc: float):
    """leave_out's answer to prediction_correlation: one panel per
    (held-out unit, held-out dataset), correlating the methods on
    samples their fit never trained on. A dataset holdout contributes
    one panel; a task holdout contributes one per member dataset (the
    "medical" fit is scored separately on pubmedqa_labeled and
    claude_anatomy — pooling them would mix two transfer stories).

    A method enters a panel only by clearing the floor TWICE: on the CV
    predictions over its training datasets, and on the panel's own
    held-out slice. One floor is not enough here — a method
    that never learned and a method that learned but did not transfer
    both produce near-noise scores off-domain, and correlating noise
    with noise says nothing about the methods.

    Yields (kind, holdout, dataset, corr, cv_auc, eval_auc), the last
    two Series over the surviving methods."""
    train = training_aucs(cv)
    for (kind, fit, dataset), g in held.groupby(
            ["holdout_kind", "fit", "dataset"], sort=True):
        ev = method_aucs(g)
        cvs = pd.Series({m: train.get((fit, m), np.nan)
                         for m in ev.index})
        keep = [m for m in ev.index
                if ev[m] >= min_auc and cvs[m] >= min_auc]
        holdout = fit.rsplit(" (", 1)[0]
        if len(keep) < 2:
            print(f"  {holdout} ({kind}) -> {dataset}: fewer than 2 "
                  f"methods clear AUC {min_auc:g} both on the training "
                  f"datasets' CV and on the held-out slice, panel "
                  f"dropped")
            continue
        yield (kind, holdout, dataset,
               prediction_correlation(g[g["method"].isin(keep)]),
               cvs[keep], ev[keep])


def holdout_correlation(args, tables, figures, stem: str) -> int:
    """The leave_out figures: one per holdout kind, panels as described
    in holdout_panels. Returns how many were written."""
    cv, held = eval_predictions("leave_out", args.tag, args.layer,
                                args.root)
    panels = list(holdout_panels(cv, held, args.min_auc))
    if not panels:
        print("  no panel survived the floor; nothing written")
        return 0
    n = 0
    for kind in sorted({p[0] for p in panels}):
        rows = [p for p in panels if p[0] == kind]
        k = max(len(c.columns) for _, _, _, c, _, _ in rows)
        fig, axes = panel_grid(len(rows), *heatmap_size(k, k))
        out = []
        for ax, (_, holdout, dataset, corr, cvs, ev) in zip(axes, rows):
            labels = [f"{m} (AUC {cvs[m]:.2f}→{ev[m]:.2f})"
                      for m in corr.index]
            labelled = corr.copy()
            labelled.index, labelled.columns = labels, labels
            # every dataset holdout, and a single-member task, evaluates
            # on a dataset of its own name — "beta → beta" says nothing
            # the bare name does not
            heatmap(labelled, vmin=-1, vmax=1, cmap="coolwarm", ax=ax,
                    title=dataset if holdout == dataset
                    else f"{holdout} → {dataset}")
            # unrounded, like the correlations themselves: redraw formats
            # these to 2dp, and rounding here first made 0.8450001 label
            # as 0.85 in this figure and 0.84 in the redrawn one
            out.append(corr.assign(holdout=holdout, dataset=dataset,
                                   cv_auc=cvs, eval_auc=ev,
                                   min_auc=args.min_auc))
        fig.suptitle("Rank correlation between methods' held-out "
                     "predictions\n")
        subtitle_corner(fig, f"{stem} — leave one {kind} out; labels "
                             f"{HOLDOUT_LABEL}, and only methods clearing "
                             f"AUC {args.min_auc:g} on both are shown")
        save(fig, figures / f"prediction_correlation_{stem}_{kind}.png")
        pd.concat(out).to_csv(
            tables / f"prediction_correlation_{stem}_{kind}.csv")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", default="per_dataset",
                    choices=("per_dataset", "leave_out", "merged"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--min-auc", type=float, default=0.7,
                    help="CV-AUC floor a (dataset, method) must clear to "
                         "enter the agreement analyses (leave_out: the "
                         "floor applies to the training datasets' CV and "
                         "the held-out "
                         "slice alike)")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    tables, figures = analysis_dirs(args.root, args.experiment)
    stem = f"{args.tag}_layer{args.layer}"
    if args.experiment == "leave_out":
        # leave_out's training sets overlap across holdouts, so there
        # is no per-sample CV frame for the error-set analyses to key
        # on; its
        # one agreement question — do the methods rank the held-out data
        # the same way? — is the correlation figure
        n = holdout_correlation(args, tables, figures, stem)
        print(f"errors: wrote {n} csv(s) + {n} figure(s) under "
              f"{tables.parent}")
        return

    preds = cv_predictions(args.experiment, args.tag, args.layer, args.root)
    preds = apply_floor(preds, args.min_auc)

    datasets = sorted(preds["dataset"].unique())
    per_ds = {d: wrong(preds[preds["dataset"] == d]) for d in datasets}
    n = max(len(w.columns) for w in per_ds.values())

    # the error-set overlap stays as a table; the FIGURE is the rank
    # correlation between the methods' predictions, with each method's
    # AUC in its tick label
    pd.concat([error_overlap(per_ds[d]).assign(dataset=d)
               for d in datasets]).to_csv(
        tables / f"error_overlap_{stem}.csv")

    corrs = []
    fig, axes = panel_grid(len(datasets), *heatmap_size(n, n))
    for ax, dataset in zip(axes, datasets):
        g = preds[preds["dataset"] == dataset]
        corr = prediction_correlation(g)
        aucs = method_aucs(g)
        corrs.append(corr.assign(dataset=dataset))
        labelled = corr.copy()
        labels = [f"{m} (AUC {aucs[m]:.2f})" for m in corr.index]
        labelled.index, labelled.columns = labels, labels
        heatmap(labelled, vmin=-1, vmax=1, cmap="coolwarm", ax=ax,
                title=dataset)
    fig.suptitle("Rank correlation between methods' CV predictions\n")
    subtitle_corner(fig, f"{stem} — only methods clearing AUC "
                         f"{args.min_auc:g} on the dataset are shown")
    save(fig, figures / f"prediction_correlation_{stem}.png")
    # min_auc rides along so redraw.py can state the floor it was drawn
    # under rather than assume the default
    pd.concat(corrs).assign(min_auc=args.min_auc).to_csv(
        tables / f"prediction_correlation_{stem}.csv")
    # the per-method error rates crowded the panels as subtitles; they
    # live in their own csv instead
    rates = pd.DataFrame({d: w.mean(axis=0) for d, w in per_ds.items()}) \
        .T.rename_axis("dataset").round(3)
    rates.to_csv(tables / f"error_rates_{stem}.csv")

    missed = missed_by_all(preds)
    missed.to_csv(tables / f"missed_by_all_{stem}.csv", index=False)
    ensemble_auc(preds).to_csv(tables / f"ensemble_auc_{stem}.csv")

    # drawn off the rounded rates, which is what redraw reads back
    fig = missed_figure(missed, rates,
                        dataset_sizes(args.root, args.experiment), stem,
                        args.min_auc)
    n_fig = 1
    if fig is not None:
        save(fig, figures / f"missed_by_all_{stem}.png")
        n_fig = 2
    print(f"errors: wrote 5 csvs + {n_fig} figure(s) under {tables.parent}")


if __name__ == "__main__":
    main()
