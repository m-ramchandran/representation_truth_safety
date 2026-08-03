"""Is transfer failure uncertain, or confidently wrong?

For each fit applied to data it never trained on (per_dataset's transfer
sets, leave_out's held-out unit, and for merged the per-dataset slices
of the pooled nested CV, where every sample is likewise scored by a fit
that did not train on it), compare the prediction distribution there
with the fit's own CV predictions, on the shared log-odds scale.

Confidence is measured on each learner's own scale, because a fixed
|log-odds| cutoff reads "never confident" and "never wrong at high
confidence" identically for a learner whose scores never leave the
boundary region. A sample's confidence quantile is therefore
where its |log-odds| falls in the fit's OWN CV |log-odds| distribution
(the empirical CDF, 0 = the fit's most hesitant call, 1 = its most
confident). The summary gives, per (method, fit, evaluated dataset):
AUC, error rate, the 10/50/90% quantiles of the log-odds — a moved
median means the scores shifted toward one class on the new data, a
narrowed 10-90 span means they shrank toward the boundary —
`confidently_wrong` (the share of samples that are wrong AND above the
fit's own CONF_Q = 80% CV confidence quantile), and
`wrong_median_conf_q` (the median confidence quantile of the wrong
samples: ~0.5 means errors look like typical calls, near 1 means the
errors are the fit's MOST confident calls, near 0 means it hesitates
exactly where it errs — "knows it doesn't know"). The same columns on
the fit's own CV predictions (suffix _cv) are the baseline.

One figure. A per-learner dot plot of the confidently-wrong RATE (CV
diamond against one dot per evaluated dataset) was dropped as redundant:
the grid already shows, per method, how much held-out mass lands past
the confidence cutoff on the wrong side. The rate itself is not gone —
margin_summary_<stem>.csv still carries `confidently_wrong` and
`confidently_wrong_cv` per (method, fit, dataset), which is where to
read magnitudes the normalized panels cannot show. It is legible on the
grid too: blue and red are normalized over the panel's evaluated samples
together, so red's area past the dashed cutoff is `confidently_wrong`
itself.

`margin_grid` is the distribution `confidently_wrong` is computed from:
rows = evaluated unit, columns = method, each panel annotated with the
AUC there. Comparing methods is the point of the experiment, so they
share a figure — leave_out gets one per holdout kind. Its bin counts go
to margin_hist_<stem>[_<kind>].csv, which is what lets redraw.py rebuild
the grid without the npz (and so without a GPU node); --hist narrows the
drawn columns but never that csv.

    python -m classifiers.analysis.margins --experiment leave_out \
        --tag raw_last --layer 9 [--hist l2reg lda] [--root ...]

Outputs under <root>/<experiment>/analysis/. knn has no margin and only
appears in the AUC / error-rate columns.
"""

from __future__ import annotations

import argparse
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import COLLECTIONS
from ..fig_style import FS, FS_SMALL
from .load_results import (add_data_args, analysis_dirs, auc_or_nan,
                           eval_predictions, is_wrong, resolve_root)
from .plots import save, subtitle_corner

CONF_Q = 0.8   # "confident" = above this quantile of the fit's own
               # CV |log-odds| distribution (its own confidence scale)
DATA_TOP = 0.82  # bars and rules stop at this fraction of a panel; the
                 # strip above it is left clear for the AUC label


def conf_quantiles(logodds: pd.Series, ref: np.ndarray) -> np.ndarray:
    """Each sample's |log-odds| as a quantile of the reference |log-odds|
    distribution (the fit's own CV scores, sorted ascending)."""
    return np.searchsorted(ref, logodds.abs().to_numpy(),
                           side="right") / len(ref)


def stats(g: pd.DataFrame, ref: np.ndarray = None) -> pd.Series:
    wrong = is_wrong(g["proba"], g["y"])
    q10, q50, q90 = g["logodds"].quantile([.1, .5, .9])
    cw, wq = np.nan, np.nan
    if ref is not None and len(ref) and g["logodds"].notna().any():
        q = conf_quantiles(g["logodds"], ref)
        cw = (wrong & (q > CONF_Q)).mean()
        if wrong.any():
            wq = float(np.median(q[wrong]))
    return pd.Series({
        "auc": auc_or_nan(g["y"], g["proba"]),
        "error_rate": wrong.mean(),
        "q10": q10, "q50": q50, "q90": q90,
        "confidently_wrong": cw,
        "wrong_median_conf_q": wq,
    })


def summary(cv: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Held-out stats per (method, fit, evaluated dataset), with the
    fit's own CV stats alongside (columns suffixed _cv). Confidence
    quantiles are always taken against the fit's own CV |log-odds| —
    for the held-out rows and the CV baseline alike."""
    refs = {key: np.sort(g["logodds"].abs().dropna().to_numpy())
            for key, g in cv.groupby(["method", "fit"])}
    h = held.groupby(["method", "fit", "dataset"]).apply(
        lambda g: stats(g, refs.get(g.name[:2])))
    c = cv.groupby(["method", "fit"]).apply(
        lambda g: stats(g, refs.get(g.name)))
    return h.join(c, rsuffix="_cv").round(3)


def _fit_order(fits, order: list = None) -> list:
    """Learners in the collection's display order — per_dataset fits are
    the training dataset, leave_out's are "<holdout> (dataset|task)", so
    the dataset holdouts come first, then the task ones, each in
    display order with unknown names last."""
    rank = {d: i for i, d in enumerate(order or [])}
    return sorted(fits, key=lambda f: (f.endswith("(task)"),
                                       rank.get(f.split(" (")[0], len(rank)),
                                       f))


# Which samples the gray CV histogram covers. It is NOT the same sample
# set in each experiment, which is exactly what a bare "own nested CV"
# hid: per_dataset trains on the row's dataset, leave_out on every
# dataset BUT it. Two further wording corrections follow from that:
# "everything but this unit" had nothing on the figure to point at, so
# the grid puts the unit on the row and the wording points there; and
# "pool" is reserved for merged, where the pool IS the whole
# corpus — calling leave_out's training set a pool read as "trained on
# everything", the opposite of the truth, so it is spelled out instead.
CV_SAMPLES = {
    "per_dataset": "nested CV on the training\ndataset itself",
    "leave_out": "nested CV on all the training\ndatasets — every one "
                 "except\nthis row's",
    "merged": "nested CV over the whole\npooled corpus",
}


def cv_label(experiment: str) -> str:
    return CV_SAMPLES.get(experiment, "own nested CV")

# margin_grid's row axis, and what the blue/red in a row is measured on.
# A bare unit name means the opposite thing in the two experiments —
# per_dataset's is what the fit trained ON, leave_out's is what it was
# trained WITHOUT — so the row axis has to say which.
GRID_ROWS = {"per_dataset": "trained on", "leave_out": "held out",
             "merged": "evaluated on"}
EVAL_SAMPLES = {
    "per_dataset": "the transfer sets — every\ndataset except the row's",
    "leave_out": "the held-out unit itself",
    "merged": "the row's dataset",
}
GRID_NOTE = {
    "per_dataset": "a row is one training dataset",
    "leave_out": "a row is one held-out unit; the fit in that row trained "
                 "on every other dataset",
    "merged": "a row is one dataset — the same pooled fit in every row, "
              "only the samples it is scored on change",
}


def margin_methods(held: pd.DataFrame, methods: list) -> list:
    """Those of `methods` that store a margin, in the order given —
    "all" (or an empty list) means every method in the frame that has
    one. knn is the standing example of one that does not."""
    have = [m for m in held["method"].unique()
            if held.loc[held["method"] == m, "logodds"].notna().any()]
    if not methods or methods == ["all"]:
        return sorted(have)
    return [m for m in methods if m in have]


def _row_slices(c: pd.DataFrame, h: pd.DataFrame, row: str,
                experiment: str) -> tuple:
    """(held samples, CV log-odds) for one row of one method's column.
    merged's rows are datasets scored by the single pooled fit, so every
    row shares that fit's CV; elsewhere the row IS the fit."""
    if experiment == "merged":
        return h[h["dataset"] == row], c["logodds"].dropna()
    return h[h["fit"] == row], c[c["fit"] == row]["logodds"].dropna()


SERIES = ("cv", "correct", "wrong")   # gray, blue, red
SERIES_COLOR = {"cv": "0.55", "correct": "tab:blue", "wrong": "tab:red"}
N_BINS = 40


def margin_hist_table(cv: pd.DataFrame, held: pd.DataFrame, methods: list,
                      order: list = None, experiment: str = None,
                      kind: str = None) -> pd.DataFrame:
    """The margin grid as BIN COUNTS — one row per (method, row unit,
    series, bin), with the panel's AUC and confidence threshold beside
    it. Long form, in display order: the reader of this csv takes rows
    and columns in order of first appearance.

    This is the sidecar that lets the grid be redrawn without a GPU
    node. Every other figure in results/ is a pure function of a csv in
    the same tables/ directory, so `redraw.py` regenerates it in
    seconds; this one alone read per-sample scores from the npz, so
    every label tweak used to cost a job.
    Counts are all a histogram needs, and ~12 methods x ~17 fits x 3
    series x 40 bins is a few hundred KB.

    `margin_grid` draws FROM this table rather than from the scores, so
    the figure and the sidecar cannot drift apart.

    Bins are per method (see margin_grid); `n_samples` is that series'
    in-range total, and every denominator margin_grid divides by is
    built from those, so the areas it draws are exact against what the
    bins hold. The outermost 0.5% of each method's span falls outside
    the bins and outside `n_samples` alike.
    """
    if kind is not None:
        held = held[held["holdout_kind"] == kind]
    cols = margin_methods(held, methods)
    if not cols:
        raise ValueError(f"none of {', '.join(methods) or 'the methods'} "
                         "stores a margin, so there is no distance to a "
                         "boundary to tabulate")
    if experiment == "merged":
        seen = set(held["dataset"])
        rows = [d for d in (order or []) if d in seen]
        rows += sorted(d for d in seen if d not in rows)
        labels = dict(zip(rows, rows))
    else:
        rows = _fit_order(held["fit"].unique(), order)
        # with one figure per kind the "(dataset)" / "(task)" suffix is
        # the figure, not the row
        labels = {r: (r.rsplit(" (", 1)[0] if kind else r) for r in rows}

    out = []
    for method in cols:
        c = cv[cv["method"] == method]
        h = held[held["method"] == method]
        span = pd.concat([c["logodds"], h["logodds"]]).dropna()
        edges = np.linspace(*span.quantile([0.005, 0.995]), N_BINS + 1)
        for row in rows:
            hf, cvf = _row_slices(c, h, row, experiment)
            wrong = is_wrong(hf["proba"], hf["y"])
            series = {"cv": cvf,
                      "correct": hf["logodds"][~wrong].dropna(),
                      "wrong": hf["logodds"][wrong].dropna()}
            auc = auc_or_nan(hf["y"], hf["proba"])
            thr = float(np.quantile(cvf.abs(), CONF_Q)) if len(cvf) else np.nan
            for name, values in series.items():
                count = np.histogram(values, bins=edges)[0]
                out.append(pd.DataFrame({
                    "method": method, "row": labels[row], "series": name,
                    "bin_left": edges[:-1], "bin_right": edges[1:],
                    "count": count, "n_samples": count.sum(),
                    "auc": auc, "thr": thr,
                }))
    # full float repr tripled the file for digits no histogram can show
    return pd.concat(out, ignore_index=True).round(
        {"bin_left": 6, "bin_right": 6, "auc": 4, "thr": 6})


def margin_grid(table: pd.DataFrame, stem: str = "",
                experiment: str = None, kind: str = None):
    """Where the evaluated scores land relative to each fit's decision
    boundary: one ROW per evaluated unit, one COLUMN per method, drawn
    from `margin_hist_table`'s counts.

    The point of the experiment is comparing methods, so the methods
    have to share a figure — one figure per method could not be read
    against another, which is why this replaced an earlier per-method
    figure of per-fit panels. leave_out gets one figure per holdout
    kind, so a row is a dataset in one and a task in the other rather
    than the two interleaved down a single column.

    In every panel: gray is the fit's own nested-CV scores (the
    training-side reference — for leave_out that fit trained on all the
    OTHER datasets, never on the row's unit), blue and red are the
    samples it never trained on, split correct / wrong. Each panel is
    annotated with the AUC actually achieved there, so a suggestive
    shape can be read against whether the method transferred at all.

    The x scale is the calibrated log-odds, i.e. the signed distance to
    the boundary. Training fixes that boundary; what this shows is how
    the NEW cloud sits against it, and three failures that share an
    error rate look nothing alike: the cloud shifted off center (the fit
    calls almost everything one class on the new data), the cloud
    collapsed onto zero (the direction carries no signal there —
    failing, but hesitantly), or red mass far out on the wrong side
    (confidently wrong). The dashed pair at +/- the fit's CONF_Q CV
    |log-odds| quantile is the cutoff `confidently_wrong` counts
    against, so the red area outside those lines IS that statistic,
    drawn.

    That last sentence is only true because blue and red share ONE
    denominator, the panel's evaluated total. Normalized separately, as
    they were at first, each enclosed area 1 and red's tail read as a
    share of the ERRORS — 1/error_rate times
    too big, 20x at a 5% error rate — while `confidently_wrong` is a
    share of all evaluated samples. Sharing the denominator also puts
    the correct/wrong proportion on the figure at all: red's total area
    is now the error rate, printed beside the AUC as well so the panels
    where red is a sliver still say how thin. Gray stays on its own,
    being a different sample set whose size means nothing here (under
    leave_out it is the other datasets entirely).

    Limits and bins are per COLUMN: the log-odds scale is shared in
    principle, but one heavy-tailed method (qda, xgboost) sets a range
    that flattens every other column to a spike. Rows within a column
    are therefore directly comparable, columns only in shape. Heights
    still are not: y is rescaled per panel to its own tallest series.
    """
    cols = list(dict.fromkeys(table["method"]))
    rows = list(dict.fromkeys(table["row"]))
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                             sharex="col", layout="constrained",
                             figsize=(2.5 * len(cols) + 2.2,
                                      1.5 * len(rows) + 2.0))
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.06)
    for j, method in enumerate(cols):
        col = table[table["method"] == method]
        lo, hi = col["bin_left"].min(), col["bin_right"].max()
        for i, row in enumerate(rows):
            ax = axes[i][j]
            panel = col[col["row"] == row]
            # blue and red are a PARTITION of one sample set, so they
            # share its total as their denominator; gray is a different
            # set entirely (and under leave_out one whose size has no
            # relation to the row's), so it keeps its own.
            n = {name: int(g["n_samples"].iloc[0])
                 for name, g in panel.groupby("series")}
            n_eval = n.get("correct", 0) + n.get("wrong", 0)
            peak = 0.0
            for name in SERIES:
                s = panel[panel["series"] == name]
                denom = n.get(name, 0) if name == "cv" else n_eval
                if s.empty or not denom:
                    continue
                width = s["bin_right"] - s["bin_left"]
                density = s["count"] / (denom * width)
                peak = max(peak, density.max())
                # at a low error rate red is a sliver against blue —
                # true, and the point, but it has to stay visible, so
                # it carries an edge the filled series do not
                ax.stairs(density, list(s["bin_left"]) + [hi], fill=True,
                          color=SERIES_COLOR[name], alpha=.55)
                if name == "wrong":
                    ax.stairs(density, list(s["bin_left"]) + [hi],
                              color=SERIES_COLOR[name], lw=0.9)
            # x in data coords, y in axes fraction, so every rule stops
            # at DATA_TOP instead of running the full height: at full
            # height a dashed line crossed the AUC label in 40% of
            # panels (worst on lda / l1reg / svm_linear, whose CV
            # cutoff sits near the edge of their held-out range).
            rule = ax.get_xaxis_transform()
            ax.plot([0, 0], [0, DATA_TOP], color="k", lw=1,
                    transform=rule)               # the boundary itself
            thr = panel["thr"].iloc[0] if len(panel) else np.nan
            if not np.isnan(thr):
                for x in (-thr, thr):
                    ax.plot([x, x], [0, DATA_TOP], color="0.35",
                            ls=(0, (4, 3)), lw=1.4, transform=rule)
            auc = panel["auc"].iloc[0] if len(panel) else np.nan
            ax.text(0.03, 0.97,
                    "AUC —" if np.isnan(auc) else f"AUC {auc:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=FS_SMALL["cell"], color="0.2")
            # the red area IS this number; printing it too means a
            # 2%-error panel, where red is a sliver, is still readable
            if n_eval:
                ax.text(0.97, 0.97, f"err {n.get('wrong', 0) / n_eval:.0%}",
                        transform=ax.transAxes,
                        ha="right", va="top", fontsize=FS_SMALL["cell"],
                        color="tab:red")
            ax.set_xlim(lo, hi)
            # the bars are scaled into DATA_TOP as well, so the label
            # strip stays clear of a tall bin too
            if peak:
                ax.set_ylim(0, peak / DATA_TOP)
            ax.set_yticks([])                   # densities are not compared
            ax.tick_params(labelsize=FS_SMALL["tick"])
            if i == 0:
                ax.set_title(method, fontsize=FS_SMALL["title"])
            if j == 0:
                # horizontal, so long dataset names stay readable
                ax.set_ylabel(row, rotation=0, ha="right", va="center",
                              fontsize=FS_SMALL["label"])
    ev = EVAL_SAMPLES.get(experiment, "the evaluated samples")
    handles = [
        matplotlib.patches.Patch(color="0.55", alpha=.55,
                                 label="TRAINING-SIDE reference:\n"
                                       + cv_label(experiment)),
        matplotlib.patches.Patch(color="tab:blue", alpha=.55,
                                 label=f"EVALUATED on {ev}:\ncorrect"),
        matplotlib.patches.Patch(color="tab:red", alpha=.55,
                                 label="wrong — blue and red share\none "
                                       "denominator, so red's area\nis the "
                                       "error rate"),
        plt.Line2D([], [], color="k", lw=1, label="the decision boundary"),
        plt.Line2D([], [], color="0.35", ls=(0, (4, 3)), lw=1.4,
                   label=f"+/- the fit's own {CONF_Q:.0%}\nCV confidence "
                         f"quantile"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
               fontsize=FS["legend"], frameon=False)
    # "log-odds = distance to boundary" as an axis label repeats a
    # sentence under every column; the gloss belongs in the subtitle,
    # said once
    fig.supxlabel("log-odds", fontsize=FS["label"])
    fig.supylabel(GRID_ROWS.get(experiment, "fit"), fontsize=FS["label"])
    # normalizing every series to its own area 1 made a tall red peak
    # mean "concentrated errors" and not "many errors" — a distinction
    # no reader made. Blue and red share the evaluated total instead, so
    # the areas ARE the proportions.
    sub = (f"{stem} — {GRID_NOTE.get(experiment, '')}"
           + (f", leaving one {kind} out" if kind else "")
           + "\nlog-odds = the signed distance to the fit's boundary; AUC is "
             "that fit's AUC on the panel's own samples"
             "\nblue and red are normalized together over the panel's "
             "evaluated samples, so red's area is the error rate and the red "
             "mass beyond the dashed lines is the confidently-wrong share; "
             "gray is normalized to its own samples, and the x range is per "
             "column")
    # the grid is as wide as it has methods, so the subtitle has to wrap
    # to it — at three columns these lines ran out under the legend
    # (~8.6 chars/inch at the subtitle size, as in plots.line_titles)
    width = max(60, int(8.6 * fig.get_size_inches()[0]))
    sub = "\n".join(part for line in sub.split("\n")
                    for part in textwrap.wrap(line, width))
    subtitle_corner(fig, sub)
    # the corner subtitle is free-floating fig.text, so constrained
    # layout cannot see it — the title clears it by carrying that many
    # blank lines itself
    fig.suptitle("\n" * sub.count("\n") + "Where the scores land",
                 fontsize=FS["suptitle"])
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiment", default="leave_out",
                    choices=("per_dataset", "leave_out", "merged"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hist", metavar="METHOD", nargs="+", default=[],
                    help="restrict the score-distribution grid to these "
                         "methods (default: every method that stores a "
                         "margin). The bin-count csv behind the grid is "
                         "always written for every such method, so redraw.py "
                         "can rebuild it later whatever this says")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)

    cv, held = eval_predictions(args.experiment, args.tag, args.layer,
                                args.root)
    tables, figures = analysis_dirs(args.root, args.experiment)
    stem = f"{args.tag}_layer{args.layer}"

    # the summary keeps confidently_wrong (+ _cv) as numbers; the dot
    # plot that drew them was dropped as redundant with the grid, so
    # this csv is the only place the rate is reported
    s = summary(cv, held)
    s.to_csv(tables / f"margin_summary_{stem}.csv")
    # the csv covers every margin-carrying method whatever --hist says:
    # it is the standalone copy of the scores, and a narrowed one would
    # silently make those methods un-redrawable for that layer
    every = margin_methods(held, [])
    cols = margin_methods(held, args.hist)
    dropped = [m for m in args.hist if m not in cols and m != "all"]
    if dropped:
        print(f"  margins --hist: no margin stored by {', '.join(dropped)}, "
              f"dropped from the grid")
    # one figure per holdout kind: a row means a dataset in one and a
    # task in the other, and interleaving them down one column asks the
    # reader to keep two transfer stories apart
    kinds = sorted(held["holdout_kind"].unique()) \
        if args.experiment == "leave_out" else [None]
    for kind in kinds:
        suffix = f"_{kind}" if kind else ""
        # the bin counts are written whether or not the png is drawn
        # here: they are what makes the grid redrawable without a GPU
        # node, and a run that skipped them would leave that layer
        # permanently tied to one
        table = margin_hist_table(cv, held, every,
                                  COLLECTIONS[args.data].datasets,
                                  args.experiment, kind)
        table.to_csv(tables / f"margin_hist_{stem}{suffix}.csv", index=False)
        save(margin_grid(table[table["method"].isin(cols)], stem,
                         args.experiment, kind),
             figures / f"margins_{stem}{suffix}.png")
    print(f"margins: wrote margin_summary + {len(kinds)} margin_hist csv(s) "
          f"over {', '.join(every)}, {len(kinds)} score-distribution "
          f"grid(s) over {', '.join(cols)}, under {tables.parent}")


if __name__ == "__main__":
    main()
