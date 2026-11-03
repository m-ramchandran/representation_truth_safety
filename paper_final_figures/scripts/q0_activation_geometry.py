"""Q0 figure: the raw activation geometry the probes operate in.

One PCA per model, fit on every Original-prompt sample of the paper's
12-dataset roster at the matched 88% depth, with every sample of both
framings projected through that same basis. Nothing is subsampled -
neither the fit nor the plot. Rows are models; the two views are the
Original samples colored by dataset and the same points colored by their
own binary label.

Each panel carries the held-out supervised readout of the map it shows:
a multinomial logistic regression predicting dataset identity and a
binary one predicting the label, each fit on the two components in the
picture and again on all of them. Since PCA is a rotation, the all-PC
number is the linear probe on the whole raw activation, so the pair
brackets how much of the readable structure the picture is showing.

The framing comparison lives in supplementary_figure1_framing, on a
basis of its own: a second PCA fit on the two framings pooled, since
asking where the framed prompts sit only means something on a map they
helped define.

The main figure reads the `figure1_activation_*` caches; its framing
supplement reads `supplementary_figure1_framing_*`.  They are written on
compute cluster by full_activation_pca.py and joint_framing_pca.py because the
activation caches live there and the full-rank fit needs ~60 GB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import fig_common as F

LAYER = {"gemma": 29, "qwen": 24}          # matched 88% depth, 0-indexed
LABEL_COLOR = {1: F.TEAL, 0: F.ORANGE}
LABEL_TEXT = {1: "Label 1 (true / yes / safe / harmless)",
              0: "Label 0 (false / no / unsafe / harmful)"}
POINT = dict(s=1.6, alpha=0.05, linewidths=0, rasterized=True)
# Overlapping groups: opaque tiny points, so a region holding both groups
# speckles in the two colours instead of blending into a third one. At
# alpha 0.05 two hues drawn on top of each other mix - blue over red comes
# out purple - and a two-colour panel reads as four.
POINT_CAT = dict(s=0.6, alpha=1.0, linewidths=0, rasterized=True,
                 antialiased=False)
LEGEND_STYLE = dict(fontsize=F.pt("label"), labelspacing=0.34,
                    handletextpad=0.4)


def _points(stem: str) -> pd.DataFrame:
    d = np.load(C.DATA / f"{stem}_points.npz", allow_pickle=False)
    return pd.DataFrame({k: d[k] for k in d.files})


def _readout_text(r: pd.DataFrame, model: str, target: str) -> str:
    """The panel title: what the two plotted components separate, and what
    the whole representation separates, for the same target. Which metric
    these are is named once in the column heading rather than repeated
    over every panel.

    Dataset identity is a 12-way choice, so it is scored by accuracy
    rather than AUC: the question is whether you can name the dataset a
    point came from, not whether you can rank points by affinity to it.
    The second figure averages the 12 datasets equally instead of
    weighting each by its size - without it, a model that learned only
    the five largest datasets and got the other seven wrong every time
    would still score 92%, because those five are 92% of the corpus. The
    label is binary and exactly 50/50, so AUC says it all there.
    """
    s = r[(r.model == model) & (r.target == target)].set_index("features")
    two, full = s.loc["first 2 PCs"], s.loc["all PCs"]
    if target == "dataset identity":
        return (f"Top 2 PCs: {two.accuracy:.0%} "
                f"({two.balanced_accuracy:.0%} dataset-averaged)\n"
                f"All PCs: {full.accuracy:.0%} "
                f"({full.balanced_accuracy:.0%} dataset-averaged)")
    return (f"Top 2 PCs: {two.auc:.3f}\n"
            f"All PCs: {full.auc:.3f}")


PERT_COLOR = {"bare": F.BLUE, "true_false": F.RED}
PERT_TEXT = {"bare": "Original prompts", "true_false": "True/False framing"}


def _supplement() -> None:
    """The same cloud three times, coloured by a different variable each
    time: dataset, prompt framing, true label.

    Its own PCA, not the main figure's. These panels carry every sample
    of BOTH framings - 430,230 per model - and the question they ask is
    where the wrapped prompts sit relative to the bare ones, so the basis
    is fit on the two framings pooled rather than on the Original samples
    alone. On the Original-only basis the framed points would be a
    projection into a map they had no part in defining, and how far they
    landed from it would be partly an artifact of that.

    social_iqa is dropped from the framing panel only - it has no bare
    variant, so its two framings are the same activations and colouring
    them differently would be drawing a distinction that is not there.
    (It is in the fit once for the same reason; see
    joint_framing_pca.py.)
    """
    pts = _points("supplementary_figure1_framing")
    var = pd.read_csv(C.DATA / "supplementary_figure1_framing_variance.csv")
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 3, figsize=F.canvas(0.72),
                             sharex="row", sharey="row")
    fig.subplots_adjust(left=0.135, right=0.988, top=0.860, bottom=0.30,
                        hspace=0.30, wspace=0.18)
    head = ("Colored by dataset", "Colored by prompt framing",
            "Colored by true label")
    for row, model in enumerate(("gemma", "qwen")):
        sel = pts[pts.model == model]
        sel = sel.iloc[rng.permutation(len(sel))]   # no framing drawn on top
        ev = (var[var.model == model].set_index("pc").explained_variance)
        # social_iqa has no bare variant: its two framings are byte-identical
        # activations (centroid shift exactly 0, where every other dataset
        # moves by thousands). Drawn in the framing panel it is 70,728 points
        # plotted twice in two colours at one set of coordinates - a 50/50
        # overlay that reads as a third colour and means nothing. Its framing
        # value is undefined, not small, so that panel leaves it out. The
        # other two panels, where it is meaningful, keep it.
        no_sq = sel[sel.dataset != "social_iqa"]
        for col, (frame, colors) in enumerate((
                (sel, [F.DATASET_COLOR[d] for d in sel.dataset]),
                (no_sq, [PERT_COLOR[p] for p in no_sq.perturbation]),
                (sel, [LABEL_COLOR[y] for y in sel.label]))):
            ax = axes[row, col]
            ax.scatter(frame.pc1, frame.pc2, c=colors, **POINT_CAT)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.axhline(0, color=F.GRID, lw=0.7, zorder=0)
            ax.axvline(0, color=F.GRID, lw=0.7, zorder=0)
            ax.set_xlabel(f"PC1 ({ev.loc[1]:.1%})")
            if row == 0:
                pos = ax.get_position()
                fig.text(pos.x0 - 0.016, 0.910, f"{'abc'[col]})",
                         fontsize=F.pt("title"), fontweight="bold",
                         ha="right", va="bottom", color=F.INK)
                fig.text(pos.x0, 0.910, head[col],
                         fontsize=F.pt("title"), ha="left", va="bottom",
                         color=F.INK)
        axes[row, 0].set_ylabel(f"PC2 ({ev.loc[2]:.1%})")
        pos = axes[row, 0].get_position()
        fig.text(0.018, (pos.y0 + pos.y1) / 2, C.MODEL_LABEL[model],
                 rotation=90, ha="center", va="center",
                 fontsize=F.pt("title"), color=F.INK)

    ds_handles = F.dataset_legend_handles(set(pts.dataset), C.DATASET_LABEL,
                                          pad_blocks=True)
    leg = fig.legend(handles=ds_handles, loc="upper left",
                     bbox_to_anchor=(0.045, 0.245), ncol=5, frameon=False,
                     columnspacing=1.3, **LEGEND_STYLE)
    F.bold_domain_headers(leg)
    # Framing and label keys share one row immediately below the grouped
    # dataset legend, avoiding two extra stacked legend rows.
    secondary = [
        plt.Line2D([], [], marker="o", ls="", ms=F.pt("legend") * 0.55,
                   color=PERT_COLOR[k], label=PERT_TEXT[k])
        for k in ("bare", "true_false")
    ] + [
        plt.Line2D([], [], marker="o", ls="", ms=F.pt("legend") * 0.55,
                   color=LABEL_COLOR[y], label=LABEL_TEXT[y])
        for y in (1, 0)
    ]
    fig.legend(handles=secondary, loc="upper center",
               bbox_to_anchor=(0.5, 0.098), ncol=4, frameon=False,
               columnspacing=1.8, **LEGEND_STYLE)
    F.save(fig, "supplementary_figure1_framing", F.caption_text("supplementary_figure1_framing"))


def main() -> None:
    F.setup()
    pts = _points("figure1_activation")
    var = pd.read_csv(C.DATA / "figure1_activation_variance.csv")
    read = pd.read_csv(C.DATA / "figure1_activation_readouts.csv")
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.4),
                             sharex="row", sharey="row")
    # the legend lives in the right margin now, so the figure carries no
    # bottom legend row and is a third shorter than the old layout
    fig.subplots_adjust(left=0.118, right=0.755, top=0.825, bottom=0.075,
                        hspace=0.40, wspace=0.12)
    for col, (letter, name) in enumerate(
            (("a", "Colored by dataset (accuracy)"),
             ("b", "Colored by true label (AUC)"))):
        pos = axes[0, col].get_position()          # exact, not guessed
        fig.text(pos.x0 - 0.016, 0.895, f"{letter})",
                 fontsize=F.pt("title"), fontweight="bold",
                 ha="right", va="bottom", color=F.INK)
        fig.text(pos.x0, 0.895, name, fontsize=F.pt("title"),
                 ha="left", va="bottom", color=F.INK)
    for row, model in enumerate(("gemma", "qwen")):
        sel = pts[(pts.model == model) & (pts.perturbation == "bare")]
        sel = sel.iloc[rng.permutation(len(sel))]      # no dataset on top
        ev = (var[var.model == model].set_index("pc").explained_variance)

        axes[row, 0].scatter(sel.pc1, sel.pc2,
                             c=[F.DATASET_COLOR[d] for d in sel.dataset],
                             **POINT)
        axes[row, 1].scatter(sel.pc1, sel.pc2,
                             c=[LABEL_COLOR[y] for y in sel.label], **POINT)
        for col in range(2):
            ax = axes[row, col]
            ax.axhline(0, color=F.GRID, lw=0.7, zorder=0)
            ax.axvline(0, color=F.GRID, lw=0.7, zorder=0)
            ax.set_xlabel(f"PC1 ({ev.loc[1]:.1%})")
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
            ax.set_title(_readout_text(read, model,
                                       ("dataset identity", "label")[col]),
                         loc="left", fontsize=F.pt("tick"), color=F.INK,
                         linespacing=1.4, pad=6)
        axes[row, 0].set_ylabel(f"PC2 ({ev.loc[2]:.1%})")
        pos = axes[row, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, C.MODEL_LABEL[model],
                 rotation=90, ha="center", va="center",
                 fontsize=F.pt("title"), color=F.INK)

    n = int((pts.perturbation == "bare").sum() / 2)
    # one single-column legend on the right, at label size rather than
    # legend size: datasets grouped by domain, then the two label colours
    handles = F.dataset_legend_handles(set(pts.dataset), C.DATASET_LABEL)
    handles += [plt.Line2D([], [], ls="", marker="", label=" "),
                plt.Line2D([], [], ls="", marker="", label="Label")]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=11,
                           color=LABEL_COLOR[y], label="  " + LABEL_TEXT[y])
                for y in (1, 0)]
    leg = fig.legend(handles=handles, loc="center left",
                     bbox_to_anchor=(0.762, 0.46), frameon=False,
                     **LEGEND_STYLE)
    F.bold_domain_headers(leg)
    for txt in leg.get_texts():
        if txt.get_text() == "Label":
            txt.set_fontweight("bold")

    F.save(fig, "figure1_activation_geometry", F.caption_text("figure1_activation_geometry"))

    _supplement()


if __name__ == "__main__":
    main()
