"""Shared figure style for the deep dive: big text, short titles,
side legends, and print-stable colors shared by the final figures."""
from __future__ import annotations

import colorsys

import matplotlib as mpl
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from pathlib import Path

FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS_PDF = Path(__file__).resolve().parents[1] / "figures_pdf"

INK = "#0A0A0A"
MUTED = "#667085"
GRID = "#D8DEE9"
BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
RED = "#C81E1E"
GRAY = "#8A94A6"
GREEN = "#109618"
FAMILY_COLOR = {"linear": BLUE, "nonlinear": ORANGE, "pooled_extra": TEAL}
DOMAIN_COLOR = {"true_false": "#3366CC", "medical": "#109618",
                "jailbreak": "#DC3912", "social": "#990099",
                "statistics": "#0099C6"}
COLLECTION_MARKER = {("gemma", "bare"): "o", ("gemma", "true_false"): "s",
                     ("qwen", "bare"): "^", ("qwen", "true_false"): "D"}
COLLECTION_LABEL = {("gemma", "bare"): "Gemma / Original",
                    ("gemma", "true_false"): "Gemma / T-F framing",
                    ("qwen", "bare"): "Qwen / Original",
                    ("qwen", "true_false"): "Qwen / T-F framing"}
MODEL_COLOR = {"gemma": "#2563EB", "qwen": "#EA580C"}

# Per-dataset colors: one hue per domain, lightness varied within it, so two
# datasets from the same domain read as closer than two from different ones.
DOMAIN_MEMBERS = {
    "true_false": ("got", "common_claim", "counterfact"),
    "medical": ("pubmedqa_artificial", "pubmedqa_labeled", "claude_anatomy",
                "claude_blood_test_ranges"),
    "jailbreak": ("harmful_harmless", "wildjailbreak_vanilla",
                  "wildjailbreak_adversarial", "xstest"),
    "social": ("social_iqa",),
    "statistics": ("claude_vds",),
}


def _shades(base: str, n: int) -> list[str]:
    h, l, s = colorsys.rgb_to_hls(*mpl.colors.to_rgb(base))
    if n == 1:
        return [mpl.colors.to_hex(colorsys.hls_to_rgb(h, l, s))]
    out = []
    for i in range(n):
        t = i / (n - 1)
        li = min(0.74, max(0.24, l * (0.65 + 0.85 * t)))
        # ease off saturation as it lightens, or the pale end goes neon
        out.append(mpl.colors.to_hex(
            colorsys.hls_to_rgb(h, li, s * (1.0 - 0.28 * t))))
    return out


DATASET_COLOR = {d: c for dom, members in DOMAIN_MEMBERS.items()
                 for d, c in zip(members, _shades(DOMAIN_COLOR[dom],
                                                  len(members)))}
# domain order, then within-domain order — for legends that group by domain
DATASET_ORDER = [d for members in DOMAIN_MEMBERS.values() for d in members]
DOMAIN_LABEL = {"true_false": "True/False", "medical": "Medical",
                "jailbreak": "Jailbreak", "social": "Social",
                "statistics": "Statistics"}


def dataset_legend_handles(present, labels, pad_blocks=False) -> list:
    """Domain-grouped dataset legend: domain header rows with members
    indented below. Default: one column, blank spacer rows between
    blocks. With pad_blocks=True, blocks are instead padded to equal
    length (no separators), so a legend with ncol = number of domains
    puts one domain per column. Pass the result through
    bold_domain_headers() after creating the legend."""
    blank = lambda: plt.Line2D([], [], ls="", marker="", label=" ")
    blocks = []
    for dom, members in DOMAIN_MEMBERS.items():
        members = [d for d in members if d in present]
        if not members:
            continue
        block = [plt.Line2D([], [], ls="", marker="",
                            label=DOMAIN_LABEL[dom])]
        block += [plt.Line2D([], [], marker="o", ls="", ms=9,
                             color=DATASET_COLOR[d], label="  " + labels[d])
                  for d in members]
        blocks.append(block)
    if pad_blocks:
        n = max(len(b) for b in blocks)
        return [h for b in blocks for h in b + [blank()] * (n - len(b))]
    handles = []
    for b in blocks:
        if handles:
            handles.append(blank())
        handles += b
    return handles


def bold_domain_headers(legend) -> None:
    for txt in legend.get_texts():
        if txt.get_text() in DOMAIN_LABEL.values():
            txt.set_fontweight("bold")


# --------------------------------------------------------------- sizing
# The deep dive's figures are composed on a wide canvas -- CANVAS_W inches
# -- with text at the point sizes in PRINT_PT. That ratio is what makes
# them readable when rendered at a smaller display size; it is the ratio,
# not the canvas, that carries over.
CANVAS_W = 16.0
PRINT_PT = {"tick": 15.0, "label": 17.0, "title": 19.0, "suptitle": 21.0,
            "legend": 15.0, "body": 17.0, "caption": 11.0}


def pt(role: str) -> float:
    """Point size for text in `role`."""
    return PRINT_PT[role]


def canvas(aspect: float, width_frac: float = 1.0) -> tuple[float, float]:
    """figsize for a figure `aspect` times as tall as it is wide, occupying
    `width_frac` of the standard canvas width."""
    w = CANVAS_W * width_frac
    return (w, w * aspect)


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": pt("body"), "axes.titlesize": pt("title"),
        "axes.labelsize": pt("label"),
        "xtick.labelsize": pt("tick"), "ytick.labelsize": pt("tick"),
        "legend.fontsize": pt("legend"), "figure.titlesize": pt("suptitle"),
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
    })


TEXT_STROKE_PT = 0.0


def _embolden(fig) -> None:
    """Stroke every glyph in its own color, a synthetic weight between
    regular and bold: at column width the plain regular face washes out.
    Stroked text is written to the PDF as outlines rather than as text,
    so at 0 the stroke is skipped entirely and the glyphs stay text."""
    if TEXT_STROKE_PT <= 0:
        return
    for t in fig.findobj(mtext.Text):
        if t.get_text().strip() and not t.get_path_effects():
            t.set_path_effects([mpe.withStroke(linewidth=TEXT_STROKE_PT,
                                               foreground=t.get_color())])


def save(fig, name: str, caption: str | None = None) -> None:
    """Write the caption beside the figure rather than drawing it on-canvas."""
    _embolden(fig)
    FIGS.mkdir(parents=True, exist_ok=True)
    FIGS_PDF.mkdir(parents=True, exist_ok=True)
    if caption:
        (FIGS / f"{name}.caption.txt").write_text(caption + "\n")
    fig.savefig(FIGS_PDF / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png + figures_pdf/{name}.pdf"
          + (" + .caption.txt" if caption else ""))


def caption_text(name: str) -> str:
    """Read the hand-edited caption stored beside a figure."""
    return (FIGS / f"{name}.caption.txt").read_text().rstrip()
