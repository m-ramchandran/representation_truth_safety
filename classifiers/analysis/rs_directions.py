"""Random-slopes deviations vs the single-dataset directions.

A random-slopes fit decomposes each group's rule into shared weights
plus a penalized per-group deviation: the effective direction for group
g is shared + dev_w[g]. The per_dataset experiment learns each dataset's
direction independently. This module asks: does a group's deviation
actually move the pooled shared direction toward the datasets' own
single-fit directions — and how far does the deviation alone point that
way?

Per (random-slopes variant, single-fit method m, dataset D), three plain
cosines against D's own per_dataset direction (g = D itself for
random_slopes, task_of(D) for random_slopes_task):

  shared           cos(shared, single_m(D)) — the no-deviation baseline
  shared_plus_dev  cos(shared + dev_g, single_m(D)) — the model's
                   effective direction for D's group
  dev              cos(dev_g, single_m(D)) — the deviation alone

The random-slopes fits come from the MERGED run (its full_ arrays) —
the one experiment where every dataset / task has a deviation. All
vectors are mapped to cache units first (each fit's own scaler / PCA
basis, via directions.to_cache_units), so the comparison is scaler-free;
binned_activations tags are refused there (each fit's bins are different
features), and sparse tags align on the union of kept features.

    python -m classifiers.analysis.rs_directions --tag raw_last \
        --layer 9 [--methods l2reg diff ...] [--data ...] [--root ...]

Outputs under <root>/coefficients/: rs_vs_single_<tag>_layer<L>.csv
(long form: variant, method, dataset, group, comparison, value) and one
figure per variant present (rs_vs_single_<tag>_layer<L>_<variant>.png):
a panel per single-fit method, rows = the three comparisons, columns =
datasets. Variants without a merged fit yet are skipped with a note.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import task_of
from ..fig_style import heatmap_size
from .directions import (DIRECTION_KEYS, LINEAR_METHODS, load_direction,
                         pair_cos, raw_direction, to_cache_units)
from .load_results import (add_data_args, analysis_dirs, load_metrics,
                           load_prediction, npz_path, resolve_root)
from .plots import heatmap, panel_grid, save, subtitle_corner

RS_VARIANTS = ("random_slopes", "random_slopes_task")
COMPARISONS = ("shared", "shared_plus_dev", "dev")
RS_KEYS = DIRECTION_KEYS | {"full_dev_w", "full_group_labels"}


def load_rs(tag: str, layer: int, variant: str, root):
    """The merged run's random-slopes directions in cache units:
    (shared (w, idx), {group -> deviation (w, idx)}). The shared weights
    and every deviation live in the same fit's standardized space, so
    one mapping serves all of them (the map is linear — mapping then
    summing equals summing then mapping)."""
    d = load_prediction(npz_path("merged", tag, layer, variant, root=root),
                        keys=RS_KEYS)
    shared = raw_direction(d, variant, prefix="full_")
    dev_w = np.atleast_2d(np.asarray(d["full_dev_w"], dtype=float))
    W, idx = to_cache_units(dev_w, d, prefix="full_")
    devs = {str(g): (W[i], idx)
            for i, g in enumerate(d["full_group_labels"])}
    return shared, devs


def rs_figure(df: pd.DataFrame, stem: str, variant: str):
    """One variant's figure from the long csv: a panel per single-fit
    method, rows = the three comparisons, columns = datasets. Also
    called by redraw.py on the saved csv."""
    df = df[df["variant"] == variant]
    methods = sorted(df["method"].unique())
    mats = {m: df[df["method"] == m]
            .pivot(index="comparison", columns="dataset", values="value")
            .reindex(list(COMPARISONS)) for m in methods}
    n_ds = max(len(m.columns) for m in mats.values())
    fig, axes = panel_grid(len(methods),
                           *heatmap_size(len(COMPARISONS), n_ds))
    for ax, m in zip(axes, methods):
        heatmap(mats[m], vmin=-1, vmax=1, cmap="coolwarm", title=m, ax=ax)
    fig.suptitle(f"{variant} directions vs each dataset's own single fit")
    subtitle_corner(fig, f"{stem} — rows: shared weights / shared + the "
                         "dataset's group deviation / the deviation alone")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=list(LINEAR_METHODS),
                    choices=LINEAR_METHODS,
                    help="the single-fit methods to compare against")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    stem = f"{args.tag}_layer{args.layer}"
    tables, figures = analysis_dirs(args.root)

    metrics = load_metrics("per_dataset", args.root)
    metrics = metrics[(metrics["features"] == args.tag)
                      & (metrics["layer"] == args.layer)]

    # the single-fit directions, loaded once and shared by both variants
    singles = {}
    for method in args.methods:
        for ds in sorted(metrics[metrics["method"] == method]
                         ["train_dataset"].unique()):
            try:
                singles[(method, ds)] = load_direction(
                    "per_dataset", args.tag, args.layer, method, ds,
                    root=args.root)
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  {method} {ds}: skipped ({e})")
    if not singles:
        raise FileNotFoundError(f"no per_dataset fits with directions for "
                                f"tag={args.tag!r} layer={args.layer}")

    rows = []
    for variant in RS_VARIANTS:
        try:
            shared, devs = load_rs(args.tag, args.layer, variant, args.root)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  {variant}: skipped ({e})")
            continue
        for (method, ds), single in singles.items():
            group = ds if variant == "random_slopes" \
                else task_of(ds, args.data)
            if group not in devs:
                print(f"  {variant} {ds}: group {group!r} not in the "
                      "merged fit, skipped")
                continue
            dev = devs[group]
            effective = (shared[0] + dev[0], dev[1])
            for comparison, direction in (("shared", shared),
                                          ("shared_plus_dev", effective),
                                          ("dev", dev)):
                rows.append({"variant": variant, "method": method,
                             "dataset": ds, "group": group,
                             "comparison": comparison,
                             "value": round(pair_cos(direction, single), 4)})
    if not rows:
        raise FileNotFoundError("no merged random-slopes fits for "
                                f"tag={args.tag!r} layer={args.layer}")

    df = pd.DataFrame(rows)
    df.to_csv(tables / f"rs_vs_single_{stem}.csv", index=False)
    for variant in sorted(df["variant"].unique()):
        save(rs_figure(df, stem, variant),
             figures / f"rs_vs_single_{stem}_{variant}.png")
    print(f"rs_directions: {df['variant'].nunique()} variant(s), "
          f"{df['method'].nunique()} single methods -> {tables.parent}")


if __name__ == "__main__":
    main()
