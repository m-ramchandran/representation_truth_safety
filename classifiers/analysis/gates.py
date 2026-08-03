"""What did the moe gate learn?

Two analyses of the merged whole-corpus mixture-of-experts fit (the
canonical pooled model). Its gate and experts are linear on the
standardized features and fully saved (full_gate_w / full_gate_b /
full_expert_w / full_expert_b + the scaler and featurization), so
per-sample routing is recomputed here from the activations — no runner
change needed. Works for any tag: the activations are pushed through
the fit's own saved featurization first. Only the datasets in the fit's
training pool (the npz's cv_dataset) enter the analysis — eval-only or
excluded units the merged run never pooled are dropped with a note.

1. Structure (in the fit's standardized feature space): principal
   angles between span(gate rows) and the between-class-mean subspace
   of the pooled features under four labelings — dataset, (dataset,
   label), task, label — telling which structure the gate's input
   directions align with; plus routing — each sample's argmax expert,
   cross-tabulated against dataset, task, and label.
2. Gate vs the per-dataset l1reg directions (both mapped to cache
   units): principal angles between span(gate rows) and span(l1
   directions), the expert x dataset cosine matrices for gate rows and
   expert vectors, and routing consistency — is each dataset's majority
   expert the one whose expert vector best aligns with that dataset's
   own l1 direction? This half needs a common cache space, so it skips
   the binned_activations and pca_dict tags with a note.

    python -m classifiers.analysis.gates --tag raw_last \
        [--layers 17 22 29 33] [--root ...]

Csvs per layer (gates_angles / gates_routing / gates_experts /
gates_consistency, suffixed _<tag>_layer<L>); one routing figure
holds all requested layers as panels (gates_routing_<tag>.png).
Outputs join merged/tables and merged/figures.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import subspace_angles

from ..data import COLLECTIONS, task_of
from . import directions
from .directions import (ROLES, apply_featurization, load_direction,
                         stack_directions, to_cache_units, unit_rows)
from .load_results import (add_data_args, analysis_dirs,
                           load_prediction, npz_path,
                           resolve_root)
from .plots import layer_panels, save


def angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between the row spans."""
    return np.degrees(subspace_angles(np.asarray(A, float).T,
                                      np.asarray(B, float).T))[::-1]


def class_means(Xs, labels: np.ndarray) -> np.ndarray:
    """Between-class subspace: one row per class, class mean minus the
    overall mean (sparse input comes out dense)."""
    mean = lambda M: np.asarray(M.mean(axis=0)).ravel()
    overall = mean(Xs)
    return np.stack([mean(Xs[labels == c]) - overall
                     for c in np.unique(labels)])


def crosstab(assign: np.ndarray, groups: np.ndarray,
             K: int) -> pd.DataFrame:
    """group x expert fraction table (rows sum to 1)."""
    out = pd.DataFrame(
        {f"expert{k}": [np.mean(assign[groups == g] == k)
                        for g in np.unique(groups)]
         for k in range(K)}, index=np.unique(groups))
    out.index.name = "group"
    return out


def standardized_features(d: dict, tag: str, units: dict,
                          names: list[str]):
    """The pooled corpus exactly as the merged fit saw it: each unit's
    blocks through the saved featurization, stacked, then the saved
    scaler (mean absent = sparse variance-only scaling)."""
    feats = [apply_featurization(d, tag, units[n]["blocks"], "full_")
             for n in names]
    X = sp.vstack(feats, format="csr") if sp.issparse(feats[0]) \
        else np.concatenate([np.asarray(f) for f in feats])
    scale = np.asarray(d["full_standardize_scale"], float)
    if sp.issparse(X):
        return X.multiply(1.0 / scale).tocsr()
    mean = np.asarray(d.get("full_standardize_mean",
                            np.zeros_like(scale)), float)
    return (X - mean) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="default: the collection's layers")
    add_data_args(ap)
    args = ap.parse_args()
    args.root = resolve_root(args)
    layers = args.layers or list(COLLECTIONS[args.data].layers)

    routing_by_layer = {}
    for layer in layers:
        try:
            run_layer(args, layer, routing_by_layer)
        except FileNotFoundError as e:
            print(f"  layer {layer}: skipped ({e})")

    # one routing figure across the layers, a panel each: fewer files,
    # and panel counts that stay legible on a slide
    if routing_by_layer:
        _, figures = analysis_dirs(args.root, "merged")
        save(layer_panels(routing_by_layer,
                          "Share of each dataset routed to each expert",
                          args.tag, vmin=0, vmax=1, cmap="magma"),
             figures / f"gates_routing_{args.tag}.png")


def run_layer(args, layer: int, routing_by_layer: dict) -> None:
    d = load_prediction(npz_path("merged", args.tag, layer, "moe",
                                 root=args.root))
    gate_w = np.asarray(d["full_gate_w"], float)
    gate_b = np.asarray(d["full_gate_b"], float)
    expert_w = np.asarray(d["full_expert_w"], float)
    K = len(gate_w)

    features, raw = args.tag.rsplit("_", 1)
    units = directions.load_units(layer, raw, ROLES[features],
                                  args.data)
    # only the datasets the moe was actually fit on (the npz's training
    # pool) — the collection may hold eval-only or excluded units the
    # merged run never pooled, and mixing those into the routing /
    # subspace analysis would misread the gate
    pool = set(np.unique(np.asarray(d["cv_dataset"]).astype(str)))
    if dropped := sorted(set(units) - pool):
        print(f"  not in the moe fit's pool, dropped: {dropped}")
    names = sorted(pool & set(units))
    if missing := sorted(pool - set(units)):
        raise FileNotFoundError(f"pool datasets missing from the "
                                f"activation caches: {missing}")
    y = np.concatenate([units[n]["y"] for n in names])
    dataset = np.concatenate([np.full(len(units[n]["y"]), n)
                              for n in names])
    task = np.array([task_of(n, args.data) for n in dataset])
    joint = np.char.add(np.char.add(dataset, ":"), y.astype(str))
    Xs = standardized_features(d, args.tag, units, names)

    tables, figures = analysis_dirs(args.root, "merged")
    stem = f"{args.tag}_layer{layer}"

    # 1. subspace structure + routing (standardized feature space)
    ang_rows = [{"compared_to": name, "angle": i + 1,
                 "degrees": round(a, 2)}
                for name, labels in (("dataset", dataset),
                                     ("dataset_label", joint),
                                     ("task", task),
                                     ("label", y.astype(str)))
                if len(np.unique(labels)) >= 2
                for i, a in enumerate(angles_deg(
                    gate_w, class_means(Xs, labels)))]

    assign = np.argmax(np.asarray(Xs @ gate_w.T) + gate_b, axis=1)
    tabs = {name: crosstab(assign, groups, K)
            for name, groups in (("dataset", dataset), ("task", task),
                                 ("label", y.astype(str)))}
    pd.concat([t.assign(grouping=name) for name, t in tabs.items()]) \
        .round(3).to_csv(tables / f"gates_routing_{stem}.csv")
    routing_by_layer[layer] = tabs["dataset"]

    # 2. gate vs per-dataset l1 directions (cache units)
    cons = None
    try:
        G, gidx = to_cache_units(gate_w, d, "full_")
        E, _ = to_cache_units(expert_w, d, "full_")
        l1_dirs, l1_names = [], []
        for ds in names:
            try:
                l1_dirs.append(load_direction("per_dataset", args.tag,
                                              layer, "l1reg", ds,
                                              root=args.root))
                l1_names.append(ds)
            except (FileNotFoundError, KeyError) as e:
                print(f"  l1reg {ds}: skipped ({e})")
        if len(l1_names) < 2:
            raise ValueError("fewer than 2 l1reg fits with coefficients")
    except ValueError as e:
        print(f"  gate-vs-l1 half skipped ({e})")
    else:
        # one aligned space for gate rows, expert vectors, l1 directions
        A, _ = stack_directions([(g, gidx) for g in G]
                                + [(e, gidx) for e in E] + l1_dirs)
        G_al, E_al, L_al = (unit_rows(A[:K]), unit_rows(A[K:2 * K]),
                            unit_rows(A[2 * K:]))
        ang_rows += [{"compared_to": "l1_directions", "angle": i + 1,
                      "degrees": round(a, 2)}
                     for i, a in enumerate(angles_deg(G_al, L_al))]

        cos_g, cos_e = G_al @ L_al.T, E_al @ L_al.T   # K x datasets
        pd.concat([pd.DataFrame(c, columns=l1_names)
                     .assign(kind=kind, expert=range(K))
                   for kind, c in (("gate", cos_g), ("expert", cos_e))]) \
            .round(3).to_csv(tables / f"gates_experts_{stem}.csv",
                             index=False)

        counts = {ds: np.bincount(assign[dataset == ds], minlength=K)
                  for ds in l1_names}
        cons = pd.DataFrame({
            "majority_expert": [counts[ds].argmax() for ds in l1_names],
            "majority_share": [counts[ds].max() / counts[ds].sum()
                               for ds in l1_names],
            "best_aligned_expert": cos_e.argmax(axis=0),
            "best_cos": cos_e.max(axis=0).round(3),
        }, index=pd.Index(l1_names, name="dataset"))
        cons["match"] = (cons["majority_expert"]
                         == cons["best_aligned_expert"])
        cons.round(3).to_csv(tables / f"gates_consistency_{stem}.csv")

    pd.DataFrame(ang_rows).to_csv(tables / f"gates_angles_{stem}.csv",
                                  index=False)
    print(f"gates layer {layer}: wrote angles, routing"
          f"{', experts, consistency' if cons is not None else ''} "
          f"under {tables.parent}")


if __name__ == "__main__":
    main()
