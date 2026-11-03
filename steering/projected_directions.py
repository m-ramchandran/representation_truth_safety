"""Build the direction file for the projected-coefficient steering run.

The question this run asks: a fitted classifier's coefficient vector does
not move the model's answer when it is added to the activations, although
it predicts the label well. Its class-separating content sits in a few
high-variance directions of its training data, while most of its length
sits in low-variance directions that the covariance-inverse in the
regression inflated. So: keep only the part of the coefficient vector that
lies in the top k principal components of the rows it was trained on, and
steer with that. If the projected vector moves the answer like the
class-mean difference does, the classifier knew the direction all along
and the steering failure was a units problem in the tail.

For each of the six target datasets, five saved coefficient vectors, all
in raw activation units (coefficient divided by the fit's standardization
scale, which is the direction the classifier reads along):

  own_l2      logistic regression fitted on the target's own rows
  pooled_l2   logistic regression on the combined pool (the expanded pool
              for the two datasets the standard pool never contained)
  pooled_rs   the hierarchical fit on that pool: shared vector plus the
              target's own adjustment
  famout_l2   logistic regression on the pool with the target's whole
              family removed (for claude_vds, which has no family, the
              standard pool, which never contained it)
  famout_rs   the hierarchical fit on that same reduced pool, shared
              vector only

Each is projected onto the top 2, 10 and 50 principal components of ITS
OWN TRAINING ROWS: the target's rows for own_l2, the pool's rows for the
pooled fits, the reduced pool's rows for the family-removed fits. The
components are those of the covariance of all training rows together,
centered on their overall mean (both classes together, the same matrix as
every earlier principal-component table in this folder). The projected
vector is rescaled to length 1 and, like every real vector in the earlier
sweeps, flipped to point toward True on the target if needed, with the
flip recorded. Nothing is refit; every input is a saved file.

References in the same file, so the whole figure comes from one sweep:
the target's own class-mean difference and the family-removed difference
(dataset-removed for claude_vds), exactly the vectors of the earlier runs.

Matched comparisons: eight random vectors projected onto the same top k
components of each of the three training sets (own rows, pool, reduced
pool), so a projected coefficient vector is judged against "any vector in
that same subspace"; and eight shuffled-label differences, the comparison
for the difference lines (same seeds as directions.py, so the same
vectors).

Outputs under steering/out/ (the sweep reads the first two by tag):
  directions_{tag}_layer{L}.npz     names, W (unit rows)
  manifest_{tag}_layer{L}.csv       one row per vector (manifest_schema)
  projected_geometry_{tag}_layer{L}.csv
                                    every vector, plus the unprojected
                                    coefficient vectors for reference:
                                    kept share, cosines with the two
                                    differences, AUC on the target rows
                                    and on the training rows
  directions_meta_{tag}_layer{L}.json

    python steering/projected_directions.py --collection gemma_tf_a --layer 29 --tag gemmaproj
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from classifiers.data import _load_units  # noqa: E402
import steering.manifest_schema as MS  # noqa: E402
from steering.directions import (POOL_EXCLUDED, SHORT, TARGETS, TASK,  # noqa: E402
                                 rs_parts)
from steering.pc_alignment import cell_stats, combine  # noqa: E402
from steering.pc_capture import read_fit  # noqa: E402

RESULTS = REPO / "classifiers" / "results"
OUT = REPO / "steering" / "out"
KS = (2, 10, 50)
N_RAND = 8
N_SHUF = 8
SOURCES = ("own_l2", "pooled_l2", "pooled_rs", "famout_l2", "famout_rs")
SOURCE_PROSE = {
    "own_l2": "logistic-regression coefficients fitted on the target's own rows",
    "pooled_l2": "logistic-regression coefficients fitted on the combined pool",
    "pooled_rs": "the hierarchical fit on the combined pool, shared vector "
                 "plus the target's own adjustment",
    "famout_l2": "logistic-regression coefficients fitted on the pool with "
                 "the target's whole family removed",
    "famout_rs": "the shared vector of the hierarchical fit on the pool with "
                 "the target's whole family removed",
}


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0:
        raise ValueError("zero or non-finite vector")
    return v / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", default="gemma_tf_a")
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--tag", default="gemmaproj")
    args = ap.parse_args()
    coll, L, tag = args.collection, args.layer, args.tag
    P = RESULTS / coll
    PE = RESULTS / f"{coll}_expanded_pool"

    def merged(root: Path, method: str) -> Path:
        return root / "merged" / "predictions" / f"raw_last_layer{L}_{method}.npz"

    def single(method: str, t: str) -> Path:
        return P / "per_dataset" / "predictions" / f"raw_last_layer{L}_{method}_{t}.npz"

    def leave_task(method: str, fam: str) -> Path:
        return P / "leave_out" / "predictions" / f"raw_last_layer{L}_{method}_task_{fam}.npz"

    # ---- the saved vectors and the rows each was trained on ----------------
    # training sets are keyed by name so a pool shared by several targets is
    # decomposed once
    sets: dict[str, list[str]] = {}
    per_target: dict[str, dict] = {}
    for t in TARGETS:
        fam = TASK[t]
        excluded = t in POOL_EXCLUDED
        pool_root = PE if excluded else P
        pool_name = "pool:expanded" if excluded else "pool:standard"
        own = read_fit(single("l2reg", t), "")
        pl = read_fit(merged(pool_root, "l2reg"), "full_")
        rs_shared, rs_dev = rs_parts(merged(pool_root, "random_slopes"), "full_")
        if t not in rs_dev:
            raise SystemExit(f"{t}: the {pool_name} hierarchical fit has no "
                             f"{t} adjustment")
        if t == "claude_vds":
            # no family in the pool: the standard pool, which never held
            # it, is the cleanest held-out set it has
            fo = read_fit(merged(P, "l2reg"), "full_")
            fo_rs_shared, fo_rs_dev = rs_parts(merged(P, "random_slopes"), "full_")
            fo_name, fo_kind_diff = "pool:standard", "datasetout_diff"
            fo_diff = read_fit(merged(P, "diff"), "full_")["coef_raw"]
            fo_prose = ("The training set is the standard pool, which never "
                        "contained claude_vds; it belongs to no family there.")
        else:
            fo = read_fit(leave_task("l2reg", fam), "")
            fo_rs_shared, fo_rs_dev = rs_parts(leave_task("random_slopes", fam), "")
            fo_name, fo_kind_diff = f"famout:{fam}", "taskout_diff"
            fo_diff = read_fit(leave_task("diff", fam), "")["coef_raw"]
            fo_prose = (f"The training set had every {fam} dataset removed "
                        "before fitting.")
        if t in fo_rs_dev:
            raise SystemExit(f"{t}: the held-out hierarchical fit still carries "
                             f"a {t} adjustment; it is not held out")
        sets.setdefault(f"own:{t}", [t])
        sets.setdefault(pool_name, sorted(pl["roster"]))
        sets.setdefault(fo_name, sorted(fo["roster"]))
        if sorted(pl["roster"]) != sets[pool_name] or sorted(fo["roster"]) != sets[fo_name]:
            raise SystemExit(f"{t}: two fits claim the same training set with "
                             "different rosters")
        if t in fo["roster"]:
            raise SystemExit(f"{t}: the held-out fit's roster contains {t}")
        per_target[t] = {
            "vectors": {
                "own_l2": (own["coef_raw"], f"own:{t}", "yes"),
                "pooled_l2": (pl["coef_raw"], pool_name, "yes"),
                "pooled_rs": (rs_shared + rs_dev[t], pool_name, "yes"),
                "famout_l2": (fo["coef_raw"], fo_name, "no"),
                "famout_rs": (fo_rs_shared, fo_name, "no"),
            },
            "own_diff": read_fit(single("diff", t), "")["coef_raw"],
            "fo_diff": fo_diff, "fo_kind_diff": fo_kind_diff,
            "fo_name": fo_name, "pool_name": pool_name, "fo_prose": fo_prose,
            "rosters": {pool_name: pl["roster"], fo_name: fo["roster"]},
        }
        print(f"  {t}: fits read; pool={pool_name}, held-out={fo_name}", flush=True)

    # ---- rows: every dataset any training set names -------------------------
    needed = sorted({m for members in sets.values() for m in members})
    units = _load_units("last", L, coll, groups=False)
    missing = [m for m in needed if m not in units]
    if missing:
        raise SystemExit(f"activation cache has no rows for {missing}")
    rows = {m: (np.asarray(units[m]["X"], dtype=np.float32),
                np.asarray(units[m]["y"], dtype=int)) for m in needed}
    del units
    for t in TARGETS:
        for sname, roster in per_target[t]["rosters"].items():
            bad = {m: (n, len(rows[m][1])) for m, n in roster.items()
                   if n != len(rows[m][1])}
            if bad:
                raise SystemExit(f"{sname}: the fit's stored row counts differ "
                                 f"from the cache: {bad}. The components would "
                                 "not be those of the training rows.")
    d = next(iter(rows.values()))[0].shape[1]
    print(f"rows for {len(needed)} datasets, d={d}", flush=True)

    # ---- top-50 components of every training set ----------------------------
    stats = {m: cell_stats(X, y) for m, (X, y) in rows.items()}
    comps: dict[str, np.ndarray] = {}
    set_meta: dict[str, dict] = {}
    kmax = max(KS)
    for sname, members in sets.items():
        N, mu, S = combine([stats[m][c] for m in members for c in (0, 1)])
        w, V = eigh(S, subset_by_index=[d - kmax, d - 1])
        order = np.argsort(w)[::-1]
        comps[sname] = np.ascontiguousarray(V[:, order])      # PC1 first
        trace = float(np.trace(S))
        share = np.cumsum(w[order]) / trace
        set_meta[sname] = {"members": members, "n_rows": int(N),
                           "variance_share_top": {str(k): round(float(share[k - 1]), 4)
                                                  for k in KS}}
        print(f"  {sname}: {N} rows, {len(members)} datasets, variance share "
              f"top 2/10/50 = {share[1]:.3f}/{share[9]:.3f}/{share[49]:.3f}",
              flush=True)

    def project(w: np.ndarray, sname: str, k: int) -> tuple[np.ndarray, float]:
        Vk = comps[sname][:, :k]
        head = Vk @ (Vk.T @ w)
        return head, float(head @ head) / float(w @ w)

    def train_auc(u: np.ndarray, sname: str) -> float:
        s, y = [], []
        for m in sets[sname]:
            X, ym = rows[m]
            s.append(X @ u.astype(np.float32)); y.append(ym)
        return float(roc_auc_score(np.concatenate(y), np.concatenate(s)))

    # ---- assemble --------------------------------------------------------------
    vecs: dict[str, np.ndarray] = {}
    man: list[dict] = []
    geom: list[dict] = []

    for ti, t in enumerate(TARGETS):
        s = SHORT[t]
        info = per_target[t]
        X, y = rows[t]
        Xf = X.astype(np.float64)
        u_own = unit(info["own_diff"])
        u_fo = unit(info["fo_diff"])

        def gap_of(u):
            p = Xf @ u
            return float(p[y == 1].mean() - p[y == 0].mean())

        def add(name, v, *, kind, uses, source, description, orient,
                training_set="", k=0, kept=np.nan, w_raw=None):
            if name in vecs:
                raise ValueError(f"duplicate {name}")
            u = unit(np.asarray(v, dtype=np.float64))
            gap, flipped = gap_of(u), False
            if orient and gap < 0:
                u, gap, flipped = -u, -gap, True
            vecs[name] = u
            man.append({"name": name, "target": t, "kind": kind,
                        "uses_target_data": uses, "description": description,
                        "source": source, "covariance": "",
                        "gap": round(gap, 3), "flipped": flipped,
                        "training_set": training_set, "k": k,
                        "kept_share": round(kept, 6) if np.isfinite(kept) else np.nan})
            geom.append(describe(name, kind, uses, u, training_set, k, kept,
                                 flipped, in_sweep=True))

        def describe(name, kind, uses, u, training_set, k, kept, flipped, in_sweep):
            a = float(roc_auc_score(y, Xf @ u))
            return {"target": t, "name": name, "kind": kind,
                    "uses_target_data": uses, "in_sweep": in_sweep,
                    "training_set": training_set, "k": k,
                    "kept_share": kept, "flipped": flipped,
                    "gap_target": round(gap_of(u), 3),
                    "auc_target": round(a, 4),
                    "auc_train": (round(train_auc(u, training_set), 4)
                                  if training_set and uses != "comparison" else np.nan),
                    "cos_own_diff": round(float(u @ u_own), 4),
                    "cos_famout_diff": round(float(u @ u_fo), 4)}

        # references
        add(f"{s}_own_diff", info["own_diff"], kind="own_diff", uses="yes",
            source="per_dataset/diff", orient=True, training_set=f"own:{t}",
            description=f"The True-minus-False average difference over {t}'s "
                        "own rows.")
        add(f"{s}_{info['fo_kind_diff']}", info["fo_diff"], kind=info["fo_kind_diff"],
            uses="no", source=f"{info['fo_name']}/diff", orient=True,
            training_set=info["fo_name"],
            description="The True-minus-False average difference over the "
                        "held-out training set. " + info["fo_prose"])

        # the projected coefficient vectors, and the unprojected ones for the
        # geometry table only
        for src in SOURCES:
            w, sname, uses = info["vectors"][src]
            u_raw = unit(w)
            u_raw = -u_raw if gap_of(u_raw) < 0 else u_raw
            geom.append(describe(f"{s}_{src}_raw", f"{src}_raw", uses, u_raw,
                                 sname, 0, 1.0, False, in_sweep=False))
            for k in KS:
                head, kept = project(w, sname, k)
                add(f"{s}_{src}_pc{k}", head, kind=f"{src}_pc{k}", uses=uses,
                    source=f"{sname}/{src}", orient=True, training_set=sname,
                    k=k, kept=kept,
                    description=f"{SOURCE_PROSE[src]}, keeping only its part "
                                f"inside the top {k} principal components of "
                                f"the rows it was trained on ({sname}), "
                                "rescaled to length 1."
                                + (" " + info["fo_prose"] if src.startswith("famout") else ""))

        # matched comparisons: random vectors through the same projection
        set_kinds = {"own": f"own:{t}", "pool": info["pool_name"],
                     "famout": info["fo_name"]}
        draws = [np.random.default_rng([4000 + ti, j]).standard_normal(d)
                 for j in range(N_RAND)]
        for skind, sname in set_kinds.items():
            for k in KS:
                for j, g in enumerate(draws):
                    head, kept = project(g, sname, k)
                    add(f"{s}_rand_{skind}_pc{k}_{j}", head,
                        kind=f"rand_{skind}_pc{k}", uses="comparison",
                        source=f"random draw {j} projected on {sname} top {k}",
                        orient=False, training_set=sname, k=k, kept=kept,
                        description="A vector of random numbers projected onto "
                                    f"the top {k} principal components of "
                                    f"{sname} (draw {j} of {N_RAND}). What a "
                                    "projected coefficient vector from that "
                                    "training set has to beat.")
        for j in range(N_SHUF):
            yp = np.random.default_rng([3000 + ti, j]).permutation(y)
            dv = Xf[yp == 1].mean(0) - Xf[yp == 0].mean(0)
            add(f"{s}_shufdiff_{j}", dv, kind="shufdiff", uses="comparison",
                source=f"label-shuffled average difference, draw {j}",
                orient=False, training_set=f"own:{t}",
                description="The average-difference recipe on the same rows "
                            f"after shuffling the labels (draw {j} of {N_SHUF}). "
                            "The comparison for the real differences.")
        print(f"  {t}: {sum(r['target'] == t for r in man)} vectors", flush=True)

    manifest = pd.DataFrame(man)
    geometry = pd.DataFrame(geom)
    MS.validate(manifest, "projected_directions.py")
    names = list(manifest.name)
    W = np.vstack([vecs[n] for n in names]).astype(np.float32)
    norms = np.linalg.norm(W.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise AssertionError(f"not unit norm: {norms.min()} .. {norms.max()}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"directions_{tag}_layer{L}.npz",
                        names=np.array(names), W=W,
                        collection=np.array(coll), layer=np.array(L))
    manifest.to_csv(OUT / f"manifest_{tag}_layer{L}.csv", index=False)
    geometry.to_csv(OUT / f"projected_geometry_{tag}_layer{L}.csv", index=False)
    meta = {"collection": coll, "layer": L, "ks": list(KS), "n_random": N_RAND,
            "n_shufdiff": N_SHUF, "training_sets": set_meta,
            "flipped": sorted(manifest[manifest.flipped].name.tolist())}
    (OUT / f"directions_meta_{tag}_layer{L}.json").write_text(json.dumps(meta, indent=2))

    n_real = int((manifest.uses_target_data != "comparison").sum())
    print(f"\nwrote directions_{tag}_layer{L}.npz: {len(names)} unit vectors "
          f"({n_real} real, {len(names) - n_real} comparison), d={W.shape[1]}, "
          f"{int(manifest.flipped.sum())} flipped to point toward True", flush=True)
    pd.set_option("display.width", 250)
    real = geometry[geometry.uses_target_data != "comparison"]
    cols = ["name", "kind", "k", "kept_share", "auc_target", "auc_train",
            "cos_own_diff", "cos_famout_diff", "flipped", "in_sweep"]
    for t, g in real.groupby("target", sort=False):
        print(f"\n-- {t} --")
        print(g[cols].to_string(index=False))
        c = geometry[(geometry.target == t) & (geometry.uses_target_data == "comparison")]
        print("   comparison sets: median |cos own diff| / median |cos held-out diff| / median AUC:")
        for k_, gg in c.groupby("kind"):
            a = gg.auc_target
            print(f"     {k_:18s} n={len(gg):2d}  {gg.cos_own_diff.abs().median():.3f} / "
                  f"{gg.cos_famout_diff.abs().median():.3f} / "
                  f"{np.maximum(a, 1 - a).median():.3f}")


if __name__ == "__main__":
    main()
