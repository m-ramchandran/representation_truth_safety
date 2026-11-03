"""How the class-mean difference lines up with the top principal component,
for every set the steering runs drew a difference vector from.

Question this answers: the target's own class-mean
difference nearly coincides with the target's top principal component, yet
a difference computed on a pool that never contained the target steers the
target just as well. Is the pool's difference the same direction? If not,
what is it aligned with?

For every (collection, layer) the steering sweeps ran at, this script
rebuilds, from the stored activation cache and nothing else:

  * each target dataset's class-mean difference and top principal
    components (all of its rows, both classes together — the same rows and
    covariance directions.py used);
  * the same for every SOURCE set a steering difference came from: the
    standard pool, the expanded pool, the pool with one dataset removed,
    and the pool with one family removed — rosters read from the saved fit
    file's `cv_dataset` array, so the set is exactly the one the saved
    vector was computed on;
  * three centerings of the covariance per set: total (both classes
    together, the L29/L24 dial), class-centered (each class about its
    set-wide mean, the L22/L18 dial), and cell-centered (each
    dataset-by-class cell about its own mean).

Every rebuilt difference is checked against the saved fit vector it should
reproduce (cosine printed; the job fails if any is below 0.999).

Nothing is fitted. Outputs under steering/out/:
  pc_alignment_pairs_{tag}_layer{L}.csv    target x source: the cosines
  pc_alignment_sets_{tag}_layer{L}.csv     one row per set: variance shares
  pc_alignment_datasets_{tag}_layer{L}.csv per pool member: own diff vs own PC1
  pc_alignment_diffcos_{tag}_layer{L}.csv  pairwise cosines of per-dataset diffs
  pc_vectors_{tag}_layer{L}.npz            diff / pc1..3 per set, for reuse

    python steering/pc_alignment.py --collection gemma_tf_a --layer 29 --tag gemma
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from classifiers.data import _load_units, COLLECTIONS  # noqa: E402

RESULTS = REPO / "classifiers" / "results"
OUT = REPO / "steering" / "out"

TASK = {"common_claim": "tf", "claude_vds": "tf",
        "wildjailbreak_vanilla": "jailbreak", "xstest": "jailbreak",
        "pubmedqa_labeled": "medical", "wildjailbreak_adversarial": "jailbreak"}
POOL_EXCLUDED = ("claude_vds", "wildjailbreak_adversarial")
ALL_TARGETS = ("common_claim", "claude_vds", "wildjailbreak_vanilla",
               "xstest", "pubmedqa_labeled", "wildjailbreak_adversarial")
CHUNK = 4096


# ---- sufficient statistics --------------------------------------------------

def cell_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """class -> (n, mean, scatter about the class mean), float64."""
    d = X.shape[1]
    out = {}
    for c in (0, 1):
        Xc = X[y == c]
        n = len(Xc)
        s = np.zeros(d)
        for i in range(0, n, CHUNK):
            s += Xc[i:i + CHUNK].astype(np.float64).sum(0)
        m = s / n
        S = np.zeros((d, d))
        for i in range(0, n, CHUNK):
            Z = Xc[i:i + CHUNK].astype(np.float64) - m
            S += Z.T @ Z
        out[c] = (n, m, S)
    return out


def combine(cells: list[tuple[int, np.ndarray, np.ndarray]]):
    """Pool cells: (N, mean, scatter about the pooled mean)."""
    N = sum(n for n, _, _ in cells)
    mu = sum(n * m for n, m, _ in cells) / N
    S = sum(S + n * np.outer(m - mu, m - mu) for n, m, S in cells)
    return N, mu, S


def top_eig(S: np.ndarray, k: int = 3):
    """(eigenvalues descending, eigenvectors as columns) for the top k."""
    d = S.shape[0]
    w, V = eigh(S, subset_by_index=[d - k, d - 1])
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def unit(v):
    return v / np.linalg.norm(v)


def acos(a, b) -> float:
    return float(abs(unit(a) @ unit(b)))


def set_summary(name: str, members: list[str], stats: dict) -> dict:
    """Everything about one set, from its member datasets' cell stats."""
    cells = [stats[m][c] for m in members for c in (0, 1)]
    N, mu, S_tot = combine(cells)
    trace = np.trace(S_tot)
    # class means over the whole set, and the class-centered scatter
    per_class = {c: combine([stats[m][c] for m in members]) for c in (0, 1)}
    diff = per_class[1][1] - per_class[0][1]
    S_class = per_class[0][2] + per_class[1][2]
    # cell-centered: every dataset-by-class cell about its own mean
    S_cell = sum(S for _, _, S in cells)
    # between-dataset scatter (dataset means, both classes, about mu)
    ds_means = [combine([stats[m][0], stats[m][1]]) for m in members]
    S_between = sum(n * np.outer(m - mu, m - mu) for n, m, _ in ds_means)
    n0, n1 = per_class[0][0], per_class[1][0]
    S_sep = (n0 * n1 / N) * np.outer(diff, diff)   # between-class scatter

    w_tot, V_tot = top_eig(S_tot)
    w_cls, V_cls = top_eig(S_class)
    w_cel, V_cel = top_eig(S_cell)
    u = unit(diff)
    row = {
        "set": name, "n_rows": int(N), "n_datasets": len(members),
        "n_label1": int(n1),
        "pc1_share_total": float(w_tot[0] / trace),
        "pc2_share_total": float(w_tot[1] / trace),
        "diff_dir_share_total": float(u @ S_tot @ u / trace),
        "between_class_share": float(np.trace(S_sep) / trace),
        "between_dataset_share": float(np.trace(S_between) / trace),
        "cos_diff_pc1_total": acos(diff, V_tot[:, 0]),
        "cos_diff_pc2_total": acos(diff, V_tot[:, 1]),
        "cos_diff_pc3_total": acos(diff, V_tot[:, 2]),
        "cos_diff_pc1_class": acos(diff, V_cls[:, 0]),
        "cos_diff_pc1_cell": acos(diff, V_cel[:, 0]),
        "cos_pc1_total_pc1_class": acos(V_tot[:, 0], V_cls[:, 0]),
        "pc1_share_class": float(w_cls[0] / np.trace(S_class)),
    }
    if len(members) > 1:
        w_b, V_b = top_eig(S_between, k=1)
        row["cos_pc1_total_between_dataset_pc1"] = acos(V_tot[:, 0], V_b[:, 0])
        row["cos_diff_between_dataset_pc1"] = acos(diff, V_b[:, 0])
    vecs = {"diff": diff, "pc1_total": V_tot[:, 0], "pc2_total": V_tot[:, 1],
            "pc3_total": V_tot[:, 2], "pc1_class": V_cls[:, 0],
            "pc1_cell": V_cel[:, 0]}
    return row, vecs


# ---- saved fits: rosters and vectors ------------------------------------------

def saved(path: Path, prefix: str = "full_"):
    """(roster {dataset: n_rows}, saved unit vector) of one diff fit."""
    with np.load(path, allow_pickle=True) as z:
        key = prefix + "coef" if prefix + "coef" in z.files else "coef"
        w = np.asarray(z[key], dtype=np.float64).ravel()
        if "cv_dataset" in z.files:
            ds, cnt = np.unique(z["cv_dataset"].astype(str), return_counts=True)
            roster = dict(zip(ds.tolist(), cnt.tolist()))
        else:
            roster = None
    return roster, unit(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="gemma_tf_a")
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--tag", default="gemma")
    ap.add_argument("--targets", default="")
    args = ap.parse_args()
    coll, L = args.collection, args.layer
    targets = (tuple(x for x in args.targets.split(",") if x)
               if args.targets else ALL_TARGETS)
    expanded = f"{coll}_expanded_pool"
    P = RESULTS / coll
    print(f"{coll} layer {L}: targets {targets}", flush=True)

    # ---- which sets, from the saved fits' own rosters -------------------
    std_roster, std_vec = saved(P / "merged/predictions" / f"raw_last_layer{L}_diff.npz")
    need_exp = any(t in POOL_EXCLUDED for t in targets)
    if need_exp:
        exp_roster, exp_vec = saved(
            RESULTS / expanded / "merged/predictions" / f"raw_last_layer{L}_diff.npz")
    sets: dict[str, tuple[list[str], np.ndarray | None]] = {
        "std_pool": (sorted(std_roster), std_vec)}
    if need_exp:
        sets["exp_pool"] = (sorted(exp_roster), exp_vec)
    rosters = {"std_pool": std_roster}
    if need_exp:
        rosters["exp_pool"] = exp_roster
    pairs_spec = []   # (target, role, set name)
    for t in targets:
        pairs_spec.append((t, "own_diff", t))
        if t in POOL_EXCLUDED:
            pairs_spec.append((t, "pooled_diff", "exp_pool"))
            pairs_spec.append((t, "datasetout_diff", "std_pool"))
        else:
            pairs_spec.append((t, "pooled_diff", "std_pool"))
            nm = f"std_minus_{t}"
            r, v = saved(P / "leave_out/predictions"
                         / f"raw_last_layer{L}_diff_dataset_{t}.npz", "")
            sets[nm], rosters[nm] = (sorted(r), v), r
            pairs_spec.append((t, "datasetout_diff", nm))
        if t != "claude_vds":
            fam = TASK[t]
            nm = f"std_minus_task_{fam}"
            if nm not in sets:
                r, v = saved(P / "leave_out/predictions"
                             / f"raw_last_layer{L}_diff_task_{fam}.npz", "")
                sets[nm], rosters[nm] = (sorted(r), v), r
            pairs_spec.append((t, "taskout_diff", nm))
    for t in targets:
        _, v = saved(P / "per_dataset/predictions"
                     / f"raw_last_layer{L}_diff_{t}.npz", "")
        sets[t] = ([t], v)
    needed = sorted({m for members, _ in sets.values() for m in members})
    print("datasets needed:", needed, flush=True)

    # ---- activations -> per-cell sufficient statistics --------------------
    units = _load_units("last", L, coll, groups=False)
    stats = {}
    for m in needed:
        if m not in units:
            raise SystemExit(f"{m} missing from the {coll} cache at layer {L}")
        X = np.asarray(units[m]["X"])
        y = np.asarray(units[m]["y"], dtype=int)
        for nm, roster in rosters.items():
            if m in roster and roster[m] != len(y):
                raise SystemExit(f"{nm}: saved fit used {roster[m]} rows of "
                                 f"{m}, cache has {len(y)}")
        stats[m] = cell_stats(X, y)
        print(f"  {m}: {len(y)} rows ({int(y.sum())} label-1)", flush=True)
    del units
    # every pool member gets its own row too (diff vs own PC1 inside the pool)
    for m in needed:
        if m not in sets:
            sets[m] = ([m], None)

    # ---- per-set summaries -----------------------------------------------
    set_rows, vecs = [], {}
    for nm, (members, saved_vec) in sets.items():
        row, v = set_summary(nm, members, stats)
        row["members"] = "+".join(members)
        if saved_vec is not None:
            row["cos_rebuilt_vs_saved_diff"] = acos(v["diff"], saved_vec)
            if row["cos_rebuilt_vs_saved_diff"] < 0.999:
                raise SystemExit(f"{nm}: rebuilt diff does not match the saved "
                                 f"fit (cos {row['cos_rebuilt_vs_saved_diff']:.4f})")
        set_rows.append(row)
        vecs[nm] = v
        print(f"  {nm:32s} n={row['n_rows']:7d} pc1 share {row['pc1_share_total']:.3f} "
              f"between-dataset {row['between_dataset_share']:.3f} "
              f"cos(diff,pc1) {row['cos_diff_pc1_total']:.3f}"
              + (f"  vs saved {row['cos_rebuilt_vs_saved_diff']:.4f}"
                 if saved_vec is not None else ""), flush=True)

    # ---- target x source pairs ------------------------------------------
    pair_rows = []
    for t, role, s in pairs_spec:
        vt, vs = vecs[t], vecs[s]
        pair_rows.append({
            "target": t, "role": role, "source": s,
            "source_members": "+".join(sets[s][0]),
            "cos_srcdiff_srcpc1_total": acos(vs["diff"], vs["pc1_total"]),
            "cos_srcdiff_tgtpc1_total": acos(vs["diff"], vt["pc1_total"]),
            "cos_srcdiff_tgtdiff": acos(vs["diff"], vt["diff"]),
            "cos_srcpc1_tgtpc1_total": acos(vs["pc1_total"], vt["pc1_total"]),
            "cos_tgtdiff_tgtpc1_total": acos(vt["diff"], vt["pc1_total"]),
            "cos_srcdiff_srcpc1_class": acos(vs["diff"], vs["pc1_class"]),
            "cos_srcdiff_tgtpc1_class": acos(vs["diff"], vt["pc1_class"]),
            "cos_srcpc1_tgtpc1_class": acos(vs["pc1_class"], vt["pc1_class"]),
            "cos_tgtdiff_tgtpc1_class": acos(vt["diff"], vt["pc1_class"]),
            "cos_srcdiff_tgtpc2_total": acos(vs["diff"], vt["pc2_total"]),
            "cos_srcdiff_tgtpc3_total": acos(vs["diff"], vt["pc3_total"]),
        })

    # ---- pool members: own diff vs own PC1, and vs the pool's vectors ----
    members_all = [m for m in needed]
    ds_rows = []
    for m in members_all:
        v = vecs[m]
        r = next(r for r in set_rows if r["set"] == m)
        ds_rows.append({
            "dataset": m, "n_rows": r["n_rows"],
            "in_std_pool": m in sets["std_pool"][0],
            "cos_owndiff_ownpc1_total": r["cos_diff_pc1_total"],
            "cos_owndiff_ownpc1_class": r["cos_diff_pc1_class"],
            "pc1_share_total": r["pc1_share_total"],
            "between_class_share": r["between_class_share"],
            "cos_owndiff_stddiff": acos(v["diff"], vecs["std_pool"]["diff"]),
            "cos_owndiff_stdpc1_total": acos(v["diff"], vecs["std_pool"]["pc1_total"]),
            "cos_ownpc1_stdpc1_total": acos(v["pc1_total"], vecs["std_pool"]["pc1_total"]),
            "diff_norm": float(np.linalg.norm(v["diff"])),
        })
    names = members_all
    M = np.array([[acos(vecs[a]["diff"], vecs[b]["diff"]) for b in names]
                  for a in names])
    diffcos = pd.DataFrame(M, index=names, columns=names)

    tag = f"_{args.tag}" if args.tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(OUT / f"pc_alignment_pairs{tag}_layer{L}.csv", index=False)
    pd.DataFrame(set_rows).to_csv(OUT / f"pc_alignment_sets{tag}_layer{L}.csv", index=False)
    pd.DataFrame(ds_rows).to_csv(OUT / f"pc_alignment_datasets{tag}_layer{L}.csv", index=False)
    diffcos.to_csv(OUT / f"pc_alignment_diffcos{tag}_layer{L}.csv")
    np.savez_compressed(OUT / f"pc_vectors{tag}_layer{L}.npz",
                        **{f"{nm}__{k}": v.astype(np.float32)
                           for nm, vv in vecs.items() for k, v in vv.items()},
                        sets=np.array(list(vecs)),
                        collection=np.array(coll), layer=np.array(L))

    pd.set_option("display.width", 250)
    print("\n== target x source ==")
    print(pd.DataFrame(pair_rows).round(3).to_string(index=False))
    print("\n== sets ==")
    print(pd.DataFrame(set_rows).drop(columns=["members"]).round(3).to_string(index=False))
    print("\n== pool members ==")
    print(pd.DataFrame(ds_rows).round(3).to_string(index=False))
    print("\n== pairwise cosines of per-dataset differences ==")
    print(diffcos.round(2).to_string())


if __name__ == "__main__":
    main()
