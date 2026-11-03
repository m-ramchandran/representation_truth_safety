"""How much of a direction lives in the top few principal components of the
set it was built on: class-mean difference versus regression coefficients.

For every set the steering sweeps drew a vector from (each target dataset,
the standard pool, the expanded pool, the pool with one dataset removed,
the pool with one family removed), this rebuilds the set's covariance from
the stored activation cache, takes its top K eigenvectors, and reports the
share of each vector's squared length that falls inside the top k of them,
for k = 1 .. K. Vectors compared, all from saved files (nothing is fit):

  diff      the class-mean difference (rebuilt; checked against the saved fit)
  l2_raw    the penalized logistic-regression coefficients in raw activation
            units (coef / standardize_scale) — the vector the p=0 steering
            arm injected and the direction the classifier reads along
  l2_std    the same coefficients as fitted, in standardized units
  rs_raw    the random-slopes fit's shared coefficient vector, raw units
  random    the expectation for a direction with no relation to the set: k/d

Also reported: the cumulative variance share of the top k components, so
"the difference is inside the top 3 PCs" can be read against "the top 3 PCs
hold 40% of the variance".

Outputs under steering/out/:
  pc_capture_{tag}_layer{L}.csv       long table: set, vector, k, captured
  pc_capture_summary_{tag}_layer{L}.csv  captured at k = 1,2,3,5,10,20,50,100,K

    python steering/pc_capture.py --collection gemma_tf_a --layer 29 --tag gemma
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
from classifiers.data import _load_units  # noqa: E402
from steering.pc_alignment import cell_stats, combine, unit  # noqa: E402

RESULTS = REPO / "classifiers" / "results"
OUT = REPO / "steering" / "out"
TASK = {"common_claim": "tf", "claude_vds": "tf", "wildjailbreak_vanilla": "jailbreak",
        "xstest": "jailbreak", "pubmedqa_labeled": "medical",
        "wildjailbreak_adversarial": "jailbreak"}
POOL_EXCLUDED = ("claude_vds", "wildjailbreak_adversarial")
ALL_TARGETS = ("common_claim", "claude_vds", "wildjailbreak_vanilla",
               "xstest", "pubmedqa_labeled", "wildjailbreak_adversarial")
KS = (1, 2, 3, 5, 10, 20, 50, 100, 200)


def read_fit(path: Path, prefix: str) -> dict:
    """{'roster', 'coef_std', 'coef_raw'} for one saved fit file.
    coef_raw = coef / standardize_scale (raw activation units)."""
    with np.load(path, allow_pickle=True) as z:
        key = prefix + "coef" if prefix + "coef" in z.files else "coef"
        w = np.asarray(z[key], dtype=np.float64).ravel()
        scale = None
        for k in (prefix + "standardize_scale", "standardize_scale"):
            if k in z.files:
                scale = np.asarray(z[k], dtype=np.float64)
                break
        roster = None
        if "cv_dataset" in z.files:
            ds, cnt = np.unique(z["cv_dataset"].astype(str), return_counts=True)
            roster = dict(zip(ds.tolist(), cnt.tolist()))
    return {"roster": roster, "coef_std": w,
            "coef_raw": w / scale if scale is not None else w}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="gemma_tf_a")
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--tag", default="gemma")
    ap.add_argument("--targets", default="")
    ap.add_argument("--k", type=int, default=200)
    args = ap.parse_args()
    coll, L, K = args.collection, args.layer, args.k
    targets = (tuple(x for x in args.targets.split(",") if x)
               if args.targets else ALL_TARGETS)
    P = RESULTS / coll
    PE = RESULTS / f"{coll}_expanded_pool"

    def merged(root, method):
        return root / "merged/predictions" / f"raw_last_layer{L}_{method}.npz"

    def single(method, t):
        return P / "per_dataset/predictions" / f"raw_last_layer{L}_{method}_{t}.npz"

    def leave_out(method, kind, name):
        return P / "leave_out/predictions" / f"raw_last_layer{L}_{method}_{kind}_{name}.npz"

    # set name -> {members, vectors {name: vec}}
    sets: dict[str, dict] = {}

    def add_set(name, diff_path, l2_path, rs_path, prefix, members=None):
        if name in sets:
            return
        d = read_fit(diff_path, prefix)
        vec = {"diff_saved": unit(d["coef_raw"])}
        roster = d["roster"]
        for label, path in (("l2", l2_path), ("rs", rs_path)):
            try:
                f = read_fit(path, prefix)
                vec[f"{label}_raw"] = unit(f["coef_raw"])
                if label == "l2":
                    vec["l2_std"] = unit(f["coef_std"])
            except FileNotFoundError:
                print(f"  {name}: no {label} fit at {path.name}, skipped", flush=True)
        sets[name] = {"members": members or sorted(roster), "roster": roster,
                      "vec": vec}

    add_set("std_pool", merged(P, "diff"), merged(P, "l2reg"),
            merged(P, "random_slopes"), "full_")
    if any(t in POOL_EXCLUDED for t in targets):
        add_set("exp_pool", merged(PE, "diff"), merged(PE, "l2reg"),
                merged(PE, "random_slopes"), "full_")
    for t in targets:
        add_set(t, single("diff", t), single("l2reg", t),
                single("random_slopes", t), "", members=[t])
        if t not in POOL_EXCLUDED:
            add_set(f"std_minus_{t}", leave_out("diff", "dataset", t),
                    leave_out("l2reg", "dataset", t),
                    leave_out("random_slopes", "dataset", t), "")
        if t != "claude_vds":
            fam = TASK[t]
            add_set(f"std_minus_task_{fam}", leave_out("diff", "task", fam),
                    leave_out("l2reg", "task", fam),
                    leave_out("random_slopes", "task", fam), "")
    needed = sorted({m for s in sets.values() for m in s["members"]})
    print("datasets needed:", needed, flush=True)

    units = _load_units("last", L, coll, groups=False)
    stats = {}
    for m in needed:
        X = np.asarray(units[m]["X"])
        y = np.asarray(units[m]["y"], dtype=int)
        for nm, s in sets.items():
            r = s["roster"]
            if r and m in r and r[m] != len(y):
                raise SystemExit(f"{nm}: saved fit used {r[m]} rows of {m}, "
                                 f"cache has {len(y)}")
        stats[m] = cell_stats(X, y)
        print(f"  {m}: {len(y)} rows", flush=True)
    del units
    d = next(iter(stats.values()))[0][1].shape[0]
    K = min(K, d)

    long_rows, summ_rows = [], []
    for nm, s in sets.items():
        cells = [stats[m][c] for m in s["members"] for c in (0, 1)]
        N, mu, S = combine(cells)
        per_class = {c: combine([stats[m][c] for m in s["members"]]) for c in (0, 1)}
        diff = per_class[1][1] - per_class[0][1]
        chk = abs(unit(diff) @ s["vec"]["diff_saved"])
        if chk < 0.999:
            raise SystemExit(f"{nm}: rebuilt diff vs saved fit cos {chk:.4f}")
        w, V = eigh(S, subset_by_index=[d - K, d - 1])
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        var_share = np.cumsum(w) / np.trace(S)
        vecs = {"diff": unit(diff), **{k: v for k, v in s["vec"].items()
                                       if k != "diff_saved"}}
        for label, v in vecs.items():
            proj = (V.T @ v) ** 2
            cum = np.cumsum(proj)
            for k in range(1, K + 1):
                long_rows.append({"set": nm, "vector": label, "k": k,
                                  "captured": float(cum[k - 1]),
                                  "var_share": float(var_share[k - 1])})
            row = {"set": nm, "n_rows": int(N), "n_datasets": len(s["members"]),
                   "vector": label}
            for k in KS:
                if k <= K:
                    row[f"k{k}"] = round(float(cum[k - 1]), 4)
            row["k_for_half"] = int(np.searchsorted(cum, 0.5) + 1) if cum[-1] >= 0.5 else -1
            row["k_for_90pct"] = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else -1
            summ_rows.append(row)
        row = {"set": nm, "n_rows": int(N), "n_datasets": len(s["members"]),
               "vector": "variance_share"}
        for k in KS:
            if k <= K:
                row[f"k{k}"] = round(float(var_share[k - 1]), 4)
        summ_rows.append(row)
        row = {"set": nm, "n_rows": int(N), "n_datasets": len(s["members"]),
               "vector": "random_expectation"}
        for k in KS:
            if k <= K:
                row[f"k{k}"] = round(k / d, 4)
        summ_rows.append(row)
        print(f"  {nm}: done", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(long_rows).to_csv(OUT / f"pc_capture{tag}_layer{L}.csv", index=False)
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT / f"pc_capture_summary{tag}_layer{L}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
