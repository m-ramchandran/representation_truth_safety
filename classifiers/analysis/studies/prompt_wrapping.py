"""E. Wrapped (tf_wrapped_a) vs unwrapped (data_unwrap), paired cell by cell.

social_iqa is loaded from the WRAPPED tree in BOTH collections (data.py), so
its per_dataset rows are the same activations twice: a built-in noise floor.
"""
import numpy as np
import pandas as pd
from scipy import stats
from .common import metrics, hdr, fmt, TASK_OF, LAYERS, DATASETS

pd.set_option("display.width", 300, "display.max_columns", 80,
              "display.max_rows", 600)
m = metrics()
pdd = m[m.experiment == "per_dataset"]
cv, tr = pdd[pdd.kind == "cv"], pdd[pdd.kind == "transfer"]
lo = m[m.experiment == "leave_out"]
mg = m[m.experiment == "merged"]


def pair(df, keys):
    p = df.pivot_table(index=keys, columns="collection", values="auc")
    return p.dropna()


hdr("E0. NOISE FLOOR: social_iqa is the identical (wrapped) cache in both "
    "collections. Its per_dataset CV delta bounds what 'wrapping effect' means.")
d = pair(cv[(cv.train_dataset == "social_iqa")],
         ["features", "layer", "method"])
d["delta"] = d["tf_wrapped_a"] - d["data_unwrap"]
print(fmt(d.describe().loc[["count", "mean", "std", "min", "max"]], 4))
print("\nsame for social_iqa as a transfer TARGET (source dataset does differ):")
d2 = pair(tr[tr.test_dataset == "social_iqa"],
          ["features", "layer", "method", "train_dataset"])
print(f"  n={len(d2)}, mean delta="
      f"{(d2['tf_wrapped_a'] - d2['data_unwrap']).mean():.4f}")

hdr("E1. WRAPPED - UNWRAPPED, paired, by regime (all feature sets)")
rows = []
for name, df, keys in [
        ("within_cv", cv, ["features", "layer", "method", "train_dataset"]),
        ("transfer", tr, ["features", "layer", "method", "train_dataset",
                          "test_dataset"]),
        ("LOO", lo[lo.holdout_kind == "dataset"],
         ["features", "layer", "method", "holdout", "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"],
         ["features", "layer", "method", "holdout", "test_dataset"]),
        ("merged_overall", mg[mg.kind == "overall"],
         ["features", "layer", "method"]),
        ("merged_per_dataset", mg[mg.kind.isin(["per_dataset", "eval_only"])],
         ["features", "layer", "method", "test_dataset"])]:
    p = pair(df, keys)
    delta = p["tf_wrapped_a"] - p["data_unwrap"]
    w = stats.wilcoxon(p["tf_wrapped_a"], p["data_unwrap"])
    rows.append({"regime": name, "n": len(p),
                 "wrapped": p["tf_wrapped_a"].mean(),
                 "unwrapped": p["data_unwrap"].mean(),
                 "delta": delta.mean(), "median_delta": delta.median(),
                 "wrapped_wins": (delta > 0).mean(), "p": w.pvalue})
print(fmt(pd.DataFrame(rows).set_index("regime"), 4))

hdr("E2. Same, restricted to raw_last, split by regime x layer")
for name, df, keys in [
        ("within_cv", cv, ["method", "train_dataset"]),
        ("transfer", tr, ["method", "train_dataset", "test_dataset"]),
        ("LOO", lo[lo.holdout_kind == "dataset"],
         ["method", "holdout", "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"],
         ["method", "holdout", "test_dataset"])]:
    s = df[df.features == "raw_last"]
    out = {}
    for L in LAYERS:
        p = pair(s[s.layer == L], keys)
        out[L] = (p["tf_wrapped_a"] - p["data_unwrap"]).mean()
    print(f"  {name:12s} " + "  ".join(f"L{L}={v:+.3f}" for L, v in out.items()))

hdr("E3. Wrapping effect PER DATASET (raw_last, mean over layers & methods)")
rows = []
for d in DATASETS:
    r = {"dataset": d, "task": TASK_OF[d]}
    a = pair(cv[(cv.features == "raw_last") & (cv.train_dataset == d)],
             ["layer", "method"])
    r["within_cv"] = (a["tf_wrapped_a"] - a["data_unwrap"]).mean() if len(a) else np.nan
    b = pair(tr[(tr.features == "raw_last") & (tr.test_dataset == d)],
             ["layer", "method", "train_dataset"])
    r["as_target"] = (b["tf_wrapped_a"] - b["data_unwrap"]).mean() if len(b) else np.nan
    c_ = pair(tr[(tr.features == "raw_last") & (tr.train_dataset == d)],
              ["layer", "method", "test_dataset"])
    r["as_source"] = (c_["tf_wrapped_a"] - c_["data_unwrap"]).mean() if len(c_) else np.nan
    e = pair(lo[(lo.features == "raw_last") & (lo.holdout_kind == "dataset")
                & (lo.test_dataset == d)], ["layer", "method"])
    r["LOO"] = (e["tf_wrapped_a"] - e["data_unwrap"]).mean() if len(e) else np.nan
    f = pair(lo[(lo.features == "raw_last") & (lo.holdout_kind == "task")
                & (lo.test_dataset == d)], ["layer", "method"])
    r["LTO"] = (f["tf_wrapped_a"] - f["data_unwrap"]).mean() if len(f) else np.nan
    rows.append(r)
print(fmt(pd.DataFrame(rows).set_index(["task", "dataset"])))

hdr("E4. Wrapping effect PER METHOD (raw_last)")
rows = []
for name, df, keys in [
        ("within_cv", cv, ["layer", "train_dataset"]),
        ("transfer", tr, ["layer", "train_dataset", "test_dataset"]),
        ("LOO", lo[lo.holdout_kind == "dataset"], ["layer", "holdout",
                                                   "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"], ["layer", "holdout",
                                                "test_dataset"])]:
    s = df[df.features == "raw_last"]
    for meth in sorted(set(s.method)):
        p = pair(s[s.method == meth], keys)
        if len(p) < 4:
            continue
        rows.append({"regime": name, "method": meth, "n": len(p),
                     "delta": (p["tf_wrapped_a"] - p["data_unwrap"]).mean()})
r = pd.DataFrame(rows)
print(fmt(r.pivot_table(index="method", columns="regime", values="delta")
          [["within_cv", "transfer", "LOO", "LTO"]]))

hdr("E5. Wrapping effect PER FEATURE SET")
rows = []
for name, df, keys in [
        ("within_cv", cv, ["layer", "method", "train_dataset"]),
        ("transfer", tr, ["layer", "method", "train_dataset", "test_dataset"]),
        ("LOO", lo[lo.holdout_kind == "dataset"],
         ["layer", "method", "holdout", "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"],
         ["layer", "method", "holdout", "test_dataset"])]:
    for f in sorted(set(df.features)):
        p = pair(df[df.features == f], keys)
        if len(p) < 4:
            continue
        rows.append({"regime": name, "features": f, "n": len(p),
                     "delta": (p["tf_wrapped_a"] - p["data_unwrap"]).mean()})
r = pd.DataFrame(rows)
print(fmt(r.pivot_table(index="features", columns="regime", values="delta")
          [["within_cv", "transfer", "LOO", "LTO"]]))

hdr("E6. THE JAILBREAK INVERSION: leave-one-task-out on the jailbreak task, "
    "raw_last, every method x layer")
j = lo[(lo.features == "raw_last") & (lo.holdout_kind == "task")
       & (lo.holdout == "jailbreak")]
p = j.pivot_table(index=["test_dataset", "method"],
                  columns=["collection", "layer"], values="auc")
print(fmt(p))
print("\nmean over methods & layers per (collection, test_dataset):")
print(fmt(j.pivot_table(index="test_dataset", columns="collection",
                        values="auc")))
print("\nfraction of jailbreak-LTO cells below 0.5:")
print(fmt(j.groupby("collection")["auc"].apply(lambda s: (s < .5).mean())))
