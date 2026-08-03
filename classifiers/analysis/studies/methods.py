"""B. Methods: in-domain vs out-of-domain, and the gap between them."""
import pandas as pd
from .common import (metrics, hdr, fmt, LINEAR, NONLINEAR, POOLED_ONLY,
                    COLLS, XGB_TFW_MISSING)

pd.set_option("display.width", 260, "display.max_columns", 60,
              "display.max_rows", 500)
m = metrics()
raw = m[m.features == "raw_last"]

FAM = {**{k: "linear" for k in LINEAR}, **{k: "nonlinear" for k in NONLINEAR},
       **{k: "pooled" for k in POOLED_ONLY}}

hdr("B1. per_dataset, raw_last: in-domain CV vs mean transfer AUC, per method")
pd_ = raw[raw.experiment == "per_dataset"]
cv = pd_[pd_.kind == "cv"]
tr = pd_[pd_.kind == "transfer"]
rows = []
for c in COLLS:
    for meth in sorted(set(pd_.method)):
        a = cv[(cv.collection == c) & (cv.method == meth)]
        b = tr[(tr.collection == c) & (tr.method == meth)]
        if not len(a):
            continue
        rows.append({"collection": c, "method": meth, "family": FAM[meth],
                     "n_cv": len(a), "cv": a.auc.mean(),
                     "n_transfer": len(b), "transfer": b.auc.mean(),
                     "transfer_med": b.auc.median(),
                     "frac_below_.6": (b.auc < .6).mean(),
                     "frac_below_.5": (b.auc < .5).mean(),
                     "gap": a.auc.mean() - b.auc.mean()})
t = pd.DataFrame(rows)
for c in COLLS:
    print(f"\n--- {c} (all 4 layers pooled) ---")
    print(fmt(t[t.collection == c].sort_values("transfer", ascending=False)
              .set_index("method").drop(columns="collection")))

hdr("B2. SAME as B1 but restricted to the datasets xgboost actually has "
    "in tf_wrapped_a, so the tree/linear comparison is apples-to-apples")
keep = [d for d in set(pd_.train_dataset) if d not in XGB_TFW_MISSING]
cvk = cv[cv.train_dataset.isin(keep)]
trk = tr[tr.train_dataset.isin(keep) & tr.test_dataset.isin(keep)]
rows = []
for c in COLLS:
    for meth in sorted(set(pd_.method)):
        a = cvk[(cvk.collection == c) & (cvk.method == meth)]
        b = trk[(trk.collection == c) & (trk.method == meth)]
        if not len(a):
            continue
        rows.append({"collection": c, "method": meth, "family": FAM[meth],
                     "cv": a.auc.mean(), "transfer": b.auc.mean(),
                     "gap": a.auc.mean() - b.auc.mean(), "n_tr": len(b)})
t2 = pd.DataFrame(rows)
print(fmt(t2.pivot_table(index=["family", "method"],
                         columns="collection", values=["cv", "transfer", "gap"])))

hdr("B3. PAIRED method comparison on transfer: for each identical "
    "(collection, layer, train, test) cell, is method X > method Y?")
piv = tr.pivot_table(index=["collection", "layer", "train_dataset",
                            "test_dataset"], columns="method", values="auc")
base = "l2reg"
rows = []
for meth in piv.columns:
    if meth == base:
        continue
    d = piv[[base, meth]].dropna()
    if not len(d):
        continue
    rows.append({"method": meth, "family": FAM[meth], "n_paired": len(d),
                 "mean_delta_vs_l2reg": (d[meth] - d[base]).mean(),
                 "win_rate": (d[meth] > d[base]).mean(),
                 "median_delta": (d[meth] - d[base]).median()})
print(fmt(pd.DataFrame(rows).sort_values("mean_delta_vs_l2reg",
                                         ascending=False).set_index("method")))

hdr("B4. Same paired test, per collection")
for c in COLLS:
    p = tr[tr.collection == c].pivot_table(
        index=["layer", "train_dataset", "test_dataset"],
        columns="method", values="auc")
    rows = []
    for meth in p.columns:
        if meth == base:
            continue
        d = p[[base, meth]].dropna()
        if not len(d):
            continue
        rows.append({"method": meth, "n": len(d),
                     "delta": (d[meth] - d[base]).mean(),
                     "win": (d[meth] > d[base]).mean()})
    print(f"\n--- {c} vs {base} ---")
    print(fmt(pd.DataFrame(rows).sort_values("delta", ascending=False)
              .set_index("method")))

hdr("B5. leave_out, raw_last: held-out AUC per method (the real OOD test)")
lo = raw[raw.experiment == "leave_out"]
for kind in ["dataset", "task"]:
    print(f"\n--- holdout_kind = {kind} ---")
    k = lo[lo.holdout_kind == kind]
    t = k.pivot_table(index="method", columns="collection", values="auc",
                      aggfunc=["mean", "median", "min"])
    print(fmt(t))

hdr("B6. leave_out paired vs l2reg (same collection/layer/holdout/test cell)")
for kind in ["dataset", "task"]:
    print(f"\n--- {kind} ---")
    p = lo[lo.holdout_kind == kind].pivot_table(
        index=["collection", "layer", "holdout", "test_dataset"],
        columns="method", values="auc")
    rows = []
    for meth in p.columns:
        if meth == base:
            continue
        d = p[[base, meth]].dropna()
        if not len(d):
            continue
        rows.append({"method": meth, "family": FAM[meth], "n": len(d),
                     "delta": (d[meth] - d[base]).mean(),
                     "win": (d[meth] > d[base]).mean()})
    print(fmt(pd.DataFrame(rows).sort_values("delta", ascending=False)
              .set_index("method")))

hdr("B7. merged (pooled nested CV), raw_last: overall + per-dataset by method")
mg = raw[raw.experiment == "merged"]
print("\n-- overall AUC --")
print(fmt(mg[mg.kind == "overall"].pivot_table(index="method",
                                               columns=["collection", "layer"],
                                               values="auc")))

hdr("B8. FAMILY SUMMARY: linear vs nonlinear, per regime "
    "(raw_last, mean over layers & cells)")
rows = []
for c in COLLS:
    for name, sub in [
        ("within_cv", cv),
        ("transfer(per_dataset)", tr),
        ("leave_one_out", lo[lo.holdout_kind == "dataset"]),
        ("leave_task_out", lo[lo.holdout_kind == "task"]),
        ("merged_overall", mg[mg.kind == "overall"]),
    ]:
        s = sub[sub.collection == c]
        for fam in ["linear", "nonlinear", "pooled"]:
            ms = [x for x in set(s.method) if FAM.get(x) == fam]
            v = s[s.method.isin(ms)]
            if not len(v):
                continue
            rows.append({"collection": c, "regime": name, "family": fam,
                         "mean_auc": v.auc.mean(), "n": len(v)})
r = pd.DataFrame(rows)
print(fmt(r.pivot_table(index=["regime"], columns=["collection", "family"],
                        values="mean_auc")))
