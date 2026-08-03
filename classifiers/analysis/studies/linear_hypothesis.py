"""F. Direct tests bearing on the linear-direction hypothesis."""
import pandas as pd
from .common import metrics, hdr, fmt, COLLS, LINEAR, NONLINEAR

pd.set_option("display.width", 300, "display.max_columns", 80,
              "display.max_rows", 600)
m = metrics()
raw = m[m.features == "raw_last"]
pdd = raw[raw.experiment == "per_dataset"]
cv, tr = pdd[pdd.kind == "cv"], pdd[pdd.kind == "transfer"]
lo = raw[raw.experiment == "leave_out"]
mg = raw[raw.experiment == "merged"]

hdr("F1. THE 'diff' TEST. diff is the raw class-mean difference — the "
    "hypothesis' own object. Discriminative linear fits (l2reg/svm_linear) "
    "add data-fitted rotation. Where does each win?")
rows = []
for name, df, keys in [
        ("within_cv", cv, ["layer", "train_dataset"]),
        ("transfer", tr, ["layer", "train_dataset", "test_dataset"]),
        ("LOO", lo[lo.holdout_kind == "dataset"], ["layer", "holdout",
                                                   "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"], ["layer", "holdout",
                                                "test_dataset"])]:
    for c in COLLS:
        p = df[df.collection == c].pivot_table(index=keys, columns="method",
                                               values="auc")
        if "diff" not in p or "l2reg" not in p:
            continue
        d = p[["diff", "l2reg"]].dropna()
        rows.append({"regime": name, "collection": c, "n": len(d),
                     "diff": d["diff"].mean(), "l2reg": d["l2reg"].mean(),
                     "diff-l2reg": (d["diff"] - d["l2reg"]).mean(),
                     "diff_wins": (d["diff"] > d["l2reg"]).mean()})
print(fmt(pd.DataFrame(rows).set_index(["regime", "collection"])))

hdr("F2. Does the discriminative advantage REVERSE as the target gets more "
    "out of domain? (diff - l2reg, by how far the target is)")
for c in COLLS:
    s = tr[tr.collection == c].copy()
    s["same_task"] = s.train_task == s.task
    p = s.pivot_table(index=["layer", "train_dataset", "test_dataset",
                             "same_task"], columns="method", values="auc")
    d = p[["diff", "l2reg"]].dropna().reset_index()
    print(f"\n--- {c} ---")
    print(fmt(d.groupby("same_task").apply(
        lambda g: pd.Series({"n": len(g), "diff": g["diff"].mean(),
                             "l2reg": g["l2reg"].mean(),
                             "delta": (g["diff"] - g["l2reg"]).mean()}),
        include_groups=False)))

hdr("F3. IS THERE ONE DIRECTION? If yes, a model trained on ANY dataset "
    "should transfer. Distribution of transfer AUC per source (linear "
    "methods, mean over layers)")
lin = tr[tr.method.isin(["l1reg", "l2reg", "svm_linear"])]
for c in COLLS:
    s = lin[lin.collection == c]
    g = s.groupby("train_dataset")["auc"].agg(
        n="size", mean="mean", median="median", q10=lambda x: x.quantile(.1),
        q90=lambda x: x.quantile(.9), frac_lt_50=lambda x: (x < .5).mean(),
        frac_gt_80=lambda x: (x > .8).mean())
    print(f"\n--- {c} ---")
    print(fmt(g))

hdr("F4. Pooling more data: does it help, and WHO does it help? "
    "merged_cv - within_cv per dataset (a shared direction would mean "
    "pooling never hurts, and helps small datasets a lot)")
rows = []
for c in COLLS:
    a = cv[cv.collection == c].groupby(["method", "layer", "train_dataset"])["auc"].mean()
    b = (mg[(mg.collection == c) & mg.kind.isin(["per_dataset", "eval_only"])]
         .groupby(["method", "layer", "test_dataset"])["auc"].mean())
    j = pd.concat([a.rename("within"), b.rename("merged")], axis=1).dropna()
    j = j.reset_index()
    j.columns = ["method", "layer", "dataset", "within", "merged"]
    j["collection"] = c
    rows.append(j)
J = pd.concat(rows)
print(fmt(J.pivot_table(index="dataset", columns="collection",
                        values=["within", "merged"], aggfunc="mean")))
print("\ndelta (merged - within) per dataset:")
J["delta"] = J["merged"] - J["within"]
print(fmt(J.pivot_table(index="dataset", columns="collection", values="delta")))
print("\ndelta by method (mean over datasets):")
print(fmt(J.pivot_table(index="method", columns="collection", values="delta")))

hdr("F5. THE POOLED MODEL'S CEILING. If one direction served all 11 "
    "datasets, merged overall AUC would approach the mean within_cv. "
    "It does not — where is the shortfall?")
for c in COLLS:
    a = cv[(cv.collection == c) & (cv.method == "l2reg")].groupby("train_dataset")["auc"].mean()
    b = mg[(mg.collection == c) & (mg.method == "l2reg")
           & mg.kind.isin(["per_dataset", "eval_only"])].groupby("test_dataset")["auc"].mean()
    o = mg[(mg.collection == c) & (mg.method == "l2reg")
           & (mg.kind == "overall")].auc.mean()
    print(f"\n--- {c} (l2reg, raw_last) ---")
    print(f"  merged OVERALL auc               = {o:.3f}")
    print(f"  mean of per-dataset within_cv    = {a.mean():.3f}")
    print(f"  mean of merged per-dataset slice = {b.mean():.3f}")
    print("  NOTE overall > slice mean means the pooled AUC is inflated by "
          "between-dataset separation, not within-dataset discrimination.")

hdr("F6. HOW MUCH OF MERGED AUC IS BETWEEN-DATASET STRUCTURE? "
    "overall AUC vs the mean per-dataset slice, per method")
rows = []
for c in COLLS:
    o = mg[(mg.collection == c) & (mg.kind == "overall")].groupby(["method", "layer"])["auc"].mean()
    s = (mg[(mg.collection == c) & (mg.kind == "per_dataset")]
         .groupby(["method", "layer"])["auc"].mean())
    j = pd.concat([o.rename("overall"), s.rename("slice_mean")], axis=1).dropna()
    j["inflation"] = j["overall"] - j["slice_mean"]
    j["collection"] = c
    rows.append(j.reset_index())
R = pd.concat(rows)
print(fmt(R.pivot_table(index="method", columns="collection",
                        values=["overall", "slice_mean", "inflation"])))

hdr("F7. NONLINEAR GAIN vs DISTANCE FROM DOMAIN. If the property were a "
    "linear direction, extra capacity should help LESS as you go OOD.")
rows = []
for name, df, keys in [
        ("within_cv", cv, ["layer", "train_dataset"]),
        ("transfer_same_task", None, None),
        ("transfer_cross_task", None, None),
        ("LOO", lo[lo.holdout_kind == "dataset"], ["layer", "holdout",
                                                   "test_dataset"]),
        ("LTO", lo[lo.holdout_kind == "task"], ["layer", "holdout",
                                                "test_dataset"])]:
    for c in COLLS:
        if df is None:
            s = tr[tr.collection == c].copy()
            s["same_task"] = s.train_task == s.task
            s = s[s.same_task == (name == "transfer_same_task")]
            p = s.pivot_table(index=["layer", "train_dataset", "test_dataset"],
                              columns="method", values="auc")
        else:
            p = df[df.collection == c].pivot_table(index=keys, columns="method",
                                                   values="auc")
        best_lin = p[[x for x in LINEAR if x in p]].max(axis=1)
        best_non = p[[x for x in NONLINEAR if x in p]].max(axis=1)
        d = pd.concat([best_lin.rename("lin"), best_non.rename("non")],
                      axis=1).dropna()
        rows.append({"regime": name, "collection": c, "n": len(d),
                     "best_linear": d["lin"].mean(),
                     "best_nonlinear": d["non"].mean(),
                     "nonlinear_gain": (d["non"] - d["lin"]).mean(),
                     "nl_wins": (d["non"] > d["lin"]).mean()})
print(fmt(pd.DataFrame(rows).set_index(["regime", "collection"])))
