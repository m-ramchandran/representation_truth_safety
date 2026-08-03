"""D. Feature sets, layers, and the wrapped/unwrapped contrast."""
import pandas as pd
from scipy import stats
from .common import metrics, hdr, fmt, COLLS, LAYERS, FEATURES

pd.set_option("display.width", 300, "display.max_columns", 80,
              "display.max_rows", 600)
m = metrics()
pdd = m[m.experiment == "per_dataset"]
cv, tr = pdd[pdd.kind == "cv"], pdd[pdd.kind == "transfer"]
lo = m[m.experiment == "leave_out"]
mg = m[m.experiment == "merged"]

# ---------------------------------------------------------------- FEATURES
hdr("D1. FEATURE SETS: is raw always best?  PAIRED against raw_last on the "
    "identical (collection, layer, method, train, test) cell")


def paired_vs_raw(df, keys):
    p = df.pivot_table(index=keys, columns="features", values="auc")
    if "raw_last" not in p:
        return pd.DataFrame()
    rows = []
    for f in p.columns:
        if f == "raw_last":
            continue
        d = p[["raw_last", f]].dropna()
        if len(d) < 5:
            continue
        w = stats.wilcoxon(d[f], d["raw_last"])
        rows.append({"features": f, "n": len(d),
                     "mean_delta_vs_raw": (d[f] - d["raw_last"]).mean(),
                     "median_delta": (d[f] - d["raw_last"]).median(),
                     "win_rate": (d[f] > d["raw_last"]).mean(),
                     "p_wilcoxon": w.pvalue})
    return pd.DataFrame(rows).sort_values("mean_delta_vs_raw", ascending=False)


for c in COLLS:
    print(f"\n### {c}")
    print("\n-- in-domain CV (per_dataset) --")
    print(fmt(paired_vs_raw(cv[cv.collection == c],
                            ["layer", "method", "train_dataset"]), 4))
    print("\n-- transfer (per_dataset off-diagonal) --")
    print(fmt(paired_vs_raw(tr[tr.collection == c],
                            ["layer", "method", "train_dataset",
                             "test_dataset"]), 4))
    print("\n-- leave-one-dataset-out --")
    print(fmt(paired_vs_raw(lo[(lo.collection == c)
                               & (lo.holdout_kind == "dataset")],
                            ["layer", "method", "holdout", "test_dataset"]), 4))
    print("\n-- leave-one-task-out --")
    print(fmt(paired_vs_raw(lo[(lo.collection == c)
                               & (lo.holdout_kind == "task")],
                            ["layer", "method", "holdout", "test_dataset"]), 4))
    print("\n-- merged overall --")
    print(fmt(paired_vs_raw(mg[(mg.collection == c) & (mg.kind == "overall")],
                            ["layer", "method"]), 4))

hdr("D2. Feature set x method: where does a non-raw set actually WIN? "
    "(mean delta vs raw on transfer, per method)")
for c in COLLS:
    s = tr[tr.collection == c]
    p = s.pivot_table(index=["layer", "method", "train_dataset",
                             "test_dataset"], columns="features", values="auc")
    out = {}
    for f in p.columns:
        if f == "raw_last":
            continue
        d = p[["raw_last", f]].dropna()
        out[f] = (d[f] - d["raw_last"]).groupby(level="method").mean()
    print(f"\n--- {c}: mean(transfer AUC - raw) by method ---")
    print(fmt(pd.DataFrame(out)))

hdr("D3. Absolute AUC by feature set and regime (mean over everything present)")
rows = []
for c in COLLS:
    for f in FEATURES:
        r = {"collection": c, "features": f}
        r["within_cv"] = cv[(cv.collection == c) & (cv.features == f)].auc.mean()
        r["transfer"] = tr[(tr.collection == c) & (tr.features == f)].auc.mean()
        s = lo[(lo.collection == c) & (lo.features == f)]
        r["LOO"] = s[s.holdout_kind == "dataset"].auc.mean()
        r["LTO"] = s[s.holdout_kind == "task"].auc.mean()
        r["merged"] = mg[(mg.collection == c) & (mg.features == f)
                         & (mg.kind == "overall")].auc.mean()
        rows.append(r)
print(fmt(pd.DataFrame(rows).set_index(["collection", "features"])))

# ------------------------------------------------------------------ LAYERS
hdr("D4. LAYERS: mean AUC by layer x regime x collection (raw_last)")
rows = []
for c in COLLS:
    for L in LAYERS:
        r = {"collection": c, "layer": L}
        r["within_cv"] = cv[(cv.collection == c) & (cv.features == "raw_last")
                            & (cv.layer == L)].auc.mean()
        r["transfer"] = tr[(tr.collection == c) & (tr.features == "raw_last")
                           & (tr.layer == L)].auc.mean()
        s = lo[(lo.collection == c) & (lo.features == "raw_last")
               & (lo.layer == L)]
        r["LOO"] = s[s.holdout_kind == "dataset"].auc.mean()
        r["LTO"] = s[s.holdout_kind == "task"].auc.mean()
        r["merged"] = mg[(mg.collection == c) & (mg.features == "raw_last")
                         & (mg.layer == L) & (mg.kind == "overall")].auc.mean()
        rows.append(r)
LY = pd.DataFrame(rows)
print(fmt(LY.set_index(["collection", "layer"])))

hdr("D5. LAYER trend PAIRED within cell (raw_last): layer L vs layer 17, "
    "same (method, train, test). Does the sign replicate across collections?")
for regime, df, keys in [
        ("within_cv", cv[cv.features == "raw_last"], ["method", "train_dataset"]),
        ("transfer", tr[tr.features == "raw_last"],
         ["method", "train_dataset", "test_dataset"]),
        ("LOO", lo[(lo.features == "raw_last") & (lo.holdout_kind == "dataset")],
         ["method", "holdout", "test_dataset"]),
        ("LTO", lo[(lo.features == "raw_last") & (lo.holdout_kind == "task")],
         ["method", "holdout", "test_dataset"]),
        ("merged", mg[(mg.features == "raw_last") & (mg.kind == "overall")],
         ["method"])]:
    print(f"\n--- {regime}: mean(AUC@L - AUC@17), paired ---")
    out = {}
    for c in COLLS:
        p = df[df.collection == c].pivot_table(index=keys, columns="layer",
                                               values="auc")
        if 17 not in p.columns:
            continue
        out[c] = {L: (p[L] - p[17]).dropna().mean()
                  for L in LAYERS if L in p.columns}
    print(fmt(pd.DataFrame(out)))

hdr("D6. Best layer per (collection, regime, dataset) -- does the winner agree?")
rows = []
for c in COLLS:
    for regime, df, tcol in [("within_cv", cv[cv.features == "raw_last"],
                              "train_dataset"),
                             ("LOO", lo[(lo.features == "raw_last")
                                        & (lo.holdout_kind == "dataset")],
                              "test_dataset"),
                             ("LTO", lo[(lo.features == "raw_last")
                                        & (lo.holdout_kind == "task")],
                              "test_dataset")]:
        s = df[df.collection == c]
        g = s.groupby([tcol, "layer"])["auc"].mean().unstack()
        for d, row in g.iterrows():
            rows.append({"collection": c, "regime": regime, "dataset": d,
                         "best_layer": row.idxmax(),
                         "spread": row.max() - row.min()})
B = pd.DataFrame(rows)
piv = B.pivot_table(index=["regime", "dataset"], columns="collection",
                    values="best_layer", aggfunc="first")
piv["agree"] = piv["tf_wrapped_a"] == piv["data_unwrap"]
print(fmt(piv))
print(f"\nagreement rate on best layer: {piv['agree'].mean():.2%}")
print("\nmean spread (max-min over layers) per regime:")
print(fmt(B.pivot_table(index="regime", columns="collection", values="spread")))
