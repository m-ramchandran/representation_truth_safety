"""C. Datasets: what is learnable, what transfers, what is hard OOD."""
import numpy as np
import pandas as pd
from .common import metrics, hdr, fmt, TASK_OF, COLLS, DATASETS

pd.set_option("display.width", 300, "display.max_columns", 80,
              "display.max_rows", 600)
m = metrics()
raw = m[m.features == "raw_last"]
pdd = raw[raw.experiment == "per_dataset"]
cv, tr = pdd[pdd.kind == "cv"], pdd[pdd.kind == "transfer"]
lo = raw[raw.experiment == "leave_out"]
mg = raw[raw.experiment == "merged"]
ORDER = list(DATASETS)

hdr("C1. THE GENERALIZATION GRADIENT per dataset "
    "(raw_last, mean over layers, mean over all methods present)")
rows = []
for c in COLLS:
    for d in ORDER:
        r = {"collection": c, "dataset": d, "task": TASK_OF[d]}
        a = cv[(cv.collection == c) & (cv.train_dataset == d)]
        r["within_cv"] = a.auc.mean() if len(a) else np.nan
        b = tr[(tr.collection == c) & (tr.test_dataset == d)]
        r["as_target(mean)"] = b.auc.mean() if len(b) else np.nan
        b2 = tr[(tr.collection == c) & (tr.train_dataset == d)]
        r["as_source(mean)"] = b2.auc.mean() if len(b2) else np.nan
        e = mg[(mg.collection == c) & (mg.test_dataset == d)
               & mg.kind.isin(["per_dataset", "eval_only"])]
        r["merged_cv"] = e.auc.mean() if len(e) else np.nan
        f = lo[(lo.collection == c) & (lo.holdout_kind == "dataset")
               & (lo.test_dataset == d)]
        r["leave_one_out"] = f.auc.mean() if len(f) else np.nan
        g = lo[(lo.collection == c) & (lo.holdout_kind == "task")
               & (lo.test_dataset == d)]
        r["leave_task_out"] = g.auc.mean() if len(g) else np.nan
        rows.append(r)
G = pd.DataFrame(rows)
cols = ["within_cv", "merged_cv", "leave_one_out", "leave_task_out",
        "as_target(mean)", "as_source(mean)"]
for c in COLLS:
    print(f"\n--- {c} ---")
    print(fmt(G[G.collection == c].set_index(["task", "dataset"])[cols]))

hdr("C2. Gradient DELTAS: how much each dataset loses as the model knows less")
for c in COLLS:
    s = G[G.collection == c].set_index(["task", "dataset"])
    d = pd.DataFrame({
        "within_cv": s["within_cv"],
        "d_merged": s["merged_cv"] - s["within_cv"],
        "d_LOO": s["leave_one_out"] - s["within_cv"],
        "d_LTO": s["leave_task_out"] - s["within_cv"],
        "LOO->LTO": s["leave_task_out"] - s["leave_one_out"]})
    print(f"\n--- {c} ---")
    print(fmt(d))

hdr("C3. Transfer matrix (train -> test), raw_last, mean over layers & methods "
    "-- the source of the 'what transfers' claim")
for c in COLLS:
    p = tr[tr.collection == c].pivot_table(index="train_dataset",
                                           columns="test_dataset", values="auc")
    p = p.reindex(index=[d for d in ORDER if d in p.index],
                  columns=[d for d in ORDER if d in p.columns])
    print(f"\n--- {c} ---")
    print(fmt(p, 2))

hdr("C4. WITHIN-task vs BETWEEN-task transfer (excludes self)")
rows = []
for c in COLLS:
    s = tr[tr.collection == c].copy()
    s["within_task"] = s.train_task == s.task
    for wt, g in s.groupby("within_task"):
        rows.append({"collection": c, "within_task": bool(wt),
                     "mean_auc": g.auc.mean(), "median": g.auc.median(),
                     "frac<0.5": (g.auc < .5).mean(), "n": len(g)})
print(fmt(pd.DataFrame(rows)))
print("\nper task-pair (train_task -> test_task), mean transfer AUC:")
for c in COLLS:
    p = tr[tr.collection == c].pivot_table(index="train_task",
                                           columns="task", values="auc")
    print(f"\n--- {c} ---")
    print(fmt(p, 2))

hdr("C5. The best/worst SOURCE->TARGET pairs (mean over layers, linear methods "
    "l1reg/l2reg/svm_linear only, so the pair effect is not a method effect)")
lin = tr[tr.method.isin(["l1reg", "l2reg", "svm_linear"])]
for c in COLLS:
    s = (lin[lin.collection == c]
         .groupby(["train_dataset", "test_dataset"])["auc"].mean()
         .sort_values())
    print(f"\n--- {c}: 15 WORST (inverted rules) ---")
    print(fmt(s.head(15)))
    print(f"\n--- {c}: 10 BEST ---")
    print(fmt(s.tail(10)))

hdr("C6. Transfer ASYMMETRY: auc(A->B) vs auc(B->A), linear methods")
for c in COLLS:
    s = (lin[lin.collection == c]
         .groupby(["train_dataset", "test_dataset"])["auc"].mean())
    rows = []
    ds = sorted(set(lin.train_dataset))
    for i, a in enumerate(ds):
        for b in ds[i + 1:]:
            if (a, b) in s.index and (b, a) in s.index:
                rows.append({"A": a, "B": b, "A->B": s[(a, b)],
                             "B->A": s[(b, a)],
                             "asym": abs(s[(a, b)] - s[(b, a)])})
    r = pd.DataFrame(rows).sort_values("asym", ascending=False)
    print(f"\n--- {c}: 12 most asymmetric pairs ---")
    print(fmt(r.head(12).set_index(["A", "B"])))
    print(f"  mean |asymmetry| over {len(r)} pairs = {r.asym.mean():.3f}")
