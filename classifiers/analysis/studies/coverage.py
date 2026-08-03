"""A. What fits actually exist. Guard against reading gaps as results."""
import pandas as pd
from .metrics_frame import load_metrics

pd.set_option("display.width", 250, "display.max_columns", 60,
              "display.max_rows", 400)

m = load_metrics()

print("=" * 90)
print("A1. per_dataset: which (features x layer x method) grids ran, per collection")
print("=" * 90)
pd_ = m[m.experiment == "per_dataset"]
g = (pd_.groupby(["collection", "features", "method"])["layer"]
     .nunique().unstack(0))
print(g.fillna(0).astype(int).to_string())

print()
print("=" * 90)
print("A2. per_dataset: n distinct train_datasets per (collection, features, method)")
print("   (12 trainable expected; a smaller number = missing fits)")
print("=" * 90)
g = (pd_.groupby(["collection", "features", "method", "layer"])["train_dataset"]
     .nunique().unstack("collection"))
bad = g[(g.min(axis=1) < 12)]
print(bad.to_string() if len(bad) else "  all grids have 12 training datasets")

print()
print("=" * 90)
print("A3. leave_out: holdouts present per (collection, features, method, kind)")
print("=" * 90)
lo = m[m.experiment == "leave_out"]
g = (lo.groupby(["collection", "features", "method", "holdout_kind"])["holdout"]
     .nunique().unstack(["holdout_kind", "collection"]))
print(g.fillna(0).astype(int).to_string())

print()
print("=" * 90)
print("A4. merged: per_dataset slices present")
print("=" * 90)
mg = m[(m.experiment == "merged")]
g = (mg[mg.kind.isin(["per_dataset", "eval_only"])]
     .groupby(["collection", "features", "method", "layer"])["test_dataset"]
     .nunique().unstack("collection"))
bad = g[(g.min(axis=1) < 13)]
print(bad.to_string() if len(bad) else "  all merged grids score 13 datasets")

print()
print("=" * 90)
print("A5. METHOD x FEATURE coverage matrix (any layer, any experiment), per collection")
print("=" * 90)
for c in ["tf_wrapped_a", "data_unwrap"]:
    print(f"\n--- {c} ---")
    sub = m[m.collection == c]
    print((sub.groupby(["method", "features"]).size().unstack()
           .notna().replace({True: "Y", False: "."}).fillna(".")).to_string())

print()
print("=" * 90)
print("A6. Duplicate rows check (same key, >1 row)")
print("=" * 90)
keys = {"per_dataset": ["collection", "features", "layer", "method",
                        "train_dataset", "test_dataset", "kind"],
        "merged": ["collection", "features", "layer", "method", "kind",
                   "test_dataset"],
        "leave_out": ["collection", "features", "layer", "method",
                      "holdout_kind", "holdout", "test_dataset"]}
for exp, k in keys.items():
    sub = m[m.experiment == exp]
    d = sub.duplicated(subset=k, keep=False)
    print(f"  {exp}: {d.sum()} duplicated rows of {len(sub)}")
    if d.sum():
        print(sub[d].sort_values(k).head(20).to_string())
