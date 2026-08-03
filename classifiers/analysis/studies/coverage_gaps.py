"""A2. Pin down exactly which fits are missing, so nothing is read as a result."""
import pandas as pd
from .metrics_frame import load_metrics

pd.set_option("display.width", 250, "display.max_columns", 60,
              "display.max_rows", 500)
m = load_metrics()

print("=" * 90)
print("Which xgboost per_dataset fits are MISSING in tf_wrapped_a (raw_last)?")
print("=" * 90)
pdm = m[(m.experiment == "per_dataset") & (m.method == "xgboost")
        & (m.features == "raw_last")]
have = (pdm.groupby(["collection", "layer"])["train_dataset"]
        .apply(lambda s: sorted(set(s))))
allds = sorted(set(pdm[pdm.collection == "data_unwrap"].train_dataset))
for (c, l), v in have.items():
    print(f"  {c:14s} L{l}: n={len(v):2d}  missing={sorted(set(allds) - set(v))}")

print()
print("=" * 90)
print("per_dataset: any (collection, features, method, layer) with <12 train datasets")
print("=" * 90)
g = (m[m.experiment == "per_dataset"]
     .groupby(["collection", "features", "method", "layer"])["train_dataset"]
     .nunique())
print(g[g < 12].to_string())

print()
print("=" * 90)
print("per_dataset: n test datasets per fit (13 expected: 12 train + eval_only)")
print("=" * 90)
g = (m[m.experiment == "per_dataset"]
     .groupby(["collection", "features", "method", "layer", "train_dataset"])
     ["test_dataset"].nunique())
print(f"  distribution: {g.value_counts().to_dict()}")
print(g[g < 13].groupby(["collection", "features", "method"]).size().to_string()
      if (g < 13).any() else "  all complete")

print()
print("=" * 90)
print("merged: n scored units (expect 11 per_dataset + 1 eval_only = 12; "
      "blood_test_ranges is --excluded)")
print("=" * 90)
mg = m[(m.experiment == "merged") & m.kind.isin(["per_dataset", "eval_only"])]
g = mg.groupby(["collection", "features", "method", "layer"])["test_dataset"].nunique()
print(f"  distribution: {g.value_counts().to_dict()}")
print(g[g != 12].to_string() if (g != 12).any() else "  all 12")
print("  units:", sorted(set(mg.test_dataset)))

print()
print("=" * 90)
print("leave_out: what does each holdout evaluate on? (dataset vs task kind)")
print("=" * 90)
lo = m[(m.experiment == "leave_out") & (m.features == "raw_last")
       & (m.method == "l2reg") & (m.layer == 29)]
for c in ["tf_wrapped_a", "data_unwrap"]:
    print(f"\n--- {c} ---")
    print(lo[lo.collection == c][["holdout_kind", "holdout", "test_dataset",
                                  "n_test", "auc"]].to_string(index=False))

print()
print("=" * 90)
print("MISSING xgboost in data_unwrap leave_out / sparse sets (documented: raw only)")
print("=" * 90)
xg = m[m.method == "xgboost"]
print(xg.groupby(["collection", "experiment", "features"]).size().unstack(0).to_string())
