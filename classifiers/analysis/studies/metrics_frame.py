"""Load every metrics_* csv from tf_wrapped_a and data_unwrap into one frame.

    python -m classifiers.analysis.studies.metrics_frame
"""
from pathlib import Path

import pandas as pd

# repo_root/classifiers/results — this file lives at
# repo_root/classifiers/analysis/studies/, so go up two.
ROOT = Path(__file__).resolve().parents[2] / "results"
COLLECTIONS = ("tf_wrapped_a", "data_unwrap")  # `original` is deliberately excluded
EXPERIMENTS = ("per_dataset", "merged", "leave_out")


def load_metrics(root: Path = ROOT,
                 collections: tuple[str, ...] = COLLECTIONS) -> pd.DataFrame:
    rows = []
    for collection in collections:
        for experiment in EXPERIMENTS:
            tables = root / collection / experiment / "tables"
            for path in sorted(tables.glob("metrics_*.csv")):
                frame = pd.read_csv(path)
                frame["collection"] = collection
                frame["experiment"] = experiment
                frame["file"] = path.name
                rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No metrics_*.csv files found below {root}")
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    m = load_metrics()
    print("shape", m.shape)
    print("\ncolumns:", list(m.columns))
    print("\n--- rows per (collection, experiment) ---")
    print(m.groupby(["collection", "experiment"]).size())
    print("\n--- kinds ---")
    print(m.groupby(["experiment", "kind"], dropna=False).size())
    print("\n--- features ---")
    print(m.groupby(["collection", "features"]).size().unstack(0))
    print("\n--- methods ---")
    print(m.groupby(["collection", "method"]).size().unstack(0))
    print("\n--- layers ---")
    print(m.groupby(["collection", "layer"]).size().unstack(0))
    print("\n--- n_test == 0 (placeholder rows!) ---")
    z = m[m["n_test"] == 0]
    print("count:", len(z))
    if len(z):
        print(z.groupby(["collection", "experiment", "features", "method"]).size())
    print("\n--- auc NaN ---")
    print(m["auc"].isna().sum())
    out = Path(__file__).with_name("metrics.parquet")
    m.to_parquet(out)
    print(f"\nwrote {out}")
