"""Run ON compute cluster: figure 4 panel b's coefficient alignment under the
regularized Mahalanobis cosine, not the plain one.

Panel b asks how close each part of the pooled fit is to the direction
fitted on one dataset alone. The plain cosine treats every activation
direction as equally important; the Mahalanobis cosine

    cos_S(a, b) = a' S b / sqrt((a' S a) (b' S b))

weights directions by how much the activations actually vary along them,
so two coefficient vectors count as aligned when they order the data the
same way rather than when they happen to point the same way in raw
coordinates. S is the Ledoit-Wolf-shrunk covariance already cached under
classifiers/results/<collection>/covariances/raw_last_layer<L>.npz -- the
same estimate, and the same two measures, that
coef_similarity.mahalanobis_reg_cov_{pool,test} use for the dataset-pair
similarities in supplementary figure 4:

  plain                     the unweighted cosine retained as a diagnostic
  mahalanobis_reg_cov_pool  whitened by the pooled corpus covariance
  mahalanobis_reg_cov_test  whitened by the covariance of the dataset whose
                            own direction is the comparison target

ROSTER CAVEAT, reported in the log: the cached cov_pool was estimated on
2026-08-08 over 13 datasets, Blood ranges included, while the paper roster
and the expanded pool are the other 12. The per-dataset covariances
(mahalanobis_reg_cov_test) are unaffected. Re-estimating cov_pool over 12
means reloading every activation cache, so this script uses the cache and
says so rather than silently refitting.

Directions, all mapped to cache units by directions.to_cache_units:

  shared / dev / shared_plus_dev  the expanded-pool merged random_slopes fit
                                  (full_ arrays), group = the dataset itself
  pooled_l2                       the expanded-pool merged l2reg fit
  (compared against)              that dataset's own per_dataset l2reg fit

and one extra row per collection, shared_vs_pooled_l2 (dataset ""), the direct
cosine between the shared random-slopes direction and pooled L2 that the
figure 4 caption quotes.

    python paper_final_figures/scripts/mahalanobis_alignment.py \
        <collection> <layer>

from the repository root (or with REPO_ROOT set). Reads the stitched root
classifiers/results/_rs_dir_expanded_pool/<collection> that
sbatch/rs_directions_expanded_pool.sbatch builds (merged -> the expanded-pool
fit, per_dataset -> the base collection's single fits); rebuilds it if it is
missing. Writes
classifiers/results/_mahalanobis_alignment/<collection>_layer<L>.csv with one
row per (dataset, comparison, measure). Nothing under results/<collection>/
is written to. The available per-collection files are also consolidated into
paper_final_figures/data/figure4_mahalanobis_alignment.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("REPO_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(ROOT))
from classifiers.analysis_scripts.directions import (              # noqa: E402
    DIRECTION_KEYS, load_direction, raw_direction, stack_directions,
    to_cache_units)
from classifiers.analysis_scripts.load_results import (            # noqa: E402
    load_prediction, npz_path)

RESULTS = ROOT / "classifiers" / "results"
OUT = RESULTS / "_mahalanobis_alignment"
PAPER_OUT = ROOT / "paper_final_figures" / "data" / "figure4_mahalanobis_alignment.csv"
STITCHED = RESULTS / "_rs_dir_expanded_pool"
TAG = "raw_last"
VARIANT = "random_slopes"      # panel b draws the per-dataset deviations
RS_KEYS = DIRECTION_KEYS | {"full_dev_w", "full_group_labels"}
COMPARISONS = ("shared", "dev", "shared_plus_dev", "pooled_l2")


def stitched_root(collection: str) -> Path:
    """merged/ -> the expanded-pool fit, per_dataset/ -> the base
    collection's single fits, as rs_directions_expanded_pool.sbatch lays
    it out. Recreated here so the script does not depend on that job
    having been run in this filesystem's lifetime."""
    root = STITCHED / collection
    (root).mkdir(parents=True, exist_ok=True)
    for name, target in (("merged", RESULTS / f"{collection}_expanded_pool"
                          / "merged"),
                         ("per_dataset", RESULTS / collection
                          / "per_dataset")):
        link = root / name
        if not target.exists():
            raise FileNotFoundError(f"missing {target}")
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(target)
    return root


def load_rs(layer: int, root: Path):
    """The merged random-slopes fit in cache units: (shared, {group ->
    deviation}). Same construction as analysis_scripts.rs_directions --
    shared weights and every deviation live in one standardized space, so
    a single mapping serves all of them."""
    d = load_prediction(npz_path("merged", TAG, layer, VARIANT, root=root),
                        keys=RS_KEYS)
    shared = raw_direction(d, VARIANT, prefix="full_")
    dev_w = np.atleast_2d(np.asarray(d["full_dev_w"], dtype=float))
    W, idx = to_cache_units(dev_w, d, prefix="full_")
    return shared, {str(g): (W[i], idx)
                    for i, g in enumerate(d["full_group_labels"])}


def cos_sigma(a: np.ndarray, b: np.ndarray, S: np.ndarray | None) -> float:
    """Mahalanobis cosine under S; the plain cosine when S is None."""
    if S is None:
        num, na, nb = a @ b, a @ a, b @ b
    else:
        Sa, Sb = S @ a, S @ b
        num, na, nb = a @ Sb, a @ Sa, b @ Sb
    return float(num / np.sqrt(na * nb))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: mahalanobis_alignment.py <collection> <layer>")
    collection, layer = sys.argv[1], int(sys.argv[2])
    root = stitched_root(collection)

    cov_path = RESULTS / collection / "covariances" / f"{TAG}_layer{layer}.npz"
    covs = np.load(cov_path, allow_pickle=True)
    pool_names = [str(x) for x in covs["pool_names"]]
    print(f"covariance cache {cov_path.name}: pool over {len(pool_names)} "
          f"datasets -> {pool_names}")
    cov_pool = np.asarray(covs["cov_pool"], dtype=np.float64)

    shared, devs = load_rs(layer, root)
    pooled = load_direction("merged", TAG, layer, "l2reg", root=root)
    # the datasets are whichever have BOTH a per_dataset l2reg fit and a
    # deviation in the merged fit; the pool roster drives the rest
    per_dataset_dir = root / "per_dataset" / "predictions"
    available = sorted({f.name.split(f"{TAG}_layer{layer}_l2reg_")[1][:-4]
                        for f in per_dataset_dir.glob(
                            f"{TAG}_layer{layer}_l2reg_*.npz")})
    datasets = [d for d in available if d in devs]
    missing = [d for d in available if d not in devs]
    if missing:
        print(f"  no deviation in the merged fit, skipped: {missing}")
    print(f"  {len(datasets)} datasets: {datasets}")

    rows = []
    for ds in datasets:
        single = load_direction("per_dataset", TAG, layer, "l2reg", ds,
                                root=root)
        dev = devs[ds]
        effective = (shared[0] + dev[0], dev[1])
        cov_test = np.asarray(covs[f"cov_{ds}"], dtype=np.float64) \
            if f"cov_{ds}" in covs else None
        for comparison, direction in (("shared", shared),
                                      ("dev", dev),
                                      ("shared_plus_dev", effective),
                                      ("pooled_l2", pooled)):
            W, _ = stack_directions([direction, single])
            for measure, S in (("plain", None),
                               ("mahalanobis_reg_cov_pool", cov_pool),
                               ("mahalanobis_reg_cov_test", cov_test)):
                if measure == "mahalanobis_reg_cov_test" and S is None:
                    continue
                rows.append({"collection": collection, "layer": layer,
                             "dataset": ds, "comparison": comparison,
                             "measure": measure,
                             "value": round(cos_sigma(W[0], W[1], S), 4)})

    # the caption quotes the shared random-slopes direction against pooled
    # L2 directly -- one value per collection, not one per dataset, so it is
    # stored with dataset = "" rather than repeated on every row
    W, _ = stack_directions([shared, pooled])
    for measure, S in (("plain", None), ("mahalanobis_reg_cov_pool", cov_pool)):
        rows.append({"collection": collection, "layer": layer, "dataset": "",
                     "comparison": "shared_vs_pooled_l2", "measure": measure,
                     "value": round(cos_sigma(W[0], W[1], S), 4)})

    df = pd.DataFrame(rows)
    df["pool_roster_n"] = len(pool_names)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{collection}_layer{layer}.csv"
    df.to_csv(path, index=False)
    combined = pd.concat(
        [pd.read_csv(source) for source in sorted(OUT.glob("*_layer*.csv"))],
        ignore_index=True,
    )
    combined.to_csv(PAPER_OUT, index=False)
    print(f"\nwrote {path}  ({len(df)} rows)")
    print(f"consolidated {len(combined)} rows -> {PAPER_OUT}")
    print(df[df.dataset != ""].pivot_table(
        index="dataset", columns=["measure", "comparison"],
        values="value").round(3).to_string())
    print("\nshared vs pooled L2 directly:")
    print(df[df.dataset == ""][["measure", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
