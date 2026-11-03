"""Run ON compute cluster (CPU): full-spectrum covariance comparisons between
datasets, beyond top-k principal subspaces.

All measures work in the pooled-whitened space W_d = L^-1 C_d L^-T
(L = chol of the pooled Ledoit-Wolf covariance), so "shared geometry" is
factored out and low-variance structure is not automatically discounted:

  loge_dist       full-spectrum log-Euclidean SPD distance
                  ||log W_a - log W_b||_F  (ratio-sensitive: tail
                  directions count as much as dominant ones)
  resid_cos       Frobenius cosine between the whitened covariance
                  residuals W_d - I (each dataset's shape deviation from
                  the pooled geometry)
  band affinities mean squared cosine between eigenspaces of W_a and W_b
                  restricted to spectral bands top1-20 / 21-100 /
                  101-500 (where in the spectrum do they share
                  structure?)

Reads only results/<coll>/covariances/raw_last_layer<L>.npz.
Writes <out>/geometry_fullcov.csv.

    python geometry_fullcov.py <results_root> <out_dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

COLLS = ("gemma_original", "gemma_tf_a", "qwen_original", "qwen_tf_a")
LAYERS = {"gemma_original": (17, 22, 29, 33), "gemma_tf_a": (17, 22, 29, 33),
          "qwen_original": (14, 18, 24, 27), "qwen_tf_a": (14, 18, 24, 27)}
BANDS = ((0, 20), (20, 100), (100, 500))
EPS = 1e-9


def main() -> None:
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for coll in COLLS:
        for layer in LAYERS[coll]:
            path = os.path.join(root, coll, "covariances",
                                f"raw_last_layer{layer}.npz")
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=True) as d:
                names = sorted(k[4:] for k in d.files if k.startswith("cov_")
                               and k != "cov_pool")
                pool = np.asarray(d["cov_pool"], dtype=np.float64)
                covs = {n: np.asarray(d[f"cov_{n}"], dtype=np.float64)
                        for n in names}
            L = np.linalg.cholesky(pool)
            spect = {}
            for n in names:
                w = np.linalg.solve(L, np.linalg.solve(L, covs[n]).T).T
                w = (w + w.T) / 2
                lam, vec = np.linalg.eigh(w)
                lam = np.clip(lam, EPS, None)
                # descending order for band selection
                order = np.argsort(lam)[::-1]
                spect[n] = (lam[order], vec[:, order], w)
            for i, a in enumerate(names):
                la, va, wa = spect[a]
                loga = (va * np.log(la)) @ va.T
                ra = wa - np.eye(len(wa))
                for b in names[i + 1:]:
                    lb, vb, wb = spect[b]
                    logb = (vb * np.log(lb)) @ vb.T
                    rb = wb - np.eye(len(wb))
                    row = {"collection": coll, "layer": layer,
                           "dataset_a": a, "dataset_b": b,
                           "loge_dist": float(np.linalg.norm(loga - logb)),
                           "resid_cos": float(np.sum(ra * rb) /
                                              (np.linalg.norm(ra)
                                               * np.linalg.norm(rb)))}
                    for lo, hi in BANDS:
                        m = (va[:, lo:hi].T @ vb[:, lo:hi]) ** 2
                        row[f"band_{lo+1}_{hi}"] = float(m.sum() / (hi - lo))
                    rows.append(row)
            print(f"{coll} layer {layer}: done", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "geometry_fullcov.csv"), index=False)


if __name__ == "__main__":
    main()
