"""Leave-one-dataset-out designs (multi-dataset runs).

For a chosen method + config (not the whole grid), hold each dataset out in turn:

  design 1 (shared PCA)   : fit PCA without the held-out dataset, then cluster
                            ALL data on those PCs. See where the held-out points land.
  design 2 (LOO PCA)      : fit PCA and cluster on the rest only, then assign the
                            held-out points. For GMM/MFA also score their log-likelihood.
  design 3 (LOO raw)      : same as 2 but in raw space (no PCA).

Only GMM, MFA, k-means — they assign new points naturally (Leiden/HDBSCAN don't).
The held-out log-likelihood (designs 2-3, GMM/MFA) says how surprising a dataset
is to the structure the others define.

Config-driven runs (a `leave_one_out:` section next to a merged `data:` section):

    python -m clustering.scripts.leave_one_out config.yaml

Every fit is saved under results/<scope>/runs/L<layer>/loo/<run>/ — labels for rest
and held-out points, soft probabilities and log-likelihoods, the fitted model,
and the PCA — and already-saved runs are skipped on re-run.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml

from . import pca
from .load_data import load_merged
from .methods import fit_gmm
from .run_clustering import _logger, _save_samples


@dataclass
class LOOResult:
    design: str
    held_out: object
    labels_rest: np.ndarray          # cluster labels for the non-held-out points
    labels_heldout: np.ndarray       # cluster labels for the held-out points
    loglik_heldout: np.ndarray = None  # per-point log-lik of held-out (GMM/MFA)
    result: object = None            # the underlying ClusterResult (fitted model)
    pca_components: object = None    # the fitted PCAComponents, if the design uses one


def _loglik(result, data):
    try:
        return result.log_likelihood(data)
    except TypeError:
        return None  # non-probabilistic method (k-means)


def design1_shared_pca(X, dataset_id, held_out, fit_fn, n_components, **kw) -> LOOResult:
    """PCA without held-out; cluster ALL data on those PCs."""
    rest = dataset_id != held_out
    comps = pca.fit(X[rest], n_components)
    Z = comps.transform(X)
    res = fit_fn(Z, **kw)
    return LOOResult("shared_pca", held_out, res.labels[rest], res.labels[~rest],
                     result=res, pca_components=comps)


def design2_loo_pca(X, dataset_id, held_out, fit_fn, n_components, **kw) -> LOOResult:
    """PCA and clustering on the rest; assign held-out; score its log-lik."""
    rest = dataset_id != held_out
    comps = pca.fit(X[rest], n_components)
    Z_rest, Z_held = comps.transform(X[rest]), comps.transform(X[~rest])
    res = fit_fn(Z_rest, **kw)
    return LOOResult("loo_pca", held_out, res.labels,
                     res.model.predict(Z_held), _loglik(res, Z_held),
                     result=res, pca_components=comps)


def design3_loo_raw(X, dataset_id, held_out, fit_fn, **kw) -> LOOResult:
    """Cluster the rest in raw space; assign held-out; score its log-lik."""
    rest = dataset_id != held_out
    res = fit_fn(X[rest], **kw)
    return LOOResult("loo_raw", held_out, res.labels,
                     res.model.predict(X[~rest]), _loglik(res, X[~rest]),
                     result=res)


def run_all(X, dataset_id, fit_fn, n_components, raw=False, **kw):
    """Every held-out dataset x the applicable designs."""
    results = []
    for d in np.unique(dataset_id):
        results.append(design1_shared_pca(X, dataset_id, d, fit_fn, n_components, **kw))
        results.append(design2_loo_pca(X, dataset_id, d, fit_fn, n_components, **kw))
        if raw:
            results.append(design3_loo_raw(X, dataset_id, d, fit_fn, **kw))
    return results


DESIGNS = {"shared_pca": design1_shared_pca, "loo_pca": design2_loo_pca,
           "loo_raw": design3_loo_raw}


def save_loo(loo: LOOResult, rest_mask, X_heldout, run_dir) -> None:
    """One LOO fit -> assignments.npz + model.joblib + pca.npz + config.json.

    `X_heldout` is the held-out data in the model's own space (PC scores for the
    PCA designs, raw activations for loo_raw) — used for soft probabilities.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"rest_mask": rest_mask, "labels_rest": loo.labels_rest,
              "labels_heldout": loo.labels_heldout, **loo.result.scores}
    if loo.loglik_heldout is not None:
        arrays["loglik_heldout"] = loo.loglik_heldout
    if hasattr(loo.result.model, "predict_proba"):
        arrays["proba_heldout"] = loo.result.model.predict_proba(X_heldout).astype(np.float32)
    np.savez(run_dir / "assignments.npz", **arrays)
    if loo.pca_components is not None:
        loo.pca_components.save(run_dir / "pca.npz")
    joblib.dump(loo.result.model, run_dir / "model.joblib")
    (run_dir / "config.json").write_text(json.dumps(
        {"design": loo.design, "held_out": str(loo.held_out),
         "method": loo.result.method, **loo.result.params}, indent=2))


def main(config_path):
    """Leave-one-dataset-out over the small GMM grid in the config.

    Needs a merged `data:` section (the datasets define what is held out) and a
    `leave_one_out:` section, e.g.

        leave_one_out:
          designs: [shared_pca, loo_pca]   # loo_raw is expensive; diag cov only
          pc: 50
          gmm: {covariance_type: [diag, full], k: [8, 16, 32]}
          seed: 0
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    d, lo = cfg["data"], cfg["leave_one_out"]
    scope = cfg.get("scope", f"merged_{d['variant']}")
    out_root = Path(cfg.get("output", "clustering/results"))
    seed = lo.get("seed", 0)

    for layer in d["layers"]:
        print(f"[{scope} loo] layer {layer}: loading data", flush=True)
        data = load_merged(d["datasets"], d["variant"], layer)
        X, dataset_id = data["X"], data["dataset_id"]
        loo_dir = out_root / scope / "runs" / f"L{layer}" / "loo"
        loo_dir.mkdir(parents=True, exist_ok=True)
        _save_samples(data, loo_dir.parent)  # shared with the main grid's runs
        log = _logger(loo_dir / "run.log")

        for cov, k in itertools.product(lo["gmm"]["covariance_type"], lo["gmm"]["k"]):
            for held in np.unique(dataset_id):
                for design in lo["designs"]:
                    name = (f"{design}__heldout={held}__gmm__cov={cov}"
                            f"__pc={lo['pc']}__k={k}__seed{seed}")
                    run_dir = loo_dir / name
                    if (run_dir / "config.json").exists():
                        continue
                    t0 = time.time()
                    rest = dataset_id != held
                    fit_fn = lambda Z: fit_gmm(Z, k, cov, seed)
                    if design == "loo_raw":
                        loo_res = design3_loo_raw(X, dataset_id, held, fit_fn)
                        X_held = X[~rest]
                    else:
                        loo_res = DESIGNS[design](X, dataset_id, held, fit_fn, lo["pc"])
                        X_held = loo_res.pca_components.transform(X[~rest])
                    save_loo(loo_res, rest, X_held, run_dir)
                    ll = ("" if loo_res.loglik_heldout is None
                          else f"  heldout mean log-lik={loo_res.loglik_heldout.mean():.1f}")
                    log(f"[{scope} L{layer}] {name}{ll}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
