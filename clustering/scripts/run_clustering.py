"""Config-driven clustering run: one YAML file = one experiment.

For each layer it loads the data, builds the PCA projections (full-data plus the
bootstrap resamples), then fits every method across its grid and every seed,
saving each fit as its own artifact and collecting a metrics row. The result is
the uniform `results/<scope>/runs/L<layer>/<run>/` tree the plots read; the
per-layer summary goes to `results/<scope>/tables/L<layer>/metrics.csv`.

    python -m clustering.scripts.run_clustering config.yaml

Data comes either from local paths (`inputs:`) or from the HuggingFace repo
(`data:` — datasets + variant + layers, merged or one run per dataset); see
`configs/example_config.yaml`. A fit whose artifacts already exist is skipped and its
saved metrics row reused, so an interrupted run resumes where it left off.
Leave-one-dataset-out designs live in `leave_one_out.py`; figures in `plots.py`.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import metrics, pca
from .load_data import load, load_hf, load_merged
from .methods import fit_gmm, fit_hdbscan, fit_kmeans, fit_leiden, leiden_graph
from .mfa import fit_mfa


def _grid(d: dict):
    """Cartesian product of a dict of lists -> list of dicts. Empty -> no runs."""
    if not d:
        return []
    keys = list(d)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(d[k] for k in keys))]


def _save_samples(data, layer_dir: Path) -> None:
    """Per-sample metadata, row-aligned with every fit's assignments.npz, so the
    results tree is self-contained (no reload needed to interpret a run).
    Written once; parallel jobs on the same layer must not rewrite it."""
    if (layer_dir / "samples.npz").exists():
        return
    arrays = {k: data[k] for k in ("sample_id", "labels", "dataset_id", "topic", "prompts")
              if k in data}
    if arrays:
        np.savez_compressed(layer_dir / "samples.npz", **arrays)


def _logger(path: Path):
    """Print + append to a log file, so progress is visible while a run goes."""
    def log(msg):
        stamped = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(stamped, flush=True)
        with open(path, "a") as f:
            f.write(stamped + "\n")
    return log


def _score(result, Z_or_X, labels):
    """All metrics for one fit, flattened into a row."""
    row = {"n_clusters": result.n_clusters}
    row.update(metrics.model_selection(result, Z_or_X))
    row.update(metrics.geometry_metrics(Z_or_X, result.labels))
    if labels is not None:
        row.update({f"label_{k}": v for k, v in metrics.info_metrics(result.labels, labels).items()})
    return row


def run_layer(cfg, scope, layer, data, out_root: Path):
    X, labels = data["X"], data.get("labels")
    seeds = cfg["seeds"]
    layer_dir = out_root / scope / "runs" / f"L{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    _save_samples(data, layer_dir)
    log = _logger(layer_dir / "run.log")

    # PCA components: full-data per PC count, plus bootstrap resamples per PC
    # count. Each is saved the moment it's fit and reloaded on rerun — at
    # merged-data scale one PCA is many minutes, so these cache like the fits.
    def cached_pca(name, fit_fn):
        path = layer_dir / f"{name}.npz"
        if path.exists():
            log(f"[{scope} L{layer}] {name}: loaded from cache")
            return pca.PCAComponents.load(path)
        log(f"[{scope} L{layer}] {name}: fitting...")
        t0 = time.time()
        comps = fit_fn()
        comps.save(path)
        log(f"[{scope} L{layer}] {name}: done ({time.time() - t0:.0f}s)")
        return comps

    n95 = None
    def n_for(p):
        """'95%' costs a full-spectrum PCA; resolve it once, from disk if possible."""
        nonlocal n95
        if p != "95%":
            return int(p)
        if n95 is None:
            saved = layer_dir / "pca__pc=95%.npz"
            if saved.exists():
                n95 = pca.PCAComponents.load(saved).n_components
            else:
                log(f"[{scope} L{layer}] scanning full PCA spectrum for the 95% cut...")
                t0 = time.time()
                X_std = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
                n95 = pca.n_components_for_variance(X_std)
                log(f"[{scope} L{layer}] 95% variance at {n95} components ({time.time() - t0:.0f}s)")
        return n95

    pc_grid = cfg["pca"]["components"]
    pca_full = {str(p): cached_pca(f"pca__pc={p}", lambda p=p: pca.fit(X, n_for(p)))
                 for p in pc_grid}
    pca_boot = {
        (str(p), b): cached_pca(f"pca__pc={p}__boot{b}",
                               lambda p=p, b=b: pca.fit_bootstrap(X, n_for(p), seed=b))
        for p in pc_grid for b in range(cfg.get("pca_bootstrap", 0))
    }

    # Collect every fit as (name, space, thunk) first so progress can say X/total.
    # Loop variables are bound via lambda defaults — the thunks run later.
    fits = []
    m = cfg["methods"]

    # --- PCA-space methods: GMM, k-means(pca). Main arm (full PCA) x seeds,
    #     plus the PCA-bootstrap arm (cluster full data on bootstrap PCs). ---
    for p, comps in pca_full.items():
        Z = comps.transform(X)
        for g in _grid(m.get("gmm", {})):
            for s in seeds:
                fits.append((f"gmm__cov={g['covariance_type']}__pc={p}__k={g['k']}__seed{s}",
                             Z, lambda Z=Z, g=g, s=s: fit_gmm(Z, g["k"], g["covariance_type"], s)))
        if "pca" in m.get("kmeans", {}).get("k", {}):
            for k in m["kmeans"]["k"]["pca"]:
                for s in seeds:
                    fits.append((f"kmeans__space=pca__pc={p}__k={k}__seed{s}",
                                 Z, lambda Z=Z, k=k, s=s: fit_kmeans(Z, k, s)))

    for (p, b), comps in pca_boot.items():
        Z = comps.transform(X)
        for g in _grid(m.get("gmm", {})):
            fits.append((f"gmm__cov={g['covariance_type']}__pc={p}__k={g['k']}__seed{seeds[0]}__boot{b}",
                         Z, lambda Z=Z, g=g: fit_gmm(Z, g["k"], g["covariance_type"], seeds[0])))

    # --- Raw-space methods: MFA (always raw), k-means(raw). ---
    for g in _grid(m.get("mfa", {})):
        for s in seeds:
            fits.append((f"mfa__nf={g['n_factors']}__k={g['k']}__seed{s}",
                         X, lambda g=g, s=s: fit_mfa(X, g["k"], g["n_factors"], s)))
    if "raw" in m.get("kmeans", {}).get("k", {}):
        for k in m["kmeans"]["k"]["raw"]:
            for s in seeds:
                fits.append((f"kmeans__space=raw__k={k}__seed{s}",
                             X, lambda k=k, s=s: fit_kmeans(X, k, s)))

    # --- Own-k methods: Leiden, HDBSCAN (run on PCA if configured, else raw). ---
    space_for = {"pca": next(iter(pca_full.values())).transform(X) if pca_full else X, "raw": X}
    # The kNN graph depends only on (space, n_neighbors): build it lazily on the
    # first fit that needs it and share it across resolutions and seeds.
    graphs = {}
    def graph_for(space, nn):
        if (space, nn) not in graphs:
            graphs[(space, nn)] = leiden_graph(space_for[space], nn)
        return graphs[(space, nn)]

    for g in _grid(m.get("leiden", {})):
        sp = g.get("space", "raw")
        Z = space_for[sp]
        for s in seeds:
            fits.append((f"leiden__res={g['resolution']}__nn={g['n_neighbors']}__seed{s}",
                         Z, lambda Z=Z, g=g, s=s, sp=sp: fit_leiden(
                             Z, g["resolution"], g["n_neighbors"], s,
                             graph=graph_for(sp, g["n_neighbors"]))))
    for g in _grid(m.get("hdbscan", {})):
        Z = space_for[g.get("space", "raw")]
        fits.append((f"hdbscan__mcs={g['min_cluster_size']}",
                     Z, lambda Z=Z, g=g: fit_hdbscan(Z, g["min_cluster_size"])))

    log(f"[{scope} L{layer}] PCA done; {len(fits)} fits to run")
    rows, n_skipped, n_failed = [], 0, 0
    for i, (save_name, space, fit_fn) in enumerate(fits, 1):
        run_dir = layer_dir / save_name
        metrics_file = run_dir / "metrics.json"
        error_file = run_dir / "error.json"
        if metrics_file.exists():
            rows.append(json.loads(metrics_file.read_text()))
            n_skipped += 1
            continue
        if error_file.exists():   # failed before; delete error.json to retry
            n_failed += 1
            continue
        t0 = time.time()
        try:
            result = fit_fn()
        except Exception as e:
            # One degenerate fit (e.g. a collapsed GMM component) must not
            # kill the whole run.
            log(f"[{scope} L{layer}] fit {i}/{len(fits)}: {save_name}  FAILED: {e}")
            run_dir.mkdir(parents=True, exist_ok=True)
            error_file.write_text(json.dumps({"run": save_name, "error": str(e)}))
            n_failed += 1
            continue
        result.save(run_dir)
        row = {"run": save_name, "method": result.method, **result.params}
        row.update(_score(result, space, labels))
        metrics_file.write_text(json.dumps(row))
        rows.append(row)
        log(f"[{scope} L{layer}] fit {i}/{len(fits)}: {save_name}  "
            f"n_clusters={result.n_clusters}  ({time.time() - t0:.1f}s)")

    if n_skipped:
        log(f"[{scope} L{layer}] reused {n_skipped}/{len(fits)} already-saved fits")
    if n_failed:
        log(f"[{scope} L{layer}] {n_failed}/{len(fits)} fits failed (see error.json files)")
    _write_metrics(layer_dir.parents[1] / "tables" / layer_dir.name / "metrics.csv", rows)
    return rows


def _write_metrics(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def iter_scopes(cfg):
    """Yield (scope, layer, load_fn) for every run the config asks for.

    `inputs:` — local .npz paths, one per layer, under cfg['scope'].
    `data:`   — HF datasets: merged into one scope (merge: true, the default
                for several datasets) or one full run per dataset.
    """
    if "inputs" in cfg:
        for entry in cfg["inputs"]:
            yield cfg["scope"], entry["layer"], lambda p=entry["path"]: load(p)
    if "data" in cfg:
        d = cfg["data"]
        datasets, variant, layers = d["datasets"], d["variant"], d["layers"]
        if d.get("merge", len(datasets) > 1):
            scope = cfg.get("scope", f"merged_{variant}")
            for layer in layers:
                yield scope, layer, lambda l=layer: load_merged(datasets, variant, l)
        else:
            for name in datasets:
                for layer in layers:
                    yield f"{name}_{variant}", layer, lambda n=name, l=layer: load_hf(n, variant, l)


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = Path(cfg.get("output", "clustering/results"))
    for scope, layer, load_fn in iter_scopes(cfg):
        print(f"[{scope}] layer {layer}: loading data", flush=True)
        data = load_fn()
        print(f"[{scope}] layer {layer}: N={len(data['X'])}, D={data['X'].shape[1]}", flush=True)
        rows = run_layer(cfg, scope, layer, data, out_root)
        print(f"[{scope}] layer {layer}: {len(rows)} fits -> {out_root / scope / ('L' + str(layer))}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
