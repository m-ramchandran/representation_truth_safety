"""Config-driven t-SNE / UMAP embeddings with NESS stability, after a clustering run.

Embeddings are for viewing, not clustering. This runs on a finished
run_clustering results tree: for each layer it projects the activations onto
the run's saved PCA (nothing re-fits), embeds at every connectivity value x
seed, then scores NESS stability across the seed ensemble.

    python -m clustering.run_embeddings config.yaml tsne
    python -m clustering.run_embeddings config.yaml umap [n_neighbors]

The optional last argument restricts the run to one connectivity value, so
slow settings (seeded UMAP is single-threaded) can run as parallel jobs.
The method's stability-curve json is only written on full-grid runs;
the curve can always be rebuilt from the per-value ness files.

Embeddings run on the PCA projections only (never raw), at every PC count in
the config's space_pc list — the same counts the clusterings used.

Writes results/<scope>/runs/L<layer>/embeddings/:
    tsne__pc=15__perp=30__seed0.npz  coords (N, 2) float32; skip-if-saved resume
    ness_tsne__pc=15__perp=30.npz    per-point local stability across the seeds
    ness_tsne__pc=15.json            global stability per perplexity (the curve)
and the same for umap, keyed nn= instead of perp=.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import ness
from .embeddings import embed_tsne, embed_umap
from .pca import PCAComponents
from .run_clustering import _logger, iter_scopes

METHODS = {
    "tsne": (embed_tsne, "perplexity", "perp"),
    "umap": (embed_umap, "n_neighbors", "nn"),
}


def run_layer(cfg, method, scope, layer, data, out_root: Path, only_value=None):
    embed_fn, param_name, tag = METHODS[method]
    ecfg = cfg["embeddings"]
    grid, seeds = ecfg[method][param_name], ecfg["seeds"]
    if only_value is not None:
        grid = [only_value]
    layer_dir = out_root / scope / "runs" / f"L{layer}"
    emb_dir = layer_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    log = _logger(emb_dir / "run.log")

    pcs = ecfg["space_pc"]
    for pc in ([pcs] if isinstance(pcs, int) else pcs):
        Z = PCAComponents.load(layer_dir / f"pca__pc={pc}.npz").transform(data["X"])
        log(f"[{scope} L{layer}] {method} pc={pc}: N={len(Z)}; "
            f"{len(grid)} {param_name} values x {len(seeds)} seeds")

        curve = {}
        for v in grid:
            coords = []
            for s in seeds:
                path = emb_dir / f"{method}__pc={pc}__{tag}={v}__seed{s}.npz"
                if path.exists():
                    coords.append(np.load(path)["coords"])
                    continue
                t0 = time.time()
                c = np.asarray(embed_fn(Z, seed=s, **{param_name: v}), np.float32)
                np.savez_compressed(path, coords=c)
                coords.append(c)
                log(f"[{scope} L{layer}] {path.name}  ({time.time() - t0:.0f}s)")
            local = ness.local_stability(coords, k=ecfg.get("ness_k", 15))
            np.savez_compressed(emb_dir / f"ness_{method}__pc={pc}__{tag}={v}.npz",
                                local_stability=local)
            curve[v] = float(local.mean())
            log(f"[{scope} L{layer}] {method} pc={pc} {param_name}={v}: "
                f"global stability {curve[v]:.3f}")
        if only_value is None:   # partial runs must not clobber the full curve
            (emb_dir / f"ness_{method}__pc={pc}.json").write_text(json.dumps(curve))


def main(config_path, method, only_value=None):
    if method not in METHODS:
        sys.exit(f"method must be one of {list(METHODS)}, got {method!r}")
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_root = Path(cfg.get("output", "clustering/results"))
    for scope, layer, load_fn in iter_scopes(cfg):
        print(f"[{scope}] layer {layer}: loading data", flush=True)
        data = load_fn()
        run_layer(cfg, method, scope, layer, data, out_root, only_value)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         yaml.safe_load(sys.argv[3]) if len(sys.argv) > 3 else None)
