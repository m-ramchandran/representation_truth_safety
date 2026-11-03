"""Joint-framing activation PCA, run on compute cluster.

The basis behind supplementary_figure1_framing. That figure asks where the
True/False-framed prompts land relative to the Original ones, so the map
it is drawn on has to be defined by both: this fits one PCA per model on
the Original and True/False samples pooled, and projects every sample of
both through it. (full_activation_pca.py fits on the Original samples
alone - that basis stays as it is, and is what the main fig12 panel and
its readouts use.)

social_iqa enters the fit once, not twice. It has no Original variant, so
its two framings are byte-identical activations; counting both copies
would give one dataset 141,456 of 430,230 rows purely because it was
loaded twice. Every other dataset contributes both of its framings. The
projected points file still carries both copies, so the figure can draw
whatever it wants.

Per model (Gemma hidden state 30 of 34, Qwen 25 of 28 - matched 88%
depth) it writes:

  supplementary_figure1_framing_points.npz     projected samples
  supplementary_figure1_framing_variance.csv   complete eigenspectrum

    sbatch sbatch/joint_framing_pca.sbatch
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psc_full_activation_pca import (LAYER, N_KEEP, OUT,  # noqa: E402
                                     _load, _project)

CHUNK = 20_000


def _pca_pooled(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]:
    """Exact PCA of the rows of every array stacked, without stacking them.

    Same accumulation as the single-matrix version in
    psc_full_activation_pca: the mean and the covariance are built in
    float64 over row chunks, and eigh gives the complete spectrum. Taking
    a list keeps the two framings in the memory they were loaded into
    instead of making a 430k-row copy of them.
    """
    d = arrays[0].shape[1]
    n = sum(len(X) for X in arrays)
    total = np.zeros(d, dtype=np.float64)
    for X in arrays:
        for i in range(0, len(X), CHUNK):
            total += X[i:i + CHUNK].sum(axis=0, dtype=np.float64)
    mean = total / n

    cov = np.zeros((d, d), dtype=np.float64)
    for X in arrays:
        for i in range(0, len(X), CHUNK):
            chunk = X[i:i + CHUNK].astype(np.float64) - mean
            cov += chunk.T @ chunk
    cov /= n - 1

    ev, vec = np.linalg.eigh(cov)
    ev, vec = ev[::-1], vec[:, ::-1]                    # descending
    ev = np.clip(ev, 0, None)
    # deterministic sign: largest-magnitude loading of each component is +
    flip = np.sign(vec[np.abs(vec).argmax(axis=0), np.arange(d)])
    vec = vec * flip
    return mean, vec, ev / ev.sum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, spectra = [], []
    for model in ("gemma", "qwen"):
        print(f"=== {model} layer {LAYER[model]} ===", flush=True)
        loaded = {p: _load(model, p) for p in ("bare", "true_false")}

        # social_iqa's framed copy is the same activations as its bare one
        keep = loaded["true_false"]["dataset"] != "social_iqa"
        fit_on = [loaded["bare"]["X"], loaded["true_false"]["X"][keep]]
        print(f"  fitting on {sum(len(X) for X in fit_on)} rows "
              f"({len(fit_on[0])} Original + {len(fit_on[1])} True/False, "
              f"social_iqa's duplicate framed copy excluded)", flush=True)
        mean, vec, ratio = _pca_pooled(fit_on)
        del fit_on
        print(f"  PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}, 90% at "
              f"{int(np.searchsorted(np.cumsum(ratio), 0.9)) + 1} PCs",
              flush=True)
        spectra.append(pd.DataFrame({
            "model": model, "layer": LAYER[model],
            "pc": np.arange(1, len(ratio) + 1),
            "explained_variance": ratio}))

        for pert in ("bare", "true_false"):
            d = loaded[pert]
            Z = _project(d["X"], mean, vec, N_KEEP)
            points.append(pd.DataFrame({
                "model": model, "perturbation": pert,
                "dataset": d["dataset"], "label": d["y"],
                **{f"pc{i + 1}": Z[:, i] for i in range(N_KEEP)}}))
            d["X"] = None
            del Z
        del loaded, mean, vec

    pts = pd.concat(points, ignore_index=True)
    np.savez_compressed(
        OUT / "supplementary_figure1_framing_points.npz",
        model=pts.model.to_numpy().astype("U8"),
        perturbation=pts.perturbation.to_numpy().astype("U10"),
        dataset=pts.dataset.to_numpy().astype("U26"),
        label=pts.label.to_numpy().astype(np.int8),
        **{c: pts[c].to_numpy(np.float32) for c in
           [f"pc{i + 1}" for i in range(N_KEEP)]})
    pd.concat(spectra, ignore_index=True).to_csv(
        OUT / "supplementary_figure1_framing_variance.csv", index=False)
    print(f"\nwrote {len(pts)} projected samples", flush=True)


if __name__ == "__main__":
    main()
