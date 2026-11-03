"""Full-sample activation PCA + supervised readouts, run on compute cluster.

Replaces the subsampled basis behind fig12. The old figure drew 400
samples per dataset and fit its PCA on that same draw (2.4% of the
corpus); this fits on every Original-prompt sample of the 12-dataset
paper roster and projects every sample of both framings through it.

Per model (Gemma hidden state 30 of 34, Qwen 25 of 28 - matched 88%
depth) it writes:

  figure1_activation_points.npz      pc1..pc4 for every sample
  figure1_activation_variance.csv    the complete eigenspectrum
  figure1_activation_readouts.csv    held-out supervised readouts

The readouts answer "how much of the structure is in the picture?": a
multinomial logistic regression predicting dataset identity and a binary
one predicting the label, each fit on the top two PCs (what the figure
shows) and on all D PCs (the whole representation - PCA is a rotation,
so this is the linear probe on the raw activation). Grouped 5-fold CV
throughout, folds split on the stored source ids so the pubmedqa /
social_iqa 1-to-2 expansions and the wildjailbreak shared prompts never
straddle a split.

    sbatch sbatch/full_activation_pca.sbatch
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classifiers.data import load_datasets                      # noqa: E402

OUT = ROOT / "paper_final_figures" / "data"
LAYER = {"gemma": 29, "qwen": 24}
COLLECTION = {("gemma", "bare"): "gemma_original",
              ("gemma", "true_false"): "gemma_tf_a",
              ("qwen", "bare"): "qwen_original",
              ("qwen", "true_false"): "qwen_tf_a"}
# the paper roster: every unit except claude_blood_test_ranges
ROSTER = ("got", "common_claim", "counterfact", "pubmedqa_artificial",
          "pubmedqa_labeled", "claude_anatomy", "harmful_harmless",
          "wildjailbreak_vanilla", "wildjailbreak_adversarial", "xstest",
          "social_iqa", "claude_vds")
N_FOLDS = 5
N_KEEP = 4                 # components stored per sample
SEED = 20260828


def _load(model: str, pert: str) -> dict:
    """Every sample of the roster at this model's layer, in ROSTER order.

    groups=True derives the fold-group id from each source's stored ids,
    which is what keeps a wrapped true/false pair - or two rows of one
    wildjailbreak prompt - inside a single fold.
    """
    units = load_datasets(LAYER[model], "last", ("raw",),
                          COLLECTION[(model, pert)], groups=True)
    X, y, ds, grp = [], [], [], []
    for name in ROSTER:
        if name not in units:
            print(f"  !! {model}/{pert}: {name} missing, skipped", flush=True)
            continue
        u = units[name]
        X.append(np.asarray(u["blocks"]["raw"], dtype=np.float32))
        y.append(np.asarray(u["y"], dtype=np.int8))
        ds.append(np.full(len(u["y"]), name))
        grp.append(np.asarray(u["group"]).astype(str))
    out = {"X": np.concatenate(X), "y": np.concatenate(y),
           "dataset": np.concatenate(ds), "group": np.concatenate(grp)}
    print(f"  {model}/{pert}: {out['X'].shape[0]} samples x "
          f"{out['X'].shape[1]} dims", flush=True)
    return out


def _pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact PCA of every row: (mean, all eigenvectors, variance ratios).

    The covariance is accumulated in float64 over row chunks - 200k x 3584
    in one float64 block would be 5.7 GB, the covariance is 103 MB - and
    eigh gives the complete spectrum, so the same call serves the figure's
    top two components and the "all PCs" readout.
    """
    n, d = X.shape
    mean = X.mean(axis=0, dtype=np.float64)
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(0, n, 20_000):
        chunk = X[i:i + 20_000].astype(np.float64) - mean
        cov += chunk.T @ chunk
    cov /= n - 1
    ev, vec = np.linalg.eigh(cov)
    ev, vec = ev[::-1], vec[:, ::-1]                    # descending
    ev = np.clip(ev, 0, None)
    # deterministic sign: largest-magnitude loading of each component is +
    flip = np.sign(vec[np.abs(vec).argmax(axis=0), np.arange(d)])
    vec = vec * flip
    return mean, vec, ev / ev.sum()


def _project(X: np.ndarray, mean: np.ndarray, vec: np.ndarray,
             k: int | None = None) -> np.ndarray:
    """Scores on the first k components (all of them when k is None)."""
    v = vec if k is None else vec[:, :k]
    out = np.empty((len(X), v.shape[1]), dtype=np.float32)
    for i in range(0, len(X), 20_000):
        out[i:i + 20_000] = ((X[i:i + 20_000].astype(np.float64) - mean)
                             @ v).astype(np.float32)
    return out


def _fit_readouts(Zs: np.ndarray, y: np.ndarray, ds: np.ndarray,
                  group: np.ndarray, model: str) -> list[dict]:
    """Grouped 5-fold held-out readouts of the PCA map.

    `Zs` arrives already divided by the standard deviation of PC1 - one
    scalar for the whole matrix, so the top-2 and all-PC feature sets stay
    nested and the L2 penalty means the same thing on Gemma (PC1 sd in the
    thousands) and Qwen (tens). Per-component standardization would
    whiten, which is a different model, not a rescaling.
    """
    ds_code = pd.Categorical(ds, categories=list(ROSTER)).codes
    folds = list(GroupKFold(n_splits=N_FOLDS).split(Zs, ds_code, group))
    rows = []
    for view, k in (("first 2 PCs", 2), ("all PCs", Zs.shape[1])):
        F = Zs[:, :k]
        for target, t in (("dataset identity", ds_code), ("label", y)):
            acc, bal, f1, auc = [], [], [], []
            for tr, te in folds:
                clf = LogisticRegression(max_iter=3000, C=1.0,
                                         random_state=SEED)
                clf.fit(F[tr], t[tr])
                p = clf.predict(F[te])
                acc.append(accuracy_score(t[te], p))
                bal.append(balanced_accuracy_score(t[te], p))
                f1.append(f1_score(t[te], p, average="macro"))
                if target == "label":
                    auc.append(roc_auc_score(
                        t[te], clf.predict_proba(F[te])[:, 1]))
            base = pd.Series(t).value_counts(normalize=True).iloc[0]
            rows.append({"model": model, "target": target, "features": view,
                         "n_features": k, "n": len(t),
                         "accuracy": float(np.mean(acc)),
                         "accuracy_sd": float(np.std(acc)),
                         "balanced_accuracy": float(np.mean(bal)),
                         "macro_f1": float(np.mean(f1)),
                         "auc": float(np.mean(auc)) if auc else np.nan,
                         "majority_baseline": float(base)})
            print(f"    {model} {target:16s} {view:11s} "
                  f"acc={rows[-1]['accuracy']:.3f} "
                  f"bal={rows[-1]['balanced_accuracy']:.3f} "
                  f"auc={rows[-1]['auc']:.3f}", flush=True)
    return rows


def _within_dataset(Zs: np.ndarray, y: np.ndarray, ds: np.ndarray,
                    group: np.ndarray, model: str) -> list[dict]:
    """The label readout again, one dataset at a time.

    The pooled label number is diluted by design: the label means a
    different thing per dataset (the statement is true / the answer is yes
    / the request is safe), so a single global boundary is not the thing
    the representation would have to encode.
    """
    rows = []
    for name in ROSTER:
        m = ds == name
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        Zd, t, g = Zs[m], y[m], group[m]
        n_splits = min(N_FOLDS, len(np.unique(g)))
        if n_splits < 2:
            continue
        for view, k in (("first 2 PCs", 2), ("all PCs", Zs.shape[1])):
            auc = []
            for tr, te in GroupKFold(n_splits=n_splits).split(Zd, t, g):
                if len(np.unique(t[tr])) < 2 or len(np.unique(t[te])) < 2:
                    continue
                clf = LogisticRegression(max_iter=3000, C=1.0,
                                         random_state=SEED)
                clf.fit(Zd[tr, :k], t[tr])
                auc.append(roc_auc_score(
                    t[te], clf.predict_proba(Zd[te, :k])[:, 1]))
            if auc:
                rows.append({"model": model, "dataset": name,
                             "features": view, "n": int(m.sum()),
                             "auc": float(np.mean(auc))})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, spectra, readouts, within = [], [], [], []
    for model in ("gemma", "qwen"):
        print(f"=== {model} layer {LAYER[model]} ===", flush=True)
        bare = _load(model, "bare")
        mean, vec, ratio = _pca(bare["X"])
        print(f"  PCA on all {len(bare['X'])} Original samples; "
              f"PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}, "
              f"90% at {int(np.searchsorted(np.cumsum(ratio), 0.9)) + 1} PCs",
              flush=True)
        spectra.append(pd.DataFrame({
            "model": model, "layer": LAYER[model],
            "pc": np.arange(1, len(ratio) + 1),
            "explained_variance": ratio}))

        Z_all = _project(bare["X"], mean, vec)          # every component
        points.append(pd.DataFrame({
            "model": model, "perturbation": "bare",
            "dataset": bare["dataset"], "label": bare["y"],
            **{f"pc{i + 1}": Z_all[:, i].copy() for i in range(N_KEEP)}}))
        bare["X"] = None                                # 2.9 GB, done with it
        Z_all /= Z_all[:, 0].std()                      # in place: no copy
        readouts += _fit_readouts(Z_all, bare["y"], bare["dataset"],
                                  bare["group"], model)
        within += _within_dataset(Z_all, bare["y"], bare["dataset"],
                                  bare["group"], model)
        del Z_all, bare

        framed = _load(model, "true_false")
        Zf = _project(framed["X"], mean, vec, N_KEEP)
        points.append(pd.DataFrame({
            "model": model, "perturbation": "true_false",
            "dataset": framed["dataset"], "label": framed["y"],
            **{f"pc{i + 1}": Zf[:, i] for i in range(N_KEEP)}}))
        del Zf, framed, vec

    pts = pd.concat(points, ignore_index=True)
    np.savez_compressed(
        OUT / "figure1_activation_points.npz",
        model=pts.model.to_numpy().astype("U8"),
        perturbation=pts.perturbation.to_numpy().astype("U10"),
        dataset=pts.dataset.to_numpy().astype("U26"),
        label=pts.label.to_numpy().astype(np.int8),
        **{c: pts[c].to_numpy(np.float32) for c in
           [f"pc{i + 1}" for i in range(N_KEEP)]})
    pd.concat(spectra, ignore_index=True).to_csv(
        OUT / "figure1_activation_variance.csv", index=False)
    pd.DataFrame(readouts).to_csv(
        OUT / "figure1_activation_readouts.csv", index=False)
    print(f"\nwrote {len(pts)} projected samples", flush=True)
    print(pd.DataFrame(readouts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
