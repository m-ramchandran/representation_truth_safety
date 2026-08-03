"""Fixed-hyperparameter full-data fits behind the layer-17 boundary figures.

Needs a GPU. One invocation handles one scope, so the four scopes can run as
independent jobs. Each reuses the hyperparameters already selected by the
main runs (no CV here), persists reloadable model state, evaluates every
classifier on an all-data PCA mean-plane grid, and writes a compact plotting
artifact. ``--combine`` assembles the four artifacts into the final figure.

Usage:
    python -m classifiers.analysis.studies.boundary_refit --scope merged
    python -m classifiers.analysis.studies.boundary_refit --combine
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifiers.data import load_datasets

REPO_ROOT = Path(__file__).resolve().parents[3]

COLLECTION = "data_unwrap"
LAYER = 17
SCOPES = ("common_claim", "harmful_harmless", "wildjailbreak", "merged")
MERGED_ORDER = (
    "common_claim", "counterfact", "got", "harmful_harmless",
    "pubmedqa_artificial", "pubmedqa_labeled", "wildjailbreak", "xstest",
    "social_iqa", "claude_anatomy", "claude_vds",
)
PARAMS = {
    "common_claim": {
        "l1reg": {"C": .026826957952797246},
        "l2reg": {"C": .001},
        "svm_linear": {"C": .0001},
        "qda": {"shrinkage": .6}, "knn": {"n_neighbors": 100},
        "svm_rbf": {"C": 3.1622776601683795},
        "mlp": {"hidden": 64, "weight_decay": 1e-4},
    },
    "harmful_harmless": {
        "l1reg": {"C": .1}, "l2reg": {"C": .001},
        "svm_linear": {"C": .0001},
        "qda": {"shrinkage": .9}, "knn": {"n_neighbors": 100},
        "svm_rbf": {"C": 1.},
        "mlp": {"hidden": 256, "weight_decay": .01},
    },
    "wildjailbreak": {
        "l1reg": {"C": .1}, "l2reg": {"C": .0031622776601683794},
        "svm_linear": {"C": .0007196856730011522},
        "qda": {"shrinkage": "auto"}, "knn": {"n_neighbors": 25},
        "svm_rbf": {"C": 3.1622776601683795},
        "mlp": {"hidden": 256, "weight_decay": 1e-4},
    },
    "merged": {
        "l1reg": {"C": .0517947467923121},
        "l2reg": {"C": .0031622776601683794},
        "svm_linear": {"C": .0007196856730011522},
        "qda": {"shrinkage": .15}, "knn": {"n_neighbors": 25},
        "svm_rbf": {"C": .31622776601683794},
        "mlp": {"hidden": 64, "weight_decay": 1e-4},
    },
}
METHOD_STYLE = [
    ("diff", "diff in means", "#172033", "-", 1.8),
    ("l1reg", "l1 logistic", "#DC2626", "--", 2.0),
    ("l2reg", "l2 logistic", "#2563EB", "-", 2.3),
    ("svm_linear", "linear SVM", "#0891B2", "--", 2.0),
    ("lda", "LDA", "#0F766E", "--", 2.1),
    ("qda", "QDA", "#EA580C", "-.", 2.2),
    ("knn", "kNN", "#667085", ":", 2.2),
    ("svm_rbf", "RBF SVM", "#7C3AED", "-", 2.5),
    ("mlp", "MLP", "#DB2777", (0, (5, 1)), 2.1),
    ("xgboost", "XGBoost", "#65A30D", (0, (3, 1, 1, 1)), 2.1),
]
OUTPUT_ROOT = Path(__file__).resolve().parent
OUT = REPO_ROOT / "classifiers" / "results" / COLLECTION / "boundary_refits"
FIGURES = OUTPUT_ROOT / "figures"
TABLES = OUTPUT_ROOT / "tables"
PRODUCTION = REPO_ROOT / "classifiers" / "results" / COLLECTION


def load_scope(scope: str) -> tuple[np.ndarray, ...]:
    data = load_datasets(LAYER, "last", ("raw",), COLLECTION)
    names = MERGED_ORDER if scope == "merged" else (scope,)
    X = np.concatenate([
        np.asarray(data[n]["blocks"]["raw"], dtype=np.float32) for n in names
    ])
    y = np.concatenate([np.asarray(data[n]["y"], dtype=np.int8) for n in names])
    sample_id = np.concatenate([
        np.asarray(data[n]["sample_id"]).astype(str) for n in names
    ])
    dataset = np.concatenate([
        np.full(len(data[n]["y"]), n) for n in names
    ])
    return X, y, sample_id, dataset


def display_plane(X: np.ndarray) -> tuple:
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X).astype(np.float32)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    X2 = pca.fit_transform(Z).astype(np.float32)
    xlo, xhi = np.quantile(X2[:, 0], [.005, .995])
    ylo, yhi = np.quantile(X2[:, 1], [.005, .995])
    padx, pady = .1 * (xhi - xlo), .1 * (yhi - ylo)
    xx, yy = np.meshgrid(np.linspace(xlo - padx, xhi + padx, 85),
                         np.linspace(ylo - pady, yhi + pady, 85))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    plane = scaler.inverse_transform(pca.inverse_transform(grid)
                                     ).astype(np.float32)
    return scaler, pca, X2, xx, yy, grid, plane


def save_linear_state(path: Path, model, mean, scale, **extra) -> None:
    import cupy as cp

    np.savez(path, coef=cp.asnumpy(model.coef_).ravel(),
             intercept=float(cp.asnumpy(model.intercept_).ravel()[0]),
             standardize_mean=cp.asnumpy(mean),
             standardize_scale=cp.asnumpy(scale), **extra)


def fit_gpu_linear(X: np.ndarray, y: np.ndarray, plane: np.ndarray,
                   kind: str, C: float) -> tuple:
    import cupy as cp
    from cuml.linear_model import LogisticRegression as CumlLogisticRegression
    from cuml.svm import LinearSVC as CumlLinearSVC
    from classifiers.methods.fit_classifiers import (
        _apply_scale, _fit_scale, _margin, _to_gpu,
    )

    Xg = _to_gpu(X)
    mean, scale = _fit_scale(Xg)
    Xs = _apply_scale(Xg, mean, scale)
    if kind in ("l1reg", "l2reg"):
        model = CumlLogisticRegression(
            C=C, penalty=kind[:2], max_iter=1000, output_type="cupy")
    else:
        model = CumlLinearSVC(C=C, output_type="cupy")
    model.fit(Xs, cp.asarray(y))
    scores = []
    for start in range(0, len(plane), 512):
        Pg = _apply_scale(_to_gpu(plane[start:start + 512]), mean, scale)
        scores.append(_margin(model, Pg))
    return model, mean, scale, np.concatenate(scores)


def qda_grid_scores(model, pca: PCA, display_scaler: StandardScaler,
                    grid: np.ndarray) -> np.ndarray:
    # The restriction of QDA's score to an affine 2-D plane is quadratic.
    probe = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.],
                      [0., -1.], [1., 1.]])
    raw = display_scaler.inverse_transform(pca.inverse_transform(probe))
    f = model.decision_function(raw)
    c = f[0]
    l1, q11 = (f[1] - f[2]) / 2, (f[1] + f[2]) / 2 - c
    l2, q22 = (f[3] - f[4]) / 2, (f[3] + f[4]) / 2 - c
    q12 = f[5] - c - l1 - l2 - q11 - q22
    z1, z2 = grid[:, 0], grid[:, 1]
    return c + l1*z1 + l2*z2 + q11*z1*z1 + q22*z2*z2 + q12*z1*z2


def production_prediction(scope: str, method: str) -> Path:
    experiment = "merged" if scope == "merged" else "per_dataset"
    suffix = "" if scope == "merged" else f"_{scope}"
    return (PRODUCTION / experiment / "predictions"
            / f"raw_last_layer17_{method}{suffix}.npz")


def knn_grid_scores(X: np.ndarray, y: np.ndarray, grid: np.ndarray,
                    display_scaler: StandardScaler, pca: PCA, k: int,
                    state_path: Path) -> np.ndarray:
    scope = state_path.stem.split("__", 1)[0]
    npz = production_prediction(scope, "knn")
    with np.load(npz, allow_pickle=True) as d:
        prefix = "full_" if scope == "merged" else ""
        mean = np.asarray(d[prefix + "standardize_mean"])
        scale = np.asarray(d[prefix + "standardize_scale"])
    Xs = (X - mean) / scale
    raw_offset = display_scaler.mean_ + display_scaler.scale_ * pca.mean_
    raw_basis = pca.components_ * display_scaler.scale_[None, :]
    a = (raw_offset - mean) / scale
    B = raw_basis / scale[None, :]
    u = Xs - a
    base_sq = np.einsum("ij,ij->i", u, u)
    proj = u @ B.T
    G = B @ B.T
    out = []
    for start in range(0, len(grid), 12):
        z = grid[start:start + 12]
        z2 = np.einsum("bi,ij,bj->b", z, G, z)
        d2 = base_sq[None, :] + z2[:, None] - 2 * z @ proj.T
        ids = np.argpartition(d2, k - 1, axis=1)[:, :k]
        out.append(y[ids].mean(axis=1))
    np.savez(state_path, n_neighbors=k, standardize_mean=mean,
             standardize_scale=scale, y=y)
    return np.concatenate(out)


def fit_rbf(X: np.ndarray, y: np.ndarray, plane: np.ndarray, C: float,
            state_path: Path) -> np.ndarray:
    import cupy as cp
    from cuml.svm import LinearSVC as CumlLinearSVC
    from classifiers.methods.fit_classifiers import (
        NYSTROM_M, _apply_scale, _fit_scale, _margin, _rbf_kernel, _to_gpu,
    )

    Xg = _to_gpu(X)
    mean, scale = _fit_scale(Xg)
    Xs = _apply_scale(Xg, mean, scale)
    n, d = Xs.shape
    gamma = 1.0 / (d * float(Xs.var()))
    if n <= NYSTROM_M:
        L = Xs
    else:
        ids = np.random.default_rng(0).choice(n, NYSTROM_M, replace=False)
        L = Xs[cp.asarray(ids)]
    K = _rbf_kernel(L, L, gamma)
    K = (K + K.T) / 2
    lam, U = cp.linalg.eigh(K)
    keep = lam > 1e-6 * float(lam[-1])
    W = U[:, keep] / cp.sqrt(lam[keep])
    del K, U, lam

    def transform(A, chunk=4096):
        return cp.concatenate([
            _rbf_kernel(A[i:i + chunk], L, gamma) @ W
            for i in range(0, len(A), chunk)
        ])

    Phi = transform(Xs)
    model = CumlLinearSVC(C=C, output_type="cupy")
    model.fit(Phi, cp.asarray(y))
    del Phi
    scores = []
    for start in range(0, len(plane), 256):
        Pg = _apply_scale(_to_gpu(plane[start:start + 256]), mean, scale)
        scores.append(_margin(model, transform(Pg)))
    np.savez(state_path, landmarks=cp.asnumpy(L), whitening=cp.asnumpy(W),
             gamma=gamma, coef=cp.asnumpy(model.coef_).ravel(),
             intercept=float(cp.asnumpy(model.intercept_).ravel()[0]),
             standardize_mean=cp.asnumpy(mean),
             standardize_scale=cp.asnumpy(scale), C=C)
    return np.concatenate(scores)


def fit_mlp(X: np.ndarray, y: np.ndarray, plane: np.ndarray, hidden: int,
            weight_decay: float, state_path: Path) -> np.ndarray:
    import torch
    from classifiers.methods.fit_torch import (
        _MLP, _as_tensor, _standardize_fit, _train,
    )

    Xt = _as_tensor(X)
    mean, scale = _standardize_fit(Xt)
    yt = torch.as_tensor(y.astype(np.float32), device=Xt.device)
    torch.manual_seed(0)
    model = _MLP(X.shape[1], hidden).to(Xt.device)
    model = _train(model, (Xt - mean) / scale, yt, seed=0,
                   weight_decay=weight_decay)
    scores = []
    with torch.no_grad():
        for start in range(0, len(plane), 1024):
            P = _as_tensor(plane[start:start + 1024])
            scores.append(model.margin((P - mean) / scale
                                       ).double().cpu().numpy())
    torch.save({
        "model_type": "mlp", "hidden": hidden,
        "weight_decay": weight_decay,
        "state_dict": {k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "standardize_mean": mean.cpu(),
        "standardize_scale": scale.cpu(),
    }, state_path)
    return np.concatenate(scores)


def fit_scope(scope: str) -> None:
    import cupy as cp
    import torch
    import xgboost

    OUT.mkdir(parents=True, exist_ok=True)
    models = OUT / "models"
    models.mkdir(exist_ok=True)
    X, y, sample_id, dataset = load_scope(scope)
    display_scaler, pca, X2, xx, yy, grid, plane = display_plane(X)
    score = {}
    threshold = {}

    # Diff-in-means.
    direction = X[y == 1].mean(0) - X[y == 0].mean(0)
    direction /= np.linalg.norm(direction) + 1e-12
    projected = X @ direction
    cut = .5 * (projected[y == 0].mean() + projected[y == 1].mean())
    score["diff"] = plane @ direction - cut
    threshold["diff"] = 0.
    np.savez(models / f"{scope}__diff.npz", direction=direction,
             threshold=cut)

    for method in ("l1reg", "l2reg", "svm_linear"):
        model, mean, scale, values = fit_gpu_linear(
            X, y, plane, method, PARAMS[scope][method]["C"])
        score[method], threshold[method] = values, 0.
        save_linear_state(models / f"{scope}__{method}.npz",
                          model, mean, scale, C=PARAMS[scope][method]["C"])
        del model
        cp.get_default_memory_pool().free_all_blocks()

    # LDA and QDA are the canonical sklearn pipelines.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lda = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        lda.fit(X, y)
        score["lda"] = lda.decision_function(plane)
        threshold["lda"] = 0.
        joblib.dump(lda, models / f"{scope}__lda.joblib", compress=0)

        qda = make_pipeline(
            StandardScaler(),
            QuadraticDiscriminantAnalysis(
                solver="eigen",
                shrinkage=PARAMS[scope]["qda"]["shrinkage"]))
        qda.fit(X, y)
        score["qda"] = qda_grid_scores(qda, pca, display_scaler, grid)
        threshold["qda"] = 0.
        joblib.dump(qda, models / f"{scope}__qda.joblib", compress=0)

    score["knn"] = knn_grid_scores(
        X, y, grid, display_scaler, pca,
        PARAMS[scope]["knn"]["n_neighbors"],
        models / f"{scope}__knn.npz")
    threshold["knn"] = .5

    score["svm_rbf"] = fit_rbf(
        X, y, plane, PARAMS[scope]["svm_rbf"]["C"],
        models / f"{scope}__svm_rbf.npz")
    threshold["svm_rbf"] = 0.
    cp.get_default_memory_pool().free_all_blocks()

    score["mlp"] = fit_mlp(
        X, y, plane, PARAMS[scope]["mlp"]["hidden"],
        PARAMS[scope]["mlp"]["weight_decay"],
        models / f"{scope}__mlp.pt")
    threshold["mlp"] = 0.
    torch.cuda.empty_cache()

    # Reuse the exact production booster.
    p = production_prediction(scope, "xgboost")
    with np.load(p, allow_pickle=True) as d:
        key = "full_booster_ubj" if scope == "merged" else "booster_ubj"
        raw = np.asarray(d[key], dtype=np.uint8).tobytes()
    booster = xgboost.Booster()
    booster.load_model(bytearray(raw))
    score["xgboost"] = booster.predict(
        xgboost.DMatrix(plane), output_margin=True)
    threshold["xgboost"] = 0.

    payload = {
        "scope": np.array(scope), "X2": X2, "y": y,
        "sample_id": sample_id, "dataset": dataset,
        "xx": xx, "yy": yy, "grid": grid,
        "pca_variance_ratio": pca.explained_variance_ratio_,
    }
    for method, values in score.items():
        payload[f"score_{method}"] = np.asarray(values, dtype=np.float32)
        payload[f"threshold_{method}"] = np.array(threshold[method])
    np.savez_compressed(OUT / f"{scope}.npz", **payload)
    print(f"saved {scope}: n={len(y):,}, methods={len(score)}", flush=True)


def combine() -> None:
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 10.3),
                             layout="constrained")
    rows = []
    for ax, scope in zip(axes.ravel(), SCOPES):
        with np.load(OUT / f"{scope}.npz", allow_pickle=True) as d:
            X2, y, xx, yy = d["X2"], d["y"], d["xx"], d["yy"]
            evr = d["pca_variance_ratio"]
            size = .45 if len(y) > 100_000 else (1.1 if len(y) > 20_000 else 5)
            alpha = .09 if len(y) > 100_000 else (.14 if len(y) > 20_000 else .28)
            ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm",
                       vmin=0, vmax=1, s=size, alpha=alpha,
                       edgecolor="none", rasterized=True, zorder=1)
            missing = []
            for method, display, color, linestyle, width in METHOD_STYLE:
                values = d[f"score_{method}"]
                cut = float(d[f"threshold_{method}"])
                crosses = values.min() <= cut <= values.max()
                if crosses:
                    ax.contour(xx, yy, values.reshape(xx.shape),
                               levels=[cut], colors=[color],
                               linestyles=[linestyle],
                               linewidths=[width], zorder=3)
                else:
                    missing.append(display)
                rows.append({
                    "scope": scope, "method": method, "layer": LAYER,
                    "n_points": len(y), "pca_variance_2d": evr.sum(),
                    "boundary_crosses_mean_plane": crosses,
                })
        if missing:
            ax.text(.02, .02, "No crossing: " + ", ".join(missing),
                    transform=ax.transAxes, fontsize=8, color="#172033")
        ax.set_title(f"{scope.replace('_', ' ')} · all {len(y):,} points",
                     fontsize=15, color="#172033", fontweight="bold")
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=width,
               label=display)
        for _, display, color, linestyle, width in METHOD_STYLE
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, handlelength=3.2, fontsize=9.5)
    fig.suptitle(
        "All ten full-data classifiers sliced through their all-data PCA mean plane",
        fontsize=18, color="#172033", fontweight="bold")
    stem = "decision-boundary-overlays-fullspace-all-data"
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(
        TABLES / "decision_boundary_overlays_fullspace_all_data.csv",
        index=False)
    print(FIGURES / f"{stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=SCOPES)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    if args.combine:
        combine()
    elif args.scope:
        fit_scope(args.scope)
    else:
        ap.error("pass --scope or --combine")


if __name__ == "__main__":
    main()
