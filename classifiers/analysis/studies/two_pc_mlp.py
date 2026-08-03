"""Fit fixed-hyperparameter MLPs using only the displayed PC1/PC2 plane.

The corresponding full-dimensional MLP has already been evaluated on this
plane by ``boundary_refit.py``.  This script supplies the missing
comparison: five-fold out-of-fold predictions from an MLP trained on X2, then
a full X2 fit scored on the exact same plotting grid.

The width and Adam weight decay match the selected full-D MLP for each scope.
Holding them fixed makes the comparison isolate input dimension rather than
retuning a different model family for the two-PC representation.

    python -m classifiers.analysis.studies.two_pc_mlp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .boundary_common import ARTIFACTS, SCOPES

PARAMS = {
    "common_claim": {"hidden": 64, "weight_decay": 1e-4},
    "harmful_harmless": {"hidden": 256, "weight_decay": 1e-2},
    "wildjailbreak": {"hidden": 256, "weight_decay": 1e-4},
    "merged": {"hidden": 64, "weight_decay": 1e-4},
}

# These match classifiers/methods/fit_torch.py.  They are repeated here so
# the two-dimensional CPU fallback does not import the package's CuPy-only
# classifiers through classifiers.methods.__init__.
LR = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 200
PATIENCE = 10
VAL_FRAC = .1
MIN_AUC_GAIN = 1e-4


class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def margin(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(1)


def train(model: MLP, X: torch.Tensor, y: torch.Tensor, *, seed: int,
          weight_decay: float) -> MLP:
    y_int = y.cpu().numpy().astype(int)
    train_ids, val_ids = train_test_split(
        np.arange(len(y_int)), test_size=VAL_FRAC, random_state=seed,
        stratify=y_int)
    tr = torch.as_tensor(train_ids, device=X.device)
    val = torch.as_tensor(val_ids, device=X.device)
    Xtr, ytr = X[tr], y[tr]
    Xval, yval = X[val], y_int[val_ids]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_auc, best_state, bad = -np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        order = torch.randperm(len(tr), generator=generator).to(X.device)
        for start in range(0, len(tr), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model.margin(Xtr[batch]), ytr[batch])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            margin = model.margin(Xval).double().cpu().numpy()
        auc = roc_auc_score(yval, margin)
        if auc > best_auc + MIN_AUC_GAIN:
            best_auc, bad = auc, 0
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def standardize_fit(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = X.mean(0)
    scale = X.std(0, unbiased=False)
    return mean, torch.where(scale == 0, torch.ones_like(scale), scale)


def fit_one(X: torch.Tensor, y: np.ndarray, train_ids: np.ndarray,
            test_ids: np.ndarray, *, hidden: int, weight_decay: float,
            seed: int) -> np.ndarray:
    tr = torch.as_tensor(train_ids, device=X.device)
    te = torch.as_tensor(test_ids, device=X.device)
    mean, scale = standardize_fit(X[tr])
    yt = torch.as_tensor(y[train_ids].astype(np.float32), device=X.device)
    torch.manual_seed(seed)
    model = MLP(hidden).to(X.device)
    model = train(model, (X[tr] - mean) / scale, yt, seed=seed,
                  weight_decay=weight_decay)
    with torch.no_grad():
        return model.margin((X[te] - mean) / scale).cpu().numpy()


def fit(scope: str, device: str) -> Path:
    source = ARTIFACTS / f"{scope}.npz"
    with np.load(source, allow_pickle=True) as data:
        X2 = np.asarray(data["X2"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int8)
        grid = np.asarray(data["grid"], dtype=np.float32)

    chosen = PARAMS[scope]
    Xt = torch.as_tensor(X2, device=device)
    oof = np.empty(len(y), dtype=np.float32)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train_ids, test_ids) in enumerate(folds.split(X2, y), 1):
        seed = 1009 * (fold - 1)
        oof[test_ids] = fit_one(
            Xt, y, train_ids, test_ids, hidden=chosen["hidden"],
            weight_decay=chosen["weight_decay"], seed=seed)
        print(f"{scope}: fold {fold}/5 complete", flush=True)

    mean, scale = standardize_fit(Xt)
    yt = torch.as_tensor(y.astype(np.float32), device=device)
    torch.manual_seed(0)
    model = MLP(chosen["hidden"]).to(device)
    model = train(model, (Xt - mean) / scale, yt, seed=0,
                  weight_decay=chosen["weight_decay"])
    with torch.no_grad():
        grid_t = torch.as_tensor(grid, device=device)
        grid_score = model.margin(
            (grid_t - mean) / scale).cpu().numpy().astype(np.float32)

    auc = float(roc_auc_score(y, oof))
    target = ARTIFACTS / f"{scope}_two_pc_mlp.npz"
    np.savez_compressed(
        target, score=grid_score, oof_score=oof, cv_auc=auc,
        hidden=np.int32(chosen["hidden"]),
        weight_decay=np.float32(chosen["weight_decay"]),
        standardize_mean=mean.cpu().numpy().astype(np.float32),
        standardize_scale=scale.cpu().numpy().astype(np.float32),
    )
    print(f"{scope}: wrote {target} (CV AUC {auc:.6f})", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="auto uses CUDA when available and otherwise CPU")
    parser.add_argument("--threads", type=int, default=5,
                        help="PyTorch CPU threads (ignored by CUDA kernels)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu"
              ) if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    fit(args.scope, device)


if __name__ == "__main__":
    main()
