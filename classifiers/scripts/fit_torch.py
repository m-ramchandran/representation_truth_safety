"""Torch methods: a small MLP, random slopes, and a mixture of experts.

All three train on the GPU with PyTorch and follow the package's shared
fit signature (numpy in / numpy out, dense input only — the runners
densify). The MLP is an ordinary method and joins the per-dataset grid;
random_slopes and moe exist for pooled multi-dataset corpora:
random_slopes needs a `groups` array naming each sample's grouping unit
— its source dataset, or its task for the runners' per-task variant
(random_slopes_task), which fits one deviation per task instead — and
moe's expert-count grid is chosen by the runner from the pool's
composition (it never sees dataset identity itself, so it transfers to
held-out datasets unchanged).

Margins (native log-odds for all three, so proba = sigmoid(margin), no
Platt calibration):

  - mlp : the pre-sigmoid output logit.
  - random_slopes : the linear score. Samples from a known dataset get
        shared weights + that dataset's deviations; samples whose dataset
        is unknown (a held-out dataset) get the shared weights only.
  - moe : the log-odds of the mixture probability
        p = sum_k gate_k(x) * sigmoid(expert_k(x)), computed stably from
        logits via logsumexp.

Training (`_train`, shared by all three): Adam on shuffled minibatches,
early stopping on a stratified 10% slice of the data passed in — during
CV tuning that data is the fold's train part, so the fold's held-out part
never touches training. The final model likewise trains on 90% of X_train
with the remaining 10% deciding when to stop (unlike the cuML methods,
which refit on 100% — early stopping needs something to watch). Seeds fix
the initialization, the stopping split, and the batch order; GPU
reductions are still not bitwise reproducible, so repeated runs agree
statistically, not to the last digit.

CV tuning follows the package rule: every candidate is fit on every fold,
the winner is the best mean per-fold AUC on margins, and the winner's
fold predictions are returned as the CV scores — nothing is refit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .base import DEFAULT_INNER_CV, FittedClassifier, Fold, ensure_folds

DEVICE = "cuda"

# --- hyperparameter grids ----------------------------------------------------

# MLP: one hidden layer; the grid crosses width with Adam weight decay.
DEFAULT_MLP_HIDDENS = (64, 256)
DEFAULT_MLP_WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)

# Random slopes: L2 strength on the shared weights, and the deviation
# penalty as a multiple of it (deviations are always penalized harder, so
# they only pick up what genuinely differs between datasets). The shared
# grid is chosen to match the effective range of the l2reg C grid: the
# penalty here sits next to a MEAN loss, so lambda corresponds to
# 1 / (2 C n) — at pooled n ~ 128k this grid spans C ~ 0.0004-39, the
# informative middle of DEFAULT_L2_CS. (The weaker end of the C grid is
# partly covered by early stopping's own regularization.)
DEFAULT_SHARED_L2 = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_DEV_MULTIPLIERS = (10.0, 100.0)

# MoE: expert counts. Pooled runners extend this with the training pool's
# number of tasks and number of datasets (features.moe_k_grid).
DEFAULT_MOE_KS = (2, 4)
MOE_WEIGHT_DECAY = 1e-4
# Load-balancing strength: penalizes concentration of the average gate
# weights (negative entropy) so the gate cannot collapse onto one expert.
MOE_BALANCE = 0.01

# --- training loop constants -------------------------------------------------

LR = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 200
PATIENCE = 10          # epochs without validation-AUC improvement
VAL_FRAC = 0.1         # early-stopping slice
MIN_AUC_GAIN = 1e-4    # improvement below this doesn't reset patience


def _as_tensor(X) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=DEVICE)


def _standardize_fit(X: torch.Tensor):
    """Per-column (mean, scale); zero-variance columns get scale 1, as in
    the cuML methods' _fit_scale."""
    mean = X.mean(0)
    scale = X.std(0, unbiased=False)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return mean, scale


# --- the three models --------------------------------------------------------
#
# Each implements margin(X, g=None) — the log-odds score — and
# loss(X, y, g=None) — the per-batch training loss — so one training loop
# and one wrapper serve all three. g is the per-sample dataset code (int64)
# and is ignored by the models that don't use it.

class _MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def margin(self, X, g=None):
        return self.net(X).squeeze(1)

    def loss(self, X, y, g=None):
        return F.binary_cross_entropy_with_logits(self.margin(X), y)


class _RandomSlopes(nn.Module):
    """Logistic regression with shared weights plus per-dataset slope and
    intercept deviations, both deviation blocks L2-penalized more strongly
    than the shared part. A sample with code -1 (unknown dataset) is scored
    with the shared weights only."""

    def __init__(self, d: int, n_groups: int, lam_shared: float,
                 lam_dev: float):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(1))
        self.dw = nn.Parameter(torch.zeros(n_groups, d))
        self.db = nn.Parameter(torch.zeros(n_groups))
        self.lam_shared = lam_shared
        self.lam_dev = lam_dev

    def margin(self, X, g=None):
        m = X @ self.w + self.b
        if g is not None:
            known = g >= 0
            gk = g.clamp(min=0)          # safe index; masked out below
            dev = (X * self.dw[gk]).sum(1) + self.db[gk]
            m = m + torch.where(known, dev, torch.zeros_like(dev))
        return m

    def loss(self, X, y, g=None):
        bce = F.binary_cross_entropy_with_logits(self.margin(X, g), y)
        return (bce + self.lam_shared * (self.w ** 2).sum()
                + self.lam_dev * ((self.dw ** 2).sum()
                                  + (self.db ** 2).sum()))


class _MoE(nn.Module):
    """K linear-logistic experts and a linear softmax gate, both on the
    features — the gate never sees dataset identity. log p and log(1-p) of
    the mixture come from logsumexp over experts."""

    def __init__(self, d: int, k: int):
        super().__init__()
        self.experts = nn.Linear(d, k)
        self.gate = nn.Linear(d, k)

    def _log_pq(self, X):
        g = F.log_softmax(self.gate(X), dim=1)
        z = self.experts(X)
        lp = torch.logsumexp(g + F.logsigmoid(z), dim=1)     # log p
        lq = torch.logsumexp(g + F.logsigmoid(-z), dim=1)    # log (1 - p)
        return lp, lq

    def margin(self, X, g=None):
        lp, lq = self._log_pq(X)
        return lp - lq

    def loss(self, X, y, g=None):
        lp, lq = self._log_pq(X)
        nll = -(y * lp + (1 - y) * lq).mean()
        return nll + MOE_BALANCE * self._balance(X)

    def _balance(self, X):
        """Load balancing: the average gate distribution should not
        concentrate on one expert, so its negative entropy is penalized.
        Shared with the hard-routing variant so the two differ ONLY in
        routing."""
        gbar = F.softmax(self.gate(X), dim=1).mean(0)
        return (gbar * torch.log(gbar + 1e-12)).sum()


class _MoEHard(_MoE):
    """The same K experts and gate, routed HARD: each sample is scored by
    its single arg-max expert, so the margin is that expert's own logit.

    A straight-through estimator keeps the gate trainable — forward uses
    the one-hot selection, backward the softmax it came from. Everything
    else matches `_MoE`, so the pair isolates the routing. Balancing
    matters more here: an expert that stops winning any sample stops
    receiving gradient at all."""

    def _route(self, X):
        """(straight-through one-hot routing weights, expert logits)."""
        soft = F.softmax(self.gate(X), dim=1)
        hard = F.one_hot(soft.argmax(dim=1), soft.shape[1]).to(soft.dtype)
        return hard + soft - soft.detach(), self.experts(X)

    def margin(self, X, g=None):
        w, z = self._route(X)
        return (w * z).sum(dim=1)

    def loss(self, X, y, g=None):
        bce = F.binary_cross_entropy_with_logits(self.margin(X), y)
        return bce + MOE_BALANCE * self._balance(X)


# --- training and tuning -----------------------------------------------------

def _train(model, X, y, g=None, *, seed: int, weight_decay: float = 0.0):
    """Adam on shuffled minibatches with early stopping: a seeded,
    stratified VAL_FRAC slice of the given data is held out, training runs
    on the rest, and the epoch with the best validation margin AUC wins
    (stop after PATIENCE epochs without improvement). Stratification is on
    (dataset, label) when g is given, else on the label. Returns the model
    with the best epoch's weights restored."""
    y_int = y.cpu().numpy().astype(int)
    strat = y_int if g is None else 2 * g.cpu().numpy() + y_int
    counts = np.bincount(strat)
    if counts[counts > 0].min() < 2:     # a (dataset, label) cell of one
        strat = y_int                    # sample can't stratify; fall back
    tr, val = train_test_split(np.arange(len(y_int)), test_size=VAL_FRAC,
                               random_state=seed, stratify=strat)
    tr_t = torch.as_tensor(tr, device=X.device)
    val_t = torch.as_tensor(val, device=X.device)
    Xtr, ytr = X[tr_t], y[tr_t]
    Xval = X[val_t]
    yval = y_int[val]
    gtr = g[tr_t] if g is not None else None
    gval = g[val_t] if g is not None else None

    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, bad = -np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        perm = torch.randperm(len(tr), generator=gen).to(X.device)
        for i in range(0, len(tr), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            model.loss(Xtr[b], ytr[b], gtr[b] if gtr is not None else None
                       ).backward()
            opt.step()
        with torch.no_grad():
            m = model.margin(Xval, gval).double().cpu().numpy()
        auc = roc_auc_score(yval, m)
        if auc > best_auc + MIN_AUC_GAIN:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def _tune_cv_torch(fit_candidate, grid, Xt, y, folds, seed, g=None):
    """CV-tune the torch methods, mirroring _tune_cv_gpu: standardize each
    fold on its train part, train one candidate per grid value via
    `fit_candidate(value, Xtr, ytr, gtr, seed)`, and score the held-out
    part. The winner is the best mean per-fold margin AUC; its fold
    predictions are kept as the CV scores — nothing is refit. Returns
    (best_index, cv_scores)."""
    yt = torch.as_tensor(y.astype(np.float32), device=DEVICE)
    cv = [{"proba": np.empty(len(y)), "margin": np.empty(len(y))}
          for _ in grid]
    aucs = np.empty((len(grid), len(folds)))
    for f, (tr, te) in enumerate(folds):
        tr_t = torch.as_tensor(tr, device=DEVICE)
        te_t = torch.as_tensor(te, device=DEVICE)
        mean, scale = _standardize_fit(Xt[tr_t])
        Xtr = (Xt[tr_t] - mean) / scale
        Xte = (Xt[te_t] - mean) / scale
        gtr = g[tr_t] if g is not None else None
        gte = g[te_t] if g is not None else None
        for i, value in enumerate(grid):
            model = fit_candidate(value, Xtr, yt[tr_t], gtr,
                                  seed + 1009 * f + i)
            with torch.no_grad():
                m = model.margin(Xte, gte).double().cpu().numpy()
            cv[i]["margin"][te] = m
            cv[i]["proba"][te] = expit(m)
            aucs[i, f] = roc_auc_score(y[te], m)
    best = int(aucs.mean(axis=1).argmax())
    return best, cv[best]


def _fit_final(fit_candidate, value, Xt, y_train, g, seed: int):
    """Train the winning candidate on all of X_train, standardized on all
    of it (early stopping still carves out its internal validation slice).
    Returns (model, mean, scale) for a TorchClassifier."""
    mean, scale = _standardize_fit(Xt)
    yt = torch.as_tensor(y_train.astype(np.float32), device=DEVICE)
    model = fit_candidate(value, (Xt - mean) / scale, yt, g, seed)
    return model, mean, scale


# --- fitted wrappers ---------------------------------------------------------

class TorchClassifier(FittedClassifier):
    """Wraps a trained torch model and the standardization fit on its
    training data. `predict` takes numpy input and returns numpy score
    arrays; the margin is a native log-odds for all three methods."""

    def __init__(self, model, mean, scale, model_type: str):
        self.model = model
        self.mean = mean
        self.scale = scale
        self.model_type = model_type

    def _margin(self, X, g=None) -> np.ndarray:
        with torch.no_grad():
            Xs = (_as_tensor(X) - self.mean) / self.scale
            return self.model.margin(Xs, g).double().cpu().numpy()

    def predict(self, X) -> dict:
        m = self._margin(X)
        return {"proba": expit(m), "margin": m}

    def save_arrays(self) -> dict:
        out = super().save_arrays()
        out["standardize_mean"] = self.mean.cpu().numpy()
        out["standardize_scale"] = self.scale.cpu().numpy()
        if isinstance(self.model, _MoE):
            # Gate and experts are both linear on the standardized
            # features: weight rows are (K, d), biases (K,).
            out["gate_w"] = self.model.gate.weight.detach().cpu().numpy()
            out["gate_b"] = self.model.gate.bias.detach().cpu().numpy()
            out["expert_w"] = self.model.experts.weight.detach().cpu().numpy()
            out["expert_b"] = self.model.experts.bias.detach().cpu().numpy()
        return out


class RandomSlopesClassifier(TorchClassifier):
    """Random slopes: plain `predict` scores with the shared weights only
    (the right call for data from datasets the model never saw);
    `predict_with_groups` applies each known dataset's deviations, falling
    back to shared-only per sample for unknown names."""

    def __init__(self, model, mean, scale, group_labels: np.ndarray):
        super().__init__(model, mean, scale, "random_slopes")
        self.group_labels = group_labels     # dataset name -> deviation row

    def get_coef(self) -> np.ndarray:
        # The shared weights (standardized-feature space) — the part of
        # the model that transfers to unseen datasets; the per-dataset
        # deviations are excluded.
        return self.model.w.detach().cpu().numpy()

    def save_arrays(self) -> dict:
        # The deviations ride along for the what-do-the-deviations-absorb
        # analyses: dev_w[i] / dev_b[i] belong to dataset group_labels[i].
        out = super().save_arrays()
        out["dev_w"] = self.model.dw.detach().cpu().numpy()
        out["dev_b"] = self.model.db.detach().cpu().numpy()
        out["group_labels"] = np.asarray(self.group_labels)
        return out

    def predict_with_groups(self, X, groups) -> dict:
        codes = np.searchsorted(self.group_labels, groups)
        codes = np.where(
            (codes < len(self.group_labels))
            & (self.group_labels[codes.clip(max=len(self.group_labels) - 1)]
               == np.asarray(groups)),
            codes, -1)
        g = torch.as_tensor(codes.astype(np.int64), device=DEVICE)
        m = self._margin(X, g)
        return {"proba": expit(m), "margin": m}


# --- fit functions -----------------------------------------------------------

def fit_mlp(
    X_train, y_train: np.ndarray,
    X_test=None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    hiddens=DEFAULT_MLP_HIDDENS,
    weight_decays=DEFAULT_MLP_WEIGHT_DECAYS,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, TorchClassifier]:
    """One-hidden-layer MLP. CV over `folds` picks (hidden width, weight
    decay) from the crossed grids. `stratify_on` sets what self-built folds
    are stratified on (default y_train); ignored when `folds` is given."""
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    grid = [(h, wd) for h in hiddens for wd in weight_decays]

    def fit_candidate(value, Xtr, ytr, gtr, cand_seed):
        h, wd = value
        torch.manual_seed(cand_seed)
        model = _MLP(Xtr.shape[1], h).to(DEVICE)
        return _train(model, Xtr, ytr, seed=cand_seed, weight_decay=wd)

    Xt = _as_tensor(X_train)
    best, cv = _tune_cv_torch(fit_candidate, grid, Xt, y_train, folds, seed)
    h, wd = grid[best]
    model, mean, scale = _fit_final(fit_candidate, grid[best], Xt, y_train,
                                    None, seed)
    fitted = TorchClassifier(model, mean, scale, "mlp")
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"hidden": h, "weight_decay": wd}, fitted


def fit_random_slopes(
    X_train, y_train: np.ndarray,
    X_test=None,
    *, seed: int,
    groups: np.ndarray,
    groups_test: Optional[np.ndarray] = None,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    shared_l2=DEFAULT_SHARED_L2,
    dev_multipliers=DEFAULT_DEV_MULTIPLIERS,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, RandomSlopesClassifier]:
    """Random slopes on a pooled corpus: `groups` names each training
    sample's grouping unit (its source dataset — or its task, for the
    per-task variant). CV over `folds` picks (shared L2, deviation
    L2 = multiplier * shared) from the crossed grids; CV predictions use
    the deviations (every fold sees all datasets). X_test is scored with
    deviations for the names in `groups_test` when given (unknown names
    fall back to shared-only per sample), else with the shared weights
    only — the right call for held-out datasets. Self-built folds are
    stratified on the joint (dataset, label) key unless `stratify_on`
    overrides."""
    groups = np.asarray(groups).astype(str)
    group_names, codes = np.unique(groups, return_inverse=True)
    if stratify_on is None:
        stratify_on = np.char.add(groups, y_train.astype(str))
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    grid = [(l2, l2 * m) for l2 in shared_l2 for m in dev_multipliers]
    g = torch.as_tensor(codes.astype(np.int64), device=DEVICE)

    def fit_candidate(value, Xtr, ytr, gtr, cand_seed):
        lam_shared, lam_dev = value
        torch.manual_seed(cand_seed)
        model = _RandomSlopes(Xtr.shape[1], len(group_names), lam_shared,
                              lam_dev).to(DEVICE)
        return _train(model, Xtr, ytr, gtr, seed=cand_seed)

    Xt = _as_tensor(X_train)
    best, cv = _tune_cv_torch(fit_candidate, grid, Xt, y_train, folds, seed,
                              g=g)
    lam_shared, lam_dev = grid[best]
    model, mean, scale = _fit_final(fit_candidate, grid[best], Xt, y_train,
                                    g, seed)
    fitted = RandomSlopesClassifier(model, mean, scale, group_names)
    test = None
    if X_test is not None:
        test = (fitted.predict_with_groups(X_test, groups_test)
                if groups_test is not None else fitted.predict(X_test))
    return (cv, test,
            {"shared_l2": lam_shared, "dev_l2": lam_dev}, fitted)


def _fit_moe(
    cls, name: str,
    X_train, y_train: np.ndarray,
    X_test=None,
    *, seed: int,
    folds: Optional[list[Fold]] = None,
    stratify_on: Optional[np.ndarray] = None,
    Ks=DEFAULT_MOE_KS,
    inner_cv: int = DEFAULT_INNER_CV,
    inner_n_jobs: int = 1,      # unused; kept so every method takes the same arguments
) -> tuple[dict, Optional[dict], dict, TorchClassifier]:
    """Shared body of fit_moe / fit_moe_hard: the two differ only in the
    module class, so the tuning grid, weight decay, and saved arrays stay
    identical by construction."""
    folds = ensure_folds(X_train, y_train, folds, seed=seed,
                         n_splits=inner_cv, stratify_on=stratify_on)
    grid = sorted({int(k) for k in Ks})

    def fit_candidate(k, Xtr, ytr, gtr, cand_seed):
        torch.manual_seed(cand_seed)
        model = cls(Xtr.shape[1], k).to(DEVICE)
        return _train(model, Xtr, ytr, seed=cand_seed,
                      weight_decay=MOE_WEIGHT_DECAY)

    Xt = _as_tensor(X_train)
    best, cv = _tune_cv_torch(fit_candidate, grid, Xt, y_train, folds, seed)
    model, mean, scale = _fit_final(fit_candidate, grid[best], Xt, y_train,
                                    None, seed)
    fitted = TorchClassifier(model, mean, scale, name)
    test = fitted.predict(X_test) if X_test is not None else None
    return cv, test, {"K": grid[best]}, fitted


def fit_moe(X_train, y_train, X_test=None, **kwargs):
    """Mixture of experts, SOFT routing: every sample is scored by the
    gate-weighted mixture of all K experts. CV over `folds` picks the
    expert count from `Ks`; weight decay is fixed (MOE_WEIGHT_DECAY).
    `stratify_on` sets what self-built folds are stratified on (default
    y_train); ignored when `folds` is given."""
    return _fit_moe(_MoE, "moe", X_train, y_train, X_test, **kwargs)


def fit_moe_hard(X_train, y_train, X_test=None, **kwargs):
    """Mixture of experts, HARD routing: each sample is scored by its
    single arg-max expert, trained through a straight-through estimator
    (see `_MoEHard`). Same grid and same saved arrays as `fit_moe`, so the
    pair isolates what routing choice costs or buys."""
    return _fit_moe(_MoEHard, "moe_hard", X_train, y_train, X_test, **kwargs)
