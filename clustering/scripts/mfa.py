"""Mixture of factor analyzers (MFA), hand-written EM on the GPU when available.

Each component c models activations as low-rank + diagonal noise:
    x ~ N(mu_c, Lam_c Lam_c^T + Psi),   Lam_c is (D, q),  Psi diagonal, shared.

Efficiency comes from never touching a D x D covariance:
  - Woodbury identity inverts the q x q matrix M = I + Lam^T Psi^-1 Lam instead.
  - matrix-determinant lemma gives log|Sigma| = sum(log Psi) + log|M|.
So cost scales with the factor count q (6-18), not D (2560). Everything is
batched over components and points and chunked over N to bound memory.

`Psi` is shared across components (Ghahramani & Hinton 1996). Means and loadings
are updated jointly via the augmented [Lam | mu] with z augmented by a constant 1.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.cluster import KMeans

from .methods import ClusterResult

LOG2PI = math.log(2 * math.pi)


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MFA:
    """Fitted MFA. Exposes predict / score_samples like an sklearn estimator."""

    def __init__(self, pi, mu, Lam, Psi, device, dtype):
        self.pi, self.mu, self.Lam, self.Psi = pi, mu, Lam, Psi  # torch tensors
        self.device, self.dtype = device, dtype

    # --- per-component Woodbury pieces, recomputed from current params ---
    def _woodbury(self):
        Lam, Psi = self.Lam, self.Psi
        q = Lam.shape[-1]
        psi_inv = 1.0 / Psi                                   # (D,)
        LtPsiInv = Lam.transpose(1, 2) * psi_inv              # (C, q, D)
        M = torch.eye(q, device=self.device, dtype=self.dtype) + LtPsiInv @ Lam  # (C,q,q)
        L = torch.linalg.cholesky(M)
        logdetM = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)    # (C,)
        Minv = torch.cholesky_inverse(L)                     # (C, q, q)
        beta = Minv @ LtPsiInv                               # (C, q, D)  = Lam^T Sigma^-1
        return psi_inv, LtPsiInv, Minv, logdetM, beta

    def _log_component_density(self, X):
        """log N(x_n | component c) for every n, c -> (C, N)."""
        psi_inv, LtPsiInv, Minv, logdetM, _ = self._woodbury()
        D = X.shape[1]
        logdetSigma = torch.log(self.Psi).sum() + logdetM    # (C,)
        const = D * LOG2PI + logdetSigma                     # (C,)
        out = []
        for xb in torch.split(X, 4096):                      # chunk over N
            diff = xb.unsqueeze(0) - self.mu.unsqueeze(1)    # (C, B, D)
            a = torch.einsum("cqd,cbd->cbq", LtPsiInv, diff)  # (C, B, q)
            quad = (diff * diff * psi_inv).sum(-1) - torch.einsum(
                "cbq,cqr,cbr->cb", a, Minv, a
            )                                                # (C, B)
            out.append(-0.5 * (const.unsqueeze(1) + quad))
        return torch.cat(out, dim=1)                         # (C, N)

    def _log_resp(self, X):
        logw = torch.log(self.pi).unsqueeze(1) + self._log_component_density(X)  # (C,N)
        return logw - torch.logsumexp(logw, dim=0, keepdim=True)

    def predict(self, X):
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._log_resp(X).argmax(0).cpu().numpy()

    def predict_proba(self, X):
        """Per-point component responsibilities -> (N, C)."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        return self._log_resp(X).exp().T.cpu().numpy()

    def score_samples(self, X):
        """Per-point log-likelihood log sum_c pi_c N(x|c)."""
        X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        logw = torch.log(self.pi).unsqueeze(1) + self._log_component_density(X)
        return torch.logsumexp(logw, dim=0).cpu().numpy()

    def _n_parameters(self) -> float:
        """Free parameters: weights + means + loadings (less rotational
        redundancy q(q-1)/2 per component) + shared diagonal noise."""
        C, D, q = self.Lam.shape
        loadings = C * (D * q - q * (q - 1) / 2)
        return (C - 1) + C * D + loadings + D

    def aic(self, X) -> float:
        return 2 * self._n_parameters() - 2 * self.score_samples(X).sum()

    def bic(self, X) -> float:
        n = len(np.asarray(X))
        return self._n_parameters() * math.log(n) - 2 * self.score_samples(X).sum()


def _fit(X, C, q, seed, max_iter, tol, device, dtype, psi_floor):
    N, D = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Init: means from k-means, small random loadings, global-variance noise.
    km = KMeans(n_clusters=C, random_state=seed, n_init=10).fit(X.cpu().numpy())
    mu = torch.as_tensor(km.cluster_centers_, device=device, dtype=dtype)
    Lam = 0.1 * torch.randn(C, D, q, generator=g).to(device=device, dtype=dtype)
    Psi = X.var(0).clamp_min(psi_floor).to(device=device, dtype=dtype)
    pi = torch.full((C,), 1.0 / C, device=device, dtype=dtype)
    model = MFA(pi, mu, Lam, Psi, device, dtype)

    prev = -math.inf
    for it in range(max_iter):
        psi_inv, LtPsiInv, Minv, logdetM, beta = model._woodbury()
        logdetSigma = torch.log(Psi).sum() + logdetM
        const = D * LOG2PI + logdetSigma

        # Accumulators for the M-step.
        Nc = torch.zeros(C, device=device, dtype=dtype)
        Sxz = torch.zeros(C, D, q + 1, device=device, dtype=dtype)   # sum r x [z;1]^T
        Szz = torch.zeros(C, q + 1, q + 1, device=device, dtype=dtype)  # sum r [z;1][z;1]^T
        total_ll = 0.0

        for xb in torch.split(X, 4096):
            B = xb.shape[0]
            diff = xb.unsqueeze(0) - mu.unsqueeze(1)                  # (C, B, D)
            a = torch.einsum("cqd,cbd->cbq", LtPsiInv, diff)
            quad = (diff * diff * psi_inv).sum(-1) - torch.einsum("cbq,cqr,cbr->cb", a, Minv, a)
            logN = -0.5 * (const.unsqueeze(1) + quad)                 # (C, B)
            logw = torch.log(pi).unsqueeze(1) + logN
            lse = torch.logsumexp(logw, dim=0)                        # (B,)
            total_ll += lse.sum().item()
            r = torch.exp(logw - lse.unsqueeze(0))                    # (C, B)

            Ez = torch.einsum("cqd,cbd->cbq", beta, diff)            # (C, B, q)
            Ezz = Minv.unsqueeze(1) + Ez.unsqueeze(-1) * Ez.unsqueeze(-2)  # (C,B,q,q)
            ones = torch.ones(C, B, 1, device=device, dtype=dtype)
            zt = torch.cat([Ez, ones], dim=-1)                       # (C, B, q+1)
            # E[z~ z~^T]: block [[Ezz, Ez],[Ez^T, 1]]
            ztzt = torch.cat([
                torch.cat([Ezz, Ez.unsqueeze(-1)], dim=-1),
                torch.cat([Ez.unsqueeze(-2), ones.unsqueeze(-1)], dim=-1),
            ], dim=-2)                                               # (C,B,q+1,q+1)

            Nc += r.sum(1)
            Sxz += torch.einsum("cb,bd,cbe->cde", r, xb, zt)
            Szz += torch.einsum("cb,cbij->cij", r, ztzt)

        # M-step. Augmented loading Lam~ = Sxz Szz^-1, then split into Lam, mu.
        Lam_aug = Sxz @ torch.linalg.inv(Szz)                        # (C, D, q+1)
        Lam = Lam_aug[:, :, :q].contiguous()
        mu = Lam_aug[:, :, q].contiguous()
        pi = Nc / N
        # Shared Psi = diag( sum_n x x^T - sum_c Lam~_c (sum_n r x [z;1]^T)^T ) / N
        cross = torch.einsum("cde,cde->d", Lam_aug, Sxz)            # diag of sum_c Lam~ Sxz^T
        Psi = ((X * X).sum(0) - cross).clamp_min(psi_floor) / N
        model = MFA(pi, mu, Lam, Psi, device, dtype)

        if abs(total_ll - prev) < tol * abs(prev):
            break
        prev = total_ll

    return model, total_ll, it + 1


def fit_mfa(X, k, n_factors, seed=0, max_iter=200, tol=1e-4,
            device=None, dtype=torch.float32, psi_floor=1e-4) -> ClusterResult:
    """Fit an MFA with `k` components and `n_factors` factors each on raw X."""
    device = _pick_device(device)
    Xt = torch.as_tensor(np.asarray(X), device=device, dtype=dtype)
    model, ll, n_iter = _fit(Xt, k, n_factors, seed, max_iter, tol, device, dtype, psi_floor)
    return ClusterResult(
        "mfa",
        {"k": k, "n_factors": n_factors, "seed": seed, "loglik": ll, "n_iter": n_iter},
        model.predict(Xt),
        model=model,
        scores={
            "proba": model.predict_proba(Xt).astype(np.float32),
            "log_likelihood": model.score_samples(Xt),
        },
    )


if __name__ == "__main__":
    # Three low-rank Gaussian blobs (the MFA generative model). EM should recover
    # the clusters and increase log-likelihood monotonically.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    D, q_true = 30, 3
    parts, y = [], []
    for c in range(3):
        Lam = rng.standard_normal((D, q_true))
        center = rng.standard_normal(D) * 6
        z = rng.standard_normal((200, q_true))
        parts.append(z @ Lam.T + center + 0.3 * rng.standard_normal((200, D)))
        y += [c] * 200
    X = np.vstack(parts).astype("float32")

    res = fit_mfa(X, k=3, n_factors=3, seed=0, dtype=torch.float64)
    from sklearn.metrics import adjusted_rand_score
    print(f"n_clusters={res.n_clusters}  n_iter={res.params['n_iter']}  "
          f"loglik={res.params['loglik']:.1f}")
    print(f"ARI vs truth = {adjusted_rand_score(y, res.labels):.3f}")
    print(f"mean per-point log-lik = {res.log_likelihood(X).mean():.3f}")
