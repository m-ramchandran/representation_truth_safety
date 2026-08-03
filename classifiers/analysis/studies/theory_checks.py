"""Numerical verification of the closed forms behind the transfer analysis.

    python -m classifiers.analysis.studies.theory_checks

Every check simulates from one generative model, a per-domain Gaussian
mixture whose label direction is domain-specific:

    domain D = d  w.p. pi_d;   label Y = +/-1  w.p. eta_d / 1-eta_d;
    x = mu_d + Y * a_d * w_d + sigma_d * xi,   xi ~ N(0, I_n),  ||w_d|| = 1.

Checks 1-4 and 7 verify the closed forms this model admits: the Bayes and
transfer AUCs, what the pooled optimum weights, and how the pooled direction
relates to the per-domain ones. Checks 5-6 are auxiliary simulations of
trained-network behaviour and nothing in the analysis rests on them.

Each check prints predicted-vs-empirical numbers and asserts agreement, so
the module doubles as a regression test on the derivations.
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import norm as gauss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
PHI = gauss.cdf


def sample_domain(m, mu, w, a, sigma, eta=0.5, rng=RNG):
    y = np.where(rng.random(m) < eta, 1.0, -1.0)
    x = mu + y[:, None] * a * w + sigma * rng.standard_normal((m, len(mu)))
    return x, y


def unit(v):
    return v / norm(v)


def fit_probe(x, y, C=1e4):
    lr = LogisticRegression(C=C, max_iter=2000).fit(x, (y > 0).astype(int))
    coef = lr.coef_[0]
    u = unit(coef)
    beta = -lr.intercept_[0] / norm(coef)   # score = <u, x> - beta
    return u, beta


# ----------------------------------------------------------------------------
# CHECK 1: closed-form transfer accuracy and AUC.
#   Probe (u, beta) trained on domain d, evaluated on domain e:
#     score s = <u, x> - beta = h + Y a_e c + sigma_e Z,
#       c = <u, w_e>,  h = <u, mu_e> - beta,  Z ~ N(0, 1)
#     Acc  = 1/2 Phi((h + a_e c)/sigma_e) + 1/2 Phi((a_e c - h)/sigma_e)
#     AUC  = Phi(sqrt(2) a_e c / sigma_e)        (threshold-free: h drops out)
# ----------------------------------------------------------------------------
def check1():
    n = 100
    a1, a2, s1, s2 = 2.0, 1.4, 1.0, 1.3          # heterogeneous margins/noise
    w1 = unit(RNG.standard_normal(n))
    w2 = unit(0.6 * w1 + np.sqrt(1 - 0.6**2) * unit(
        RNG.standard_normal(n) - (RNG.standard_normal(n) @ w1) * w1))
    mu1 = RNG.standard_normal(n) * 0.5
    mu2 = mu1 + 1.0 * unit(RNG.standard_normal(n))

    xtr, ytr = sample_domain(20000, mu1, w1, a1, s1)
    u, beta = fit_probe(xtr, ytr)

    xte, yte = sample_domain(400000, mu2, w2, a2, s2)
    emp_acc = np.mean(np.sign(xte @ u - beta) == yte)
    emp_auc = roc_auc_score(yte > 0, xte @ u)

    c, h = u @ w2, u @ mu2 - beta
    pred_acc = 0.5 * PHI((h + a2 * c) / s2) + 0.5 * PHI((a2 * c - h) / s2)
    pred_auc = PHI(np.sqrt(2) * a2 * c / s2)

    print(f"CHECK 1  transfer acc: pred {pred_acc:.4f}  emp {emp_acc:.4f}")
    print(f"CHECK 1  transfer auc: pred {pred_auc:.4f}  emp {emp_auc:.4f}")
    assert abs(pred_acc - emp_acc) < 0.005 and abs(pred_auc - emp_auc) < 0.005
    print("CHECK 1  PASS: acc = ½Φ((h+ac)/σ)+½Φ((ac−h)/σ), auc = Φ(√2·ac/σ)\n")


# ----------------------------------------------------------------------------
# CHECK 2: random-effects (partial-pooling) law of transfer.
#   w_d = sqrt(rho) vbar + sqrt(1-rho) u_d  with u_d ⟂ vbar random
#   => E<w_d, w_e> = rho (pairwise cosine of boundaries = shared fraction)
#   => mean cross-domain AUC ≈ Phi(sqrt(2) a rho / sigma)
#   Fitted probes attenuate: E cos(ŵ_d, ŵ_e) ≈ rho·kappa²,  kappa = E cos(ŵ, w).
#   Split-half cosine estimates kappa² => Spearman-style disattenuation
#   rho_hat = pairwise / splithalf recovers rho.
# ----------------------------------------------------------------------------
def check2():
    n, K, m, a, sigma = 200, 30, 2000, 2.0, 1.0
    for rho in (0.2, 0.5, 0.8):
        vbar = unit(RNG.standard_normal(n))
        ws, mus = [], []
        for _ in range(K):
            r = RNG.standard_normal(n)
            r -= (r @ vbar) * vbar
            ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * unit(r))
            mus.append(3.0 * unit(RNG.standard_normal(n)))
        ws = np.array(ws)

        cos_true = ws @ ws.T
        mean_true = cos_true[~np.eye(K, dtype=bool)].mean()

        fitted, halves, kappas = [], [], []
        for d in range(K):
            x, y = sample_domain(m, mus[d], ws[d], a, sigma)
            u, _ = fit_probe(x, y)
            fitted.append(u)
            kappas.append(u @ ws[d])
            ua, _ = fit_probe(x[: m // 2], y[: m // 2])
            ub, _ = fit_probe(x[m // 2:], y[m // 2:])
            halves.append(ua @ ub)
        fitted = np.array(fitted)
        cos_fit = fitted @ fitted.T
        mean_fit = cos_fit[~np.eye(K, dtype=bool)].mean()
        kappa2 = np.mean(kappas) ** 2
        splithalf = np.mean(halves)
        rho_hat = mean_fit / splithalf

        # transfer AUCs (via the CHECK-1-verified formula). True-direction
        # probes obey Phi(sqrt2·a·rho/sigma); fitted probes are attenuated to
        # Phi(sqrt2·a·kappa·rho/sigma).
        kappa = np.sqrt(kappa2)
        auc_true = np.mean([PHI(np.sqrt(2) * a * (ws[d] @ ws[e]) / sigma)
                            for d in range(K) for e in range(K) if d != e])
        auc_fit = np.mean([PHI(np.sqrt(2) * a * (fitted[d] @ ws[e]) / sigma)
                           for d in range(K) for e in range(K) if d != e])
        pred_true = PHI(np.sqrt(2) * a * rho / sigma)
        pred_fit = PHI(np.sqrt(2) * a * kappa * rho / sigma)

        # probit-ratio estimator of rho from the AUC matrix alone: reliability
        # kappa_d cancels in sqrt[(z_de·z_ed)/(z_dd·z_ee)], z = Phi^{-1}(AUC).
        z = np.array([[gauss.ppf(PHI(np.sqrt(2) * a * (fitted[d] @ ws[e]) / sigma))
                       for e in range(K)] for d in range(K)])
        ratios = [np.sqrt(z[d, e] * z[e, d] / (z[d, d] * z[e, e]))
                  for d in range(K) for e in range(K)
                  if d != e and z[d, e] > 0 and z[e, d] > 0]
        rho_probit = np.mean(ratios)

        print(f"CHECK 2  rho={rho}: true pairwise cos {mean_true:.3f}; "
              f"fitted {mean_fit:.3f} ≈ rho·kappa² {rho * kappa2:.3f}; "
              f"splithalf {splithalf:.3f} ≈ kappa² {kappa2:.3f}; "
              f"disattenuated rho_hat {rho_hat:.3f}")
        print(f"          cross-AUC (true dirs) {auc_true:.3f} vs Φ(√2aρ/σ) {pred_true:.3f}; "
              f"(fitted) {auc_fit:.3f} vs Φ(√2aκρ/σ) {pred_fit:.3f}; "
              f"probit-ratio rho_hat {rho_probit:.3f}")
        assert abs(mean_true - rho) < 0.04
        assert abs(mean_fit - rho * kappa2) < 0.04
        assert abs(rho_hat - rho) < 0.06
        assert abs(auc_true - pred_true) < 0.02
        assert abs(auc_fit - pred_fit) < 0.02
        assert abs(rho_probit - rho) < 0.06
    print("CHECK 2  PASS: pairwise probe cosine ≈ rho·kappa² (Spearman "
          "disattenuation recovers rho); transfer AUC ≈ Φ(√2·a·κρ/σ); the "
          "probit-ratio of the AUC matrix estimates rho with no coefficient access\n")


# ----------------------------------------------------------------------------
# CHECK 3: what a probe trained on POOLED domains learns.
#   (i) balanced labels: LDA direction = Sigma_W^{-1} dmu with
#         dmu = 2a·(mean_d w_d),  Sigma_W = sigma² I + B + a² V
#       (B, V = between-domain covariances of mu_d and a·w_d): the pooled probe
#       is the consensus direction, *shrunk* along domain-varying subspaces.
#   (ii) label-domain confounding (eta_d ≠ 1/2):
#         E[Yx] = a·mean_d w_d + sum_d pi_d (2 eta_d - 1) mu_d
#       — the probe picks up the domain-mean (Simpson/shortcut) component.
# ----------------------------------------------------------------------------
def check3():
    n, K, a, sigma = 60, 6, 1.5, 1.0
    vbar = unit(RNG.standard_normal(n)); rho = 0.5
    ws, mus = [], []
    for _ in range(K):
        r = RNG.standard_normal(n); r -= (r @ vbar) * vbar
        ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * unit(r))
        mus.append(3.0 * unit(RNG.standard_normal(n)))
    ws, mus = np.array(ws), np.array(mus)

    # (i) balanced: empirical LDA vs closed form
    m = 40000
    X = np.concatenate([sample_domain(m, mus[d], ws[d], a, sigma)[0] for d in range(K)])
    Y = np.concatenate([sample_domain(m, mus[d], ws[d], a, sigma)[1] for d in range(K)])
    # regenerate consistently (x,y pairs together)
    Xs, Ys = [], []
    for d in range(K):
        x, y = sample_domain(m, mus[d], ws[d], a, sigma)
        Xs.append(x); Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)

    mu_bar = mus.mean(0)
    B = (mus - mu_bar).T @ (mus - mu_bar) / K
    w_bar = ws.mean(0)
    V = (ws - w_bar).T @ (ws - w_bar) / K
    Sigma_W = sigma**2 * np.eye(n) + B + a**2 * V
    v_pred = unit(np.linalg.solve(Sigma_W, 2 * a * w_bar))

    Xp, Xm = X[Y > 0], X[Y < 0]
    dmu_emp = Xp.mean(0) - Xm.mean(0)
    Sw_emp = 0.5 * (np.cov(Xp.T) + np.cov(Xm.T))
    v_lda = unit(np.linalg.solve(Sw_emp, dmu_emp))
    u_log, _ = fit_probe(X, Y, C=100)

    print(f"CHECK 3  pooled LDA vs closed form (σ²I+B+a²V)⁻¹w̄: cos {v_lda @ v_pred:.4f}")
    print(f"CHECK 3  pooled logistic vs closed form:            cos {u_log @ v_pred:.4f}")
    print(f"CHECK 3  shrinkage: cos(pooled, w̄)={v_pred @ unit(w_bar):.3f}, "
          f"cos(naive w̄ itself, vbar)={unit(w_bar) @ vbar:.3f}, cos(pooled, vbar)={v_pred @ vbar:.3f}")
    assert v_lda @ v_pred > 0.995

    # (ii) Simpson term: eta alternating 0.8 / 0.2
    etas = np.array([0.8, 0.2] * (K // 2))
    Xs, Ys = [], []
    for d in range(K):
        x, y = sample_domain(m, mus[d], ws[d], a, sigma, eta=etas[d])
        Xs.append(x); Ys.append(y)
    Xc, Yc = np.concatenate(Xs), np.concatenate(Ys)
    Eyx_pred = a * w_bar + np.mean((2 * etas - 1)[:, None] * mus, 0)
    Eyx_emp = (Yc[:, None] * Xc).mean(0)
    err = norm(Eyx_pred - Eyx_emp) / norm(Eyx_pred)
    u_conf, _ = fit_probe(Xc, Yc, C=100)
    domain_axis = unit(np.mean((2 * etas - 1)[:, None] * mus, 0))
    print(f"CHECK 3  E[Yx] formula rel err {err:.3f}; "
          f"confounded probe: cos with domain-mean axis {u_conf @ domain_axis:.3f}, "
          f"balanced probe: {u_log @ domain_axis:.3f}")
    # balanced within-domain AUC of both probes
    auc_bal, auc_conf = [], []
    for d in range(K):
        x, y = sample_domain(20000, mus[d], ws[d], a, sigma)
        auc_bal.append(roc_auc_score(y > 0, x @ u_log))
        auc_conf.append(roc_auc_score(y > 0, x @ u_conf))
    print(f"CHECK 3  per-domain AUC: balanced-trained {np.mean(auc_bal):.3f}, "
          f"confounded-trained {np.mean(auc_conf):.3f}")
    assert err < 0.05 and abs(u_conf @ domain_axis) > abs(u_log @ domain_axis) + 0.2
    assert np.mean(auc_bal) > np.mean(auc_conf)
    print("CHECK 3  PASS: pooled probe = shrunken consensus; label-domain "
          "imbalance injects the domain-mean (shortcut) component\n")


# ----------------------------------------------------------------------------
# CHECK 4: the Bayes-optimal predictor is a softmax-gated mixture of linear
# experts:
#   P(Y=1|x) = sum_d g_d(x) · sigmoid(2 a <w_d, x - mu_d> / sigma²)
#   g_d(x)   = softmax_d[ <x, mu_d>/sigma² - ||mu_d||²/(2 sigma²) + log pi_d
#                         + logcosh(a <w_d, x - mu_d>/sigma²) ]
# Verified against brute-force Bayes on the 2K-component Gaussian mixture,
# and Bayes acc vs best-linear acc gap grows as domain separation grows.
# ----------------------------------------------------------------------------
def check4():
    n, K, a, sigma = 30, 3, 1.5, 1.0
    vbar = unit(RNG.standard_normal(n)); rho = 0.3
    ws, mus = [], []
    for _ in range(K):
        r = RNG.standard_normal(n); r -= (r @ vbar) * vbar
        ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * unit(r))
        mus.append(unit(RNG.standard_normal(n)))
    ws = np.array(ws)

    def gated_posterior(X, mus_, sep):
        M = np.array(mus_) * sep
        z = (X @ M.T) / sigma**2 - (M**2).sum(1) / (2 * sigma**2)
        score = (X - M[None].transpose(1, 0, 2).mean(0) * 0)  # placeholder
        s = np.stack([(X - M[d]) @ ws[d] for d in range(K)], 1) * a / sigma**2
        logit_gate = z + np.log(1.0 / K) + np.logaddexp(s, -s) - np.log(2)  # logcosh
        g = np.exp(logit_gate - logit_gate.max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        return (g / (1 + np.exp(-2 * s))).sum(1)

    def brute_posterior(X, mus_, sep):
        M = np.array(mus_) * sep
        logp = []
        for d in range(K):
            for y in (1, -1):
                c = M[d] + y * a * ws[d]
                logp.append((y, -((X - c)**2).sum(1) / (2 * sigma**2)))
        L = np.array([l for _, l in logp])
        L -= L.max(0)
        P = np.exp(L)
        num = sum(P[i] for i, (y, _) in enumerate(logp) if y == 1)
        return num / P.sum(0)

    for sep in (1.0, 4.0):
        X = np.concatenate([sample_domain(4000, np.array(mus[d]) * sep, ws[d], a, sigma)[0]
                            for d in range(K)])
        pg, pb = gated_posterior(X, mus, sep), brute_posterior(X, mus, sep)
        print(f"CHECK 4  sep={sep}: max |gated - brute-force Bayes| = {np.abs(pg - pb).max():.2e}")
        assert np.abs(pg - pb).max() < 1e-8

    # accuracy comparison at large separation
    accs = {}
    for sep in (0.5, 4.0):
        Xs, Ys = [], []
        for d in range(K):
            x, y = sample_domain(30000, np.array(mus[d]) * sep, ws[d], a, sigma)
            Xs.append(x); Ys.append(y)
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        acc_bayes = np.mean(np.sign(brute_posterior(X, mus, sep) - 0.5) == Y)
        u, beta = fit_probe(X[::3], Y[::3], C=100)
        acc_lin = np.mean(np.sign(X @ u - beta) == Y)
        oracle = PHI(a / sigma)
        accs[sep] = (acc_bayes, acc_lin, oracle)
        print(f"CHECK 4  sep={sep}: Bayes {acc_bayes:.3f}  best-linear≈ {acc_lin:.3f}  "
              f"domain-oracle Φ(a/σ) {oracle:.3f}")
    gap_small = accs[0.5][0] - accs[0.5][1]
    gap_big = accs[4.0][0] - accs[4.0][1]
    assert gap_big > gap_small and gap_big > 0.03
    assert accs[4.0][0] > PHI(a / sigma) - 0.02   # gating recovers oracle when separable
    print("CHECK 4  PASS: Bayes = softmax-gated mixture of linear experts; the "
          "linearity gap grows with domain separation\n")


# ----------------------------------------------------------------------------
# CHECK 5: architectures. A one-layer gated (GLU-style) block represents the
# gated-mixture solution exactly; trained GLU / ReLU nets become locally
# linear with per-domain gradients ≈ w_d, while a linear model can only learn
# the consensus.
# ----------------------------------------------------------------------------
def check5():
    import torch
    torch.manual_seed(0)
    n, K, a, sigma, sep, rho = 64, 4, 1.5, 1.0, 3.0, 0.5
    vbar = unit(RNG.standard_normal(n))
    ws, mus = [], []
    for _ in range(K):
        r = RNG.standard_normal(n); r -= (r @ vbar) * vbar
        ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * unit(r))
        mus.append(sep * unit(RNG.standard_normal(n)))
    ws, mus = np.array(ws), np.array(mus)

    # explicit construction: f(x) = sum_d softmax(beta <mu_d, x>)_d (<w_d, x> - <w_d, mu_d>)
    def constructed(X, beta):
        g = np.exp(beta * (X @ mus.T)); g /= g.sum(1, keepdims=True)
        s = X @ ws.T - (ws * mus).sum(1)
        return (g * s).sum(1)

    Xs, Ys = [], []
    for d in range(K):
        x, y = sample_domain(30000, mus[d], ws[d], a, sigma)
        Xs.append(x); Ys.append(y)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    for beta in (0.5, 2.0, 8.0):
        acc = np.mean(np.sign(constructed(X, beta)) == Y)
        print(f"CHECK 5  constructed gated block, beta={beta}: acc {acc:.3f} "
              f"(domain-oracle {PHI(a / sigma):.3f})")
    assert np.mean(np.sign(constructed(X, 8.0)) == Y) > PHI(a / sigma) - 0.01

    # train linear / ReLU MLP / GLU nets
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor((Y > 0).astype(np.float32))
    perm = torch.randperm(len(Xt)); Xt, Yt = Xt[perm], Yt[perm]
    dom = np.concatenate([np.full(30000, d) for d in range(K)])[perm.numpy()]

    class GLU(torch.nn.Module):
        def __init__(self, n, h):
            super().__init__()
            self.G = torch.nn.Linear(n, h); self.V = torch.nn.Linear(n, h)
            self.out = torch.nn.Linear(h, 1)
        def forward(self, x):
            return self.out(torch.nn.functional.silu(self.G(x)) * self.V(x))[:, 0]

    def train(model, steps=400):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for t in range(steps):
            idx = torch.randint(0, len(Xt), (4096,))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        return model

    lin = train(torch.nn.Sequential(torch.nn.Linear(n, 1), torch.nn.Flatten(0)))
    mlp = train(torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1), torch.nn.Flatten(0)))
    glu = train(GLU(n, 64))

    def acc(model):
        with torch.no_grad():
            return ((model(Xt) > 0).float() == Yt).float().mean().item()

    def local_alignment(model):
        cs = []
        for d in range(K):
            xs = Xt[np.where(dom == d)[0][:2000]].clone().requires_grad_(True)
            model(xs).sum().backward()
            gbar = xs.grad.mean(0).detach().numpy()
            cs.append(unit(gbar) @ ws[d])
        return np.array(cs)

    for name, model in (("linear", lin), ("relu-mlp", mlp), ("glu", glu)):
        al = local_alignment(model)
        print(f"CHECK 5  {name}: acc {acc(model):.3f}; per-domain grad·w_d = "
              f"{np.round(al, 2)} (mean {al.mean():.2f})")
    al_lin, al_mlp, al_glu = (local_alignment(m) for m in (lin, mlp, glu))
    assert acc(glu) > acc(lin) + 0.03 and acc(mlp) > acc(lin) + 0.03
    assert al_glu.mean() > al_lin.mean() + 0.1 and al_mlp.mean() > al_lin.mean() + 0.1
    # linear model's direction ≈ shrunken consensus, cos with vbar high
    wlin = unit(lin[0].weight.detach().numpy()[0])
    print(f"CHECK 5  linear model dir: cos(vbar) {wlin @ vbar:.2f}; "
          f"cos(w_d) {np.round(wlin @ ws.T, 2)}")
    print("CHECK 5  PASS: gated block realizes the Bayes solution; trained "
          "nonlinear nets are locally linear with per-domain gradients ≈ w_d; "
          "linear model learns only the consensus\n")


# ----------------------------------------------------------------------------
# CHECK 6: learning order. Training the gated architecture from small init:
# the consensus direction is a first-order (information-exponent-1) signal —
# E[Yx] = a·w̄ — while domain corrections require the gate. Prediction:
# shared component of the expert weights grows first, gate alignment second,
# idiosyncratic components last.
# ----------------------------------------------------------------------------
def check6():
    import torch
    torch.manual_seed(1)
    n, K, a, sigma, sep, rho = 64, 4, 1.5, 1.0, 3.0, 0.5
    vbar = unit(RNG.standard_normal(n))
    ws, mus, us = [], [], []
    for _ in range(K):
        r = RNG.standard_normal(n); r -= (r @ vbar) * vbar
        u = unit(r); us.append(u)
        ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * u)
        mus.append(sep * unit(RNG.standard_normal(n)))
    ws, mus, us = np.array(ws), np.array(mus), np.array(us)

    Xs, Ys = [], []
    for d in range(K):
        x, y = sample_domain(15000, mus[d], ws[d], a, sigma)
        Xs.append(x); Ys.append(y)
    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32)
    Y = torch.tensor((np.concatenate(Ys) > 0).astype(np.float32))

    h = 16
    G = torch.nn.Parameter(torch.randn(h, n) * 1e-3)
    V = torch.nn.Parameter(torch.randn(h, n) * 1e-3)
    c = torch.nn.Parameter(torch.randn(h) * 1e-3)
    opt = torch.optim.SGD([G, V, c], lr=0.5)

    def f(x):
        return (torch.softmax(x @ G.T, dim=1) * (x @ V.T) * c).sum(1)

    Pm = np.linalg.qr(mus.T)[0]          # basis of domain-mean subspace
    Pu = np.linalg.qr(us.T)[0]           # basis of idiosyncratic subspace

    def shares():
        Vn = V.detach().numpy()
        Gn = G.detach().numpy()
        vs = norm(Vn @ vbar) ** 2 / norm(Vn) ** 2
        vu = norm(Vn @ Pu) ** 2 / norm(Vn) ** 2
        gm = norm(Gn @ Pm) ** 2 / norm(Gn) ** 2
        return vs, vu, gm

    hist = []
    for t in range(1501):
        idx = torch.randint(0, len(X), (8192,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(f(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 100 == 0:
            vs, vu, gm = shares()
            with torch.no_grad():
                acc = ((f(X) > 0).float() == Y).float().mean().item()
            hist.append((t, vs, vu, gm, acc))
            print(f"CHECK 6  t={t:5d}  expert-shared {vs:.2f}  expert-idio {vu:.2f}  "
                  f"gate-domain {gm:.2f}  acc {acc:.3f}")

    def crossing(col, frac=0.5):
        final = hist[-1][col]
        for t, *vals in hist:
            if vals[col - 1] >= frac * final:
                return t
        return hist[-1][0]

    t_shared, t_idio = crossing(1), crossing(2)
    print(f"CHECK 6  50%-of-final crossing: shared at t={t_shared}, "
          f"idiosyncratic at t={t_idio}")
    assert t_shared <= t_idio
    assert hist[-1][4] > PHI(a / sigma) - 0.03
    print("CHECK 6  PASS: shared (consensus) component of the experts is "
          "learned no later than the idiosyncratic components; final acc ≈ oracle\n")


# ----------------------------------------------------------------------------
# CHECK 7: the moment structure driving the pooled-covariance framework.
#   E[Yx]      = a * mean_d w_d                      (consensus only; deviations cancel)
#   E[Y x x^T] = a * mean_d (mu_d w_d^T + w_d mu_d^T) (domain-mean paired with its
#                                                     own boundary at second order)
# ----------------------------------------------------------------------------
def check7():
    n, K, a, sigma = 40, 4, 1.3, 1.0
    vbar = unit(RNG.standard_normal(n)); rho = 0.5
    ws, mus = [], []
    for _ in range(K):
        r = RNG.standard_normal(n); r -= (r @ vbar) * vbar
        ws.append(np.sqrt(rho) * vbar + np.sqrt(1 - rho) * unit(r))
        mus.append(2.0 * unit(RNG.standard_normal(n)))
    ws, mus = np.array(ws), np.array(mus)

    M_emp = np.zeros((n, n)); v_emp = np.zeros(n); N = 0
    for d in range(K):
        for _ in range(5):
            x, y = sample_domain(50000, mus[d], ws[d], a, sigma)
            M_emp += (x * y[:, None]).T @ x
            v_emp += (y[:, None] * x).sum(0)
            N += len(y)
    M_emp /= N; v_emp /= N

    v_pred = a * ws.mean(0)
    M_pred = a * np.mean([np.outer(mus[d], ws[d]) + np.outer(ws[d], mus[d])
                          for d in range(K)], axis=0)
    e1 = norm(v_emp - v_pred) / norm(v_pred)
    e2 = norm(M_emp - M_pred) / norm(M_pred)
    print(f"CHECK 7  E[Yx] rel err {e1:.3f};  E[Yxx^T] rel err {e2:.3f}")
    assert e1 < 0.05 and e2 < 0.06
    print("CHECK 7  PASS: first moment = consensus only; second moment pairs "
          "each domain mean with its own boundary (the order-2 staircase rung)\n")


if __name__ == "__main__":
    check1()
    check2()
    check3()
    check4()
    check7()
    check5()
    check6()
    print("ALL CHECKS PASS")
