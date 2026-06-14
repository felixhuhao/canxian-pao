"""
Ladder R1 — does the coverage benefit survive PAO's CRYSTALLIZE-AND-FREEZE mechanism?
=====================================================================================
Pre-registered in ../PREREG_RDD_ladder_r1.md.

R0 (rdd_taware / rdd_capacity) found: a task-aligned coverage reward helps downstream and
the benefit scales with excess capacity — but via JOINT co-adaptation + an auxiliary loss.
PAO instead builds a library by SEQUENTIALLY crystallizing FROZEN units and admitting them
through a gate. R1 changes ONLY that mechanism (same RDD task, scalar-τ units, supervised)
and asks whether the benefit + capacity law survive.

Build = sequential residual boosting of frozen low-pass channels:
  - train a candidate channel on the current residual; FREEZE its (τ, W_in).
  - GATED arm   : admit only if it lowers held-out (val) MSE by > EPS (covers something new),
                  else reject. (PAO + coverage gate.)
  - UNGATED arm : admit every crystallized candidate. (PAO as-is -> redundant library.)
  - global readout over admitted frozen features = ridge least squares (refit each admission).
Swept over capacity M_max (candidates attempted). Metric: held-out test MSE.
"""
import sys, os, math, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
from scipy.stats import spearmanr

HIDDEN = R.HIDDEN
K = len(R.FREQS)
M_GRID = [2, 3, 5, 8, 12, 16]
CAND_STEPS = 100
LAM = 1e-2                 # ridge
EPS = 1e-3                 # coverage gate: min val-MSE improvement to admit
DEVICE = R.DEVICE


def compute_h(theta, W_in, u):
    """Low-pass features h(t) for one frozen channel. u:(B,T,1) -> (B,T,HIDDEN)."""
    a = 1.0 / torch.clamp(torch.exp(theta), min=1.05)
    h = torch.zeros(u.shape[0], HIDDEN, device=u.device)
    outs = []
    for t in range(u.shape[1]):
        h = (1 - a) * h + a * (u[:, t, :] @ W_in.t())
        outs.append(h)
    return torch.stack(outs, dim=1)


def targetY(u):
    """next-step target Y(t)=u(t+1), valid for t=0..T-2."""
    return u[:, 1:, 0]                      # (B, T-1)


def feats_valid(theta, W_in, u):
    return compute_h(theta, W_in, u)[:, :-1, :]     # (B, T-1, HIDDEN)


def ridge_fit(Xlist, Y):
    """Closed-form ridge readout over concatenated frozen features. Xlist: list of (B,T-1,H)."""
    if not Xlist:
        return None
    X = torch.cat(Xlist, dim=-1).reshape(-1, sum(x.shape[-1] for x in Xlist))   # (N,F)
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X = torch.cat([X, ones], dim=1)
    y = Y.reshape(-1, 1)
    A = X.t() @ X + LAM * torch.eye(X.shape[1], device=X.device)
    W = torch.linalg.solve(A, X.t() @ y)
    return W


def ridge_pred(Xlist, W):
    X = torch.cat(Xlist, dim=-1).reshape(-1, sum(x.shape[-1] for x in Xlist))
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X = torch.cat([X, ones], dim=1)
    return (X @ W).reshape(-1)


def mse_of(Xlist, W, Y):
    if W is None:
        return float((Y ** 2).mean())
    return float(((ridge_pred(Xlist, W) - Y.reshape(-1)) ** 2).mean())


def train_candidate(u, R_target, seed):
    """Train one channel (θ,W_in) to predict residual R_target; return frozen θ,W_in."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    theta = torch.nn.Parameter(torch.tensor(math.log(R.TAU_INIT) + 0.3 * torch.randn(1, generator=g).item(), device=DEVICE))
    W_in = torch.nn.Parameter(0.1 * torch.randn(HIDDEN, 1, generator=g).to(DEVICE))
    w = torch.nn.Parameter(0.1 * torch.randn(HIDDEN, 1, generator=g).to(DEVICE))
    b = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
    opt = torch.optim.Adam([{"params": [W_in, w, b], "lr": R.LR}, {"params": [theta], "lr": R.LR_TAU * 5}])
    for _ in range(CAND_STEPS):
        opt.zero_grad()
        h = compute_h(theta, W_in, u)[:, :-1, :]
        pred = (h @ w).squeeze(-1) + b
        loss = ((pred - R_target) ** 2).mean()
        loss.backward(); opt.step()
    return theta.detach(), W_in.detach()


def build(u_tr, u_val, u_te, M_max, gated, seed):
    Y_tr, Y_val, Y_te = targetY(u_tr), targetY(u_val), targetY(u_te)
    admitted = []                       # list of (theta, W_in)
    Ftr, Fval, Fte = [], [], []
    W = None
    for k in range(M_max):
        resid = Y_tr - (ridge_pred(Ftr, W).reshape(Y_tr.shape) if W is not None else 0.0)
        theta, W_in = train_candidate(u_tr, resid, seed * 1000 + k)
        cand_tr = feats_valid(theta, W_in, u_tr)
        cand_val = feats_valid(theta, W_in, u_val)
        cand_te = feats_valid(theta, W_in, u_te)
        W_new = ridge_fit(Ftr + [cand_tr], Y_tr)
        val_before = mse_of(Fval, W, Y_val)
        val_after = mse_of(Fval + [cand_val], W_new, Y_val)
        admit = True if not gated else (val_before - val_after > EPS)
        if admit:
            admitted.append((theta, W_in))
            Ftr.append(cand_tr); Fval.append(cand_val); Fte.append(cand_te)
            W = W_new
        elif gated:
            continue   # reject redundant candidate; keep trying remaining budget
    test_mse = mse_of(Fte, W, Y_te)
    return {"test_mse": test_mse, "n_admitted": len(admitted)}


def main(seeds):
    data = {(M, g): [] for M in M_GRID for g in (True, False)}
    for M in M_GRID:
        for gated in (True, False):
            for s in seeds:
                rng = np.random.RandomState(s)
                u_tr = R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)
                u_val = R.gen_data(200, R.FREQS, R.AMPS, np.random.RandomState(s + 5000))
                u_te = R.gen_data(R.N_TEST, R.FREQS, R.AMPS, np.random.RandomState(s + 10000))
                data[(M, gated)].append(build(u_tr, u_val, u_te, M, gated, s))
            d = data[(M, gated)]
            print(f"M_max={M:<2} {'GATED ' if gated else 'ungated'} "
                  f"test_mse={np.mean([r['test_mse'] for r in d]):.4f}  "
                  f"n_admitted={np.mean([r['n_admitted'] for r in d]):.1f}")

    def col(M, g, k): return np.array([r[k] for r in data[(M, g)]], float)
    print(f"\n{'='*78}\n  LADDER R1: coverage-GATED vs ungated sequential frozen admission (N={len(seeds)}, K={K})\n{'='*78}")
    print(f"  {'M':>3} {'M/K':>5} {'gated':>8} {'ungated':>8} {'Δ(un−ga)':>9} {'g':>7} {'p':>7} {'n_adm(ga)':>10}")
    ratios, deltas = [], []
    for M in M_GRID:
        ga = col(M, True, 'test_mse'); un = col(M, False, 'test_mse')
        delta = un.mean() - ga.mean()
        gg = R.hedges_g(ga, un); p = R.pw(ga, un, "less")
        ratios.append(M / K); deltas.append(delta)
        star = "*" if (p < 0.05 and gg <= -0.5) else ""
        print(f"  {M:>3} {M/K:>5.1f} {ga.mean():>8.4f} {un.mean():>8.4f} {delta:>+9.4f} {gg:>+7.2f} {p:>7.3f}{star:>2} "
              f"{col(M,True,'n_admitted').mean():>10.1f}")
    rho, prho = spearmanr(ratios, deltas)
    print(f"\n  TREND Spearman(M/K, Δ(ungated−gated)) = {rho:+.3f} (p={prho:.3f})")
    verdict = ("SURVIVES (gated beats ungated; gap grows with capacity)"
               if (rho > 0 and prho < 0.10 and np.mean([d for r, d in zip(ratios, deltas) if r >= 1]) > 0)
               else "DOES NOT SURVIVE")
    print(f"  -> coverage benefit under PAO-style mechanism: {verdict}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_ladder_r1.pkl"), "wb") as f:
        pickle.dump({str(k): v for k, v in data.items()}, f)
    print("\n  Saved results/rdd_ladder_r1.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(list(range(a.start, a.start + a.n)))
