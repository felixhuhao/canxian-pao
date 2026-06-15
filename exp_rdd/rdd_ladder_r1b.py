"""
Ladder R1b — is coverage-gating's payoff gated by COMBINATION FRAGILITY?
=======================================================================
Pre-registered in ../PREREG_RDD_ladder_r1b.md.

R1 found coverage-gating gives no PERFORMANCE benefit when frozen units are combined by a
ROBUST readout (ridge down-weights redundant units). Hypothesis: the benefit returns when the
combiner is FRAGILE — i.e. it cannot down-weight a bad/redundant unit. R1b changes ONLY the
combiner: ridge (robust, =R1) vs AVERAGE (fragile: fixed equal weights of each unit's standalone
prediction, no reweighting — the canonical fragile combination, matching PAO's inability to
globally correct a wrong skill applied as additive bias).

Each candidate channel is trained standalone to predict Y; ensemble prediction = mean of admitted
channels' predictions. GATED admits a candidate only if it lowers val MSE of the mean; UNGATED
admits all. Swept over capacity. Prediction: with the fragile combiner, gated beats ungated and
the gap grows with capacity (mirroring R0/PAO); with ridge it does not (R1).
"""
import sys, os, math, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
from rdd_ladder_r1 import compute_h, targetY, HIDDEN, K, M_GRID, CAND_STEPS, ridge_fit, ridge_pred, mse_of, feats_valid
from scipy.stats import spearmanr

EPS = 1e-3
DEVICE = R.DEVICE


def train_candidate_Y(u, Y, seed):
    """Train one channel standalone to predict the FULL target Y; return frozen (θ,W_in,w,b)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    theta = torch.nn.Parameter(torch.tensor(math.log(R.TAU_INIT) + 0.5 * torch.randn(1, generator=g).item(), device=DEVICE))
    W_in = torch.nn.Parameter(0.1 * torch.randn(HIDDEN, 1, generator=g).to(DEVICE))
    w = torch.nn.Parameter(0.1 * torch.randn(HIDDEN, 1, generator=g).to(DEVICE))
    b = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
    opt = torch.optim.Adam([{"params": [W_in, w, b], "lr": R.LR}, {"params": [theta], "lr": R.LR_TAU * 5}])
    for _ in range(CAND_STEPS):
        opt.zero_grad()
        h = compute_h(theta, W_in, u)[:, :-1, :]
        pred = (h @ w).squeeze(-1) + b
        loss = ((pred - Y) ** 2).mean()
        loss.backward(); opt.step()
    return theta.detach(), W_in.detach(), w.detach(), b.detach()


def pred_channel(ch, u):
    theta, W_in, w, b = ch
    h = compute_h(theta, W_in, u)[:, :-1, :]
    return (h @ w).squeeze(-1) + b           # (B, T-1)


def mse_mean(preds, Y):
    if not preds:
        return float((Y ** 2).mean())
    m = torch.stack(preds, 0).mean(0)
    return float(((m - Y) ** 2).mean())


def build_avg(u_tr, u_val, u_te, M_max, gated, seed):
    Y_tr, Y_val, Y_te = targetY(u_tr), targetY(u_val), targetY(u_te)
    ptr, pval, pte = [], [], []
    for k in range(M_max):
        ch = train_candidate_Y(u_tr, Y_tr, seed * 1000 + k)
        c_tr, c_val, c_te = pred_channel(ch, u_tr), pred_channel(ch, u_val), pred_channel(ch, u_te)
        before = mse_mean(pval, Y_val)
        after = mse_mean(pval + [c_val], Y_val)
        admit = True if not gated else (before - after > EPS)
        if admit:
            ptr.append(c_tr); pval.append(c_val); pte.append(c_te)
    return {"test_mse": mse_mean(pte, Y_te), "n_admitted": len(pte)}


def build_ridge(u_tr, u_val, u_te, M_max, gated, seed):
    """R1's robust combiner (residual boosting + ridge readout), for direct contrast."""
    from rdd_ladder_r1 import build as r1build
    return r1build(u_tr, u_val, u_te, M_max, gated, seed)


def main(seeds):
    # fragile combiner only; robust(ridge) contrast = R1 (rdd_ladder_r1, same setup, held-out seeds)
    combiners = {"fragile(avg)": build_avg}
    data = {(cn, M, g): [] for cn in combiners for M in M_GRID for g in (True, False)}
    for cn, fn in combiners.items():
        for M in M_GRID:
            for gated in (True, False):
                for s in seeds:
                    rng = np.random.RandomState(s)
                    u_tr = R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)
                    u_val = R.gen_data(200, R.FREQS, R.AMPS, np.random.RandomState(s + 5000))
                    u_te = R.gen_data(R.N_TEST, R.FREQS, R.AMPS, np.random.RandomState(s + 10000))
                    data[(cn, M, gated)].append(fn(u_tr, u_val, u_te, M, gated, s))
            d = data[(cn, M, True)]; du = data[(cn, M, False)]
            print(f"{cn:<14} M={M:<2} gated={np.mean([r['test_mse'] for r in d]):.4f}(adm {np.mean([r['n_admitted'] for r in d]):.1f}) "
                  f"ungated={np.mean([r['test_mse'] for r in du]):.4f}")

    def col(cn, M, g, k): return np.array([r[k] for r in data[(cn, M, g)]], float)
    print(f"\n{'='*82}\n  LADDER R1b: does gating's payoff depend on COMBINER fragility? (N={len(seeds)}, K={K})\n{'='*82}")
    for cn in combiners:
        print(f"\n  --- combiner = {cn} ---")
        print(f"  {'M':>3} {'M/K':>5} {'gated':>8} {'ungated':>8} {'Δ(un−ga)':>9} {'g':>7} {'p(<)':>7}")
        ratios, deltas = [], []
        for M in M_GRID:
            ga = col(cn, M, True, 'test_mse'); un = col(cn, M, False, 'test_mse')
            delta = un.mean() - ga.mean(); gg = R.hedges_g(ga, un); p = R.pw(ga, un, "less")
            ratios.append(M / K); deltas.append(delta)
            star = "*" if (p < 0.05 and gg <= -0.5) else ""
            print(f"  {M:>3} {M/K:>5.1f} {ga.mean():>8.4f} {un.mean():>8.4f} {delta:>+9.4f} {gg:>+7.2f} {p:>7.3f}{star:>2}")
        rho, prho = spearmanr(ratios, deltas)
        verdict = ("gating HELPS, scales with capacity" if (rho > 0 and prho < 0.10 and np.mean([d for r, d in zip(ratios, deltas) if r >= 1]) > 0)
                   else "gating does NOT help performance")
        print(f"  TREND Spearman(M/K, Δ) = {rho:+.3f} (p={prho:.3f}) -> {verdict}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_ladder_r1b.pkl"), "wb") as f:
        pickle.dump({str(k): v for k, v in data.items()}, f)
    print("\n  Saved results/rdd_ladder_r1b.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=400)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    main(list(range(a.start, a.start + a.n)))
