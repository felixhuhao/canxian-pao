"""
RDD task-aware repulsion — does FUNCTIONALLY TARGETED diversity help downstream?
================================================================================
Pre-registered in ../PREREG_RDD_taware.md.

rdd_li.py + rdd_crossover.py found generic lateral inhibition never helps downstream:
its repulsion is TASK-BLIND (maximizes raw τ-distance, not coverage of the signal's
timescales). This tests the constructive implication: a repulsion targeted at the
signal's actual spectrum should help — i.e. "diversity helps iff functionally targeted."

Generic LI:      Σ_{i<j} exp(−(τ_i−τ_j)²/2ℓ²)                         (distance in τ-space)
Task-aware LI:   Σ_{i<j} Σ_f Ŝ(f) · p_i(f) · p_j(f)                   (overlap in signal spectrum)
  where p_m(f) = channel m's normalized frequency response (from learned τ_m), and Ŝ(f) is the
  signal power spectrum ESTIMATED FROM TRAINING DATA via FFT (not oracle knowledge of the freqs).
This penalizes channels for co-occupying parts of the spectrum where the signal has power.
"""
import sys, os, math, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
from rdd_crossover import SSM   # SSM with symmetric / theta_jitter support

# frequency grid from a length-SEQ_LEN rFFT: f = k/SEQ_LEN, k=0..SEQ_LEN//2
FREQ_GRID = torch.tensor(np.fft.rfftfreq(R.SEQ_LEN), dtype=torch.float32, device=R.DEVICE)  # (F,)


def signal_spectrum(u):
    """Empirical power spectrum Ŝ(f) from training data (no oracle freqs). Returns (F,), sums to 1."""
    spec = torch.fft.rfft(u[:, :, 0], dim=1).abs() ** 2     # (B, F)
    s = spec.mean(0)
    return s / (s.sum() + 1e-12)


def channel_response(theta):
    """|H_m(f)|^2 for a first-order low-pass with a_m=1/τ_m, on FREQ_GRID. Returns (M, F), each row sums to 1."""
    a = (1.0 / torch.clamp(torch.exp(theta), min=1.05)).unsqueeze(1)     # (M,1)
    w = 2 * math.pi * FREQ_GRID.unsqueeze(0)                              # (1,F)
    denom = 1 - 2 * (1 - a) * torch.cos(w) + (1 - a) ** 2                 # (M,F)
    H2 = a ** 2 / (denom + 1e-8)
    return H2 / (H2.sum(dim=1, keepdim=True) + 1e-12)


def taware_loss(theta, S_hat, beta):
    if beta == 0:
        return torch.zeros((), device=theta.device)
    p = channel_response(theta)                                          # (M,F)
    M = p.shape[0]
    overlap = torch.zeros((), device=theta.device)
    for i in range(M):
        for j in range(i + 1, M):
            overlap = overlap + (S_hat * p[i] * p[j]).sum()
    return beta * overlap


def train_one(M, symmetric, li_type, beta, seed, theta_jitter, steps=R.STEPS):
    R.set_seed(seed)
    rng = np.random.RandomState(seed)
    u_tr = R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)
    u_te = R.gen_data(R.N_TEST, R.FREQS, R.AMPS, np.random.RandomState(seed + 10000))
    u_ood = R.gen_data(R.N_TEST, R.OOD_FREQS, R.OOD_AMPS, np.random.RandomState(seed + 20000))
    S_hat = signal_spectrum(u_tr).detach()
    model = SSM(M, symmetric, theta_jitter=theta_jitter).to(R.DEVICE)
    opt = torch.optim.Adam([
        {"params": [model.W_in, model.W_out, model.b], "lr": R.LR},
        {"params": [model.theta], "lr": R.LR_TAU},
    ])
    for _ in range(steps):
        opt.zero_grad()
        loss = R.mse_next(model, u_tr)
        if li_type == "generic":
            loss = loss + R.li_loss(model.theta, beta, inert=False)
        elif li_type == "taware":
            loss = loss + taware_loss(model.theta, S_hat, beta)
        elif li_type == "coverage":
            # REWARD covering the signal spectrum: at each powered freq, some channel responds.
            p = channel_response(model.theta)                 # (M,F)
            cov = (S_hat * p.max(dim=0).values).sum()
            loss = loss - beta * cov
        loss.backward(); opt.step()
    with torch.no_grad():
        tr = float(R.mse_next(model, u_tr)); te = float(R.mse_next(model, u_te))
        ood = float(R.mse_next(model, u_ood))
    return {"train_mse": tr, "test_mse": te, "ood_mse": ood,
            "div": R.diversity(model.taus()), "taus": model.taus().tolist()}


JITTER = 0.1
BETA_GEN = 10.0      # generic LI strength (reference)
COV_BETAS = (1.0, 10.0, 50.0)
CONDITIONS = []
for M in (3, 8):
    for init in ("asym", "sym"):
        sym = init == "sym"; jit = JITTER if sym else 0.0
        CONDITIONS.append((f"M{M}_{init}_noLI", M, sym, "none", 0.0, jit))
        CONDITIONS.append((f"M{M}_{init}_genLI", M, sym, "generic", BETA_GEN, jit))
        for b in COV_BETAS:
            CONDITIONS.append((f"M{M}_{init}_cov{b:g}", M, sym, "coverage", b, jit))


def main(seeds):
    data = {name: [] for name, *_ in CONDITIONS}
    for name, M, sym, lit, beta, jit in CONDITIONS:
        for s in seeds:
            data[name].append(train_one(M, sym, lit, beta, s, jit))
        d = data[name]
        print(f"{name:<16} div={np.mean([r['div'] for r in d]):6.3f}  "
              f"test_mse={np.mean([r['test_mse'] for r in d]):.4f}  "
              f"ood_mse={np.mean([r['ood_mse'] for r in d]):.4f}")

    def col(c, k): return np.array([r[k] for r in data[c]], float)
    print(f"\n{'='*84}\n  RDD COVERAGE-REWARD (N={len(seeds)}, seeds {seeds[0]}..{seeds[-1]}, device={R.DEVICE})\n{'='*84}")
    print("  Does TASK-ALIGNED coverage reward beat no-LI on test MSE? (primary: asym M=8, β=50)\n")
    for M in (3, 8):
        for init in ("asym", "sym"):
            no = col(f"M{M}_{init}_noLI", 'test_mse')
            print(f"  M={M} {init:<4} noLI={no.mean():.4f}", end="")
            for b in COV_BETAS:
                cv = col(f"M{M}_{init}_cov{b:g}", 'test_mse')
                g = R.hedges_g(cv, no); p = R.pw(cv, no, "less")
                tag = "*" if (p < 0.05 and g <= -0.5) else ""
                print(f"  | cov{b:g}={cv.mean():.4f}(g{g:+.2f},p{p:.3f}){tag}", end="")
            print()
    # primary pre-registered verdict
    no = col("M8_asym_noLI", 'test_mse'); cv = col("M8_asym_cov50", 'test_mse')
    g = R.hedges_g(cv, no); p = R.pw(cv, no, "less"); rev = R.pw(cv, no, "greater")
    verdict = ("COVERAGE HELPS" if (p < 0.05 and g <= -0.5)
               else "ANTI" if (rev < 0.05 and g >= 0.5) else "NULL")
    print(f"\n  PRIMARY (asym M=8, cov β=50 vs noLI): {cv.mean():.4f} vs {no.mean():.4f}  "
          f"g={g:+.2f} p(<)={p:.3f} -> {verdict}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_taware.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/rdd_taware.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=100)   # held-out seeds (calibration used seed 0)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(list(range(a.start, a.start + a.n)))
