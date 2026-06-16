"""
Busch E1 — variance-vs-geometry deconfound of on-/off-manifold BCI learning.

Pre-registration: ../PREREG_BUSCH_E1.md  (FROZEN 2026-06-16; metric amended same day, pre-confirmatory).

Linear-Gaussian "brain". An agent learns a linear policy (intended angle -> brain state) to drive a
BCI readout C, across trials with random targets, under a plasticity prior.

Min-plasticity-cost action achieving readout r along C:  b = r * Sigma_act C / (C^T Sigma_act C),
with cost  r^2 / (C^T Sigma_act C).  So:
  readability     (decoder SNR)         ~ (C^T Sigma_obs C) / sigma_read^2
  controllability (cheapness to drive)  ~  C^T Sigma_act C
  agent optimal gain                    g* = 1 / (1 + lam_plast * cost),  cost = 1/(C^T Sigma_act C)
  -> low controllability => agent under-drives the readout => learning fails.

PEV = C^T Sigma_obs C / tr(Sigma_obs) is Busch's "on-manifold-ness". It tracks ONLY readability and is
blind to controllability. Busch assume Sigma_obs = Sigma_act so both collapse onto PEV. We break that:
  D1 = high PEV (readable) but low controllability -> predicted to FAIL.
  D2 = low PEV but made readable + controllable    -> predicted to LEARN.
Metric: ΔControl = 100 * (mean acc last third - first third), acc = exp(-err^2 / 2 s_acc^2).
"""
import argparse
import os
import numpy as np
from scipy import stats

D = 20  # brain/latent dim (matches their 20-D embedding)


# ----------------------------- model pieces --------------------------------- #
def make_basis(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    return Q


def spectrum(rho):
    lam = rho ** np.arange(D)
    return lam / lam.mean()          # normalized eigenvalues, mean 1


def cov_from(U, lam):
    return (U * lam) @ U.T


def pev(C, Sigma_obs):
    return float(C @ Sigma_obs @ C) / float(np.trace(Sigma_obs))


def read_snr(C, Sigma_obs, sigma_read):
    return float(C @ Sigma_obs @ C) / (sigma_read ** 2)


def controllability(C, Sigma_act):
    return float(C @ Sigma_act @ C)


# ----------------------------- the learning agent --------------------------- #
def run_session(C, Sigma_obs, Sigma_act, sigma_read, lam_plast,
                n_trials, lr, rng, K=10, s_acc=0.3):
    """Agent learns scalar gain g (intended angle -> readout amplitude) by gradient on
    readout error plus a plasticity penalty on amplitude. Returns ΔControl over session."""
    SoC = max(float(C @ Sigma_obs @ C), 1e-12)     # signal variance along C (observation)
    SaC = max(float(C @ Sigma_act @ C), 1e-12)     # control variance along C (volitional)
    cost = 1.0 / SaC                               # plasticity cost per unit readout^2
    noise_sd = sigma_read / np.sqrt(SoC)           # decode noise in angle units (per subsample)

    g = 0.1
    accs = np.empty(n_trials)
    for t in range(n_trials):
        theta = rng.uniform(-1.0, 1.0)
        noise = rng.normal(0.0, noise_sd, size=K).mean()   # K sub-TR averaging
        decoded = g * theta + noise
        err = decoded - theta
        accs[t] = np.exp(-err * err / (2 * s_acc * s_acc))
        # grad of (decoded-theta)^2 + lam_plast*cost*g^2*theta^2  wrt g
        g -= lr * (err * theta + lam_plast * cost * g * theta * theta)
        g = max(g, 0.0)

    third = max(1, n_trials // 3)
    return 100.0 * (accs[-third:].mean() - accs[:third].mean())


# ----------------------------- conditions ----------------------------------- #
def build_conditions(U, lam, sigma_read):
    """name -> (C, Sigma_obs, Sigma_act, sigma_read_cond). Sigma_obs fixed; Sigma_act and
    per-condition read noise perturbed only for the deconfound cells."""
    Sigma_obs = cov_from(U, lam)
    SocTop = float(U[:, 0] @ Sigma_obs @ U[:, 0])  # readSNR scale for matching
    SocBot = float(U[:, -1] @ Sigma_obs @ U[:, -1])

    conds = {}
    # Busch regime (Sigma_act = Sigma_obs), eigen-directions
    conds["IM"] = (U[:, 0].copy(), Sigma_obs, Sigma_obs, sigma_read)
    conds["WMP"] = (U[:, 1].copy(), Sigma_obs, Sigma_obs, sigma_read)
    conds["OMP"] = (U[:, -1].copy(), Sigma_obs, Sigma_obs, sigma_read)

    # D1: high PEV (u_0, readable) but uncontrollable -> Sigma_act tiny along u_0
    lam_d1 = lam.copy(); lam_d1[0] = lam[-1]
    conds["D1_onMan_uncontrol"] = (U[:, 0].copy(), Sigma_obs, cov_from(U, lam_d1), sigma_read)

    # D2: low PEV (u_-1) but made controllable (Sigma_act large along u_-1) AND readable
    # (read noise lowered so its readSNR matches IM's).
    lam_d2 = lam.copy(); lam_d2[-1] = lam[0]
    sigma_read_d2 = sigma_read * np.sqrt(SocBot / SocTop)  # match IM readSNR
    conds["D2_offMan_usable"] = (U[:, -1].copy(), Sigma_obs, cov_from(U, lam_d2), sigma_read_d2)

    return conds, Sigma_obs


def sweep_directions(U, lam, sigma_read, rng, n=16):
    """Directions spanning readSNR x controllability: random extreme-eigendirection mixes,
    random Sigma_act perturbations, random read-noise scales. For the P4 regression."""
    out = []
    for _ in range(n):
        w = rng.standard_normal(D)
        w[3:-3] = 0.0                       # concentrate on extreme eigendirections -> spread PEV
        C = U @ w
        C = C / np.linalg.norm(C)
        lam_act = lam.copy()
        j = rng.integers(0, D)
        lam_act[j] *= rng.choice([0.03, 0.15, 1.0, 6.0, 30.0])
        sr = sigma_read * rng.choice([0.1, 0.3, 1.0, 3.0])
        out.append((C, cov_from(U, lam_act), sr))
    return out


# ----------------------------- stats helpers -------------------------------- #
def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    g = (a.mean() - b.mean()) / sp
    return g * (1 - 3 / (4 * (na + nb) - 9))


def eval_condition(C, Sob, Sact, sr, P, seeds):
    return np.array([run_session(C, Sob, Sact, sr, P["lam_plast"], P["n_trials"], P["lr"],
                                 np.random.default_rng(s)) for s in seeds])


# ----------------------------- runner --------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm"], default="confirm")
    ap.add_argument("--rho", type=float, default=0.75)
    ap.add_argument("--sigma_read", type=float, default=0.15)
    ap.add_argument("--lam_plast", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=55)
    ap.add_argument("--lr", type=float, default=0.3)
    args = ap.parse_args()

    P = dict(lam_plast=args.lam_plast, n_trials=args.n_trials, lr=args.lr)
    seeds = list(range(500, 504)) if args.mode == "calib" else list(range(600, 630))

    U = make_basis(np.random.default_rng(42))        # fixed geometry, seed-independent
    lam = spectrum(args.rho)
    conds, Sigma_obs = build_conditions(U, lam, args.sigma_read)

    print("=" * 86)
    print(f"  BUSCH E1 — deconfound  mode={args.mode}  N={len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f" D={D} rho={args.rho}")
    print(f"  knobs: sigma_read={args.sigma_read} lam_plast={args.lam_plast} "
          f"n_trials={args.n_trials} lr={args.lr}")
    print("=" * 86)
    print(f"  {'condition':22s} {'PEV':>6s} {'readSNR':>9s} {'control':>8s} "
          f"{'dCtrl':>7s} {'sem':>6s}")

    results = {}
    for name, (C, Sob, Sact, sr) in conds.items():
        d = eval_condition(C, Sob, Sact, sr, P, seeds)
        results[name] = d
        print(f"  {name:22s} {pev(C, Sob):6.3f} {read_snr(C, Sob, sr):9.1f} "
              f"{controllability(C, Sact):8.3f} {d.mean():7.1f} "
              f"{d.std(ddof=1)/np.sqrt(len(d)):6.1f}")

    def cmp(a, b, tail):
        try:
            _, p = stats.wilcoxon(results[a], results[b], alternative=tail)
        except ValueError:
            p = float("nan")
        return hedges_g(results[a], results[b]), p

    print("\n  --- pre-registered tests ---")
    for a, b in [("IM", "WMP"), ("WMP", "OMP"), ("IM", "OMP")]:
        g, p = cmp(a, b, "greater"); print(f"  P1  {a}>{b:4s}  g={g:+.2f} p={p:.4f}")
    g, p = cmp("IM", "D1_onMan_uncontrol", "greater")
    print(f"  P2  IM>D1     g={g:+.2f} p={p:.4f}   (D1 should FAIL despite high PEV)")
    g, p = cmp("D1_onMan_uncontrol", "OMP", "two-sided")
    print(f"  P2  D1~OMP    g={g:+.2f} p={p:.4f}")
    g, p = cmp("D2_offMan_usable", "OMP", "greater")
    print(f"  P3  D2>OMP    g={g:+.2f} p={p:.4f}   (D2 should LEARN despite low PEV)")
    g, p = cmp("D2_offMan_usable", "WMP", "two-sided")
    print(f"  P3  D2~WMP    g={g:+.2f} p={p:.4f}")

    # P4: ΔControl ~ readSNR + controllability over a direction sweep
    print("\n  --- P4: direction-sweep regression (standardized betas) ---")
    sweep = sweep_directions(U, lam, args.sigma_read, np.random.default_rng(7), n=16)
    X, y = [], []
    for C, Sact, sr in sweep:
        d = eval_condition(C, Sigma_obs, Sact, sr, P, seeds).mean()
        X.append([pev(C, Sigma_obs), read_snr(C, Sigma_obs, sr), controllability(C, Sact)])
        y.append(d)
    X = np.array(X); y = np.array(y)
    print("    univariate Spearman vs dCtrl:")
    for j, nm in enumerate(["PEV", "readSNR", "controllability"]):
        r, p = stats.spearmanr(X[:, j], y)
        print(f"      {nm:16s} rho={r:+.2f} p={p:.3f}")
    Xs = np.log(np.clip(X[:, 1:], 1e-9, None))           # readSNR, controllability
    Xz = (Xs - Xs.mean(0)) / Xs.std(0)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yz)), Xz]), yz, rcond=None)
    print("    multiple regression dCtrl ~ readSNR + controllability:")
    for nm, b in zip(["intercept", "readSNR", "controllability"], beta):
        print(f"      {nm:16s} beta={b:+.3f}")
    print("    (P4 predicts BOTH readSNR and controllability betas positive & non-trivial)")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/e1_{args.mode}.npz", sweep_X=X, sweep_y=y, beta=beta,
             **{k: v for k, v in results.items()})
    print(f"\n  saved results/e1_{args.mode}.npz")


if __name__ == "__main__":
    main()
