"""
Busch E2 — does nonlinear (T-PHATE-style) manifold geometry add anything beyond
readability x controllability?

Pre-registration: ../PREREG_BUSCH_E2.md.

Data from a genuinely nonlinear manifold  x = W2 tanh(gain * W1 u),  u ~ N(0, Sigma_lat).
  Sigma_obs = Cov(x)            (global; what Busch's PCA/decoder see)  -> PEV, readability
  Sigma_act = J Sigma_lat J^T   (local reachable cov; J = d x/d u|_0 = gain*W2 W1) -> controllability
Curvature makes Sigma_obs != Sigma_act, so a high-variance global direction can be locally NORMAL
(uncontrollable) -> the E1 dissociation arises for free, and is exactly what linear PCA mislabels.

Agent / learning / metric are reused unchanged from E1.
"""
import argparse
import os
import numpy as np
from scipy import stats
from scipy.linalg import eigh

from e1_deconfound import run_session, hedges_g, pev, read_snr, controllability

D = 20
d = 3
H = 64


def make_manifold(seed, gain, offset=1.5, n=8000, sigma_read=0.15):
    """offset = operating-point distance from the manifold's linear center (in latent std units).
    At u0=0 tanh is locally linear (tangent == global high-variance dirs); pushing u0 into a
    *curved/saturated* region shrinks the local tangent while global variance persists, so a
    high-PEV direction can become locally uncontrollable -- the case linear PCA mislabels."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((H, d)) / np.sqrt(d)
    W2 = rng.standard_normal((D, H)) / np.sqrt(H)
    Sigma_lat = np.diag([4.0, 2.0, 1.0])      # large latent var -> drive tanh into curvature
    Ld = np.sqrt(np.diag(Sigma_lat))

    U = rng.standard_normal((n, d)) * Ld
    X = np.tanh(gain * (U @ W1.T)) @ W2.T
    X -= X.mean(0)
    Sigma_obs = np.cov(X, rowvar=False)

    u0 = offset * Ld                          # operating point, off the linear center
    sech2 = 1.0 - np.tanh(gain * (W1 @ u0)) ** 2          # H, local saturation
    J = gain * (W2 * sech2) @ W1              # D x d, Jacobian at u0
    Sigma_act = J @ Sigma_lat @ J.T           # rank d, local reachable cov at u0
    return Sigma_obs, Sigma_act


def deconfound_dirs(Sigma_obs, Sigma_act, K=3):
    """HiPEV_uncontrol = the LEAST controllable among the top-K *high-variance* PCs (a genuinely
    high-PEV component that operating-point curvature has made locally uncontrollable -> linear PCA
    ranks it 'on-manifold' but it is unlearnable). On a 3-D manifold only the top ~3 PCs carry real
    PEV, so K=3. LoPEV_tangent = the most controllable among the bottom-K PCs (kept for symmetry)."""
    w, V = np.linalg.eigh(Sigma_obs)                      # ascending
    top = [V[:, D - 1 - k] for k in range(K)]            # highest variance
    bot = [V[:, k] for k in range(K)]                    # lowest variance
    ctrl = lambda c: float(c @ Sigma_act @ c)
    hi = min(top, key=ctrl)                               # high PEV, min controllability
    lo = max(bot, key=ctrl)                               # low PEV, max controllability
    return hi / np.linalg.norm(hi), lo / np.linalg.norm(lo)


def top_eigvec(M, which="max"):
    w, V = np.linalg.eigh(M)
    v = V[:, -1] if which == "max" else V[:, 0]
    return v / np.linalg.norm(v)


def conditions(Sigma_obs, Sigma_act):
    pcs = top_eigvec  # alias
    w, V = np.linalg.eigh(Sigma_obs)          # ascending eigvals
    IM = V[:, -1]; WMP = V[:, -2]; OMP = V[:, 0]
    hi, lo = deconfound_dirs(Sigma_obs, Sigma_act)
    TAN = top_eigvec(Sigma_act, "max")
    NOR = top_eigvec(Sigma_act, "min")        # smallest reachable variance -> normal
    return {
        "IM_lin": IM, "WMP_lin": WMP, "OMP_lin": OMP,
        "HiPEV_normal": hi, "LoPEV_tangent": lo,
        "TAN": TAN, "NOR": NOR,
    }


def eval_dir(C, Sigma_obs, Sigma_act, sigma_read, P, seeds):
    return np.array([run_session(C, Sigma_obs, Sigma_act, sigma_read, P["lam_plast"],
                                 P["n_trials"], P["lr"], np.random.default_rng(s))
                     for s in seeds])


def cmp(res, a, b, tail):
    try:
        _, p = stats.wilcoxon(res[a], res[b], alternative=tail)
    except ValueError:
        p = float("nan")
    return hedges_g(res[a], res[b]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm"], default="confirm")
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--offset", type=float, default=1.5)
    ap.add_argument("--sigma_read", type=float, default=0.15)
    ap.add_argument("--lam_plast", type=float, default=0.5)
    ap.add_argument("--n_trials", type=int, default=55)
    ap.add_argument("--lr", type=float, default=0.3)
    args = ap.parse_args()

    P = dict(lam_plast=args.lam_plast, n_trials=args.n_trials, lr=args.lr)
    seeds = list(range(750, 754)) if args.mode == "calib" else list(range(700, 730))
    sr = args.sigma_read

    Sigma_obs, Sigma_act = make_manifold(seed=43, gain=args.gain, offset=args.offset)
    conds = conditions(Sigma_obs, Sigma_act)

    print("=" * 90)
    print(f"  BUSCH E2 — nonlinear manifold  mode={args.mode}  N={len(seeds)} "
          f"seeds {seeds[0]}..{seeds[-1]}  D={D} d={d} gain={args.gain} offset={args.offset}")
    print(f"  knobs: sigma_read={sr} lam_plast={args.lam_plast} n_trials={args.n_trials} lr={args.lr}")
    print("=" * 90)
    print(f"  {'condition':16s} {'PEV':>6s} {'readSNR':>9s} {'control':>9s} {'dCtrl':>7s} {'sem':>6s}")

    res = {}
    for name, C in conds.items():
        dd = eval_dir(C, Sigma_obs, Sigma_act, sr, P, seeds)
        res[name] = dd
        print(f"  {name:16s} {pev(C, Sigma_obs):6.3f} {read_snr(C, Sigma_obs, sr):9.1f} "
              f"{controllability(C, Sigma_act):9.4f} {dd.mean():7.1f} "
              f"{dd.std(ddof=1)/np.sqrt(len(dd)):6.1f}")

    print("\n  --- pre-registered tests ---")
    g, p = cmp(res, "TAN", "NOR", "greater")
    print(f"  P1  TAN>NOR          g={g:+.2f} p={p:.4f}")
    g, p = cmp(res, "IM_lin", "HiPEV_normal", "greater")
    print(f"  P2  IM_lin>HiPEV_n   g={g:+.2f} p={p:.4f}  (HiPEV_normal should FAIL despite high PEV)")
    g, p = cmp(res, "HiPEV_normal", "OMP_lin", "two-sided")
    print(f"  P2  HiPEV_n~OMP_lin  g={g:+.2f} p={p:.4f}")
    g, p = cmp(res, "LoPEV_tangent", "OMP_lin", "greater")
    print(f"  P3  LoPEV_t>OMP_lin  g={g:+.2f} p={p:.4f}  (LoPEV_tangent should LEARN despite low PEV)")

    # P4: regression over a direction sweep
    print("\n  --- P4: direction-sweep regression (standardized betas) ---")
    rng = np.random.default_rng(7)
    X, y = [], []
    for _ in range(24):
        C = rng.standard_normal(D); C /= np.linalg.norm(C)
        dd = eval_dir(C, Sigma_obs, Sigma_act, sr, P, seeds).mean()
        X.append([pev(C, Sigma_obs), read_snr(C, Sigma_obs, sr), controllability(C, Sigma_act)])
        y.append(dd)
    X = np.array(X); y = np.array(y)
    for j, nm in enumerate(["PEV", "readSNR", "controllability"]):
        r, p = stats.spearmanr(X[:, j], y)
        print(f"    Spearman {nm:16s} rho={r:+.2f} p={p:.3f}")
    Xs = np.log(np.clip(X[:, 1:], 1e-9, None))
    Xz = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yz)), Xz]), yz, rcond=None)
    for nm, b in zip(["intercept", "readSNR", "controllability"], beta):
        print(f"    beta {nm:16s} {b:+.3f}")

    # P5: does the PEV<->controllability coupling break down with operating-point curvature?
    # On a single shared manifold PEV and controllability coincide (data lives where it is reachable),
    # so PEV predicts learning -- UNLESS the operating point sits in a curved region. We measure, over a
    # random direction sweep at each offset: corr(PEV, controllability) [coupling], and the partial
    # predictive power of controllability for learning BEYOND PEV (extra R^2).
    print("\n  --- P5: coupling sweep (PEV is a proxy only while coupled to controllability) ---")
    print(f"    {'offset':>6s} {'corr(PEV,ctrl)':>14s} {'R2(PEV)':>9s} {'R2(PEV+ctrl)':>13s} "
          f"{'dR2_ctrl':>9s} {'HiPEVunc dCtrl':>15s} {'IM dCtrl':>9s}")
    rs = np.random.default_rng(11)
    Cs = rs.standard_normal((40, D)); Cs /= np.linalg.norm(Cs, axis=1, keepdims=True)
    for offs in [0.0, 0.5, 1.0, 1.5, 2.0]:
        So, Sa = make_manifold(seed=43, gain=args.gain, offset=offs)
        pv = np.array([pev(c, So) for c in Cs])
        cl = np.array([controllability(c, Sa) for c in Cs])
        lrn = np.array([eval_dir(c, So, Sa, sr, P, seeds).mean() for c in Cs])
        coup, _ = stats.spearmanr(pv, cl)
        lp = np.log(np.clip(pv, 1e-9, None)); lc = np.log(np.clip(cl, 1e-9, None))
        z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        y2 = z(lrn)
        def r2(cols):
            M = np.column_stack([np.ones(len(y2))] + cols)
            b, *_ = np.linalg.lstsq(M, y2, rcond=None)
            return 1 - ((y2 - M @ b) ** 2).sum() / (y2 ** 2).sum()
        r2_pev = r2([z(lp)]); r2_both = r2([z(lp), z(lc)])
        _, Vo = np.linalg.eigh(So)
        hi, _ = deconfound_dirs(So, Sa)
        hi_d = eval_dir(hi, So, Sa, sr, P, seeds).mean()
        im_d = eval_dir(Vo[:, -1], So, Sa, sr, P, seeds).mean()
        print(f"    {offs:6.1f} {coup:14.2f} {r2_pev:9.2f} {r2_both:13.2f} "
              f"{r2_both - r2_pev:9.2f} {hi_d:15.1f} {im_d:9.1f}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/e2_{args.mode}.npz", sweep_X=X, sweep_y=y, beta=beta,
             **{k: v for k, v in res.items()})
    print(f"\n  saved results/e2_{args.mode}.npz")


if __name__ == "__main__":
    main()
