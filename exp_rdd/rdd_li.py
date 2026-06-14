"""
RDD lateral-inhibition NN proof-of-concept — EXTENDED to the open question.
==========================================================================
Pre-registered in ../PREREG_RDD_LI.md.

RDD (Cai, "Lateral inhibition enables escape from geometric deadlock") shows, in a
multi-channel linear SSM, that a Gaussian repulsion loss on channel timescales raises
τ-diversity (3.70 -> 9.40). The paper explicitly leaves OPEN whether that diversity
translates to **downstream performance** (prediction MSE). This experiment answers it,
and folds in the γ-inertness test (Direction 1): a β_LI gain with no actual repulsion
kernel should be INERT.

Model (RDD Table S8): M parallel first-order low-pass channels, learned τ_m = exp(θ_m):
    h_m(t+1) = (1 - a_m) h_m(t) + a_m W_in^m u(t),   a_m = 1/τ_m
    ŷ(t) = W_out · concat_m h_m(t) + b
Loss = MSE(ŷ(t), u(t+1)) + β_LI · Σ_{i<j} exp(-(τ_i - τ_j)^2 / 2ℓ^2)
Task: next-step prediction of multi-frequency sine mixtures.
"""
import sys, os, math, pickle, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

# ── RDD Table S8 config ──
FREQS = [0.02, 0.05, 0.08, 0.12, 0.18]
AMPS = [1.0, 3.0, 1.0, 1.0, 1.0]          # 0.05 has 3x amplitude
OOD_FREQS = [0.035, 0.065, 0.10, 0.15, 0.20]   # held-out frequencies (mild OOD)
OOD_AMPS = [1.0, 3.0, 1.0, 1.0, 1.0]
SEQ_LEN = 100
HIDDEN = 8
LI_SCALE = 5.0                            # ℓ for the NN PoC (Table S8)
TAU_INIT = 15.0
N_TRAIN = 500
N_TEST = 500
STEPS = 200
LR = 0.01
LR_TAU = 0.003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def gen_data(n, freqs, amps, rng):
    """n sine-mixture sequences with random per-sample phases. Returns (n, SEQ_LEN, 1)."""
    t = np.arange(SEQ_LEN)[None, :]                       # (1, T)
    u = np.zeros((n, SEQ_LEN))
    for f, a in zip(freqs, amps):
        ph = rng.uniform(0, 2 * np.pi, size=(n, 1))
        u += a * np.sin(2 * np.pi * f * t + ph)
    return torch.tensor(u[:, :, None], dtype=torch.float32, device=DEVICE)


class MultiChannelSSM(nn.Module):
    def __init__(self, M, hidden=HIDDEN, tau_init=TAU_INIT):
        super().__init__()
        self.M, self.hidden = M, hidden
        self.theta = nn.Parameter(torch.full((M,), math.log(tau_init)))  # τ=exp(θ), identical init
        self.W_in = nn.Parameter(torch.randn(M, hidden, 1) * 0.1)
        self.W_out = nn.Parameter(torch.randn(1, M * hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, u):                                  # u: (B, T, 1)
        B, T, _ = u.shape
        a = 1.0 / torch.clamp(torch.exp(self.theta), min=1.05)   # (M,) stable low-pass
        a = a.view(1, self.M, 1)
        h = torch.zeros(B, self.M, self.hidden, device=u.device)
        outs = []
        for t in range(T):
            inp = torch.einsum('mhi,bi->bmh', self.W_in, u[:, t, :])
            h = (1 - a) * h + a * inp
            outs.append(h.reshape(B, self.M * self.hidden) @ self.W_out.t() + self.b)
        return torch.stack(outs, dim=1)                   # (B, T, 1)

    def taus(self):
        return torch.exp(self.theta).detach().cpu().numpy()


def diversity(taus):
    if len(taus) < 2: return 0.0
    return float(np.mean([abs(taus[i] - taus[j])
                          for i in range(len(taus)) for j in range(i + 1, len(taus))]))


def li_loss(theta, beta, inert):
    """Gaussian repulsion. If inert=True, detach τ -> gain present but ZERO gradient."""
    if beta == 0:
        return torch.zeros((), device=theta.device)
    tau = torch.exp(theta)
    if inert:
        tau = tau.detach()
    s = torch.zeros((), device=theta.device)
    M = tau.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            s = s + torch.exp(-(tau[i] - tau[j]) ** 2 / (2 * LI_SCALE ** 2))
    return beta * s


def mse_next(model, u):
    pred = model(u)[:, :-1, :]
    target = u[:, 1:, :]
    return ((pred - target) ** 2).mean()


def train_one(M, beta, inert, seed, steps=STEPS):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    u_tr = gen_data(N_TRAIN, FREQS, AMPS, rng)
    u_te = gen_data(N_TEST, FREQS, AMPS, np.random.RandomState(seed + 10000))
    u_ood = gen_data(N_TEST, OOD_FREQS, OOD_AMPS, np.random.RandomState(seed + 20000))
    model = MultiChannelSSM(M).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": [model.W_in, model.W_out, model.b], "lr": LR},
        {"params": [model.theta], "lr": LR_TAU},
    ])
    for _ in range(steps):
        opt.zero_grad()
        loss = mse_next(model, u_tr) + li_loss(model.theta, beta, inert)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr = float(mse_next(model, u_tr)); te = float(mse_next(model, u_te))
        ood = float(mse_next(model, u_ood))
    taus = model.taus()
    return {"train_mse": tr, "test_mse": te, "ood_mse": ood,
            "div": diversity(taus), "taus": taus.tolist()}


# ── stats ──
def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return 0.0 if sp == 0 else (x.mean()-y.mean())/sp * (1 - 3/(4*(nx+ny-2)-1))

def pw(x, y, alt):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y): return 1.0
    try: return float(wilcoxon(x, y, alternative=alt).pvalue)
    except ValueError: return 1.0

def boot_ci(x, y, n=10000):
    d = np.asarray(x, float) - np.asarray(y, float); rng = np.random.RandomState(0)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


CONDITIONS = (
    [("M1", 1, 0.0, False)]
    + [("M3_noLI", 3, 0.0, False)]
    + [(f"M3_LI_b{b:g}", 3, b, False) for b in (0.1, 1.0, 10.0, 30.0)]
    + [(f"M3_inert_b{b:g}", 3, b, True) for b in (0.1, 1.0, 10.0, 30.0)]
)


def main(seeds):
    data = {name: [] for name, *_ in CONDITIONS}
    for name, M, beta, inert in CONDITIONS:
        t0 = time.time()
        for s in seeds:
            data[name].append(train_one(M, beta, inert, s))
        d = data[name]
        print(f"{name:<14} div={np.mean([r['div'] for r in d]):6.3f}  "
              f"test_mse={np.mean([r['test_mse'] for r in d]):.4f}  "
              f"ood_mse={np.mean([r['ood_mse'] for r in d]):.4f}  ({time.time()-t0:.0f}s)")

    def col(c, k): return np.array([r[k] for r in data[c]], float)
    print(f"\n{'='*78}\n  RDD-LI RESULTS (N={len(seeds)}, device={DEVICE})\n{'='*78}")

    print("\n  D3 PRIMARY — does LI-driven diversity lower downstream test MSE?")
    base = "M3_noLI"; best_li = "M3_LI_b10"
    for c in [best_li, "M1"]:
        x, y = col(best_li, 'test_mse'), col(base if c == best_li else best_li, 'test_mse')
    for lab, a, b in [("LI_b10 < noLI", "M3_LI_b10", "M3_noLI"),
                      ("LI_b10 < M1", "M3_LI_b10", "M1"),
                      ("LI_b10 < noLI (OOD)", "M3_LI_b10", "M3_noLI")]:
        k = 'ood_mse' if 'OOD' in lab else 'test_mse'
        x, y = col(a, k), col(b, k)
        g = hedges_g(x, y); lo, hi = boot_ci(x, y); p = pw(x, y, "less")
        print(f"    {lab:<22} {x.mean():.4f} vs {y.mean():.4f}  g={g:+.2f} CI[{lo:+.4f},{hi:+.4f}] p(<)={p:.3f}")

    print("\n  LI β-sweep (does benefit scale, per RDD monotonicity?)")
    for c in ["M3_noLI", "M3_LI_b0.1", "M3_LI_b1", "M3_LI_b10", "M3_LI_b30"]:
        print(f"    {c:<14} div={col(c,'div').mean():6.3f}  test_mse={col(c,'test_mse').mean():.4f}")

    print("\n  D1 γ-INERTNESS — real-LI responds to β, inert-LI does NOT")
    print(f"    {'β':>6} {'real div':>10} {'inert div':>11}")
    for b in (0.1, 1.0, 10.0, 30.0):
        rd = col(f"M3_LI_b{b:g}", 'div').mean(); idv = col(f"M3_inert_b{b:g}", 'div').mean()
        print(f"    {b:>6g} {rd:>10.3f} {idv:>11.3f}")
    real = np.array([col(f"M3_LI_b{b:g}", 'div').mean() for b in (0.1,1,10,30)])
    inert = np.array([col(f"M3_inert_b{b:g}", 'div').mean() for b in (0.1,1,10,30)])
    print(f"    real-LI div range across β = {real.max()-real.min():.3f} (responsive);  "
          f"inert-LI div range = {inert.max()-inert.min():.3f} (should be ~0)")

    print("\n  diversity↔usefulness: within M3-LI runs, corr(div, -test_mse)")
    from scipy.stats import spearmanr
    allruns = [r for b in (0.1,1,10,30) for r in data[f"M3_LI_b{b:g}"]]
    dv = [r['div'] for r in allruns]; ms = [r['test_mse'] for r in allruns]
    rho, p = spearmanr(dv, ms)
    print(f"    Spearman(div, test_mse) = {rho:+.3f} (p={p:.3f})  [negative => more diversity helps]")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_li.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/rdd_li.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    main(list(range(args.seeds)))
