"""
RDD deadlock — is lateral inhibition SPECIAL, or just one way to break symmetry?
================================================================================
Pre-registered in ../PREREG_RDD_deadlock.md. Reuses rdd_li.py (data, config, stats).

RDD's central claim: a symmetric multi-channel SSM sits in a GEOMETRIC DEADLOCK (all channels
identical, gradients identical, so they never differentiate), and lateral inhibition (Gaussian
τ-repulsion) is the force that escapes it. We test that claim two ways:

  #2 Characterize the deadlock: with perfectly symmetric init (shared W_in/W_out, identical τ) and no
     perturbation, do channels stay collapsed (diversity ~0, MSE ~ single-channel)? How does the harm
     scale with M (wasted capacity)?

  #1 Is LI special? In this architecture channels differ ONLY by τ, so only a perturbation that
     *distinguishes channels* can escape. Taxonomy:
       channel-distinguishing (predict ESCAPE):  LI repulsion | τ-init jitter | channel dropout
       symmetric              (predict NO escape): input noise | weight decay
     If cheap jitter/dropout escape the deadlock AND match LI on downstream MSE, LI is not a special
     mechanism — just one symmetry-breaker among many. `asym` (per-channel weights) is the ceiling.
"""
import sys, os, math, pickle
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
from rdd_li import DEVICE


class SSM(nn.Module):
    def __init__(self, M, symmetric, hidden=R.HIDDEN, tau_init=R.TAU_INIT, theta_jitter=0.0, dropout_p=0.0):
        super().__init__()
        self.M, self.hidden, self.symmetric, self.dropout_p = M, hidden, symmetric, dropout_p
        t0 = torch.full((M,), math.log(tau_init))
        if theta_jitter:
            t0 = t0 + theta_jitter * torch.randn(M)
        self.theta = nn.Parameter(t0)
        if symmetric:
            self.W_in = nn.Parameter(torch.randn(hidden, 1) * 0.1)        # shared across channels
            self.W_out = nn.Parameter(torch.randn(hidden, 1) * 0.1)
        else:
            self.W_in = nn.Parameter(torch.randn(M, hidden, 1) * 0.1)
            self.W_out = nn.Parameter(torch.randn(1, M * hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, u):
        B, T, _ = u.shape
        a = (1.0 / torch.clamp(torch.exp(self.theta), min=1.05)).view(1, self.M, 1)
        h = torch.zeros(B, self.M, self.hidden, device=u.device)
        outs = []
        for t in range(T):
            if self.symmetric:
                base = u[:, t, :] @ self.W_in.t()
                inp = base.unsqueeze(1).expand(B, self.M, self.hidden)
                h = (1 - a) * h + a * inp
                hd = h
                if self.training and self.dropout_p > 0:                  # channel dropout (output only)
                    mask = (torch.rand(B, self.M, 1, device=u.device) > self.dropout_p).float()
                    hd = h * mask / (1 - self.dropout_p)
                y = hd.sum(dim=1) @ self.W_out + self.b
            else:
                inp = torch.einsum('mhi,bi->bmh', self.W_in, u[:, t, :])
                h = (1 - a) * h + a * inp
                y = h.reshape(B, self.M * self.hidden) @ self.W_out.t() + self.b
            outs.append(y)
        return torch.stack(outs, dim=1)

    def taus(self):
        return torch.exp(self.theta).detach().cpu().numpy()


# arm -> (symmetric, beta, jitter, dropout, innoise, wdecay)
# Regime 1 (PURE symmetry, jitter=0): can a mechanism escape a TRUE deadlock from nothing?
# Regime 2 (SEEDED asymmetry, jitter=0.1): given a seed to amplify, does LI beat the seed alone / dropout?
ARMS = {
    "asym":      dict(sym=False, beta=0.0,  jitter=0.0, dropout=0.0, innoise=0.0, wd=0.0),  # ceiling
    # --- pure-symmetry regime ---
    "none":      dict(sym=True,  beta=0.0,  jitter=0.0, dropout=0.0, innoise=0.0, wd=0.0),  # deadlock
    "LI":        dict(sym=True,  beta=10.0, jitter=0.0, dropout=0.0, innoise=0.0, wd=0.0),
    "jitter":    dict(sym=True,  beta=0.0,  jitter=0.1, dropout=0.0, innoise=0.0, wd=0.0),
    "dropout":   dict(sym=True,  beta=0.0,  jitter=0.0, dropout=0.3, innoise=0.0, wd=0.0),
    "innoise":   dict(sym=True,  beta=0.0,  jitter=0.0, dropout=0.0, innoise=0.5, wd=0.0),
    "wdecay":    dict(sym=True,  beta=0.0,  jitter=0.0, dropout=0.0, innoise=0.0, wd=1e-2),
    # --- seeded-asymmetry regime (jitter=0.1 baseline) ---
    "seed":      dict(sym=True,  beta=0.0,  jitter=0.1, dropout=0.0, innoise=0.0, wd=0.0),  # seed only
    "LI_seed":   dict(sym=True,  beta=10.0, jitter=0.1, dropout=0.0, innoise=0.0, wd=0.0),
    "drop_seed": dict(sym=True,  beta=0.0,  jitter=0.1, dropout=0.3, innoise=0.0, wd=0.0),
}
PURE = ["LI", "jitter", "dropout", "innoise", "wdecay"]
SEEDED = ["seed", "LI_seed", "drop_seed"]


def train_one(M, arm, seed, steps=R.STEPS):
    c = ARMS[arm]
    R.set_seed(seed); rng = np.random.RandomState(seed)
    u_tr = R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)
    u_te = R.gen_data(R.N_TEST, R.FREQS, R.AMPS, np.random.RandomState(seed + 10000))
    u_ood = R.gen_data(R.N_TEST, R.OOD_FREQS, R.OOD_AMPS, np.random.RandomState(seed + 20000))
    model = SSM(M, c["sym"], theta_jitter=c["jitter"], dropout_p=c["dropout"]).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": [model.W_in, model.W_out, model.b], "lr": R.LR},
        {"params": [model.theta], "lr": R.LR_TAU, "weight_decay": c["wd"]},
    ])
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        u_in = u_tr + c["innoise"] * torch.randn_like(u_tr) if c["innoise"] else u_tr
        pred = model(u_in)[:, :-1, :]
        loss = ((pred - u_tr[:, 1:, :]) ** 2).mean() + R.li_loss(model.theta, c["beta"], inert=False)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr = float(R.mse_next(model, u_tr)); te = float(R.mse_next(model, u_te))
        ood = float(R.mse_next(model, u_ood))
    return {"train_mse": tr, "test_mse": te, "ood_mse": ood,
            "div": R.diversity(model.taus()), "taus": model.taus().tolist()}


def main(seeds, M=8, Msweep=(2, 3, 5, 8, 12)):
    def col(d, k): return np.array([r[k] for r in d], float)

    # ---- escape-mechanism comparison (fixed M) ----
    data = {arm: [train_one(M, arm, s) for s in seeds] for arm in ARMS}
    notes = {"asym": "ceiling (per-channel weights)", "none": "deadlock baseline",
             "LI": "RDD repulsion", "jitter": "τ-init jitter", "dropout": "channel dropout",
             "innoise": "input noise (symmetric)", "wdecay": "weight decay (symmetric)",
             "seed": "seed jitter only", "LI_seed": "seed + LI", "drop_seed": "seed + dropout"}

    print(f"\n{'='*78}\n  RDD DEADLOCK at M={M} (N={len(seeds)}, dev={DEVICE})\n{'='*78}")
    print(f"  {'arm':<10} {'div':>7} {'test_mse':>9} {'ood_mse':>9}   note")
    for arm in ["asym", "none"] + PURE + SEEDED:
        d = data[arm]
        print(f"  {arm:<10} {col(d,'div').mean():7.3f} {col(d,'test_mse').mean():9.4f} "
              f"{col(d,'ood_mse').mean():9.4f}   {notes[arm]}")

    none_mse = col(data["none"], 'test_mse')
    print(f"\n  #2 PURE SYMMETRY — can a mechanism escape a TRUE deadlock from nothing?"
          f"  (none div=0, mse={none_mse.mean():.4f})")
    for arm in PURE:
        dv = col(data[arm], 'div').mean(); x = col(data[arm], 'test_mse')
        tag = "ESCAPES" if dv > 1.0 else "inert" if dv < 0.05 else "weak"
        print(f"    {arm:<10} div={dv:6.2f}  test={x.mean():.4f}  -> {tag}")
    print("  => predict: LI INERT at true deadlock (repulsion grad vanishes at τ_i=τ_j); only "
          "asymmetry-injectors move.")

    print(f"\n  #1 GIVEN A SEED, IS LI SPECIAL?  (seeded regime, jitter=0.1)")
    seed_mse = col(data["seed"], 'test_mse')
    for arm in SEEDED:
        dv = col(data[arm], 'div').mean(); x = col(data[arm], 'test_mse')
        g = R.hedges_g(x, seed_mse); p = R.pw(x, seed_mse, "less") if arm != "seed" else 1.0
        print(f"    {arm:<10} div={dv:6.2f}  test={x.mean():.4f}  vs seed: g={g:+.2f} p(help)={p:.3f}")
    print("  => if LI_seed does NOT beat 'seed' on MSE, LI's amplification adds no value beyond the "
          "trivial init seed.")

    # ---- #2 deadlock severity vs M ----
    print(f"\n  #2 DEADLOCK SEVERITY vs M  (test MSE: deadlock 'none' vs ceiling 'asym')")
    print(f"    {'M':>3} {'none(div)':>10} {'none_mse':>9} {'asym_mse':>9} {'gap':>7}")
    sev = {}
    for m in Msweep:
        dn = [train_one(m, "none", s) for s in seeds]
        da = [train_one(m, "asym", s) for s in seeds]
        nm, am = col(dn, 'test_mse').mean(), col(da, 'test_mse').mean()
        sev[m] = (col(dn, 'div').mean(), nm, am)
        print(f"    {m:>3} {sev[m][0]:>10.3f} {nm:>9.4f} {am:>9.4f} {nm-am:>7.4f}")
    from scipy.stats import spearmanr
    ms = list(Msweep); gaps = [sev[m][1] - sev[m][2] for m in ms]
    rho, p = spearmanr(ms, gaps)
    print(f"    TREND Spearman(M, gap) = {rho:+.3f} (p={p:.3f})  [>0: deadlock wastes more capacity at larger M]")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_deadlock.pkl"), "wb") as f:
        pickle.dump({"escape": data, "severity": sev}, f)
    print("\n  Saved results/rdd_deadlock.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20)
    main(list(range(ap.parse_args().seeds)))
