"""
RDD crossover — WHEN does lateral inhibition help downstream?
=============================================================
Pre-registered in ../PREREG_RDD_crossover.md.

Follow-up to rdd_li.py, which found LI HURTS downstream MSE — but in a regime where
channels self-separate without it (no collapse). RDD's claim is conditional: LI helps
when channels would otherwise COLLAPSE into the same basin ("escape through symmetry
breaking"). This test constructs the collapse regime and looks for the crossover.

Two init regimes (the manipulated variable):
  asym : per-channel W_in & W_out  -> channels self-separate (no collapse). LI expected to hurt.
  sym  : SHARED W_in & W_out + identical τ init -> channels degenerate; LI is the ONLY
         symmetry-breaking force. LI expected to HELP (separate timescales -> cover the signal).
Crossed with {no-LI, LI(β=10)} and M ∈ {3, 8}. Task = rdd_li's multi-frequency next-step prediction.
"""
import sys, os, math, pickle
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R   # reuse data, config, stats


class SSM(nn.Module):
    def __init__(self, M, symmetric, hidden=R.HIDDEN, tau_init=R.TAU_INIT, theta_jitter=0.0):
        super().__init__()
        self.M, self.hidden, self.symmetric = M, hidden, symmetric
        t0 = torch.full((M,), math.log(tau_init))
        if theta_jitter:                              # tiny seed: real systems aren't perfectly symmetric
            t0 = t0 + theta_jitter * torch.randn(M)
        self.theta = nn.Parameter(t0)
        if symmetric:
            self.W_in = nn.Parameter(torch.randn(hidden, 1) * 0.1)        # shared across channels
            self.W_out = nn.Parameter(torch.randn(hidden, 1) * 0.1)       # shared across channels
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
                base = u[:, t, :] @ self.W_in.t()                 # (B, hidden) same for all channels
                inp = base.unsqueeze(1).expand(B, self.M, self.hidden)
                h = (1 - a) * h + a * inp
                y = h.sum(dim=1) @ self.W_out + self.b            # (B,1)
            else:
                inp = torch.einsum('mhi,bi->bmh', self.W_in, u[:, t, :])
                h = (1 - a) * h + a * inp
                y = h.reshape(B, self.M * self.hidden) @ self.W_out.t() + self.b
            outs.append(y)
        return torch.stack(outs, dim=1)

    def taus(self):
        return torch.exp(self.theta).detach().cpu().numpy()


def train_one(M, symmetric, beta, seed, steps=R.STEPS, theta_jitter=0.0):
    R.set_seed(seed)
    rng = np.random.RandomState(seed)
    u_tr = R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)
    u_te = R.gen_data(R.N_TEST, R.FREQS, R.AMPS, np.random.RandomState(seed + 10000))
    u_ood = R.gen_data(R.N_TEST, R.OOD_FREQS, R.OOD_AMPS, np.random.RandomState(seed + 20000))
    model = SSM(M, symmetric, theta_jitter=theta_jitter).to(R.DEVICE)
    opt = torch.optim.Adam([
        {"params": [model.W_in, model.W_out, model.b], "lr": R.LR},
        {"params": [model.theta], "lr": R.LR_TAU},
    ])
    for _ in range(steps):
        opt.zero_grad()
        loss = R.mse_next(model, u_tr) + R.li_loss(model.theta, beta, inert=False)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr = float(R.mse_next(model, u_tr)); te = float(R.mse_next(model, u_te))
        ood = float(R.mse_next(model, u_ood))
    return {"train_mse": tr, "test_mse": te, "ood_mse": ood,
            "div": R.diversity(model.taus()), "taus": model.taus().tolist()}


# sym regime gets a tiny τ-init seed (JITTER) so it is the meaningful *collapse* regime
# (near-degenerate, LI can act) rather than the LI-proof perfectly-symmetric fixed point.
JITTER = 0.1
CONDITIONS = [(f"M{M}_{init}_{'LI' if b else 'noLI'}", M, init == "sym", b,
               JITTER if init == "sym" else 0.0)
              for M in (3, 8) for init in ("asym", "sym") for b in (0.0, 10.0)]


def main(seeds):
    data = {name: [] for name, *_ in CONDITIONS}
    for name, M, sym, beta, jit in CONDITIONS:
        for s in seeds:
            data[name].append(train_one(M, sym, beta, s, theta_jitter=jit))
        d = data[name]
        print(f"{name:<16} div={np.mean([r['div'] for r in d]):6.3f}  "
              f"test_mse={np.mean([r['test_mse'] for r in d]):.4f}  "
              f"ood_mse={np.mean([r['ood_mse'] for r in d]):.4f}")

    def col(c, k): return np.array([r[k] for r in data[c]], float)
    print(f"\n{'='*80}\n  RDD CROSSOVER (N={len(seeds)}, device={R.DEVICE})\n{'='*80}")
    print("  Does LI help (test MSE) in sym (collapse) regime but hurt in asym (no-collapse)?\n")
    for M in (3, 8):
        for init in ("asym", "sym"):
            no = col(f"M{M}_{init}_noLI", 'test_mse'); li = col(f"M{M}_{init}_LI", 'test_mse')
            g = R.hedges_g(li, no); p_help = R.pw(li, no, "less"); p_hurt = R.pw(li, no, "greater")
            verdict = ("LI HELPS" if (p_help < 0.05 and g <= -0.5)
                       else "LI HURTS" if (p_hurt < 0.05 and g >= 0.5)
                       else "neutral")
            dn = col(f"M{M}_{init}_noLI", 'div').mean(); dl = col(f"M{M}_{init}_LI", 'div').mean()
            print(f"  M={M} {init:<4}  noLI {no.mean():.4f} (div {dn:5.2f}) -> LI {li.mean():.4f} "
                  f"(div {dl:5.2f})  g={g:+.2f} p(help)={p_help:.3f}  => {verdict}")
    # OOD crossover (M=8)
    print("\n  OOD (M=8):")
    for init in ("asym", "sym"):
        no = col(f"M8_{init}_noLI", 'ood_mse'); li = col(f"M8_{init}_LI", 'ood_mse')
        g = R.hedges_g(li, no); p = R.pw(li, no, "less")
        print(f"  M=8 {init:<4}  noLI {no.mean():.4f} -> LI {li.mean():.4f}  g={g:+.2f} p(help)={p:.3f}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_crossover.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/rdd_crossover.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20)
    main(list(range(ap.parse_args().seeds)))
