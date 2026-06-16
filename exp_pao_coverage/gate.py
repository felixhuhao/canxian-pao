"""
PAO learned gate — does the trigger advantage survive without the oracle?

Pre-registration: ../PREREG_PAO_GATE.md.

Replaces the trigger experiment's Bayes-optimal gate (argmax of the noisy cue) with a gate net trained by
REINFORCE from reward only (no niche labels), as a contextual bandit over the K skills, at the monolith's
episode budget. Arms: gated-learned (new), gated-Bayes (oracle upper bound), fire-all, monolith.
Reuses everything from trigger.py.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import grid as G
import r4
import trigger as T

GATE_HIDDEN = 32
GATE_LR = 5e-3
GATE_EPISODES = G.K * T.MONO_UPDATES * T.MONO_EPT   # = monolith budget (2400), matched


class GateNet(nn.Module):
    def __init__(self, hidden=GATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(G.K, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, G.K))

    def forward(self, x):
        return self.net(x)


def _skill_success(net, k, env, k_clean):
    """Deploy a (frozen) skill greedily on true task k; return 1/0 success. The skill always reads its
    own clean one-hot k_clean + the true position."""
    env.set_task(k)
    env.reset(); done = False; succ = False
    while not done:
        a = G.greedy_action(net, T.clean_obs(env.x, env.y, k_clean))
        _, _, done, info = env.step(a); succ = info["success"]
    return float(succ)


def train_gate(skills, sigma, seed, episodes=GATE_EPISODES):
    """REINFORCE contextual bandit: context = noisy cue, arm = which skill to deploy, reward = success."""
    torch.manual_seed(seed * 19 + 3)
    rng = np.random.RandomState(seed * 19 + 3)
    gate = GateNet().to(G.DEVICE)
    opt = torch.optim.Adam(gate.parameters(), lr=GATE_LR)
    env = G.GridTasks(seed)
    baseline = 0.0
    for ep in range(episodes):
        k = ep % G.K
        cue = np.zeros(G.K, dtype=np.float32); cue[k] = 1.0
        cue = cue + rng.randn(G.K).astype(np.float32) * sigma
        ct = torch.as_tensor(cue, device=G.DEVICE).unsqueeze(0)
        logits = gate(ct)
        dist = torch.distributions.Categorical(logits=logits)
        j = dist.sample()
        r = _skill_success(skills[int(j.item())]["net"], k, env, int(j.item()))
        baseline = 0.99 * baseline + 0.01 * r
        loss = -dist.log_prob(j) * (r - baseline)
        opt.zero_grad(); loss.backward(); opt.step()
    return gate


def roll_gated_learned(skills, gate, true_k, sigma, seed, n=T.EVAL_N):
    """Deploy argmax g(noisy cue); success iff it reaches the TRUE goal."""
    env = G.GridTasks(seed); env.set_task(true_k)
    rng = np.random.RandomState(seed * 7 + 1)
    s = 0
    for _ in range(n):
        cue = np.zeros(G.K, dtype=np.float32); cue[true_k] = 1.0
        cue = cue + rng.randn(G.K).astype(np.float32) * sigma
        with torch.no_grad():
            j = int(torch.argmax(gate(torch.as_tensor(cue, device=G.DEVICE).unsqueeze(0))))
        s += _skill_success(skills[j]["net"], true_k, env, j)
    return s / n


def run_seed(seed):
    skills = [r4.train_mastered(k, seed) for k in range(G.K)]
    out = {}
    for sigma in T.SIGMAS:
        gate = train_gate(skills, sigma, seed)
        mono = T.train_monolith(sigma, seed)
        gl = np.mean([roll_gated_learned(skills, gate, k, sigma, seed) for k in range(G.K)])
        gb = np.mean([T.roll_gated(skills, k, sigma, seed) for k in range(G.K)])
        fa = np.mean([T.roll_fireall(skills, k, sigma, seed) for k in range(G.K)])
        mo = np.mean([T.roll_monolith(mono, k, sigma, seed) for k in range(G.K)])
        out[sigma] = (gl, gb, fa, mo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm", "long"], default="confirm")
    args = ap.parse_args()
    seeds = {"calib": range(940, 943), "confirm": range(950, 958),
             "long": range(960, 980)}[args.mode]
    seeds = list(seeds)

    print("=" * 86)
    print(f"  PAO LEARNED GATE  mode={args.mode}  N={len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f" K={G.K} dev={G.DEVICE}")
    print(f"  sigmas={T.SIGMAS}  gate: hidden={GATE_HIDDEN} lr={GATE_LR} episodes={GATE_EPISODES} "
          f"(=mono budget)")
    print("=" * 86)

    arms = ("learned", "bayes", "fireall", "mono")
    res = {a: {s: [] for s in T.SIGMAS} for a in arms}
    for sd in seeds:
        o = run_seed(sd)
        for s in T.SIGMAS:
            gl, gb, fa, mo = o[s]
            res["learned"][s].append(gl); res["bayes"][s].append(gb)
            res["fireall"][s].append(fa); res["mono"][s].append(mo)
        print(f"  seed {sd}: " + "  ".join(
            f"s{s}:L{o[s][0]:.2f}/B{o[s][1]:.2f}/F{o[s][2]:.2f}/M{o[s][3]:.2f}" for s in T.SIGMAS))

    print(f"\n  {'sigma':>6s} {'learned':>9s} {'bayes':>9s} {'fireall':>9s} {'mono':>9s} "
          f"{'B-L gap':>8s} {'L-M g':>7s} {'L-F g':>7s}")
    summ = {}
    for s in T.SIGMAS:
        L = np.array(res["learned"][s]); B = np.array(res["bayes"][s])
        F = np.array(res["fireall"][s]); M = np.array(res["mono"][s])
        summ[s] = (L, B, F, M)
        print(f"  {s:6.1f} {L.mean():9.2f} {B.mean():9.2f} {F.mean():9.2f} {M.mean():9.2f} "
              f"{B.mean()-L.mean():8.2f} {T.hedges_g(L, M):+7.2f} {T.hedges_g(L, F):+7.2f}")

    def wil(a, b, tail):
        try:
            return stats.wilcoxon(a, b, alternative=tail)[1]
        except ValueError:
            return float("nan")

    print("\n  --- pre-registered tests ---")
    print("  P1 learned>fireall (sigma<=1.0):")
    for s in T.SIGMAS:
        if s <= 1.0:
            L, B, F, M = summ[s]
            print(f"    sigma={s}: g={T.hedges_g(L, F):+.2f} p={wil(L, F, 'greater'):.4f}")
    print("  P2 oracle shortfall (bayes-learned; small => learned recovers the oracle):")
    for s in T.SIGMAS:
        L, B, F, M = summ[s]
        print(f"    sigma={s}: B-L={B.mean()-L.mean():+.3f}  p(B>L)={wil(B, L, 'greater'):.4f}")
    print("  P3 learned vs monolith (survives without oracle?):")
    for s in T.SIGMAS:
        L, B, F, M = summ[s]
        sign = "L>M" if L.mean() > M.mean() else "L<=M"
        print(f"    sigma={s}: L-M g={T.hedges_g(L, M):+.2f} p={wil(L, M, 'two-sided'):.4f}  {sign}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/gate_{args.mode}.npz", sigmas=np.array(T.SIGMAS),
             **{f"{a}_{s}": np.array(res[a][s]) for a in arms for s in T.SIGMAS})
    print(f"\n  saved results/gate_{args.mode}.npz")


if __name__ == "__main__":
    main()
