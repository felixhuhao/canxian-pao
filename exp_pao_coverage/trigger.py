"""
PAO trigger under partial observability — is PAO's value the deployment decision?

Pre-registration: ../PREREG_PAO_TRIGGER.md.

Partial observability: the task one-hot in the observation is corrupted by per-episode Gaussian noise
(position stays clean). Three arms share nothing but the env:
  gated    -- deploy the single skill k_hat = argmax(noisy cue) (Bayes-optimal gate)   [PAO, disciplined]
  fireall  -- sum ALL skills' policy logits, argmax (cue-blind)                          [PAO, R5 liability]
  monolith -- one net trained end-to-end on the noisy task at the same sigma             [fair baseline]
The gated and fireall arms use the SAME skill library, so any gap is purely the deployment decision.
"""
import argparse
import os
import numpy as np
import torch
from scipy import stats

import grid as G
import r4

SIGMAS = [0.0, 0.3, 0.6, 1.0, 1.5]
MONO_UPDATES = 150         # multi-task updates; total eps = updates*K*EPT = 150*4*4 = 2400 (matched to skills)
MONO_EPT = 4               # episodes per task per update (bigger batch -> stable multi-task gradient)
MONO_HIDDEN = 128          # capacity-matched: one big net vs the K-network skill library
MONO_RESTARTS = 2          # retry-to-best, matching the skills' RETRY (fair: skills retry to competence)
EVAL_N = 40               # eval episodes per (niche, sigma)


# ----------------------------- partial-obs env ------------------------------ #
class NoisyGrid(G.GridTasks):
    """Task one-hot corrupted by per-episode Gaussian noise; position channels clean."""
    def __init__(self, sigma, seed=0):
        self.sigma = sigma
        super().__init__(seed)

    def reset(self):
        self.cue_noise = self.rng.randn(G.K).astype(np.float32) * self.sigma
        return super().reset()

    def _obs(self):
        oh = np.zeros(G.K, dtype=np.float32); oh[self.task] = 1.0
        oh = oh + self.cue_noise
        return np.concatenate([[self.x / (G.SIZE - 1), self.y / (G.SIZE - 1)], oh]).astype(np.float32)


def clean_obs(x, y, k):
    oh = np.zeros(G.K, dtype=np.float32); oh[k] = 1.0
    return np.concatenate([[x / (G.SIZE - 1), y / (G.SIZE - 1)], oh]).astype(np.float32)


def logits_of(net, obs):
    with torch.no_grad():
        lo, _ = net(torch.as_tensor(obs, dtype=torch.float32, device=G.DEVICE).unsqueeze(0))
    return lo.squeeze(0).cpu().numpy()


# ----------------------------- the monolith --------------------------------- #
def _train_monolith_once(sigma, seed):
    """Fair multi-task monolith: each PPO update interleaves MONO_EPT episodes from EVERY niche
    (large multi-task batch -> avoids both catastrophic forgetting and high-variance collapse).
    Capacity-matched (hidden=MONO_HIDDEN) to the K-network library; budget-matched to the skills."""
    rng = np.random.RandomState(seed)
    ag = G.PPO(seed=seed, hidden=MONO_HIDDEN)
    env = NoisyGrid(sigma, seed)
    for _ in range(MONO_UPDATES):
        for k in range(G.K):
            for _ in range(MONO_EPT):
                env.set_task(k)
                o = env.reset(); done = False
                while not done:
                    a = ag.act(o, train=True)
                    if rng.random() < r4.EPS_EXPLORE:
                        a = rng.randint(0, G.ACT_DIM)
                    o, r, done, info = env.step(a); ag.store(r, done)
        ag.finish(ent_coef=r4.ENT)
    return ag.net


def train_monolith(sigma, seed):
    """Retry-to-best, matching the skills' RETRY discipline (skills retry until competent, so the
    monolith gets the same number of shots and we keep the best by held-out validation success)."""
    best, best_acc = None, -1.0
    for r in range(MONO_RESTARTS):
        net = _train_monolith_once(sigma, seed * 977 + r * 31 + 1)
        acc = np.mean([roll_monolith(net, k, sigma, seed * 13 + 5, n=12) for k in range(G.K)])
        if acc > best_acc:
            best, best_acc = net, acc
    return best


# ----------------------------- rollouts ------------------------------------- #
def roll_gated(skills, true_k, sigma, seed, n=EVAL_N):
    """Deploy skill argmax(noisy cue); success iff it reaches the TRUE goal."""
    env = G.GridTasks(seed); env.set_task(true_k)
    rng = np.random.RandomState(seed * 7 + 1)
    s = 0
    for _ in range(n):
        cue = np.zeros(G.K); cue[true_k] = 1.0
        cue = cue + rng.randn(G.K) * sigma
        k_hat = int(np.argmax(cue))
        net = skills[k_hat]["net"]
        o = env.reset(); done = False; succ = False
        while not done:
            a = G.greedy_action(net, clean_obs(env.x, env.y, k_hat))
            o, r, done, info = env.step(a); succ = info["success"]
        s += succ
    return s / n


def roll_fireall(skills, true_k, sigma, seed, n=EVAL_N):
    """Sum all skills' logits (each fed its own clean cue), argmax; cue-blind."""
    env = G.GridTasks(seed); env.set_task(true_k)
    s = 0
    for _ in range(n):
        o = env.reset(); done = False; succ = False
        while not done:
            tot = np.zeros(G.ACT_DIM)
            for k in range(G.K):
                tot += logits_of(skills[k]["net"], clean_obs(env.x, env.y, k))
            o, r, done, info = env.step(int(np.argmax(tot))); succ = info["success"]
        s += succ
    return s / n


def roll_monolith(net, true_k, sigma, seed, n=EVAL_N):
    env = NoisyGrid(sigma, seed); env.set_task(true_k)
    s = 0
    for _ in range(n):
        o = env.reset(); done = False; succ = False
        while not done:
            o, r, done, info = env.step(G.greedy_action(net, o)); succ = info["success"]
        s += succ
    return s / n


# ----------------------------- per-seed run --------------------------------- #
def run_seed(seed):
    skills = [r4.train_mastered(k, seed) for k in range(G.K)]      # clean-cue mastered specialists
    out = {}                                                       # sigma -> (gated, fireall, mono)
    for sigma in SIGMAS:
        mono = train_monolith(sigma, seed)
        g = np.mean([roll_gated(skills, k, sigma, seed) for k in range(G.K)])
        f = np.mean([roll_fireall(skills, k, sigma, seed) for k in range(G.K)])
        m = np.mean([roll_monolith(mono, k, sigma, seed) for k in range(G.K)])
        out[sigma] = (g, f, m)
    return out


def hedges_g(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm", "long"], default="confirm")
    args = ap.parse_args()
    if args.mode == "calib":
        seeds = list(range(900, 903))
    elif args.mode == "confirm":
        seeds = list(range(910, 918))             # N=8 first look
    else:
        seeds = list(range(920, 940))             # N=20 survival run

    print("=" * 78)
    print(f"  PAO TRIGGER — partial observability  mode={args.mode}  N={len(seeds)} "
          f"seeds {seeds[0]}..{seeds[-1]}  K={G.K} dev={G.DEVICE}")
    print(f"  sigmas={SIGMAS}  mono_eps={MONO_EPS}  eval_n={EVAL_N}")
    print("=" * 78)

    # results[arm][sigma] = list over seeds
    res = {a: {s: [] for s in SIGMAS} for a in ("gated", "fireall", "mono")}
    for sd in seeds:
        o = run_seed(sd)
        for s in SIGMAS:
            g, f, m = o[s]
            res["gated"][s].append(g); res["fireall"][s].append(f); res["mono"][s].append(m)
        print(f"  seed {sd}: " + "  ".join(
            f"s{s}:G{o[s][0]:.2f}/F{o[s][1]:.2f}/M{o[s][2]:.2f}" for s in SIGMAS))

    print(f"\n  {'sigma':>6s} {'gated':>14s} {'fireall':>14s} {'monolith':>14s} "
          f"{'G-F g':>7s} {'G-M g':>7s}")
    summary = {}
    for s in SIGMAS:
        gd = np.array(res["gated"][s]); fa = np.array(res["fireall"][s]); mo = np.array(res["mono"][s])
        try:
            _, p_gf = stats.wilcoxon(gd, fa, alternative="greater")
        except ValueError:
            p_gf = float("nan")
        try:
            _, p_gm = stats.wilcoxon(gd, mo, alternative="two-sided")
        except ValueError:
            p_gm = float("nan")
        summary[s] = dict(gd=gd, fa=fa, mo=mo, p_gf=p_gf, p_gm=p_gm)
        print(f"  {s:6.1f} {gd.mean():7.2f}±{gd.std(ddof=1)/np.sqrt(len(gd)):.2f} "
              f"{fa.mean():7.2f}±{fa.std(ddof=1)/np.sqrt(len(fa)):.2f} "
              f"{mo.mean():7.2f}±{mo.std(ddof=1)/np.sqrt(len(mo)):.2f} "
              f"{hedges_g(gd, fa):+7.2f} {hedges_g(gd, mo):+7.2f}")

    print("\n  --- pre-registered tests ---")
    print("  P1 gated>fireall (informative sigma<=1.0):")
    for s in SIGMAS:
        if s <= 1.0:
            d = summary[s]
            print(f"    sigma={s}: g={hedges_g(d['gd'], d['fa']):+.2f} p={d['p_gf']:.4f}")
    print("  P3 gated vs monolith (two-sided; band where gated>mono = PAO niche):")
    for s in SIGMAS:
        d = summary[s]
        sign = "G>M" if d['gd'].mean() > d['mo'].mean() else "G<=M"
        print(f"    sigma={s}: G-M g={hedges_g(d['gd'], d['mo']):+.2f} p={d['p_gm']:.4f}  {sign}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/trigger_{args.mode}.npz",
             sigmas=np.array(SIGMAS),
             **{f"{a}_{s}": np.array(res[a][s]) for a in res for s in SIGMAS})
    print(f"\n  saved results/trigger_{args.mode}.npz")


if __name__ == "__main__":
    main()
