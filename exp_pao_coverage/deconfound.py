"""
PAO de-confound — is the trigger advantage modularity, or just factor-then-route?

Pre-registration: ../PREREG_PAO_DECONFOUND.md.

Decomposes the "gated beats monolith" advantage one ingredient at a time, gate held at the Bayes oracle:
  vanilla monolith   : end-to-end noisy net            (no clean / no factor / no modular)
  curriculum monolith: clean-pretrain -> noisy finetune (clean / no factor / no modular)
  factored (Bayes)   : Bayes gate + ONE shared clean multitask net  (clean / factor / no modular)
  gated  (Bayes)     : Bayes gate + K clean mastered skills          (clean / factor / modular)
Ladder vanilla -> curriculum -> factored -> gated adds one ingredient per step.
Floors/context: gated-learned, fire-all, random gate. Reuses trigger.py + gate.py.
"""
import argparse
import os
import numpy as np
from scipy import stats

import grid as G
import r4
import trigger as T
import gate as Gate

SIGMAS = [0.3, 0.6, 1.0]               # informative band where the effect lives


# ----------------------------- curriculum monolith -------------------------- #
def _curriculum_once(sigma, seed):
    """One end-to-end net: clean-cue pretrain (half budget) -> noisy-cue finetune (half).
    Same capacity/batch as the fair monolith; no factorization, but gets the clean-training affordance."""
    rng = np.random.RandomState(seed)
    ag = G.PPO(seed=seed, hidden=T.MONO_HIDDEN)
    half = T.MONO_UPDATES // 2
    for phase, sig in ((0, 0.0), (1, sigma)):       # phase 0 clean, phase 1 noisy
        env = T.NoisyGrid(sig, seed + phase)
        for _ in range(half):
            for k in range(G.K):
                for _ in range(T.MONO_EPT):
                    env.set_task(k)
                    o = env.reset(); done = False
                    while not done:
                        a = ag.act(o, train=True)
                        if rng.random() < r4.EPS_EXPLORE:
                            a = rng.randint(0, G.ACT_DIM)
                        o, rw, done, info = env.step(a); ag.store(rw, done)
            ag.finish(ent_coef=r4.ENT)
    return ag.net


def train_curriculum(sigma, seed):
    best, best_acc = None, -1.0
    for r in range(T.MONO_RESTARTS):
        net = _curriculum_once(sigma, seed * 911 + r * 41 + 1)
        acc = np.mean([T.roll_monolith(net, k, sigma, seed * 13 + 5, n=12) for k in range(G.K)])
        if acc > best_acc:
            best, best_acc = net, acc
    return best


# ----------------------------- factored monolith ---------------------------- #
def roll_factored(mt_net, true_k, sigma, seed, n=T.EVAL_N):
    """Bayes gate (argmax noisy cue) -> run the ONE shared clean multitask net with the chosen clean
    one-hot. Same factorization as gated, but a single shared policy instead of K modular skills."""
    env = G.GridTasks(seed); env.set_task(true_k)
    rng = np.random.RandomState(seed * 7 + 1)
    s = 0
    for _ in range(n):
        cue = np.zeros(G.K, dtype=np.float32); cue[true_k] = 1.0
        cue = cue + rng.randn(G.K).astype(np.float32) * sigma
        k_hat = int(np.argmax(cue))
        env.reset(); done = False; succ = False
        while not done:
            a = G.greedy_action(mt_net, T.clean_obs(env.x, env.y, k_hat))
            _, _, done, info = env.step(a); succ = info["success"]
        s += succ
    return s / n


def roll_random_gate(skills, true_k, sigma, seed, n=T.EVAL_N):
    """Floor: deploy a uniformly random skill (cue-blind, but commits to ONE skill, unlike fire-all)."""
    env = G.GridTasks(seed); env.set_task(true_k)
    rng = np.random.RandomState(seed * 7 + 1)
    s = 0
    for _ in range(n):
        j = rng.randint(0, G.K)
        env.reset(); done = False; succ = False
        while not done:
            a = G.greedy_action(skills[j]["net"], T.clean_obs(env.x, env.y, j))
            _, _, done, info = env.step(a); succ = info["success"]
        s += succ
    return s / n


# ----------------------------- per-seed ------------------------------------- #
def run_seed(seed):
    skills = [r4.train_mastered(k, seed) for k in range(G.K)]
    clean_mt = T.train_monolith(0.0, seed)                 # ONE shared clean multitask net (factored policy)
    out = {}
    for sigma in SIGMAS:
        vanilla = T.train_monolith(sigma, seed)
        curric = train_curriculum(sigma, seed)
        gate = Gate.train_gate(skills, sigma, seed)
        rows = dict(
            vanilla=np.mean([T.roll_monolith(vanilla, k, sigma, seed) for k in range(G.K)]),
            curric=np.mean([T.roll_monolith(curric, k, sigma, seed) for k in range(G.K)]),
            factored=np.mean([roll_factored(clean_mt, k, sigma, seed) for k in range(G.K)]),
            gated_b=np.mean([T.roll_gated(skills, k, sigma, seed) for k in range(G.K)]),
            gated_l=np.mean([Gate.roll_gated_learned(skills, gate, k, sigma, seed) for k in range(G.K)]),
            fireall=np.mean([T.roll_fireall(skills, k, sigma, seed) for k in range(G.K)]),
            randg=np.mean([roll_random_gate(skills, k, sigma, seed) for k in range(G.K)]),
        )
        out[sigma] = rows
    return out


ARMS = ["gated_b", "factored", "curric", "vanilla", "gated_l", "randg", "fireall"]
LABEL = {"gated_b": "gated(Bayes)", "factored": "factored", "curric": "curriculum",
         "vanilla": "vanilla-mono", "gated_l": "gated(learn)", "randg": "random-gate",
         "fireall": "fire-all"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm", "long"], default="confirm")
    args = ap.parse_args()
    seeds = {"calib": range(980, 983), "confirm": range(990, 998),
             "long": range(1000, 1020)}[args.mode]
    seeds = list(seeds)

    print("=" * 92)
    print(f"  PAO DE-CONFOUND  mode={args.mode}  N={len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f" K={G.K} dev={G.DEVICE}  sigmas={SIGMAS}")
    print("  ladder: vanilla -> curriculum -> factored -> gated(Bayes)  [+clean,+factor,+modular]")
    print("=" * 92)

    res = {a: {s: [] for s in SIGMAS} for a in ARMS}
    for sd in seeds:
        o = run_seed(sd)
        for s in SIGMAS:
            for a in ARMS:
                res[a][s].append(o[s][a])
        print(f"  seed {sd}: " + "  ".join(
            f"s{s}:G{o[s]['gated_b']:.2f}/Fc{o[s]['factored']:.2f}/Cu{o[s]['curric']:.2f}/"
            f"Va{o[s]['vanilla']:.2f}" for s in SIGMAS))

    def col(a, s):
        return np.array(res[a][s])

    print(f"\n  {'sigma':>6s} " + " ".join(f"{LABEL[a]:>13s}" for a in ARMS))
    for s in SIGMAS:
        print(f"  {s:6.1f} " + " ".join(f"{col(a, s).mean():13.2f}" for a in ARMS))

    def cmp(a, b, s, tail):
        A, B = col(a, s), col(b, s)
        try:
            p = stats.wilcoxon(A, B, alternative=tail)[1]
        except ValueError:
            p = float("nan")
        return T.hedges_g(A, B), p

    print("\n  --- pre-registered tests (per sigma) ---")
    print("  P1 modularity NOT the lever:  gated(Bayes) ~ factored  (small |g|, ns)")
    for s in SIGMAS:
        g, p = cmp("gated_b", "factored", s, "two-sided")
        print(f"    sigma={s}: g={g:+.2f} p={p:.4f}")
    print("  P2 factorization IS the lever:  factored > curriculum, factored > vanilla")
    for s in SIGMAS:
        g1, p1 = cmp("factored", "curric", s, "greater")
        g2, p2 = cmp("factored", "vanilla", s, "greater")
        print(f"    sigma={s}: F>Cu g={g1:+.2f} p={p1:.4f} | F>Va g={g2:+.2f} p={p2:.4f}")
    print("  P3 clean training alone insufficient:  curriculum ~ vanilla")
    for s in SIGMAS:
        g, p = cmp("curric", "vanilla", s, "greater")
        print(f"    sigma={s}: Cu>Va g={g:+.2f} p={p:.4f}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/deconfound_{args.mode}.npz", sigmas=np.array(SIGMAS),
             **{f"{a}_{s}": col(a, s) for a in ARMS for s in SIGMAS})
    print(f"\n  saved results/deconfound_{args.mode}.npz")


if __name__ == "__main__":
    main()
