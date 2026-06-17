"""
PAO online/learned skills — does factor-then-route survive without the clean-skill gift?

Pre-registration: ../PREREG_PAO_ONLINE.md.

Skills are crystallized UNDER the same partial observability they are deployed in (no clean gift): a learned
skill k is trained on episodes attributed to k by a noisy cue, so the true goal is wrong with probability
c(sigma) (the cue mis-classification rate). The skill always sees its own clean identity k + position but is
rewarded toward the (sometimes wrong) goal -> library quality degrades with sigma.

Arms: oracle/clean, oracle/learned, PAO-full (learned gate + learned skills), monolith, fire-all, random-gate.
Reuses trigger.py + gate.py.
"""
import argparse
import os
import numpy as np
from scipy import stats

import grid as G
import r4
import trigger as T
import gate as Gate
import deconfound as D

SIGMAS = [0.3, 0.6, 1.0]


def misclass_rate(sigma, n=40000, seed=0):
    """c(sigma) = 1 - P(argmax(e_0 + N(0,sigma^2 I_K)) == 0), the cue mis-classification rate."""
    if sigma == 0:
        return 0.0
    rng = np.random.RandomState(seed)
    s = np.zeros((n, G.K)); s[:, 0] = 1.0
    s = s + rng.randn(n, G.K) * sigma
    return float(1.0 - (np.argmax(s, axis=1) == 0).mean())


LEARNED_RETRIES = 2          # retry-to-best, matching the clean skills' RETRY (isolates contamination)


def _train_learned_once(k, sigma, seed, c):
    rng = np.random.RandomState(seed)
    ag = G.PPO(seed=seed)
    env = G.GridTasks(seed)
    others = [j for j in range(G.K) if j != k]
    for _ in range(r4.MATURE_EPS):
        goal = k if rng.random() > c else int(rng.choice(others))
        env.set_task(goal)
        env.reset(); done = False
        while not done:
            obs_k = T.clean_obs(env.x, env.y, k)           # identity k, regardless of true goal
            a = ag.act(obs_k, train=True)
            if rng.random() < r4.EPS_EXPLORE:
                a = rng.randint(0, G.ACT_DIM)
            _, rw, done, info = env.step(a); ag.store(rw, done)
        ag.finish(ent_coef=r4.ENT)
    return ag.net


def train_learned_skill(k, sigma, seed, c):
    """Crystallize skill k under noisy attribution: each episode the true goal is k w.p. (1-c), else a random
    wrong niche. The skill sees its own identity k (clean one-hot) + position; reward is toward the true goal.
    Retry-to-best (selected by clean niche-k success) matches the clean skills' RETRY, so the only difference
    from a clean skill is contamination, not training variance. (This favors PAO: best-case learned skills.)"""
    best, best_acc = None, -1.0
    for r in range(LEARNED_RETRIES):
        net = _train_learned_once(k, sigma, seed + r * 1009, c)
        acc = G.policy_success(net, k)
        if acc > best_acc:
            best, best_acc = net, acc
    return {"net": best, "task": k}


def run_seed(seed):
    clean = [r4.train_mastered(k, seed) for k in range(G.K)]   # reference (clean gift)
    out = {}
    for sigma in SIGMAS:
        c = misclass_rate(sigma)
        learned = [train_learned_skill(k, sigma, seed * 53 + k + 1, c) for k in range(G.K)]
        mono = T.train_monolith(sigma, seed)
        gate_l = Gate.train_gate(learned, sigma, seed)        # learned gate over the LEARNED library
        rows = dict(
            oracle_clean=np.mean([T.roll_gated(clean, k, sigma, seed) for k in range(G.K)]),
            oracle_learned=np.mean([T.roll_gated(learned, k, sigma, seed) for k in range(G.K)]),
            pao_full=np.mean([Gate.roll_gated_learned(learned, gate_l, k, sigma, seed) for k in range(G.K)]),
            monolith=np.mean([T.roll_monolith(mono, k, sigma, seed) for k in range(G.K)]),
            fireall=np.mean([T.roll_fireall(learned, k, sigma, seed) for k in range(G.K)]),
            randg=np.mean([D.roll_random_gate(learned, k, sigma, seed) for k in range(G.K)]),
            c=c,
        )
        out[sigma] = rows
    return out


ARMS = ["oracle_clean", "oracle_learned", "pao_full", "monolith", "fireall", "randg"]
LABEL = {"oracle_clean": "oracle/clean", "oracle_learned": "oracle/learn",
         "pao_full": "PAO-full", "monolith": "monolith", "fireall": "fire-all", "randg": "random-gate"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "confirm", "long"], default="confirm")
    args = ap.parse_args()
    seeds = {"calib": range(1020, 1023), "confirm": range(1030, 1038),
             "long": range(1040, 1060)}[args.mode]
    seeds = list(seeds)

    print("=" * 92)
    print(f"  PAO ONLINE/LEARNED SKILLS  mode={args.mode}  N={len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f" K={G.K} dev={G.DEVICE}  sigmas={SIGMAS}")
    print(f"  contamination c(sigma): " + " ".join(f"{s}->{misclass_rate(s):.2f}" for s in SIGMAS))
    print("=" * 92)

    res = {a: {s: [] for s in SIGMAS} for a in ARMS}
    for sd in seeds:
        o = run_seed(sd)
        for s in SIGMAS:
            for a in ARMS:
                res[a][s].append(o[s][a])
        print(f"  seed {sd}: " + "  ".join(
            f"s{s}:Oc{o[s]['oracle_clean']:.2f}/Ol{o[s]['oracle_learned']:.2f}/"
            f"P{o[s]['pao_full']:.2f}/M{o[s]['monolith']:.2f}" for s in SIGMAS))

    col = lambda a, s: np.array(res[a][s])
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
    print("  P1 learned skills degrade library:  oracle/learned < oracle/clean")
    for s in SIGMAS:
        g, p = cmp("oracle_learned", "oracle_clean", s, "less")
        print(f"    sigma={s}: g={g:+.2f} p={p:.4f}")
    print("  P2 (CRUX) factor-then-route survives no gift:  PAO-full > monolith")
    for s in SIGMAS:
        g, p = cmp("pao_full", "monolith", s, "greater")
        sign = "P>M" if col("pao_full", s).mean() > col("monolith", s).mean() else "P<=M"
        print(f"    sigma={s}: g={g:+.2f} p={p:.4f}  {sign}")
    print("  P3 attribution: library loss (Oc-Ol) vs gate loss (Ol-Pao)")
    for s in SIGMAS:
        lib = col("oracle_clean", s).mean() - col("oracle_learned", s).mean()
        gate = col("oracle_learned", s).mean() - col("pao_full", s).mean()
        print(f"    sigma={s}: library_loss={lib:+.3f}  gate_loss={gate:+.3f}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/online_{args.mode}.npz", sigmas=np.array(SIGMAS),
             **{f"{a}_{s}": col(a, s) for a in ARMS for s in SIGMAS})
    print(f"\n  saved results/online_{args.mode}.npz")


if __name__ == "__main__":
    main()
