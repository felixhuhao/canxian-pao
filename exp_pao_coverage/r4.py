"""
R4 (additive combiner) — coverage-gated skill admission in multi-task RL, the PAO transfer test.
================================================================================================
Pre-registered in ../PREREG_R4.md. Builds on the validated grid env (grid.py).

Transfers the ladder's causal law ("coverage-gating's payoff <=> the combiner is redundancy-fragile",
R1b) into PAO's regime, using PAO's OWN combiner: skills are applied as ADDITIVE BIAS to a base
policy's action-logits (P3 showed this is fragile). For eval task k, every library skill tagged to
niche k is TRIGGERED and its logits are SUMMED onto the base's logits, then argmax. Summing means junk
biases ACCUMULATE, so corruption grows with the number of admitted junk skills -> a redundancy-fragile
combiner, exactly the R1b condition.

Library:
  - base : a mediocre/forgetful base policy (round-robin-undertrained) -> needs skills on most niches.
  - one MASTERED skill per niche (fresh specialist trained to competence), plus
  - `cap` IMMATURE snapshots per niche (premature crystallization) = junk; `cap` is the capacity axis.
Arms (greedy eval):
  - noskill : base alone.
  - ungated : base + sum of ALL niche-matched skills (1 good + cap junk).
  - gated   : base + the one competent, niche-covering skill (lean).
Prediction: gated > ungated, gap GROWS with cap (more junk summed). noskill is the floor.
"""
import sys, os, copy, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import grid as G
from grid import K, GOALS, DEVICE

MATURE_EPS = 600
IMMATURE_EPS = 12
BASE_EPS = 60            # per niche, round-robin -> undertrained, mediocre base
EPS_EXPLORE = 0.2
ENT = 0.02
COMPETENT = 0.9
RETRY = 4                # retries to guarantee a mastered skill per niche


def train_specialist(task, seed, episodes):
    rng = np.random.RandomState(seed)
    ag = G.PPO(seed=seed)
    env = G.GridTasks(seed); env.set_task(task)
    for _ in range(episodes):
        G.run_episode(ag, env, train=True, eps=EPS_EXPLORE, rng=rng, ent_coef=ENT)
    return _freeze(ag.net, task)


def train_mastered(task, seed):
    for r in range(RETRY):
        net = train_specialist(task, seed * 131 + r * 17 + 1, MATURE_EPS)
        if G.policy_success(net["net"], task) >= COMPETENT:
            return net
    return net                                   # best effort (rare)


def train_base(seed):
    """Undertrained round-robin base: partially competent, needs skills."""
    rng = np.random.RandomState(seed)
    ag = G.PPO(seed=seed)
    env = G.GridTasks(seed)
    for _ in range(BASE_EPS):
        for k in range(K):
            env.set_task(k)
            G.run_episode(ag, env, train=True, eps=EPS_EXPLORE, rng=rng, ent_coef=ENT)
    return ag.net


def _freeze(net, task):
    n = copy.deepcopy(net).eval()
    for p in n.parameters():
        p.requires_grad = False
    return {"net": n, "task": task}


def logits(net, obs):
    with torch.no_grad():
        l, _ = net(torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    return l


def combined_action(base, skills, obs):
    l = logits(base, obs)
    for s in skills:
        l = l + logits(s["net"], obs)            # additive bias, SUMMED
    return int(torch.argmax(l))


def eval_arm(base, lib_by_task, seed=999, n=30):
    env = G.GridTasks(seed); out = []
    for k in range(K):
        env.set_task(k); s = 0
        sk = lib_by_task.get(k, [])
        for _ in range(n):
            o = env.reset(); done = False; succ = False
            while not done:
                o, r, done, info = env.step(combined_action(base, sk, o)); succ = info["success"]
            s += succ
        out.append(s / n)
    return np.array(out)


def build_base_good(seed):
    """Seed-only: a mediocre/forgetful base + one mastered skill per niche (the costly part)."""
    base = train_base(seed)
    good = {k: train_mastered(k, seed + 1 + k) for k in range(K)}
    return base, good


def make_junk(good, seed, cap):
    """cap MIS-ASSOCIATED skills per niche: a skill mastered on a DIFFERENT niche, tagged to k -> a
    confidently-wrong bias when it fires for task k. PAO's real failure mode (P3: wrong skill fires in
    the wrong context); over-crystallization without quality control produces more such mis-firings."""
    rng = np.random.RandomState(seed * 31 + 5)
    junk = {}
    for k in range(K):
        others = [j for j in range(K) if j != k]
        junk[k] = [{"net": good[rng.choice(others)]["net"], "task": k} for _ in range(cap)]
    return junk


def eval_run(base, good, junk, arm, seed):
    if arm == "noskill":
        lib = {}
    elif arm == "ungated":
        lib = {k: [good[k]] + junk[k] for k in range(K)}
    else:  # gated: admit one competent, niche-covering skill per niche
        lib = {}
        for k in range(K):
            comp = [s for s in [good[k]] + junk[k] if G.policy_success(s["net"], k) >= COMPETENT]
            if comp:
                lib[k] = [comp[0]]
    succ = eval_arm(base, lib, seed=seed + 7)
    return {"mean_succ": float(succ.mean()), "per_task": succ.tolist(),
            "lib": sum(len(v) for v in lib.values())}


def run(arm, cap, seed):                          # convenience for sanity checks
    base, good = build_base_good(seed)
    return eval_run(base, good, make_junk(good, seed, cap), arm, seed)


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return 0.0 if sp == 0 else (x.mean()-y.mean())/sp * (1 - 3/(4*(nx+ny-2)-1))


if __name__ == "__main__":
    import argparse
    from scipy.stats import wilcoxon, spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start", type=int, default=100)        # held-out seeds (sanity used 0)
    ap.add_argument("--caps", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    a = ap.parse_args()
    seeds = list(range(a.start, a.start + a.seeds))
    data = {(cap, arm): [] for cap in a.caps for arm in ("noskill", "ungated", "gated")}
    for s in seeds:                                          # build base+good once per seed
        base, good = build_base_good(s)
        for cap in a.caps:
            junk = make_junk(good, s, cap)
            for arm in ("noskill", "ungated", "gated"):
                data[(cap, arm)].append(eval_run(base, good, junk, arm, s))
        print(f"seed {s} done", flush=True)
    for cap in a.caps:
        for arm in ("noskill", "ungated", "gated"):
            d = data[(cap, arm)]
            print(f"cap={cap} {arm:<8} mean_succ={np.mean([r['mean_succ'] for r in d]):.3f} "
                  f"lib={np.mean([r['lib'] for r in d]):.1f}", flush=True)
    print(f"\n  cap  noskill  ungated   gated   Δ(ga-un)    g     p(ga>un)")
    deltas = []
    for cap in a.caps:
        ns = np.array([r['mean_succ'] for r in data[(cap, 'noskill')]])
        un = np.array([r['mean_succ'] for r in data[(cap, 'ungated')]])
        ga = np.array([r['mean_succ'] for r in data[(cap, 'gated')]])
        try:
            p = wilcoxon(ga, un, alternative="greater").pvalue if not np.allclose(ga, un) else 1.0
        except ValueError:
            p = 1.0
        deltas.append(ga.mean() - un.mean())
        print(f"  {cap:>3}  {ns.mean():.3f}    {un.mean():.3f}    {ga.mean():.3f}   "
              f"{ga.mean()-un.mean():+.3f}   {hedges_g(ga, un):+.2f}  {p:.3f}")
    rho, pr = spearmanr(a.caps, deltas)
    print(f"\n  TREND Spearman(cap, Δ) = {rho:+.3f} (p={pr:.3f})")
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/r4.pkl"), "wb") as f:
        pickle.dump({str(k): v for k, v in data.items()}, f)
    print("\n  Saved results/r4.pkl")
