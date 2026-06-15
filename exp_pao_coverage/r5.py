"""
R5 — is R4's harmful junk REACHABLE through PAO's own channels, or did it have to be planted?
============================================================================================
R4 showed competence-gating helps an ADDITIVE skill-combiner when junk is *confidently-wrong*
(mis-tagged competent skills), and that *weak/immature* junk did NOT corrupt the sum. R5 tests whether
the harm is reachable via PAO's two real channels, WITHOUT planting:

  Condition A (crystallization volume): junk = IMMATURE self-snapshots from a specialist's own training,
    correctly tagged to its OWN niche — what PAO's frequent event-triggered crystallization actually
    produces. Sweep how many are admitted. Arms noskill/ungated/gated.
    Prediction: ungated ≈ gated (weak same-niche skills don't corrupt the additive sum) -> the
    crystallization channel makes HARMLESS junk; admission-gating is moot here.

  Condition B (mis-trigger rate): all skills competent & correctly tagged, but the router co-fires a
    competent WRONG-niche skill with probability eps per episode. Sweep eps.
    Prediction: success degrades with eps -> the harm is governed by TRIGGERING, not crystallization
    volume; a competence gate can't fix it (every skill is competent), only a better router can.

Conclusion sought: PAO's real liability is its applicability/triggering mechanism, not its
crystallization counter; "more skills" is not itself the danger. Pre-registered in ../PREREG_R5.md.
"""
import sys, os, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import grid as G
import r4 as R4
from grid import K, DEVICE


def train_with_snaps(task, seed, snap_eps):
    """One training run; freeze immature snapshots at the given episode counts (correct niche tag)."""
    if not snap_eps:
        return []
    rng = np.random.RandomState(seed); ag = G.PPO(seed=seed)
    env = G.GridTasks(seed); env.set_task(task)
    target = set(snap_eps); snaps = {}
    for ep in range(max(snap_eps) + 1):
        G.run_episode(ag, env, train=True, eps=R4.EPS_EXPLORE, rng=rng, ent_coef=R4.ENT)
        if ep in target:
            snaps[ep] = R4._freeze(ag.net, task)
    return [snaps[e] for e in snap_eps]


# ----- Condition A: natural over-crystallization junk (immature, own-niche) -----
SNAP_EPS = [3, 6, 10, 16, 24, 36, 50, 70]                  # increasingly-trained immature snapshots


def run_A(arm, cap, seed):
    base, good = R4.build_base_good(seed)
    imm = {k: train_with_snaps(k, seed * 257 + k * 11 + 9, SNAP_EPS[:cap]) for k in range(K)}
    if arm == "noskill":
        lib = {}
    elif arm == "ungated":
        lib = {k: [good[k]] + imm[k] for k in range(K)}
    else:                                                   # gated: competence drops immature
        lib = {}
        for k in range(K):
            comp = [s for s in [good[k]] + imm[k] if G.policy_success(s["net"], k) >= R4.COMPETENT]
            if comp:
                lib[k] = [comp[0]]
    succ = R4.eval_arm(base, lib, seed=seed + 7)
    return {"mean_succ": float(succ.mean()), "per_task": succ.tolist(),
            "lib": sum(len(v) for v in lib.values())}


# ----- Condition B: mis-triggering (competent wrong-niche skill(s) co-fire) -----
def eval_misfire(base, good, eps, m, seed, n=30):
    """With prob eps per episode, co-fire m wrong-niche competent skills (with replacement) alongside
    the correct one. eps = mis-trigger RATE; m = number of simultaneous wrong skills."""
    rng = np.random.RandomState(seed); env = G.GridTasks(seed); out = []
    for k in range(K):
        env.set_task(k); s = 0; others = [x for x in range(K) if x != k]
        for _ in range(n):
            fire = [good[k]]
            if m and rng.random() < eps:
                fire = [good[k]] + [good[rng.choice(others)] for _ in range(m)]
            o = env.reset(); done = False; succ = False
            while not done:
                o, r, done, info = env.step(R4.combined_action(base, fire, o)); succ = info["success"]
            s += succ
        out.append(s / n)
    return np.array(out)


def run_B(eps, m, seed):
    base, good = R4.build_base_good(seed)
    succ = eval_misfire(base, good, eps, m, seed + 7)
    return {"mean_succ": float(succ.mean()), "per_task": succ.tolist()}


if __name__ == "__main__":
    import argparse
    from scipy.stats import wilcoxon, spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start", type=int, default=200)       # held-out (R4 used 0/100)
    ap.add_argument("--caps", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    a = ap.parse_args()
    seeds = list(range(a.start, a.start + a.seeds))
    A = {(cap, arm): [] for cap in a.caps for arm in ("noskill", "ungated", "gated")}
    Brate = {e: [] for e in a.eps}                          # rate sweep, m=1 wrong skill
    Bcount = {m: [] for m in a.ms}                          # count sweep, eps=1.0
    for s in seeds:
        for cap in a.caps:
            for arm in ("noskill", "ungated", "gated"):
                A[(cap, arm)].append(run_A(arm, cap, s))
        for e in a.eps:
            Brate[e].append(run_B(e, 1, s))
        for m in a.ms:
            Bcount[m].append(run_B(1.0, m, s))
        print(f"seed {s} done", flush=True)

    print("\n== Condition A: natural over-crystallization junk (immature, own-niche) ==")
    print("  cap  noskill  ungated   gated   Δ(ga-un)    g     p(ga>un)")
    dA = []
    for cap in a.caps:
        un = np.array([r['mean_succ'] for r in A[(cap, 'ungated')]])
        ga = np.array([r['mean_succ'] for r in A[(cap, 'gated')]])
        ns = np.array([r['mean_succ'] for r in A[(cap, 'noskill')]])
        try:
            p = wilcoxon(ga, un, alternative="greater").pvalue if not np.allclose(ga, un) else 1.0
        except ValueError:
            p = 1.0
        dA.append(ga.mean() - un.mean())
        print(f"  {cap:>3}  {ns.mean():.3f}    {un.mean():.3f}    {ga.mean():.3f}   "
              f"{ga.mean()-un.mean():+.3f}   {R4.hedges_g(ga, un):+.2f}  {p:.3f}")
    rA, pA = spearmanr(a.caps, dA)
    print(f"  TREND Spearman(cap, Δ) = {rA:+.3f} (p={pA:.3f})  [predict ~0: volume is harmless]")

    print("\n== Condition B-rate: ONE wrong skill co-fires with prob eps ==")
    print("  eps   mean_succ")
    mr = []
    for e in a.eps:
        v = np.mean([r['mean_succ'] for r in Brate[e]]); mr.append(v)
        print(f"  {e:>4}   {v:.3f}")
    rBr, pBr = spearmanr(a.eps, mr)
    print(f"  TREND Spearman(eps, succ) = {rBr:+.3f} (p={pBr:.3f})  [single mis-fire: expect mild]")

    print("\n== Condition B-count: m wrong skills co-fire every episode (eps=1.0) ==")
    print("    m   mean_succ")
    mc = []
    for m in a.ms:
        v = np.mean([r['mean_succ'] for r in Bcount[m]]); mc.append(v)
        print(f"  {m:>3}   {v:.3f}")
    rBc, pBc = spearmanr(a.ms, mc)
    print(f"  TREND Spearman(m, succ) = {rBc:+.3f} (p={pBc:.3f})  [harm scales with # simultaneous wrong]")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/r5.pkl"), "wb") as f:
        pickle.dump({"A": {str(k): v for k, v in A.items()},
                     "Brate": {str(k): v for k, v in Brate.items()},
                     "Bcount": {str(k): v for k, v in Bcount.items()}}, f)
    print("\n  Saved results/r5.pkl")
