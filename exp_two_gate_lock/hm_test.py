"""
H_M discrimination test — does manifold health predict skill usefulness?
Pre-registered in ../PREREG_HM.md (frozen 2026-06-09).
"""
import sys, os, pickle, random
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOForced

EPISODES = [3, 6, 10, 15, 22, 32, 45, 65]
SEEDS = list(range(10))
P1 = 80
APP = 0.3
N_VAL = 30
N_ROLL = 20
DWORLD = 64
EPS = 1e-6


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def torso_acts(skill_policy, obs_list):
    """64-dim policy-torso activations for a list of observations."""
    x = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
    with torch.no_grad():
        h = skill_policy.torso(x)        # (n, 64)
    return h.numpy()


def manifold_metrics(acts):
    """d_int = participation ratio; V_info = total variance (trace)."""
    X = acts - acts.mean(0, keepdims=True)
    cov = np.cov(X, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s, s2 = lam.sum(), (lam**2).sum()
    d_int = (s*s) / s2 if s2 > 0 else 0.0           # participation ratio
    V_info = float(s)                                # total variance
    H_M = np.log(1+DWORLD) * np.log(1+V_info) / np.log(1+d_int+EPS)
    return float(d_int), V_info, float(H_M)


def skill_rollouts(skill_policy, seed, n):
    """Run n skill-only rollouts on A→B; return visited obs and #successes."""
    env = TwoGateLockEnv(rule="A→B", seed=seed)
    obs_all, succ = [], 0
    for _ in range(n):
        o = env.reset(); d = False
        while not d:
            obs_all.append(o.copy())
            with torch.no_grad():
                logits, _ = skill_policy(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
            a = int(torch.distributions.Categorical(logits=logits).sample().item())
            o, r, d, info = env.step(a)
        if info.get("at_goal", False):
            succ += 1
    return obs_all, succ / n


def train_and_measure(episode, seed):
    set_seed(seed)
    ag = PAOForced(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02, crystallize_at=episode)
    ag._app_thresh = APP
    env = TwoGateLockEnv(rule="A→B", seed=0)
    for _ in range(P1):
        o = env.reset(); d = False
        while not d:
            a = ag.act(o, True); o, r, d, i = env.step(a); ag.step_end(r, d)
        ag.finish_episode()
    if ag.skill_policy is None:
        return None
    # manifold from rollout obs; Q from independent validation rollouts
    obs_m, _ = skill_rollouts(ag.skill_policy, seed + 1234, N_ROLL)
    d_int, V_info, H_M = manifold_metrics(torso_acts(ag.skill_policy, obs_m))
    _, Q = skill_rollouts(ag.skill_policy, seed + 9000, N_VAL)
    return {"episode": episode, "seed": seed, "Q": Q, "d_int": d_int, "V_info": V_info, "H_M": H_M}


def partial_spearman(x, y, z):
    """Spearman partial correlation r_xy.z."""
    rxy = spearmanr(x, y).statistic
    rxz = spearmanr(x, z).statistic
    ryz = spearmanr(y, z).statistic
    denom = np.sqrt((1-rxz**2) * (1-ryz**2))
    if denom == 0: return 0.0, 1.0
    r = (rxy - rxz*ryz) / denom
    n = len(x); dfn = n - 3
    t = r * np.sqrt(dfn / max(1e-9, 1 - r**2))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), dfn))
    return float(r), float(p)


def main():
    rows = []
    for e in EPISODES:
        for s in SEEDS:
            r = train_and_measure(e, s)
            if r: rows.append(r)
            tag = f"{r['Q']:.2f}/{r['d_int']:.1f}/{r['V_info']:.2f}/{r['H_M']:.2f}" if r else "no-skill"
            print(f"  ep={e:3d} seed={s}  Q/d_int/V/H_M = {tag}")

    Q = np.array([r["Q"] for r in rows])
    dint = np.array([r["d_int"] for r in rows])
    Vinf = np.array([r["V_info"] for r in rows])
    HM = np.array([r["H_M"] for r in rows])
    ep = np.array([r["episode"] for r in rows], float)

    print(f"\n{'='*74}\n  H_M DISCRIMINATION (N={len(rows)} skills)\n{'='*74}")
    for name, v in [("H_M", HM), ("d_int", dint), ("V_info", Vinf), ("episode", ep)]:
        rho = spearmanr(v, Q)
        print(f"  raw  Spearman({name:8s}, Q) = {rho.statistic:+.3f}  p={rho.pvalue:.4f}")
    pr, pp = partial_spearman(HM, Q, ep)
    print(f"  {'─'*70}")
    print(f"  DECISIVE  partial Spearman(H_M, Q | episode) = {pr:+.3f}  p={pp:.4f}")

    # within-episode check: does H_M separate seeds of differing Q at fixed training?
    print(f"\n  Within-episode H_M↔Q (training-amount held fixed):")
    wr = []
    for e in EPISODES:
        idx = [i for i, r in enumerate(rows) if r["episode"] == e]
        if len(idx) >= 4 and np.std(Q[idx]) > 1e-6 and np.std(HM[idx]) > 1e-6:
            rho = spearmanr(HM[idx], Q[idx]).statistic
            wr.append(rho)
            print(f"    ep={e:3d}: Q range [{Q[idx].min():.2f},{Q[idx].max():.2f}]  rho(H_M,Q)={rho:+.2f}")
    if wr:
        print(f"    mean within-episode rho(H_M,Q) = {np.mean(wr):+.3f}")

    # verdict
    valid = (abs(pr) >= 0.3 and pp < 0.05 and pr > 0)
    clock = (abs(pr) < 0.15 and pp > 0.05)
    verdict = "VALID (beyond clock)" if valid else ("INVALID (just a clock)" if clock else "AMBIGUOUS")
    print(f"\n  {'─'*70}\n  PRE-REGISTERED VERDICT: H_M is {verdict}\n  {'─'*70}")

    with open(os.path.join(os.path.dirname(__file__), "results/hm_test.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("  Saved results/hm_test.pkl")


if __name__ == "__main__":
    main()
