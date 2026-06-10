"""
P1 — CRUX: does crystallisation TIMING matter? (H1)
===================================================
Pre-registered in ../PREREG_P1.md (frozen 2026-06-07). Do not change the decision rule here.

Arms (identical mechanism, app_thresh=0.3, gate live; only the crystallisation EPISODE differs):
  Trigger       : heuristic trigger fires (signal-selected window)        [PAOLight]
  RandomLate    : random episode in [40,80)  (competence-matched control) [PAOForced]   <- decisive
  RandomUniform : random episode in [0,80)   (weak floor)                 [PAOForced]
  NoSkill       : skill never applied (bias=0) (sanity floor)             [PAONoSkill]

Primary outcome : Q(z) = skill-only validation success rate (N_val=30).
Secondary       : Phase-3 reuse (early-20), Phase-2 lock-in (late-20).
Stats           : paired Wilcoxon (one-sided) + Hedges' g + 95% bootstrap CI, BH-FDR. N=30.
"""
import sys, os, pickle, random
import numpy as np
import torch
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOLight, PAONoSkill, PAOForced

P1, P2, P3 = 80, 120, 60
SEEDS = list(range(30))
APP = 0.3
NVAL = 30
LATE_LO, LATE_HI = 40, 80      # RandomLate window
UNIF_LO, UNIF_HI = 0, 80       # RandomUniform window


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_protocol(agent):
    def phase(env, n, eps, rs):
        rng = np.random.RandomState(rs); rets = []
        for _ in range(n):
            o = env.reset(); d = False
            while not d:
                a = agent.act(o, True)
                if rng.random() < eps: a = rng.randint(0, NUM_ACTIONS)
                o, r, d, i = env.step(a); agent.step_end(r, d)
            agent.finish_episode()
            rets.append(agent.get_log()["returns"][-1])
        return rets
    env = TwoGateLockEnv(rule="A→B", seed=0)
    r1 = phase(env, P1, 0.0, 999)
    env.set_rule("B→A"); r2 = phase(env, P2, 0.1, 5000)
    env.set_rule("A→B"); r3 = phase(env, P3, 0.0, 999)
    return r1, r2, r3


def crystal_ep_and_return(agent, r1):
    eps = agent.get_log().get("skill_episodes", [])
    if not eps:
        return None, None
    ce = eps[0]
    rr = r1[ce] if ce < len(r1) else None
    return ce, rr


def measure(agent, seed):
    r1, r2, r3 = run_protocol(agent)
    qz = agent._validate_skill(n_rollouts=NVAL,
                               eval_env=TwoGateLockEnv(rule="A→B", seed=seed + 9000))
    ce, cr = crystal_ep_and_return(agent, r1)
    return {
        "Qz": qz,
        "p2_late": float(np.mean(r2[-20:])),
        "p3_early": float(np.mean(r3[:20])),
        "cryst_ep": ce, "cryst_return": cr,
    }


def build(arm, seed):
    set_seed(seed)  # global RNG seeded BEFORE construction -> identical init across arms
    if arm == "Trigger":
        ag = PAOLight(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02)
    elif arm == "NoSkill":
        ag = PAONoSkill(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02)
    elif arm == "RandomLate":
        ce = int(np.random.RandomState(seed + 7000).randint(LATE_LO, LATE_HI))
        ag = PAOForced(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02, crystallize_at=ce)
    elif arm == "RandomUniform":
        ce = int(np.random.RandomState(seed + 7000).randint(UNIF_LO, UNIF_HI))
        ag = PAOForced(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02, crystallize_at=ce)
    ag._app_thresh = APP
    return ag


# ── stats ──
def hedges_g(x, y):  # paired-ish, pooled-SD effect size
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if sp == 0: return 0.0
    d = (np.mean(x) - np.mean(y)) / sp
    return d * (1 - 3/(4*(nx+ny-2)-1))

def boot_ci(x, y, n=10000):
    x, y = np.asarray(x), np.asarray(y); diff = x - y
    rng = np.random.RandomState(0)
    bs = [np.mean(diff[rng.randint(0, len(diff), len(diff))]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def paired_wilcoxon(x, y, alt):
    x, y = np.asarray(x), np.asarray(y)
    if np.allclose(x, y): return 1.0  # no differences (e.g. saturation)
    try:
        return float(wilcoxon(x, y, alternative=alt).pvalue)
    except ValueError:
        return 1.0

def bh_fdr(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 1.0
    for i in range(m-1, -1, -1):
        idx = order[i]; val = min(prev, p[idx]*m/(i+1)); adj[idx] = val; prev = val
    return adj


def main():
    data = {arm: [] for arm in ["Trigger", "RandomLate", "RandomUniform", "NoSkill"]}
    for arm in data:
        print(f"\n── {arm} ──")
        for s in SEEDS:
            m = measure(build(arm, s), s)
            data[arm].append(m)
            print(f"  seed={s:2d}  cryst_ep={str(m['cryst_ep']):>4s} "
                  f"cryst_R={(f'{m['cryst_return']:+.2f}' if m['cryst_return'] is not None else '  NA')}  "
                  f"Qz={m['Qz']:.2f}  P3={m['p3_early']:+.3f}  P2={m['p2_late']:+.3f}")

    def col(arm, k): return np.array([d[k] for d in data[arm]], dtype=float)

    print(f"\n{'='*78}\n  P1 RESULTS (N={len(SEEDS)}, seeded, app_thresh={APP})\n{'='*78}")
    print(f"  {'Arm':<14s} {'Qz mean':>10s} {'P3 early':>12s} {'P2 late':>12s} {'cryst_ep':>10s} {'cryst_R':>9s}")
    for arm in data:
        q, p3, p2 = col(arm, 'Qz'), col(arm, 'p3_early'), col(arm, 'p2_late')
        ce = np.array([d['cryst_ep'] for d in data[arm] if d['cryst_ep'] is not None], dtype=float)
        cr = np.array([d['cryst_return'] for d in data[arm] if d['cryst_return'] is not None], dtype=float)
        print(f"  {arm:<14s} {q.mean():>9.3f} {p3.mean():>8.3f}±{p3.std(ddof=1)/np.sqrt(len(p3)):.2f} "
              f"{p2.mean():>8.3f}±{p2.std(ddof=1)/np.sqrt(len(p2)):.2f} "
              f"{(ce.mean() if len(ce) else float('nan')):>10.1f} {(cr.mean() if len(cr) else float('nan')):>9.2f}")

    print(f"\n  {'─'*74}\n  KEY COMPARISON (pre-registered): Trigger vs RandomLate\n  {'─'*74}")
    comps, ps = [], []
    for (a, b, k, alt, lbl) in [
        ("Trigger","RandomLate","Qz","greater","PRIMARY  Qz   (Trig>RandLate)"),
        ("Trigger","RandomLate","p3_early","greater","2ndary   P3   (Trig>RandLate)"),
        ("Trigger","RandomLate","p2_late","less","2ndary   P2   (Trig<RandLate)"),
        ("Trigger","RandomUniform","Qz","greater","ctrl     Qz   (Trig>RandUnif)"),
        ("Trigger","NoSkill","p3_early","greater","sanity   P3   (Trig>NoSkill)"),
        ("RandomLate","NoSkill","p3_early","greater","sanity   P3   (RandLate>NoSkill)"),
    ]:
        x, y = col(a, k), col(b, k)
        p = paired_wilcoxon(x, y, alt); g = hedges_g(x, y); lo, hi = boot_ci(x, y)
        comps.append((lbl, x.mean(), y.mean(), g, lo, hi, p)); ps.append(p)
    adj = bh_fdr(ps)
    for (lbl, mx, my, g, lo, hi, p), pa in zip(comps, adj):
        print(f"  {lbl:<32s} {mx:>7.3f} vs {my:>7.3f}  g={g:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  p={p:.3f} (FDR {pa:.3f})")

    # ── pre-registered verdict on PRIMARY (Trigger vs RandomLate, Qz) ──
    qT, qR = col("Trigger","Qz"), col("RandomLate","Qz")
    gp = hedges_g(qT, qR); pp = paired_wilcoxon(qT, qR, "greater")
    print(f"\n  {'─'*74}\n  PRE-REGISTERED VERDICT\n  {'─'*74}")
    saturated = (qT.mean() >= 0.95 and qR.mean() >= 0.95)
    if saturated:
        print(f"  Qz saturated (Trig {qT.mean():.3f}, RandLate {qR.mean():.3f} ≥0.95) → PRIMARY = H1 NOT supported.")
        print(f"  Decisive fallback = Phase-3 reuse:")
        x, y = col("Trigger","p3_early"), col("RandomLate","p3_early")
        g = hedges_g(x, y); p = paired_wilcoxon(x, y, "greater")
        verdict = ("SUPPORTED" if (p < 0.05 and g >= 0.5)
                   else "FALSIFIED" if (abs(g) < 0.3 and p > 0.05) else "INCONCLUSIVE")
        print(f"    P3: Trig {x.mean():.3f} vs RandLate {y.mean():.3f}  g={g:+.2f}  p={p:.3f}  → H1 {verdict}")
    else:
        verdict = ("SUPPORTED" if (pp < 0.05 and gp >= 0.5)
                   else "FALSIFIED" if (abs(gp) < 0.3 and pp > 0.05) else "INCONCLUSIVE")
        print(f"  PRIMARY Qz: Trig {qT.mean():.3f} vs RandLate {qR.mean():.3f}  g={gp:+.2f}  p={pp:.3f}  → H1 {verdict}")

    with open(os.path.join(os.path.dirname(__file__), "results/p1_crux.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/p1_crux.pkl")


if __name__ == "__main__":
    main()
