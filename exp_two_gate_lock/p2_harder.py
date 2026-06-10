"""
P2/Harder-Env — H1 steelman: BOCPD trigger + non-convex 2D env with Location-Shift.
Pre-registered in ../PREREG_P2_harderenv.md (frozen 2026-06-07).
"""
import sys, os, pickle, random
import numpy as np
import torch
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from env_2d_shift import TwoGate2DShiftEnv, OBS_DIM, NUM_ACTIONS
from agents import PAOLight, PAONoSkill, PAOForced, PAOBocpd

P1, P2, P3 = 300, 150, 150
SEEDS = list(range(30))
APP = 0.3
NVAL = 30
LATE_LO, LATE_HI = 100, 250
UNIF_LO, UNIF_HI = 0, 300
KW = dict(lr=3e-4, entropy_coef=0.03, trigger_entropy=1.0)  # 2D: 4 actions, entropy thresh relaxed


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
    env = TwoGate2DShiftEnv(rule="A→B", seed=0, shifted=False)
    r1 = phase(env, P1, 0.0, 999)
    env.set_rule("B→A"); env.set_shifted(True);  r2 = phase(env, P2, 0.1, 5000)
    env.set_rule("A→B"); env.set_shifted(False); r3 = phase(env, P3, 0.0, 999)
    return r1, r2, r3


def measure(agent, seed):
    r1, r2, r3 = run_protocol(agent)
    qz = agent._validate_skill(n_rollouts=NVAL,
                               eval_env=TwoGate2DShiftEnv(rule="A→B", seed=seed + 9000, shifted=False))
    eps = agent.get_log().get("skill_episodes", [])
    ce = eps[0] if eps else None
    return {"Qz": qz, "p2_late": float(np.mean(r2[-20:])), "p3_early": float(np.mean(r3[:20])),
            "cryst_ep": ce}


def build(arm, seed):
    set_seed(seed)
    if arm == "BOCPD":
        ag = PAOBocpd(OBS_DIM, NUM_ACTIONS, **KW)
    elif arm == "Heuristic":
        ag = PAOLight(OBS_DIM, NUM_ACTIONS, **KW)
    elif arm == "NoSkill":
        ag = PAONoSkill(OBS_DIM, NUM_ACTIONS, **KW)
    elif arm == "RandomLate":
        ce = int(np.random.RandomState(seed + 7000).randint(LATE_LO, LATE_HI))
        ag = PAOForced(OBS_DIM, NUM_ACTIONS, crystallize_at=ce, **KW)
    elif arm == "RandomUniform":
        ce = int(np.random.RandomState(seed + 7000).randint(UNIF_LO, UNIF_HI))
        ag = PAOForced(OBS_DIM, NUM_ACTIONS, crystallize_at=ce, **KW)
    ag._app_thresh = APP
    return ag


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if sp == 0: return 0.0
    return (np.mean(x)-np.mean(y))/sp * (1 - 3/(4*(nx+ny-2)-1))

def boot_ci(x, y, n=10000):
    diff = np.asarray(x, float) - np.asarray(y, float); rng = np.random.RandomState(0)
    bs = [np.mean(diff[rng.randint(0, len(diff), len(diff))]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def pw(x, y, alt):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y): return 1.0
    try: return float(wilcoxon(x, y, alternative=alt).pvalue)
    except ValueError: return 1.0

def bh(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p); adj = np.empty(m); prev = 1.0
    for i in range(m-1, -1, -1):
        idx = order[i]; prev = min(prev, p[idx]*m/(i+1)); adj[idx] = prev
    return adj


def main():
    arms = ["BOCPD", "Heuristic", "RandomLate", "RandomUniform", "NoSkill"]
    data = {a: [] for a in arms}
    for arm in arms:
        print(f"\n── {arm} ──")
        for s in SEEDS:
            m = measure(build(arm, s), s)
            data[arm].append(m)
            print(f"  seed={s:2d} cryst_ep={str(m['cryst_ep']):>4s} Qz={m['Qz']:.2f} "
                  f"P3={m['p3_early']:+.3f} P2={m['p2_late']:+.3f}")

    def col(a, k): return np.array([d[k] for d in data[a]], float)
    print(f"\n{'='*82}\n  P2/HARDER-ENV RESULTS (N={len(SEEDS)}, seeded, app={APP}, env=2D-shift)\n{'='*82}")
    print(f"  {'Arm':<14s} {'Qz':>8s} {'P3 early':>14s} {'P2 late':>14s} {'cryst_ep':>10s}")
    for a in arms:
        q, p3, p2 = col(a, 'Qz'), col(a, 'p3_early'), col(a, 'p2_late')
        ce = np.array([d['cryst_ep'] for d in data[a] if d['cryst_ep'] is not None], float)
        print(f"  {a:<14s} {q.mean():>8.3f} {p3.mean():>8.3f}±{p3.std(ddof=1)/np.sqrt(len(p3)):.2f} "
              f"{p2.mean():>8.3f}±{p2.std(ddof=1)/np.sqrt(len(p2)):.2f} {(ce.mean() if len(ce) else float('nan')):>10.1f}")

    print(f"\n  {'─'*78}\n  COMPARISONS (paired Wilcoxon, Hedges g, bootstrap CI, BH-FDR)\n  {'─'*78}")
    specs = [
        ("BOCPD","RandomLate","Qz","greater","PRIMARY  Qz  BOCPD>RandLate"),
        ("BOCPD","RandomLate","p3_early","greater","2ndary   P3  BOCPD>RandLate"),
        ("Heuristic","RandomLate","Qz","greater","ctx      Qz  Heur >RandLate"),
        ("BOCPD","RandomUniform","Qz","greater","ctrl     Qz  BOCPD>RandUnif"),
        ("BOCPD","NoSkill","p3_early","greater","sanity   P3  BOCPD>NoSkill"),
        ("RandomLate","NoSkill","p3_early","greater","sanity   P3  RandLate>NoSkill"),
    ]
    rows, ps = [], []
    for a, b, k, alt, lbl in specs:
        x, y = col(a, k), col(b, k); p = pw(x, y, alt); g = hedges_g(x, y); lo, hi = boot_ci(x, y)
        rows.append((lbl, x.mean(), y.mean(), g, lo, hi, p)); ps.append(p)
    for (lbl, mx, my, g, lo, hi, p), pa in zip(rows, bh(ps)):
        print(f"  {lbl:<30s} {mx:>6.3f} vs {my:>6.3f}  g={g:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={p:.3f} (FDR {pa:.3f})")

    qB, qR = col("BOCPD","Qz"), col("RandomLate","Qz")
    g = hedges_g(qB, qR); p = pw(qB, qR, "greater")
    rev = pw(qR, qB, "greater")  # reverse direction
    verdict = ("SUPPORTED" if (p < 0.05 and g >= 0.5)
               else "FALSIFIED" if ((abs(g) < 0.3 and p > 0.05) or (rev < 0.05 and g <= -0.3))
               else "INCONCLUSIVE")
    print(f"\n  {'─'*78}\n  PRE-REGISTERED VERDICT (BOCPD vs RandomLate, Qz)\n  {'─'*78}")
    print(f"  BOCPD {qB.mean():.3f} vs RandomLate {qR.mean():.3f}  g={g:+.2f}  p(>)={p:.3f}  p(reverse)={rev:.3f}  → H1 {verdict}")

    with open(os.path.join(os.path.dirname(__file__), "results/p2_harder.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/p2_harder.pkl")


if __name__ == "__main__":
    main()
