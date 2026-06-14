"""
P3 — Trigger steelman in UNPREDICTABLE, RECURRING non-stationarity.
==================================================================
Pre-registered in ../PREREG_P3_unpredictable.md.

Why this test exists: p1/p2 swapped rules on a FIXED, known schedule, where a
clock ("crystallise at ep 40") is sufficient by construction -> the trigger added
nothing. Event-triggered crystallisation/reuse can only earn its keep where a
clock CANNOT know when to act: random, unsignaled, recurring regime changes.

Regimes = ACTION-MAPPING FLIPS on the (easy, symmetric) A->B task: in a "flipped"
segment the agent's action labels are inverted before reaching the env. This makes
the two regimes (a) equally learnable (identical task, relabeled actions) and
(b) maximally antagonistic (a skill cached in one regime is exactly correct when
that regime recurs, exactly wrong otherwise). The flip is unsignaled: observations
are identical, only action semantics differ, so observations alone cannot reveal
the active regime. Change-point detection therefore becomes load-bearing for skill
RE-SELECTION, not just crystallisation. This is PAO's best honest shot.

Arms (identical library-building rule; differ ONLY in the re-selection trigger):
  BOCPD     : re-select on detected change-point            [PAOLibraryBOCPD]  <- under test
  Random    : re-select at random eps (count-matched)       [PAOLibraryRandom] <- decisive control
  Obs       : per-step obs-confidence selection (no detect) [PAOLibraryObs]    <- should fail
  NoSkill   : plain PPO, no library                         [PAONoSkill]       <- "any lead?" floor

Primary metric : adaptation speed = mean return over first ADAPT_K episodes after
                 each swap into a PREVIOUSLY-SEEN rule, averaged over swaps.
Stats          : paired Wilcoxon (one-sided) + Hedges' g + 95% bootstrap CI. N=30.
"""
import sys, os, pickle, random
import numpy as np
import torch
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOLibraryBOCPD, PAOLibraryRandom, PAOLibraryObs, PAONoSkill

# ── frozen config ──
SEEDS = list(range(30))
N_SEGMENTS = 8
SEG_LO, SEG_HI = 40, 90      # random segment length (episodes)
DEBOUNCE = 10                # min episodes between crystallisations
HAZARD = 0.01               # BOCPD hazard
REFRACTORY = 30             # min episodes between acted-on resets (anti-thrash)
ADAPT_K = 15                # episodes after a boundary used for adaptation-speed metric
EPS = 0.1                   # exploration epsilon (needed to discover the new rule post-swap)
KW = dict(lr=3e-4, entropy_coef=0.02)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def make_schedule(seed):
    """Seed-determined segment schedule (SAME across arms -> paired comparison).
    Regime = action-flip flag, alternating False/True so each recurs; segs 0,1 are
    first-occurrences. Task is always A->B (easy, symmetric)."""
    rng = np.random.RandomState(seed + 12345)
    flips = [bool(i % 2) for i in range(N_SEGMENTS)]
    lengths = [int(rng.randint(SEG_LO, SEG_HI + 1)) for _ in range(N_SEGMENTS)]
    return list(zip(flips, lengths))


def run_protocol(agent, seed, schedule):
    ep_rng = np.random.RandomState(seed + 777)
    env = TwoGateLockEnv(rule="A→B", seed=0)
    rets, boundaries, seen = [], [], set()
    gep = 0
    for seg_i, (flipped, length) in enumerate(schedule):
        boundaries.append({"ep": gep, "regime": flipped, "recur": flipped in seen, "seg": seg_i})
        seen.add(flipped)
        for _ in range(length):
            if hasattr(agent, "begin_episode"):
                agent.begin_episode()
            # no exploration noise when a deterministic cached skill is driving the
            # episode (it only derails a known-good policy); explore only on the base.
            using_skill = getattr(agent, "_episode_bias_idx", None) is not None
            eps_eff = 0.0 if using_skill else EPS
            o = env.reset(); d = False
            while not d:
                a = agent.act(o, True)
                if eps_eff > 0 and ep_rng.random() < eps_eff:
                    a = ep_rng.randint(0, NUM_ACTIONS)
                env_a = (1 - a) if flipped else a   # unsignaled action-mapping flip
                o, r, d, i = env.step(env_a); agent.step_end(r, d)
            agent.finish_episode()
            rets.append(agent.get_log()["returns"][-1])
            gep += 1
    return rets, boundaries


def adaptation_speed(rets, boundaries, k=ADAPT_K):
    vals = [np.mean(rets[b["ep"]:b["ep"] + k]) for b in boundaries
            if b["recur"] and rets[b["ep"]:b["ep"] + k]]
    return float(np.mean(vals)) if vals else float("nan")


def recovery_time(rets, boundaries, schedule, thresh=0.5):
    """Episodes after a recurrence swap until return first crosses `thresh`
    (capped at segment length = 'never recovered'). Lower = faster reuse."""
    times = []
    for b in boundaries:
        if not b["recur"]:
            continue
        s = b["ep"]; L = schedule[b["seg"]][1]
        seg = rets[s:s + L]
        t = L
        for i, r in enumerate(seg):
            if r > thresh:
                t = i; break
        times.append(t)
    return float(np.mean(times)) if times else float("nan")


def asymptotic(rets, boundaries, schedule, k=15):
    """Mean return over the LAST k episodes of each recurrence segment."""
    ends = []
    for b in boundaries:
        if not b["recur"]:
            continue
        seg_len = schedule[b["seg"]][1]
        e = b["ep"] + seg_len
        ends.append(np.mean(rets[max(b["ep"], e - k):e]))
    return float(np.mean(ends)) if ends else float("nan")


def measure(agent, seed, schedule):
    rets, boundaries = run_protocol(agent, seed, schedule)
    return {
        "adapt": adaptation_speed(rets, boundaries),
        "asymp": asymptotic(rets, boundaries, schedule),
        "total": float(np.sum(rets)),
        "lib": len(getattr(agent, "library", [])),
        "resets": int(getattr(agent, "reset_count", 0)),
        "rets": rets, "boundaries": boundaries,
    }


def build(arm, seed, schedule, bocpd_count=None):
    set_seed(seed)
    total = sum(L for _, L in schedule)
    if arm == "BOCPD":
        return PAOLibraryBOCPD(5, 2, cryst_debounce=DEBOUNCE, bocpd_hazard=HAZARD, **KW)
    if arm == "Random":
        n = int(bocpd_count or 0)
        # draw n random reset eps spaced >= REFRACTORY apart, so the agent's
        # refractory doesn't silently drop any (keeps count matched to BOCPD).
        rng = np.random.RandomState(seed + 333)
        pool = list(range(20, total)); rng.shuffle(pool)
        chosen = []
        for e in pool:
            if all(abs(e - c) >= REFRACTORY for c in chosen):
                chosen.append(e)
            if len(chosen) >= n:
                break
        return PAOLibraryRandom(5, 2, cryst_debounce=DEBOUNCE, reset_eps=set(chosen), **KW)
    if arm == "Obs":
        return PAOLibraryObs(5, 2, cryst_debounce=DEBOUNCE, **KW)
    if arm == "NoSkill":
        return PAONoSkill(5, 2, **KW)
    raise ValueError(arm)


# ── stats (same as p1) ──
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


def main():
    arms = ["BOCPD", "Random", "Obs", "NoSkill"]
    data = {a: [] for a in arms}
    bocpd_counts = {}
    for arm in arms:
        print(f"\n── {arm} ──")
        for s in SEEDS:
            sched = make_schedule(s)
            ag = build(arm, s, sched, bocpd_count=bocpd_counts.get(s))
            m = measure(ag, s, sched)
            if arm == "BOCPD":
                bocpd_counts[s] = m["resets"]
            data[arm].append(m)
            print(f"  seed={s:2d}  adapt={m['adapt']:+.3f}  asymp={m['asymp']:+.3f}  "
                  f"total={m['total']:+.1f}  lib={m['lib']}  resets={m['resets']}")

    def col(a, k): return np.array([d[k] for d in data[a]], float)
    print(f"\n{'='*84}\n  P3 RESULTS (N={len(SEEDS)}, seeded, unpredictable recurring env)\n{'='*84}")
    print(f"  {'Arm':<10s} {'adapt':>10s} {'asymp':>10s} {'total':>10s} {'lib':>6s} {'resets':>8s}")
    for a in arms:
        ad, asy, to = col(a, 'adapt'), col(a, 'asymp'), col(a, 'total')
        print(f"  {a:<10s} {ad.mean():>7.3f}±{ad.std(ddof=1)/np.sqrt(len(ad)):.2f} "
              f"{asy.mean():>7.3f}±{asy.std(ddof=1)/np.sqrt(len(asy)):.2f} "
              f"{to.mean():>10.1f} {col(a,'lib').mean():>6.1f} {col(a,'resets').mean():>8.1f}")

    print(f"\n  {'─'*80}\n  COMPARISONS (paired Wilcoxon, Hedges g, bootstrap CI) on PRIMARY metric 'adapt'\n  {'─'*80}")
    specs = [
        ("BOCPD", "NoSkill", "greater", "HEADLINE  BOCPD > NoSkill  (any lead?)"),
        ("BOCPD", "Random", "greater", "DECISIVE  BOCPD > Random   (detection load-bearing?)"),
        ("BOCPD", "Obs", "greater", "ctx       BOCPD > Obs      (detection > obs-select?)"),
        ("Random", "NoSkill", "greater", "ctx       Random > NoSkill (any library benefit?)"),
        ("Obs", "NoSkill", "greater", "ctx       Obs > NoSkill"),
    ]
    for a, b, alt, lbl in specs:
        x, y = col(a, 'adapt'), col(b, 'adapt')
        g = hedges_g(x, y); lo, hi = boot_ci(x, y); p = pw(x, y, alt)
        print(f"  {lbl:<52s} {x.mean():>6.3f} vs {y.mean():>6.3f}  g={g:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={p:.3f}")

    # ── pre-registered verdict ──
    bn = col("BOCPD", 'adapt'); ns = col("NoSkill", 'adapt'); rd = col("Random", 'adapt')
    g_bn = hedges_g(bn, ns); p_bn = pw(bn, ns, "greater"); rev_bn = pw(ns, bn, "greater")
    g_br = hedges_g(bn, rd); p_br = pw(bn, rd, "greater"); rev_br = pw(rd, bn, "greater")
    print(f"\n  {'─'*80}\n  PRE-REGISTERED VERDICT (primary metric = adapt)\n  {'─'*80}")
    if rev_bn < 0.05 and g_bn <= -0.3:
        head = "ANTI (BOCPD significantly WORSE than NoSkill)"
    elif p_bn < 0.05 and g_bn >= 0.5:
        head = "LEAD FOUND (BOCPD > NoSkill)"
    elif p_bn > 0.05 and abs(g_bn) < 0.3:
        head = "NULL (BOCPD ~ NoSkill)"
    else:
        head = "INCONCLUSIVE"
    print(f"  HEADLINE  BOCPD {bn.mean():.3f} vs NoSkill {ns.mean():.3f}  g={g_bn:+.2f} "
          f"p(>)={p_bn:.3f} p(rev)={rev_bn:.3f}  -> {head}")
    load = ("YES" if (p_br < 0.05 and g_br >= 0.5)
            else "ANTI" if (rev_br < 0.05 and g_br <= -0.3)
            else "NO")
    print(f"  DECISIVE  BOCPD {bn.mean():.3f} vs Random {rd.mean():.3f}  g={g_br:+.2f} "
          f"p(>)={p_br:.3f} p(rev)={rev_br:.3f}  -> detection load-bearing: {load}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/p3_unpredictable.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/p3_unpredictable.pkl")


if __name__ == "__main__":
    main()
