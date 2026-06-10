"""
P0.6 — Does the skill gate have ANY causal effect?
==================================================
Finding (FINDINGS.md 2026-06-07): at the default app_thresh=0.7 the ApplicabilityNet
never opens (scores collapse to [0.30,0.49]), so the crystallised skill is never applied
and PAO-light == PAO-no-skill byte-identical.

This script tests whether LOWERING the threshold makes the skill causally active.
For each condition we report Phase-2 (lock-in) and Phase-3 (reuse), plus the % of
post-crystallisation steps the skill is actually applied, and whether PAO diverges
from its own no-skill control (the clean causal test).

Seeded for reproducibility + matched across conditions.
"""
import sys, os, pickle, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOLight, PAONoSkill, FlatPPO

P1, P2, P3 = 80, 120, 60
SEEDS = list(range(10))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def run_agent(agent, count_apply=False):
    """Run the 3-phase hysteresis protocol on one agent. Returns metrics."""
    applied = [0]; postskill = [0]
    if count_apply:
        _orig = agent.act
        def act2(obs, training=True):
            if getattr(agent, 'skill_policy', None) is not None and agent.applicability is not None and training:
                postskill[0] += 1
                with torch.no_grad():
                    sc = float(agent.applicability(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item())
                if sc > getattr(agent, '_app_thresh', 0.7):
                    applied[0] += 1
            return _orig(obs, training)
        agent.act = act2

    def phase(env, n, eps=0.0, rs=999):
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
    r1 = phase(env, P1)
    env.set_rule("B→A"); r2 = phase(env, P2, eps=0.1, rs=5000)
    env.set_rule("A→B"); r3 = phase(env, P3)
    all_r = r1 + r2 + r3
    return {
        "all": all_r,
        "p2_late": float(np.mean(r2[-20:])),
        "p3_early": float(np.mean(r3[:20])),
        "apply_pct": (100.0 * applied[0] / postskill[0]) if postskill[0] else 0.0,
    }


def make(cls, app, bias=None):
    def factory(seed):
        set_seed(seed)
        ag = cls(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02)
        ag._app_thresh = app
        if bias is not None: ag.skill_bias_strength = bias
        return ag
    return factory


CONDITIONS = {
    "Flat":            (lambda s: (set_seed(s), FlatPPO(obs_dim=5, act_dim=2, lr=3e-4, entropy_coef=0.02))[1], False),
    "PAO app=0.7":     (make(PAOLight, 0.7), True),
    "NoSkill app=0.7": (make(PAONoSkill, 0.7), False),
    "PAO app=0.3":     (make(PAOLight, 0.3), True),
    "NoSkill app=0.3": (make(PAONoSkill, 0.3), False),
}


def main():
    data = {name: [] for name in CONDITIONS}
    for name, (factory, count) in CONDITIONS.items():
        print(f"\n── {name} ──")
        for s in SEEDS:
            ag = factory(s)
            m = run_agent(ag, count_apply=count)
            data[name].append(m)
            print(f"  seed={s}  P2={m['p2_late']:+.3f}  P3={m['p3_early']:+.3f}  apply={m['apply_pct']:.1f}%")

    def agg(name, k): return np.array([d[k] for d in data[name]])
    print(f"\n{'='*72}\n  P0.6 GATE CAUSALITY (N={len(SEEDS)}, seeded)\n{'='*72}")
    print(f"  {'Condition':<16s} {'P2 late':>14s} {'P3 early':>14s} {'skill-apply%':>13s}")
    for name in CONDITIONS:
        p2, p3, ap = agg(name, 'p2_late'), agg(name, 'p3_early'), agg(name, 'apply_pct')
        print(f"  {name:<16s} {p2.mean():>7.3f}±{p2.std(ddof=1)/np.sqrt(len(p2)):.3f} "
              f"{p3.mean():>7.3f}±{p3.std(ddof=1)/np.sqrt(len(p3)):.3f} {ap.mean():>12.1f}")

    # causal test: does PAO differ from its own no-skill control, per threshold?
    print(f"\n  Causal test — PAO vs its NoSkill control (byte-identical ⇒ gate inert):")
    for thr in ["0.7", "0.3"]:
        pao = [np.array(d["all"]) for d in data[f"PAO app={thr}"]]
        nos = [np.array(d["all"]) for d in data[f"NoSkill app={thr}"]]
        identical = all(np.array_equal(a, b) for a, b in zip(pao, nos))
        print(f"    app={thr}: PAO == NoSkill ? {identical}")

    with open(os.path.join(os.path.dirname(__file__), "results/p06_gate_test.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/p06_gate_test.pkl")


if __name__ == "__main__":
    main()
