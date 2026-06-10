"""
Two-Gate 2D — Rule-Swap Hysteresis Experiment
===============================================
Tests the phase-transition signature of structural hysteresis:

Phase 1 (clean, ep 0–200): Rule = A→B. PAO crystallises skill.
Phase 2 (rule swap, ep 200–400): Rule = B→A. Test structural inertia.
Phase 3 (restore, ep 400–500): Rule = A→B. Test skill reuse.

Predictions:
  Phase 2: PAO adapts SLOWER than FlatPPO (cached skill fights new rule)
  Phase 3: PAO recovers FASTER than FlatPPO (skill provides reuse advantage)
  → hysteresis loop: PAO's learning curve is asymmetric across rule changes,
    FlatPPO's is more symmetric.
"""

import argparse, sys, os, pickle
from typing import Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env_2d import TwoGate2DEnv, NUM_ACTIONS, Rule
from agents import PAOLight, FlatPPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBS_DIM = 10   # 2D env: 10-D observation (x, y, 4x relative, 4x flags)
SPATIAL_DIMS = 6  # x, y, rel_A_x, rel_A_y, rel_B_x, rel_B_y


# ─── Train Loop ──────────────────────────────────────────────────────────────

def train_phase(agent, env: TwoGate2DEnv, n_episodes: int,
                name: str = "", verbose: bool = True) -> dict:
    """Train agent on a given env for n_episodes. Returns log."""
    log = {"episode": [], "return": [], "entropy": [], "n_skills": [],
           "phase_step": []}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, training=True)
            obs, r, done, info = env.step(a)
            agent.step_end(r, done)

        agent.finish_episode()
        l = agent.get_log()
        log["episode"].append(ep)
        log["return"].append(l["returns"][-1] if l["returns"] else 0.0)
        log["entropy"].append(l["entropies"][-1] if l["entropies"] else 0.0)
        log["n_skills"].append(l.get("n_skills", 0))

        if verbose and (ep < 5 or ep % 50 == 49 or ep == n_episodes - 1):
            sk = ""
            if isinstance(agent, PAOLight) and l.get("skill_episodes"):
                sk = f" sk@[{','.join(str(e) for e in l['skill_episodes'])}]"
            print(f"  [{name}] ep={ep:4d} R={log['return'][-1]:+.3f} H={log['entropy'][-1]:.3f} "
                  f"skills={l.get('n_skills',0)}{sk}")
    return log


# ─── Multi-Phase Hysteresis ──────────────────────────────────────────────────

def run_hysteresis(agent_cls, kwargs: dict,
                   n_phase1: int, n_phase2: int, n_phase3: int,
                   seeds: list, name: str) -> dict:
    """Run 3-phase rule-swap hysteresis test."""
    results = {}
    for seed in seeds:
        print(f"\n── {name} seed={seed} ──")
        agent = agent_cls(**kwargs)

        # Phase 1: Rule A→B
        env = TwoGate2DEnv(rule="A→B", seed=seed)
        log1 = train_phase(agent, env, n_phase1, name=f"{name}-P1")

        # Record skill state end of Phase 1
        skills_p1 = agent.get_log().get("n_skills", 0)

        # Phase 2: Rule swaps to B→A
        env.set_rule("B→A")
        print(f"  ── RULE SWAP: B→A ──")
        log2 = train_phase(agent, env, n_phase2, name=f"{name}-P2")
        skills_p2 = agent.get_log().get("n_skills", 0)

        # Phase 3: Rule restores to A→B
        env.set_rule("A→B")
        print(f"  ── RULE RESTORE: A→B ──")
        log3 = train_phase(agent, env, n_phase3, name=f"{name}-P3")
        skills_p3 = agent.get_log().get("n_skills", 0)

        # Combine logs
        all_returns = log1["return"] + log2["return"] + log3["return"]
        all_entropies = log1["entropy"] + log2["entropy"] + log3["entropy"]
        all_skills = log1["n_skills"] + log2["n_skills"] + log3["n_skills"]

        # Adaptation metrics
        p2_drop_ep = 50   # check how far into Phase 2 before return recovers
        p2_recovery = np.mean(log2["return"][min(p2_drop_ep, len(log2["return"])-20):-1]) \
            if len(log2["return"]) > p2_drop_ep else 0.0
        p3_recovery = np.mean(log3["return"][-20:]) if log3["return"] else 0.0
        p1_late = np.mean(log1["return"][-20:]) if log1["return"] else 0.0

        results[seed] = {
            "all_returns": all_returns,
            "all_entropies": all_entropies,
            "all_skills": all_skills,
            "log1": log1, "log2": log2, "log3": log3,
            "skills_p1": skills_p1, "skills_p2": skills_p2, "skills_p3": skills_p3,
            "p1_late_return": p1_late,
            "p2_recovery_return": p2_recovery,
            "p3_recovery_return": p3_recovery,
        }
    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def make_plots(pao: dict, flat: dict, save_dir: str, n_p1: int, n_p2: int):
    os.makedirs(save_dir, exist_ok=True)
    seed = list(pao.keys())[0]
    pr = pao[seed]
    fr = flat[seed]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Helper to shade phases
    def shade_phases(ax):
        for i, (start, color, label) in enumerate([
            (0, "lightgreen", "Phase 1: A→B"),
            (n_p1, "lightyellow", "Phase 2: B→A"),
            (n_p1 + n_p2, "lightblue", "Phase 3: A→B"),
        ]):
            ax.axvspan(start, start + (n_p1 if i==0 else n_p2 if i==1 else len(pr["all_returns"])-n_p1-n_p2),
                       alpha=0.15, color=color, label=label)

    # Plot 1: Return curves (ENTIRE experiment, with phase shading)
    ax = axes[0, 0]
    shade_phases(ax)
    ax.plot(pr["all_returns"], label="PAO-light", color="steelblue", lw=1.5)
    ax.plot(fr["all_returns"], label="Flat PPO", color="coral", lw=1.5, alpha=0.8)
    # Mark skill crystallisation
    if "skill_episodes" in pr["log1"]:
        for sk_ep in pr["log1"]["skill_episodes"]:
            ax.axvline(sk_ep, color="steelblue", alpha=0.3, ls="--", lw=1)
            ax.scatter([sk_ep], [pr["log1"]["return"][sk_ep] if sk_ep < len(pr["log1"]["return"]) else 0],
                       c="darkblue", s=100, marker="*", zorder=5)
    ax.axvline(n_p1, color="red", ls=":", lw=2, alpha=0.5, label="Swap: B→A")
    ax.axvline(n_p1 + n_p2, color="green", ls=":", lw=2, alpha=0.5, label="Restore: A→B")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Rule-Swap Hysteresis")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    # Plot 2: Entropy
    ax = axes[0, 1]
    shade_phases(ax)
    ax.plot(pr["all_entropies"], label="PAO-light", color="steelblue", lw=1.5)
    ax.plot(fr["all_entropies"], label="Flat PPO", color="coral", lw=1.5, alpha=0.8)
    ax.axvline(n_p1, color="red", ls=":", lw=2, alpha=0.5)
    ax.axvline(n_p1 + n_p2, color="green", ls=":", lw=2, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Entropy (regime changes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Plot 3: Phase 2 adaptation rate (zoom)
    ax = axes[1, 0]
    p2_pao = pr["all_returns"][n_p1:n_p1 + n_p2]
    p2_flat = fr["all_returns"][n_p1:n_p1 + n_p2]
    ax.plot(p2_pao, label="PAO-light (slower → hysteresis)", color="steelblue", lw=1.5)
    ax.plot(p2_flat, label="Flat PPO (faster → gradient)", color="coral", lw=1.5, alpha=0.8)
    ax.set_xlabel("Episode (within Phase 2)")
    ax.set_ylabel("Return")
    ax.set_title("Phase 2: Adaptation to B→A (Inertia Test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Plot 4: Phase 3 recovery rate (zoom)
    ax = axes[1, 1]
    p3_pao = pr["all_returns"][n_p1 + n_p2:]
    p3_flat = fr["all_returns"][n_p1 + n_p2:]
    ax.plot(p3_pao, label="PAO-light (faster → skill reuse)", color="steelblue", lw=1.5)
    ax.plot(p3_flat, label="Flat PPO (slower → re-learn)", color="coral", lw=1.5, alpha=0.8)
    ax.set_xlabel("Episode (within Phase 3)")
    ax.set_ylabel("Return")
    ax.set_title("Phase 3: Recovery to A→B (Reuse Test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, "rule_swap_hysteresis.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot: {path}")
    plt.close()


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(pao: dict, flat: dict):
    def agg(d):
        vs = list(d.values())
        return {k: np.mean([v[k] for v in vs]) for k in
                ["skills_p1", "skills_p2", "skills_p3",
                 "p1_late_return", "p2_recovery_return", "p3_recovery_return"]}
    pa = agg(pao)
    fa = agg(flat)

    print("\n" + "=" * 66)
    print("  TWO-GATE 2D — RULE-SWAP HYSTERESIS SUMMARY")
    print("=" * 66)
    print(f"  {'Metric':<36s} {'PAO-light':>12s} {'FlatPPO':>12s}")
    print(f"  {'─' * 60}")
    print(f"  {'Skills Phase 1 (A→B)':<36s} {pa['skills_p1']:>12.1f} {'—':>12s}")
    print(f"  {'Skills Phase 2 (B→A)':<36s} {pa['skills_p2']:>12.1f} {'—':>12s}")
    print(f"  {'Skills Phase 3 (A→B)':<36s} {pa['skills_p3']:>12.1f} {'—':>12s}")
    print(f"  {'Return late Phase 1':<36s} {pa['p1_late_return']:>12.3f} {fa['p1_late_return']:>12.3f}")
    print(f"  {'Return Phase 2 recovery':<36s} {pa['p2_recovery_return']:>12.3f} {fa['p2_recovery_return']:>12.3f}")
    print(f"  {'Return Phase 3 recovery':<36s} {pa['p3_recovery_return']:>12.3f} {fa['p3_recovery_return']:>12.3f}")

    # Hysteresis signatures
    p2_slower = pa["p2_recovery_return"] < fa["p2_recovery_return"]
    p3_faster = pa["p3_recovery_return"] > fa["p3_recovery_return"]
    print(f"\n  {'─' * 60}")
    print(f"  Phase 2: PAO slower than FlatPPO (inertia): "
          f"{'✓ YES — hysteresis' if p2_slower else '✗ NO'}")
    print(f"  Phase 3: PAO faster than FlatPPO (reuse):   "
          f"{'✓ YES — skill reuse' if p3_faster else '✗ NO'}")
    if p2_slower and p3_faster:
        print(f"  → HYSTERESIS LOOP CONFIRMED: structural asymmetry across rule swap")
    elif p2_slower and not p3_faster:
        print(f"  → Partial: inertia present, but reuse advantage absent")
    elif not p2_slower and p3_faster:
        print(f"  → Partial: reuse advantage present, but inertia absent")
    else:
        print(f"  → No hysteresis signature detected")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    if args.quick:
        n_p1, n_p2, n_p3 = 80, 80, 50
    else:
        n_p1, n_p2, n_p3 = 200, 200, 100

    seeds = args.seeds
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    kw = {"obs_dim": OBS_DIM, "act_dim": NUM_ACTIONS,
          "lr": 3e-4, "entropy_coef": 0.02,
          "spatial_dims": SPATIAL_DIMS}

    print(f"Two-Gate 2D — Rule-Swap Hysteresis")
    print(f"  Seeds: {seeds}")
    print(f"  Phase 1 (A→B): {n_p1}ep  Phase 2 (B→A): {n_p2}ep  Phase 3 (A→B): {n_p3}ep")
    if args.quick:
        print("  ⚡ Quick mode")

    # Run PAO-light
    print("\n" + "─" * 50)
    print("  PAO-light")
    print("  " + "─" * 30)
    pao_hyst = run_hysteresis(PAOLight, kw, n_p1, n_p2, n_p3, seeds, "PAO")

    # Run FlatPPO
    print("\n" + "─" * 50)
    print("  Flat PPO (baseline)")
    print("  " + "─" * 30)
    flat_hyst = run_hysteresis(FlatPPO, kw, n_p1, n_p2, n_p3, seeds, "Flat")

    print_summary(pao_hyst, flat_hyst)
    make_plots(pao_hyst, flat_hyst, save_dir, n_p1, n_p2)

    data = {"pao": pao_hyst, "flat": flat_hyst}
    path = os.path.join(save_dir, "rule_swap_data.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n  Data: {path}")
    print("  Done.")


if __name__ == "__main__":
    main()
