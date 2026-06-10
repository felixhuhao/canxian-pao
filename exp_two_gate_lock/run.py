"""
Two-Gate Lock Experiment Runner
================================
Trains PAO-light and FlatPPO, runs hysteresis test, generates plots.

Hysteresis design:
  Phase 1 (clean): Standard environment. PAO crystallises A→B→G skill.
  Phase 2 (reward corruption): Goal reward zeroed 65% of the time.
    - PAO predictions: cached skill provides direct action bias →
      agent maintains good performance despite reward noise.
    - FlatPPO: reward noise corrupts gradient → performance degrades.
  Phase 3 (scratch in reward noise): Fresh start under same corruption.
    - Neither learns well, but PAO's cached skill gives it an edge.

Usage:
  python3 run.py                       # full run
  python3 run.py --quick               # fewer episodes
  python3 run.py --seeds 0 1           # specific seeds
"""

import argparse, sys, os, pickle
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOLight, FlatPPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Train Loop ──────────────────────────────────────────────────────────────

def train(agent, n_episodes: int, seed: int = 0,
          reward_corrupt_p: float = 0.0,
          name: str = "", verbose: bool = True) -> dict:
    """
    Train agent for n_episodes.
    reward_corrupt_p: probability of zeroing goal reward (hysteresis noise).
    """
    env = TwoGateLockEnv(seed=seed)
    rng = np.random.RandomState(seed + 999)

    log = {"episode": [], "return": [], "entropy": [], "n_skills": [], "step": []}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            a = agent.act(obs, training=True)
            obs, raw_r, done, info = env.step(a)

            # Reward corruption: zero out goal reward with probability p
            reward = raw_r
            if info.get("at_goal", False) and rng.random() < reward_corrupt_p:
                reward = 0.0  # goal reward suppressed

            agent.step_end(reward, done)
            step += 1

        agent.finish_episode()
        l = agent.get_log()

        log["episode"].append(ep)
        log["return"].append(l["returns"][-1] if l["returns"] else 0.0)
        log["entropy"].append(l["entropies"][-1] if l["entropies"] else 0.0)
        log["n_skills"].append(l.get("n_skills", 0))

        if verbose and (ep < 5 or ep % 50 == 49 or ep == n_episodes - 1):
            sk_str = ""
            if isinstance(agent, PAOLight) and l.get("skill_episodes"):
                sk_str = f" sk@[{l['skill_episodes'][-1]}]"
            print(f"  [{name}] ep={ep:4d} R={log['return'][-1]:+.3f} H={log['entropy'][-1]:.3f} "
                  f"sk={l.get('n_skills',0)}{sk_str}")
    return log


# ─── Hysteresis ──────────────────────────────────────────────────────────────

def hysteresis_test(agent_cls, kwargs: dict,
                    clean_eps: int, corrupt_eps: int, scratch_eps: int,
                    seeds: list, name: str, corrupt_p: float = 0.65) -> dict:
    """
    Phase 1: Clean training → skill formation.
    Phase 2: Continue with reward corruption → skill buffers against reward noise.
    Phase 3: Fresh start under reward corruption → no reliable learning.
    """
    results = {}
    for seed in seeds:
        print(f"\n── {name} seed={seed} ──")
        agent = agent_cls(**kwargs)

        # Phase 1: Clean
        log1 = train(agent, clean_eps, seed=seed, name=f"{name}-clean")
        skills_clean = agent.get_log().get("n_skills", 0)

        # Phase 2: Reward corruption (continue training)
        log2_data = []
        env = TwoGateLockEnv(seed=seed + 100)
        rng = np.random.RandomState(seed + 1000)
        for ep in range(corrupt_eps):
            obs = env.reset()
            done = False
            while not done:
                a = agent.act(obs, training=True)
                obs, raw_r, done, info = env.step(a)
                reward = raw_r
                if info.get("at_goal", False) and rng.random() < corrupt_p:
                    reward = 0.0
                agent.step_end(reward, done)
            agent.finish_episode()
            l = agent.get_log()
            log2_data.append(l["returns"][-1] if l["returns"] else 0.0)
        return_corrupt = np.mean(log2_data[-20:]) if log2_data else 0.0
        skills_corrupt = agent.get_log().get("n_skills", 0)
        print(f"  → Noise phase done. Return(last20)={return_corrupt:.3f} skills={skills_corrupt}")

        # Phase 3: Scratch in reward corruption
        agent_scratch = agent_cls(**kwargs)
        log3 = train(agent_scratch, scratch_eps, seed=seed + 200,
                     reward_corrupt_p=corrupt_p, name=f"{name}-scratch-corrupt")
        skills_scratch = agent_scratch.get_log().get("n_skills", 0)
        return_scratch = np.mean(log3["return"][-20:]) if log3["return"] else 0.0
        print(f"  → Scratch phase done. Return(last20)={return_scratch:.3f} skills={skills_scratch}")

        results[seed] = {
            "log1": log1, "log3": log3,
            "skills_clean": skills_clean,
            "skills_corrupt": skills_corrupt,
            "skills_scratch": skills_scratch,
            "return_clean": np.mean(log1["return"][-30:]) if log1["return"] else 0.0,
            "return_corrupt": return_corrupt,
            "return_scratch": return_scratch,
        }

    return results


# ─── Plots ───────────────────────────────────────────────────────────────────

def make_plots(pao: dict, flat: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    seed = list(pao.keys())[0]
    pr = pao[seed]
    fr = flat[seed]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Learning curves
    ax = axes[0, 0]
    ep = np.arange(len(pr["log1"]["return"]))
    ax.plot(pr["log1"]["return"], label="PAO-light", alpha=0.8, color="steelblue")
    ax.plot(fr["log1"]["return"], label="Flat PPO", alpha=0.8, color="coral")
    # Mark skill crystallisation
    if "skill_episodes" in pr["log1"]:
        for sk_ep in pr["log1"]["skill_episodes"]:
            if sk_ep < len(pr["log1"]["return"]):
                ax.axvline(sk_ep, color="steelblue", alpha=0.3, ls="--", lw=1.5)
                ax.scatter([sk_ep], [pr["log1"]["return"][sk_ep]],
                           c="steelblue", s=120, marker="*", zorder=5,
                           label="Crystallisation" if sk_ep == pr["log1"]["skill_episodes"][0] else "")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("Learning Curves (Phase 1: Clean)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Entropy
    ax = axes[0, 1]
    ax.plot(pr["log1"]["entropy"], label="PAO-light", alpha=0.8, color="steelblue")
    ax.plot(fr["log1"]["entropy"], label="Flat PPO", alpha=0.8, color="coral")
    if "skill_episodes" in pr["log1"]:
        for sk_ep in pr["log1"]["skill_episodes"]:
            if sk_ep < len(pr["log1"]["entropy"]):
                ax.axvline(sk_ep, color="steelblue", alpha=0.3, ls="--", lw=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy (drop = regime change)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Hysteresis bar chart
    ax = axes[1, 0]
    cats = ["Skills\n(Clean)", "Skills\n(Corrupt)", "Scratch\n(Corrupt)"]
    pv = [pr["skills_clean"], pr["skills_corrupt"], pr["skills_scratch"]]
    fv = [0, 0, 0]
    x = np.arange(len(cats))
    ax.bar(x - 0.15, pv, 0.3, label="PAO-light", color="steelblue")
    ax.bar(x + 0.15, fv, 0.3, label="Flat PPO", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Hysteresis: Skill Survival Under Reward Corruption")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Return table
    ax = axes[1, 1]
    ax.axis("off")
    data = [
        ["", "PAO-light", "Flat PPO"],
        ["Return (clean)", f"{pr['return_clean']:.3f}", f"{fr['return_clean']:.3f}"],
        ["Return (corrupt)", f"{pr['return_corrupt']:.3f}", f"{fr['return_corrupt']:.3f}"],
        ["Return (scratch)", f"{pr['return_scratch']:.3f}", f"{fr['return_scratch']:.3f}"],
    ]
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    ax.set_title("Return Comparison Under Reward Corruption")
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "two_gate_lock.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot: {path}")
    plt.close()


def print_summary(pao: dict, flat: dict):
    def agg(d):
        vs = list(d.values())
        return {k: np.mean([v[k] for v in vs]) for k in vs[0] if k.startswith(("skills", "return"))}
    pa = agg(pao)
    fa = agg(flat)
    print("\n" + "=" * 62)
    print("  TWO-GATE LOCK — PHASE-TRANSITION SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<32s} {'PAO-light':>12s} {'FlatPPO':>12s}")
    print(f"  {'─' * 56}")
    print(f"  {'Skills after clean':<32s} {pa['skills_clean']:>12.1f} {'—':>12s}")
    print(f"  {'Skills after corruption':<32s} {pa['skills_corrupt']:>12.1f} {'—':>12s}")
    print(f"  {'Skills scratch-in-corrupt':<32s} {pa['skills_scratch']:>12.1f} {'—':>12s}")
    print(f"  {'Return clean (last30)':<32s} {pa['return_clean']:>12.3f} {fa['return_clean']:>12.3f}")
    print(f"  {'Return corrupt (last20)':<32s} {pa['return_corrupt']:>12.3f} {fa['return_corrupt']:>12.3f}")
    print(f"  {'Return scratch-corrupt':<32s} {pa['return_scratch']:>12.3f} {fa['return_scratch']:>12.3f}")

    # Phase transition signatures
    skill_persistence = bool(pa["skills_corrupt"] > pa["skills_scratch"])
    perf_edge = bool(pa["return_corrupt"] > fa["return_corrupt"])
    print(f"\n  {'─' * 56}")
    print(f"  Skill persistence (skills survive corruption): "
          f"{'✓' if skill_persistence else '✗'}")
    print(f"  PAO perf > FlatPPO under corruption: "
          f"{'✓' if perf_edge else '✗'}")
    print(f"  {'→ Phase transition signature confirmed!' if skill_persistence else '→ Signature absent.'}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corrupt-p", type=float, default=0.65,
                        help="Probability of zeroing goal reward (hysteresis noise)")
    args = parser.parse_args()

    if args.quick:
        clean_eps, corrupt_eps, scratch_eps = 80, 40, 50
    else:
        clean_eps, corrupt_eps, scratch_eps = 300, 100, 100

    seeds = args.seeds
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    kw = {"obs_dim": 5, "act_dim": 2, "lr": 3e-4, "entropy_coef": 0.02}

    print(f"Two-Gate Lock — Phase Transition Experiment")
    print(f"  Seeds: {seeds}")
    print(f"  Clean: {clean_eps}ep  Corrupt: {corrupt_eps}ep  Scratch: {scratch_eps}ep")
    print(f"  Reward corruption p = {args.corrupt_p} (hysteresis noise)")
    if args.quick:
        print("  ⚡ Quick mode")

    # ── Hysteresis Test ──
    print("\n" + "─" * 40)
    print("  HYSTERESIS TEST")
    print("  " + "─" * 30)
    pao_hyst = hysteresis_test(
        PAOLight, kw, clean_eps, corrupt_eps, scratch_eps, seeds,
        "PAO", corrupt_p=args.corrupt_p)
    flat_hyst = hysteresis_test(
        FlatPPO, kw, clean_eps, corrupt_eps, scratch_eps, seeds,
        "Flat", corrupt_p=args.corrupt_p)

    print_summary(pao_hyst, flat_hyst)
    make_plots(pao_hyst, flat_hyst, save_dir)

    data_path = os.path.join(save_dir, "results.pkl")
    with open(data_path, "wb") as f:
        pickle.dump({"pao": pao_hyst, "flat": flat_hyst}, f)
    print(f"\n  Data: {data_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
