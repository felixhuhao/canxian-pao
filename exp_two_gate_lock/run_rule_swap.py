"""
Two-Gate 1D — Rule-Swap Hysteresis Experiment
===============================================
Tests structural hysteresis via rule swap in the 1D corridor.
Also runs ablation controls: PAO-no-dormancy, PAO-no-skill.
"""

import argparse, sys, os, pickle, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import TwoGateLockEnv, NUM_ACTIONS
from agents import PAOLight, PAONoDormancy, PAONoSkill, FlatPPO


def set_seed(seed: int):
    """Seed all global RNGs so each `seed` is a reproducible replicate.
    Without this, torch net-init, action sampling, and numpy minibatch
    shuffling run off the unseeded global RNG (see FINDINGS.md, 2026-06-07)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OBS_DIM = 5
ACT_DIM = 2


def train_phase(agent, env, n_episodes: int,
                name: str = "", verbose: bool = True,
                explore_eps: float = 0.0,
                rng_seed: int = 999) -> dict:
    rng = np.random.RandomState(rng_seed)
    log = {"return": [], "entropy": [], "n_skills": []}
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, training=True)
            if rng.random() < explore_eps:
                a = rng.randint(0, NUM_ACTIONS)
            obs, r, done, info = env.step(a)
            agent.step_end(r, done)
        agent.finish_episode()
        l = agent.get_log()
        log["return"].append(l["returns"][-1] if l["returns"] else 0.0)
        log["entropy"].append(l["entropies"][-1] if l["entropies"] else 0.0)
        log["n_skills"].append(l.get("n_skills", 0))
        if verbose and (ep < 5 or ep % 50 == 49 or ep == n_episodes - 1):
            sk = ""
            if l.get("skill_episodes"):
                sk = f" sk@[{','.join(str(e) for e in l['skill_episodes'][-3:])}]"
            print(f"  [{name}] ep={ep:4d} R={log['return'][-1]:+.3f} H={log['entropy'][-1]:.3f} "
                  f"sk={l.get('n_skills',0)}{sk}")
    return log


def run_hysteresis(agent_cls, kwargs: dict,
                   n_p1: int, n_p2: int, n_p3: int,
                   seeds: list, name: str) -> dict:
    results = {}
    for seed in seeds:
        print(f"\n── {name} seed={seed} ──")
        set_seed(seed)  # reproducible + matched across conditions
        agent = agent_cls(**kwargs)
        env = TwoGateLockEnv(rule="A→B", seed=seed)
        log1 = train_phase(agent, env, n_p1, name=f"{name}-P1")
        skills_p1 = agent.get_log().get("n_skills", 0)
        skill_eps_p1 = list(agent.get_log().get("skill_episodes", []))
        env.set_rule("B→A")
        print(f"  ── RULE SWAP: B→A (ε=0.1) ──")
        log2 = train_phase(agent, env, n_p2, name=f"{name}-P2",
                           explore_eps=0.1, rng_seed=seed + 5000)
        skills_p2 = agent.get_log().get("n_skills", 0)
        env.set_rule("A→B")
        print(f"  ── RULE RESTORE: A→B ──")
        log3 = train_phase(agent, env, n_p3, name=f"{name}-P3")
        skills_p3 = agent.get_log().get("n_skills", 0)
        all_returns = log1["return"] + log2["return"] + log3["return"]
        all_entropies = log1["entropy"] + log2["entropy"] + log3["entropy"]

        results[seed] = {
            "all_returns": all_returns, "all_entropies": all_entropies,
            "log1": log1, "log2": log2, "log3": log3,
            "skills_p1": skills_p1, "skills_p2": skills_p2, "skills_p3": skills_p3,
            "skill_eps_p1": skill_eps_p1,
            "p2_late": np.mean(log2["return"][-20:]) if len(log2["return"]) >= 20 else 0.0,
            "p3_early": np.mean(log3["return"][:20]) if len(log3["return"]) >= 20 else 0.0,
            "p3_late": np.mean(log3["return"][-20:]) if len(log3["return"]) >= 20 else 0.0,
        }
    return results


# ─── Plot (SEM error bands) ────────────────────────────────────────────────

def make_plots(pao: dict, flat: dict, save_dir: str, n_p1: int, n_p2: int, n_p3: int):
    os.makedirs(save_dir, exist_ok=True)
    seeds_pao = list(pao.keys())
    seeds_flat = list(flat.keys())

    def stack_and_sem(data, seeds, key):
        arr = np.array([data[s][key] for s in seeds])
        m = np.mean(arr, axis=0)
        s = np.std(arr, axis=0, ddof=1)
        n = len(seeds)
        return m, s / np.sqrt(n)  # SEM

    pao_m, pao_e = stack_and_sem(pao, seeds_pao, "all_returns")
    flat_m, flat_e = stack_and_sem(flat, seeds_flat, "all_returns")
    pao_em, pao_ee = stack_and_sem(pao, seeds_pao, "all_entropies")
    flat_em, flat_ee = stack_and_sem(flat, seeds_flat, "all_entropies")

    total_eps = len(pao_m)
    n_p3_actual = total_eps - n_p1 - n_p2
    blue, orange = "#1f77b4", "#ff7f0e"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def shade(ax):
        for s, c, l in [(0, "#e8f5e9", "P1: A→B"), (n_p1, "#fff8e1", "P2: B→A"),
                         (n_p1+n_p2, "#e3f2fd", "P3: A→B")]:
            e = s + (n_p1 if s==0 else n_p2 if s==n_p1 else n_p3_actual)
            ax.axvspan(s, e, alpha=0.12, color=c, label=l)

    def pline(ax):
        ax.axvline(n_p1, color="#d32f2f", ls="--", lw=1.5, alpha=0.5, label="Swap B→A")
        ax.axvline(n_p1+n_p2, color="#388e3c", ls="--", lw=1.5, alpha=0.5, label="Restore A→B")

    def band(ax, x, m, e, c, label):
        ax.plot(x, m, color=c, lw=1.5, label=label)
        ax.fill_between(x, m-e, m+e, color=c, alpha=0.15)

    x = np.arange(total_eps)
    pr0 = pao[seeds_pao[0]]

    ax = axes[0,0]; shade(ax); pline(ax)
    band(ax, x, pao_m, pao_e, blue, "PAO-light")
    band(ax, x, flat_m, flat_e, orange, "Flat PPO")
    for sk_ep in pr0["skill_eps_p1"]:
        if sk_ep < n_p1:
            ax.axvline(sk_ep, color=blue, alpha=0.2, ls=":")
            ax.scatter([sk_ep], [pr0["log1"]["return"][sk_ep]],
                       c="#0d47a1", s=100, marker="*", zorder=5, label="Crystallisation")
    ax.set_xlabel("Episode", fontsize=10); ax.set_ylabel("Return", fontsize=10)
    ax.set_title("Full Return (mean ± SEM, N=10)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2); ax.set_ylim(-1.2, 2.0)

    ax = axes[0,1]; shade(ax); pline(ax)
    band(ax, x, pao_em, pao_ee, blue, "PAO-light")
    band(ax, x, flat_em, flat_ee, orange, "Flat PPO")
    ax.set_xlabel("Episode", fontsize=10); ax.set_ylabel("Policy Entropy", fontsize=10)
    ax.set_title("Entropy", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.2)

    ax = axes[1,0]; zoom2 = 40
    x2 = np.arange(zoom2)
    band(ax, x2, pao_m[n_p1:n_p1+zoom2], pao_e[n_p1:n_p1+zoom2], blue, "PAO-light (locked)")
    band(ax, x2, flat_m[n_p1:n_p1+zoom2], flat_e[n_p1:n_p1+zoom2], orange, "Flat PPO (unstable)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.annotate("Structural Inertia", xy=(5, -0.6), fontsize=8, color=blue, fontstyle="italic")
    ax.set_xlabel("Episode in Phase 2", fontsize=10); ax.set_ylabel("Return", fontsize=10)
    ax.set_title("Phase 2: B→A (Inertia Test)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_ylim(-1.2, 1.6)

    ax = axes[1,1]; zoom3 = min(40, n_p3_actual)
    x3 = np.arange(zoom3)
    band(ax, x3, pao_m[n_p1+n_p2:n_p1+n_p2+zoom3],
         pao_e[n_p1+n_p2:n_p1+n_p2+zoom3], blue, "PAO-light (instant)")
    band(ax, x3, flat_m[n_p1+n_p2:n_p1+n_p2+zoom3],
         flat_e[n_p1+n_p2:n_p1+n_p2+zoom3], orange, "Flat PPO (partial)")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.annotate("Reuse Acceleration", xy=(5, 1.0), fontsize=8, color=blue, fontstyle="italic")
    ax.set_xlabel("Episode in Phase 3", fontsize=10); ax.set_ylabel("Return", fontsize=10)
    ax.set_title("Phase 3: Recovery to A→B (Reuse Test)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_ylim(-1.2, 2.0)

    plt.tight_layout()
    for fmt, dpi_ in [("png", 300), ("pdf", None)]:
        p = os.path.join(save_dir, f"1d_rule_swap_hysteresis.{fmt}")
        kw = {"dpi": dpi_} if dpi_ else {}
        plt.savefig(p, bbox_inches="tight", **kw)
        print(f"  Plot: {p}")
    plt.close()


# ─── Summary (SEM, Hedges' g, Phase 3 late 20) ─────────────────────────────

def print_summary(pao: dict, flat: dict, label: str = ""):
    from scipy.stats import wilcoxon
    sp, sf = list(pao.keys()), list(flat.keys())
    n = min(len(sp), len(sf))

    def v(k): return np.array([pao[s][k] for s in sp]), np.array([flat[s][k] for s in sf])

    p2p, p2f = v("p2_late"); p3p, p3f = v("p3_early"); p3lp, p3lf = v("p3_late")

    def sem(x): return np.std(x, ddof=1) / np.sqrt(len(x))
    def hedges(x, y):
        nx, ny = len(x), len(y)
        s = np.sqrt(((nx-1)*np.var(x,ddof=1)+(ny-1)*np.var(y,ddof=1))/(nx+ny-2))
        d = (np.mean(x)-np.mean(y))/s if s > 0 else 0.0
        return d * (1 - 3/(4*(nx+ny-2)-1))

    wp2 = wilcoxon(p2p[:n], p2f[:n], alternative='less')
    wp3 = wilcoxon(p3p[:n], p3f[:n], alternative='greater')
    g2, g3 = hedges(p2p, p2f), hedges(p3p, p3f)

    def f(m, se): return f"{m:.3f}±{se:.3f}"

    tag = f" {label}" if label else ""
    print(f"\n{'='*78}")
    print(f"  HYSTERESIS STATISTICS{tag} (N={len(sp)} seeds)")
    print(f"{'='*78}")
    print(f"  {'Metric':<36s} {'PAO':>18s} {'Flat':>18s}")
    print(f"  {'─'*72}")
    print(f"  {'Phase 2 (late 20)':<36s} {f(np.mean(p2p),sem(p2p)):>18s} {f(np.mean(p2f),sem(p2f)):>18s}")
    print(f"  {'Phase 3 (early 20)':<36s} {f(np.mean(p3p),sem(p3p)):>18s} {f(np.mean(p3f),sem(p3f)):>18s}")
    print(f"  {'Phase 3 (late 20)':<36s}  {f(np.mean(p3lp),sem(p3lp)):>18s} {f(np.mean(p3lf),sem(p3lf)):>18s}")
    print(f"  {'─'*72}")
    print(f"  Wilcoxon (P2 inertia, PAO<Flat): W={wp2.statistic:.1f} p={wp2.pvalue:.4f}")
    print(f"  Wilcoxon (P3 reuse, PAO>Flat):   W={wp3.statistic:.1f} p={wp3.pvalue:.4f}")
    print(f"  Hedges' g (P2): {g2:.2f}  |  Hedges' g (P3): {g3:.2f}")
    print(f"  {'─'*72}")
    # Corrected interpretations
    print(f"  Phase 2: PAO={np.mean(p2p):.3f}±{sem(p2p):.3f} (locked); "
          f"Flat={np.mean(p2f):.3f}±{sem(p2f):.3f} (fails to consolidate B→A)")
    print(f"  Phase 3: PAO={np.mean(p3p):.3f}±{sem(p3p):.3f} (instant); "
          f"Flat={np.mean(p3f):.3f}±{sem(p3f):.3f} (unstable partial, variance large)")
    if wp2.pvalue < 0.05 and wp3.pvalue < 0.05:
        print(f"  → Asymmetric adaptation consistent with hysteresis (preliminary, N={len(sp)}).")
        print(f"    Effect sizes g={g2:.2f}/{g3:.2f} require full protocol (N≥30) for definitive test.")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ablations", action="store_true",
                        help="Also run PAO-no-dormancy and PAO-no-skill")
    args = parser.parse_args()

    n_p1, n_p2, n_p3 = (80, 120, 60) if args.quick else (150, 200, 100)
    seeds = args.seeds
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    kw = {"obs_dim": OBS_DIM, "act_dim": ACT_DIM, "lr": 3e-4, "entropy_coef": 0.02}

    print(f"1D Rule-Swap Hysteresis  Seeds:{seeds}  P1:{n_p1}  P2:{n_p2}  P3:{n_p3}")
    if args.quick: print("  ⚡ Quick mode")

    # ── Main: PAO-light vs FlatPPO ──
    print("\n" + "─"*50 + "\n  PAO-light"); pao = run_hysteresis(PAOLight, kw, n_p1, n_p2, n_p3, seeds, "PAO")
    print("\n" + "─"*50 + "\n  Flat PPO"); flat = run_hysteresis(FlatPPO, kw, n_p1, n_p2, n_p3, seeds, "Flat")
    print_summary(pao, flat)
    make_plots(pao, flat, save_dir, n_p1, n_p2, n_p3)

    all_data = {"pao": pao, "flat": flat}

    # ── Ablations ──
    if args.ablations:
        print("\n" + "═"*50 + "\n  ABLATION: PAO-no-dormancy (lr_factor=1.0)")
        nodorm = run_hysteresis(PAONoDormancy, kw, n_p1, n_p2, n_p3, seeds, "NoDorm")
        print_summary(nodorm, flat, "NoDorm vs Flat")

        print("\n" + "═"*50 + "\n  ABLATION: PAO-no-skill (bias_strength=0)")
        noskill = run_hysteresis(PAONoSkill, kw, n_p1, n_p2, n_p3, seeds, "NoSkill")
        print_summary(noskill, flat, "NoSkill vs Flat")

        all_data["nodorm"] = nodorm
        all_data["noskill"] = noskill

    path = os.path.join(save_dir, "rule_swap_1d.pkl")
    with open(path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"  Data: {path}")


if __name__ == "__main__":
    main()
