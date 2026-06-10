"""
Rigidity Knob Sweep: Measure the Crystallinity Tradeoff
========================================================
Systematically varies skill rigidity and measures hysteresis vs generalisation.

Conditions (4 × 5 seeds = 20 runs):
  Soft:     app_thresh=0.3, bias=0.5
  Medium:   app_thresh=0.6, bias=1.0
  Hard:     app_thresh=0.9, bias=2.0
  Trajectory: old nearest-neighbour matching (no policy network)

Metrics per condition:
  - p1_skill_success:  held-out validation rate (skill generalisation)
  - p2_pao_return:     Phase 2 mean return (lock-in)
  - p2_flat_return:    Phase 2 FlatPPO baseline
  - hysteresis_index:  (flat_p2 - pao_p2) / (|flat_p2| + 0.01)
  - p3_recovery:       Phase 3 early return (reuse)
"""

import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from env_2d_small import TwoGate2DSmallEnv
from agents import PAOLight, FlatPPO
import torch

OBS_DIM = 33; N_ACT = 4; P1, P2, P3 = 500, 200, 150
SEEDS = [0, 1, 3, 6, 7]

# ─── Condition definitions ──────────────────────────────────────────────────
CONDITIONS = {
    "Soft":       {"app_thresh": 0.3, "bias": 0.5, "use_traj": False},
    "Medium":     {"app_thresh": 0.6, "bias": 1.0, "use_traj": False},
    "Hard":       {"app_thresh": 0.9, "bias": 2.0, "use_traj": False},
    "Trajectory": {"app_thresh": 0.0, "bias": 3.0, "use_traj": True},
}


def eval_skill_success(agent, env, n_trials=50):
    """Run skill-guided rollouts; measure success rate (return > 1.0)."""
    if agent.skill_policy is None:
        return 0.0
    successes = 0
    for _ in range(n_trials):
        obs = env.reset(); done = False; total = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            # Skill-only action (no base policy)
            with torch.no_grad():
                logits = agent.skill_policy(obs_t)
            a = torch.distributions.Categorical(logits=logits).sample()
            obs, r, done, _ = env.step(int(a.item()))
            total += r
        if total > 1.0:
            successes += 1
    return successes / n_trials


def run_condition(cond_name, cond_params, seeds):
    """Run one condition across all seeds. Returns list of per-seed results."""
    results = []
    for seed in seeds:
        kw = dict(obs_dim=OBS_DIM, act_dim=N_ACT, lr=3e-4, entropy_coef=0.03, spatial_dims=29,
                  trigger_return=1.0, trigger_entropy=1.0, trigger_sustained=4)
        # Store the threshold for act() to use
        kw["trigger_sustained"] = 4  # unchanged
        
        # Create agent with distillation (all except Trajectory use it)
        agent = PAOLight(**kw)
        agent.dormancy_lr_factor = 0.1
        agent.skill_bias_strength = cond_params["bias"]
        agent._app_thresh = cond_params["app_thresh"]
        
        if cond_params["use_traj"]:
            # Override: disable distillation, use old SkillCache
            # We create a special flag for trajectory mode
            agent._use_trajectory = True
        else:
            agent._use_trajectory = False
        
        env = TwoGate2DSmallEnv(rule="A→B", seed=seed)
        
        # Phase 1
        for ep in range(P1):
            obs = env.reset(); done = False
            while not done:
                a = agent.act(obs, training=True)
                obs, r, done, _ = env.step(a)
                agent.step_end(r, done)
            agent.finish_episode()
        
        # Evaluate skill success rate
        eval_env = TwoGate2DSmallEnv(rule="A→B", seed=seed+999)
        skill_success = eval_skill_success(agent, eval_env, n_trials=30)
        
        # Phase 2: B→A
        env.set_rule("B→A")
        rng2 = np.random.RandomState(seed+500)
        p2_returns = []
        for ep in range(P2):
            obs = env.reset(); done = False
            while not done:
                a = agent.act(obs, training=True)
                if rng2.random() < 0.10:
                    a = rng2.randint(0, N_ACT)
                obs, r, done, _ = env.step(a)
                agent.step_end(r, done)
            agent.finish_episode()
            p2_returns.append(agent.get_log()["returns"][-1])
        p2_mean = np.mean(p2_returns[-20:]) if p2_returns else 0.0
        
        # Phase 3: A→B restored
        env.set_rule("A→B")
        p3_returns = []
        for ep in range(P3):
            obs = env.reset(); done = False
            while not done:
                a = agent.act(obs, training=True)
                obs, r, done, _ = env.step(a)
                agent.step_end(r, done)
            agent.finish_episode()
            p3_returns.append(agent.get_log()["returns"][-1])
        p3_early = np.mean(p3_returns[:20]) if len(p3_returns) >= 20 else 0.0
        
        results.append({
            "seed": seed, "cond": cond_name,
            "skill_success": skill_success,
            "p2_return": p2_mean, "p3_early": p3_early,
        })
        print(f"  [{cond_name}] seed={seed}  skill_succ={skill_success:.2f}  "
              f"p2={p2_mean:.3f}  p3={p3_early:.3f}")
    return results


def run_flat_baseline(seeds):
    """FlatPPO baseline for Phase 2 comparison."""
    results = []
    for seed in seeds:
        agent = FlatPPO(obs_dim=OBS_DIM, act_dim=N_ACT, lr=3e-4, entropy_coef=0.03)
        env = TwoGate2DSmallEnv(rule="A→B", seed=seed)
        for ep in range(P1):
            obs = env.reset(); done = False
            while not done:
                a = agent.act(obs, training=True)
                obs, r, done, _ = env.step(a)
                agent.step_end(r, done)
            agent.finish_episode()
        # Phase 2
        env.set_rule("B→A")
        rng2 = np.random.RandomState(seed+500)
        p2 = []
        for ep in range(P2):
            obs = env.reset(); done = False
            while not done:
                a = agent.act(obs, training=True)
                if rng2.random() < 0.10:
                    a = rng2.randint(0, N_ACT)
                obs, r, done, _ = env.step(a)
                agent.step_end(r, done)
            agent.finish_episode()
            p2.append(agent.get_log()["returns"][-1])
        p2_mean = np.mean(p2[-20:]) if p2 else 0.0
        results.append({"seed": seed, "p2_return": p2_mean})
        print(f"  [Flat] seed={seed}  p2={p2_mean:.3f}")
    return results


def main():
    print(f"=== Rigidity Knob Sweep (N={len(SEEDS)} seeds) ===\n")
    
    # Run all PAO conditions
    all_data = {}
    for cond_name in ["Soft", "Medium", "Hard", "Trajectory"]:
        print(f"\n── {cond_name} ──")
        all_data[cond_name] = run_condition(cond_name, CONDITIONS[cond_name], SEEDS)
    
    # Run Flat baseline once
    print(f"\n── FlatPPO Baseline ──")
    flat_data = run_flat_baseline(SEEDS)
    
    # Build results table
    p2_flat = {d["seed"]: d["p2_return"] for d in flat_data}
    
    print(f"\n{'='*80}")
    print(f"  CRYSTALLINITY TRADEOFF TABLE")
    print(f"{'='*80}")
    print(f"  {'Condition':<12s} {'SkillSucc':>10s} {'P2_PAO':>8s} {'P2_Flat':>8s} "
          f"{'HystIdx':>8s} {'P3_PAO':>8s}")
    print(f"  {'─'*56}")
    
    for cond_name in ["Soft", "Medium", "Hard", "Trajectory"]:
        succs, p2s, p3s = [], [], []
        for d in all_data[cond_name]:
            s = d["seed"]
            succs.append(d["skill_success"])
            p2s.append(d["p2_return"])
            p3s.append(d["p3_early"])
        m_succ = np.mean(succs)
        m_p2 = np.mean(p2s)
        m_p3 = np.mean(p3s)
        m_f = np.mean([p2_flat[s["seed"]] for s in all_data[cond_name]])
        hyst = (m_f - m_p2) / (abs(m_f) + 0.01)
        print(f"  {cond_name:<12s} {m_succ:>9.2f}%  {m_p2:>8.3f} {m_f:>8.3f} "
              f"{hyst:>8.3f} {m_p3:>8.3f}")
    
    print(f"\n{'─'*80}")
    print(f"  Predicted negative correlation: increased skill success → decreased hysteresis.")
    print(f"  This is the Crystallinity Tradeoff: rigidity × generalisation are inverse.")
    
    # Save raw data
    out = {"flat": flat_data, "conditions": all_data, "config": CONDITIONS}
    path = os.path.join(os.path.dirname(__file__), "results/rigidity_sweep.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\n  Raw data: {path}")
    print("  Done.")


if __name__ == "__main__":
    main()
