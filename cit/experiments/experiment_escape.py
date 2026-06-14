"""
Escape from Canxianization Traps
=================================
Experiments on causal-chain reconstruction in meta-plastic systems.

Experiment 1: τ Jump vs Continuous Drift
  - Does a discrete τ jump bypass the anti-phase zone?
  - Compare: continuous drift (24→23→...→12) vs jump (24→12)
  - Test also: stochastic jumps (probabilistic large steps)

Experiment 2: Bridge Skill
  - Insert a skill whose period matches τ₀ in the deadlock zone
  - Can one resonant skill unlock the entire system?

Experiment 3: Noise-Assisted Escape
  - Add σ·ξ to τ updates in deadlock
  - Map σ → N_locked (expected inverted-U)

Usage:
    python experiment_escape.py --jump           # τ jump experiment
    python experiment_escape.py --bridge         # bridge skill experiment
    python experiment_escape.py --noise          # noise-assisted escape
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef, run_experiment


# ============================================================
# Experiment 1: τ Jump vs Continuous Drift
# ============================================================

def run_jump_experiment(seed=42, quick=True, verbose=True):
    """
    Compare three τ transition modes from 24 to 12 (C's period):
    1. Continuous drift: standard meta-δ (γ=0.05)
    2. Discrete jump: τ = 24 → 12 in one step
    3. Stochastic: τ jumps with probability p per update
    """
    skill_names_c_first = ['C', 'A', 'B', 'D', 'E']
    from experiment_order import SKILL_POOL, make_order_cfg

    results = {}

    # --- Mode 1: Continuous drift (standard) ---
    cfg1 = make_order_cfg(skill_names_c_first, meta_delta=True, gamma_delta=0.05, quick=quick)
    r1 = run_experiment(cfg1, seed=seed, verbose=False)
    results['continuous'] = {
        'locked': {n: r1['skill_summary'][n]['locked'] for n in skill_names_c_first},
        'N_locked': sum(1 for n in skill_names_c_first if r1['skill_summary'][n]['locked']),
        'delta_final': r1['delta_stats']['final'],
        'delta_traj': np.array(r1['trajectories']['delta']),
    }

    # --- Mode 2: Discrete jump ---
    # Custom config: override meta-delta to force τ = target period
    class JumpConfig(ExperimentConfig):
        pass

    cfg2 = make_order_cfg(skill_names_c_first, meta_delta=False, quick=quick)
    cfg2.meta_delta = True
    cfg2.gamma_delta = 5.0  # Very fast adaptation → instant jump

    from experiment_order import SKILL_POOL
    cfg2.skills = [SKILL_POOL[n] for n in skill_names_c_first]

    def jump_mask(phase):
        mask = np.zeros(len(skill_names_c_first), dtype=bool)
        if phase == 0:
            pass
        elif 1 <= phase <= 5:
            idx = phase - 1
            if idx < len(skill_names_c_first):
                mask[idx] = True
        elif phase == 6:
            mask[len(skill_names_c_first) - 1] = True
        return mask
    cfg2.skill_mask = jump_mask

    # Run with modified experiment that handles discrete jumps
    r2 = run_custom_jump(cfg2, seed=seed, jump_from=24, jump_to=12, jump_step=500,
                         quick=quick)
    results['jump'] = {
        'locked': {n: r2['skill_summary'][n]['locked'] for n in skill_names_c_first},
        'N_locked': sum(1 for n in skill_names_c_first if r2['skill_summary'][n]['locked']),
        'delta_final': r2['delta_stats']['final'],
        'delta_traj': r2.get('delta_traj', None),
    }

    # --- Mode 3: Stochastic jump ---
    class StochConfig(ExperimentConfig):
        pass

    cfg3 = make_order_cfg(skill_names_c_first, meta_delta=False, quick=quick)
    cfg3.meta_delta = True
    cfg3.gamma_delta = 0.05
    cfg3.jump_probability = 0.1
    cfg3.jump_target = 12

    cfg3.skills = [SKILL_POOL[n] for n in skill_names_c_first]
    cfg3.skill_mask = jump_mask

    r3 = run_custom_stochastic(cfg3, seed=seed, quick=quick)
    results['stochastic'] = {
        'locked': {n: r3['skill_summary'][n]['locked'] for n in skill_names_c_first},
        'N_locked': sum(1 for n in skill_names_c_first if r3['skill_summary'][n]['locked']),
        'delta_final': r3['delta_stats']['final'],
        'delta_traj': r3.get('delta_traj', None),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Tau Jump Experiment — C-first, 24→12")
        print(f"{'='*60}")
        print(f"{'Mode':<15} {'N_locked':<10} {'δ_final':<10} {'C':<6} {'A':<6} {'B':<6} {'D':<6} {'E':<6}")
        print('-' * 60)
        for mode, r in results.items():
            locks = r['locked']
            print(f"{mode:<15} {r['N_locked']:<10} {r['delta_final']:<10.0f} "
                  f"{'✓' if locks.get('C') else '✗':<6} {'✓' if locks.get('A') else '✗':<6} "
                  f"{'✓' if locks.get('B') else '✗':<6} {'✓' if locks.get('D') else '✗':<6} "
                  f"{'✓' if locks.get('E') else '✗':<6}")

    return results


def run_custom_jump(cfg, seed=42, jump_from=24, jump_to=12, jump_step=500, quick=True):
    """Run with a discrete τ jump at the start of Phase 1."""
    N = len(cfg.skills)
    np.random.seed(seed)
    delta = cfg.delta  # Start at configured δ
    tau_jumped = False

    x_hist = list(np.random.uniform(-0.1, 0.1, max(60, cfg.delta + 10)))
    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)
    kappa_traj = np.zeros((cfg.t_total, N))
    delta_traj = np.zeros(cfg.t_total)

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        mask = cfg.skill_mask(phase)

        # Jump at the start of Phase 1
        if phase == 1 and not tau_jumped:
            delta = jump_to
            tau_jumped = True

        # Stimulus
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / cfg.skills[i].period)
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        # State
        x_delayed = x_hist[-delta] if len(x_hist) >= delta else x_hist[0]
        total_kappa = np.mean(kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))
        x_hist.append(x_t)

        corr = np.tanh(x_t) * np.tanh(x_delayed)
        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        # κ update
        for i in range(N):
            target = np.tanh(cfg.lam * abs(C[i]))
            base_k = kappa[i] + cfg.gamma * (target - kappa[i])
            if kappa[i] > cfg.theta_k:
                comp = 0.0
                for j in range(N):
                    if j != i and kappa[j] > cfg.theta_k:
                        pr = min(cfg.skills[i].period, cfg.skills[j].period) / \
                             max(cfg.skills[i].period, cfg.skills[j].period)
                        if pr > 0.3:
                            comp += cfg.beta_comp * kappa[i] * kappa[j]
                diss = cfg.alpha_diss * (1.0/(1.0+np.exp(-10*(PE[i]-cfg.diss_threshold)))) * kappa[i]
                kappa[i] = base_k + cfg.eta * kappa[i] * (1.0 - kappa[i]) - comp - diss
            else:
                kappa[i] = base_k
            kappa[i] = np.clip(kappa[i], 0.0, 1.0)

        delta_traj[t] = delta
        kappa_traj[t] = kappa

    # Simple analysis
    skill_summary = {}
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]
    for i in range(N):
        s = cfg.skills[i]
        k_t = kappa_traj[:, i]
        active_phase = None
        for p_idx in range(8):
            if cfg.skill_mask(p_idx)[i]:
                active_phase = p_idx
                break
        if active_phase is not None:
            p_end = phase_ends[active_phase + 1]
            p_start = phase_ends[active_phase]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_end = float(np.mean(k_t[window]))
        else:
            k_at_end = float(np.mean(k_t[-500:]))
        skill_summary[s.name] = {
            'expected': s.expected_crystallize,
            'peak_kappa': float(np.max(k_t)),
            'kappa_at_active_end': k_at_end,
            'locked': k_at_end > 0.85,
        }

    return {
        'skill_summary': skill_summary,
        'delta_stats': {'final': float(delta_traj[-1]),
                        'min': float(np.min(delta_traj)),
                        'max': float(np.max(delta_traj))},
        'delta_traj': delta_traj,
    }


def run_custom_stochastic(cfg, seed=42, quick=True):
    """Run with stochastic τ jumps."""
    N = len(cfg.skills)
    np.random.seed(seed)
    delta = cfg.delta
    if not hasattr(cfg, 'jump_probability'):
        cfg.jump_probability = 0.1
    if not hasattr(cfg, 'jump_target'):
        cfg.jump_target = 12

    x_hist = list(np.random.uniform(-0.1, 0.1, max(60, cfg.delta + 10)))
    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)
    kappa_traj = np.zeros((cfg.t_total, N))
    delta_traj = np.zeros(cfg.t_total)

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        mask = cfg.skill_mask(phase)

        # Stochastic jump
        if phase == 1 and np.random.random() < cfg.jump_probability:
            delta = cfg.jump_target
        # Also slight continuous drift
        if cfg.meta_delta and t % 50 == 0 and t > cfg.t_pretrain:
            weights = kappa * kappa
            w_sum = np.sum(weights)
            if w_sum > 0.01:
                T_dom = np.average([s.period for s in cfg.skills], weights=weights)
                delta += cfg.gamma_delta * (T_dom - delta)
                delta = int(np.clip(round(delta), cfg.delta_min, cfg.delta_max))
                delta = max(delta, 1)

        # Stimulus, state, C, κ (same as run_custom_jump)
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / cfg.skills[i].period)
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        x_delayed = x_hist[-delta] if len(x_hist) >= delta else x_hist[0]
        total_kappa = np.mean(kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))
        x_hist.append(x_t)

        corr = np.tanh(x_t) * np.tanh(x_delayed)
        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        for i in range(N):
            target = np.tanh(cfg.lam * abs(C[i]))
            base_k = kappa[i] + cfg.gamma * (target - kappa[i])
            if kappa[i] > cfg.theta_k:
                comp = 0.0
                for j in range(N):
                    if j != i and kappa[j] > cfg.theta_k:
                        pr = min(cfg.skills[i].period, cfg.skills[j].period) / \
                             max(cfg.skills[i].period, cfg.skills[j].period)
                        if pr > 0.3:
                            comp += cfg.beta_comp * kappa[i] * kappa[j]
                diss = cfg.alpha_diss * (1.0/(1.0+np.exp(-10*(PE[i]-cfg.diss_threshold)))) * kappa[i]
                kappa[i] = base_k + cfg.eta * kappa[i] * (1.0 - kappa[i]) - comp - diss
            else:
                kappa[i] = base_k
            kappa[i] = np.clip(kappa[i], 0.0, 1.0)

        delta_traj[t] = delta
        kappa_traj[t] = kappa

    # Analysis
    skill_summary = {}
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]
    for i in range(N):
        s = cfg.skills[i]
        k_t = kappa_traj[:, i]
        active_phase = None
        for p_idx in range(8):
            if cfg.skill_mask(p_idx)[i]:
                active_phase = p_idx
                break
        if active_phase is not None:
            p_end = phase_ends[active_phase + 1]
            p_start = phase_ends[active_phase]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_end = float(np.mean(k_t[window]))
        else:
            k_at_end = float(np.mean(k_t[-500:]))
        skill_summary[s.name] = {
            'expected': s.expected_crystallize,
            'peak_kappa': float(np.max(k_t)),
            'locked': k_at_end > 0.85,
        }

    return {
        'skill_summary': skill_summary,
        'delta_stats': {'final': float(delta_traj[-1]),
                        'min': float(np.min(delta_traj)),
                        'max': float(np.max(delta_traj))},
        'delta_traj': delta_traj,
    }


# ============================================================
# Experiment 2: Bridge Skill
# ============================================================

def run_bridge_experiment(seed=42, quick=True, verbose=True):
    """
    Test whether introducing a bridge skill can escape deadlock.

    Deadlock τ₀ = 16 (no skill resonates).
    Bridge skill with P_b = 16 (perfect resonance at τ₀=16).
    Protocol: Bridge → A(24) → B(24,π) → C(12) → D(48) → E(30)
    """
    from experiment_order import SKILL_POOL

    # Define bridge skill
    bridge = SkillDef('Bridge', 16, 16/24, True)

    orders = {
        'no_bridge': ['C', 'A', 'B', 'D', 'E'],  # Original C-first
        'with_bridge': ['Bridge', 'A', 'B', 'C', 'D', 'E'],
    }

    results = {}
    for order_name, skill_names in orders.items():
        from experiment_order import make_order_cfg
        cfg = make_order_cfg(skill_names, meta_delta=True, gamma_delta=0.05, quick=quick)
        cfg.delta = 16  # Start in deadlock

        # Override to include bridge skill properly
        pool = dict(SKILL_POOL)
        pool['Bridge'] = bridge
        cfg.skills = [pool[n] for n in skill_names]

        def bridge_mask(phase):
            mask = np.zeros(len(skill_names), dtype=bool)
            if phase == 0:
                pass
            elif 1 <= phase <= len(skill_names):
                idx = phase - 1
                if idx < len(skill_names):
                    mask[idx] = True
            elif phase == 6:
                mask[len(skill_names) - 1] = True
            return mask
        cfg.skill_mask = bridge_mask

        r = run_experiment(cfg, seed=seed, verbose=False)

        results[order_name] = {
            'locked': {n: r['skill_summary'].get(n, {}).get('locked', False) for n in skill_names},
            'N_locked': sum(1 for n in skill_names if r['skill_summary'].get(n, {}).get('locked', False)),
            'delta_final': r['delta_stats']['final'],
        }

        if verbose:
            print(f"  {order_name}: N={results[order_name]['N_locked']}, "
                  f"locked={[n for n,v in results[order_name]['locked'].items() if v]}")

    return results


# ============================================================
# Experiment 3: Noise-Assisted Escape
# ============================================================

def run_noise_experiment(seed=42, quick=True, verbose=True):
    """Add σ·ξ noise to τ and measure escape from deadlock."""
    from experiment_order import SKILL_POOL, make_order_cfg

    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    skill_names = ['C', 'A', 'B', 'D', 'E']

    results = {}
    for sigma in sigmas:
        cfg = make_order_cfg(skill_names, meta_delta=True, gamma_delta=0.05, quick=quick)
        cfg.delta = 28  # Deadlock τ₀
        cfg.noise_sigma_tau = sigma

        cfg.skills = [SKILL_POOL[n] for n in skill_names]

        def noise_mask(phase):
            mask = np.zeros(len(skill_names), dtype=bool)
            if phase == 0:
                pass
            elif 1 <= phase <= 5:
                idx = phase - 1
                if idx < len(skill_names):
                    mask[idx] = True
            elif phase == 6:
                mask[len(skill_names) - 1] = True
            return mask
        cfg.skill_mask = noise_mask

        # We need to run with noise. Since the standard run_experiment doesn't
        # support noise on τ, let me run it with the custom noisy runner.
        r = run_custom_noisy(cfg, seed=seed, noise_sigma=sigma, quick=quick)

        results[sigma] = {
            'locked': {n: r['skill_summary'].get(n, {}).get('locked', False) for n in skill_names},
            'N_locked': sum(1 for n in skill_names if r['skill_summary'].get(n, {}).get('locked', False)),
            'delta_final': r['delta_stats']['final'],
        }

        if verbose:
            print(f"  σ={sigma:5.1f}: N={results[sigma]['N_locked']}, "
                  f"locked={[n for n,v in results[sigma]['locked'].items() if v]}")

    return results


def run_custom_noisy(cfg, seed=42, noise_sigma=0.5, quick=True):
    """Run with noise injected into τ dynamics."""
    N = len(cfg.skills)
    np.random.seed(seed)
    delta = cfg.delta

    x_hist = list(np.random.uniform(-0.1, 0.1, max(60, cfg.delta + 10)))
    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)
    kappa_traj = np.zeros((cfg.t_total, N))
    delta_traj = np.zeros(cfg.t_total)

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        mask = cfg.skill_mask(phase)

        # Stimulus
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / cfg.skills[i].period)
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        x_delayed = x_hist[-delta] if len(x_hist) >= delta else x_hist[0]
        total_kappa = np.mean(kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))
        x_hist.append(x_t)

        corr = np.tanh(x_t) * np.tanh(x_delayed)
        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        for i in range(N):
            target = np.tanh(cfg.lam * abs(C[i]))
            base_k = kappa[i] + cfg.gamma * (target - kappa[i])
            if kappa[i] > cfg.theta_k:
                comp = 0.0
                for j in range(N):
                    if j != i and kappa[j] > cfg.theta_k:
                        pr = min(cfg.skills[i].period, cfg.skills[j].period) / \
                             max(cfg.skills[i].period, cfg.skills[j].period)
                        if pr > 0.3:
                            comp += cfg.beta_comp * kappa[i] * kappa[j]
                diss = cfg.alpha_diss * (1.0/(1.0+np.exp(-10*(PE[i]-cfg.diss_threshold)))) * kappa[i]
                kappa[i] = base_k + cfg.eta * kappa[i] * (1.0 - kappa[i]) - comp - diss
            else:
                kappa[i] = base_k
            kappa[i] = np.clip(kappa[i], 0.0, 1.0)

        # Meta-δ with noise
        if cfg.meta_delta and t % 50 == 0 and t > cfg.t_pretrain:
            weights = kappa * kappa
            w_sum = np.sum(weights)
            if w_sum > 0.01:
                T_dom = np.average([s.period for s in cfg.skills], weights=weights)
                delta += cfg.gamma_delta * (T_dom - delta)
            delta += np.random.normal(0, noise_sigma)
            delta = int(np.clip(round(delta), cfg.delta_min, cfg.delta_max))
            delta = max(delta, 1)

        delta_traj[t] = delta
        kappa_traj[t] = kappa

    skill_summary = {}
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]
    for i in range(N):
        s = cfg.skills[i]
        k_t = kappa_traj[:, i]
        active_phase = None
        for p_idx in range(8):
            if cfg.skill_mask(p_idx)[i]:
                active_phase = p_idx
                break
        if active_phase is not None:
            p_end = phase_ends[active_phase + 1]
            p_start = phase_ends[active_phase]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_end = float(np.mean(k_t[window]))
        else:
            k_at_end = float(np.mean(k_t[-500:]))
        skill_summary[s.name] = {
            'expected': s.expected_crystallize,
            'peak_kappa': float(np.max(k_t)),
            'locked': k_at_end > 0.85,
        }

    return {
        'skill_summary': skill_summary,
        'delta_stats': {'final': float(delta_traj[-1]),
                        'min': float(np.min(delta_traj)),
                        'max': float(np.max(delta_traj))},
        'delta_traj': delta_traj,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jump', action='store_true')
    parser.add_argument('--bridge', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if not any([args.jump, args.bridge, args.noise]):
        print("Running all experiments...")
        args.jump = args.bridge = args.noise = True

    if args.jump:
        print(f"\n{'='*60}")
        print("Experiment 1: τ Jump vs Continuous Drift")
        print(f"{'='*60}")
        r = run_jump_experiment(seed=42, quick=args.quick, verbose=True)

    if args.bridge:
        print(f"\n{'='*60}")
        print("Experiment 2: Bridge Skill Escape from Deadlock")
        print(f"{'='*60}")
        r = run_bridge_experiment(seed=42, quick=args.quick, verbose=True)

    if args.noise:
        print(f"\n{'='*60}")
        print("Experiment 3: Noise-Assisted Escape")
        print(f"{'='*60}")
        r = run_noise_experiment(seed=42, quick=args.quick, verbose=True)
