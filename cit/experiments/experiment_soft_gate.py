"""
Soft Gating & Critical Slowing Down
=====================================
Replaces the hard threshold (step-function locking) with a sigmoid soft gate,
eliminating artificial freeze mechanisms and allowing the system to discover
critical slowing near the phase boundary.

Key changes from standard model:
  L_i = σ(λ(κ_i - θ))  instead of  1(κ_i > θ)
  
When λ → ∞:  recovers the original hard-gate model
When λ moderate:  sub-threshold κ still exerts weak "exploration gravity"
When λ small:  near-linear gating, system never truly crystallizes

Predictions:
  1. Moderate λ (5-20) eliminates deadlock: system always has non-zero exploration
  2. Near the λ-phase transition (~λ=15 under default params), system shows
     critical slowing — τ drift slows down dramatically but doesn't stop
  3. The artificial T_freeze can be removed entirely under soft gating:
     γ_eff = γ₀·exp(-a·max L_i) emerges naturally from the soft gate dynamics

Usage:
    python experiment_soft_gate.py --scan-lambda
    python experiment_soft_gate.py --critical-slow
    python experiment_soft_gate.py --eliminate-freeze
"""

import numpy as np
import time
import os
import sys
import json
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef, sigmoid, run_experiment
from experiment_order import SKILL_POOL, make_order_cfg


# ============================================================
# Config with soft gating
# ============================================================

def sigmoid_steep(x: float, lam: float) -> float:
    """Sigmoid with tunable steepness: σ(λ·x)"""
    return 1.0 / (1.0 + np.exp(-np.clip(lam * x, -50, 50)))


class SoftGateConfig(ExperimentConfig):
    """Extension of standard config with soft gating parameters."""
    lam_s: float = 15.0       # Steepness of the sigmoid gate (λ_s)
    use_soft_gate: bool = True  # True: soft; False: original hard threshold
    eliminate_freeze: bool = False  # True: remove T_freeze entirely


def run_soft_gate_experiment(cfg: Optional[SoftGateConfig] = None,
                              seed: int = 42, verbose: bool = True) -> Dict:
    """
    Run experiment with soft gating instead of hard threshold.
    The main change is in the κ update: the autocatalytic term,
    competition term, and dissolution term are all gated by
    L_i = σ(λ_s(κ_i - θ_k)) instead of 1(κ_i > θ_k).
    """
    if cfg is None:
        cfg = SoftGateConfig()

    np.random.seed(seed)
    N = cfg.N

    # State initialisation (same as standard)
    x_hist = list(np.random.uniform(-0.1, 0.1, cfg.delta))
    delta_current = cfg.delta
    
    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)
    
    kappa_traj = np.zeros((cfg.t_total, N))
    C_traj = np.zeros((cfg.t_total, N))
    L_traj = np.zeros((cfg.t_total, N))   # Track gate activation
    tau_traj = np.zeros(cfg.t_total)
    PE_traj = np.zeros((cfg.t_total, N))
    reward_traj = np.zeros((cfg.t_total, N))
    phase_traj = np.zeros(cfg.t_total, dtype=int)
    delta_traj = np.zeros(cfg.t_total)

    # Skill phase offsets (same as standard)
    # A=0, B=π, C=0, D=0, E=0
    skill_phase_offsets = [0.0, np.pi, 0.0, 0.0, 0.0] if N >= 5 else [0.0] * N

    t0_wall = time.time()

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        phase_traj[t] = phase
        mask = cfg.skill_mask(phase)

        # --- 1. Stimulus (same as standard) ---
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_i = np.sin(2 * np.pi * t / cfg.skills[i].period
                             + skill_phase_offsets[i])
                S_active += S_i
                reward_traj[t, i] = 1.0
            else:
                reward_traj[t, i] = 0.0
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        # --- 2. State update (same as standard) ---
        x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
        total_kappa = np.mean(kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))
        x_hist.append(x_t)

        # --- 3. Self-correlation C_i (same as standard) ---
        corr = np.tanh(x_t) * np.tanh(x_delayed)
        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        # --- 4. PE (same as standard) ---
        for i in range(N):
            expected = np.sin(2 * np.pi * t / cfg.skills[i].period
                              + skill_phase_offsets[i])
            if mask[i]:
                pe_instant = 0.01 * (1.0 - abs(np.tanh(x_t)))
            else:
                # Soft-gate version: even weakly gated skills have mild PE
                if cfg.use_soft_gate:
                    gate_i = sigmoid_steep(kappa[i] - cfg.theta_k, cfg.lam_s)
                    pe_instant = 0.5 * gate_i + 0.3 * gate_i * abs(expected)
                else:
                    if kappa[i] > cfg.theta_k:
                        pe_instant = 0.5 + 0.3 * abs(expected)
                    else:
                        pe_instant = 0.0
            PE[i] = (1.0 - cfg.pe_decay) * PE[i] + cfg.pe_decay * pe_instant

        # --- 5. Kappa update with SOFT GATING ---
        for i in range(N):
            target = np.tanh(cfg.lam * abs(C[i]))
            base_k = kappa[i] + cfg.gamma * (target - kappa[i])

            if cfg.use_soft_gate:
                # Soft gate: continuous activation level
                gate_i = sigmoid_steep(kappa[i] - cfg.theta_k, cfg.lam_s)
                L_i = gate_i
                
                # Competition (gated by activation level)
                comp = 0.0
                for j in range(N):
                    if j != i:
                        gate_j = sigmoid_steep(kappa[j] - cfg.theta_k, cfg.lam_s)
                        period_ratio = min(cfg.skills[i].period,
                                           cfg.skills[j].period) / \
                                       max(cfg.skills[i].period,
                                           cfg.skills[j].period)
                        if period_ratio > 0.3:
                            comp += cfg.beta_comp * gate_i * gate_j * kappa[i] * kappa[j]

                # Dissolution (gated by activation level × PE)
                diss = (cfg.alpha_diss
                        * sigmoid(10 * (PE[i] - cfg.diss_threshold))
                        * gate_i
                        * kappa[i])

                # Autocatalytic consolidation (gated)
                kappa[i] = (base_k
                            + cfg.eta * gate_i * kappa[i] * (1.0 - kappa[i])
                            - comp
                            - diss)
            else:
                # Original hard threshold (reproduce baseline)
                L_i = 1.0 if kappa[i] > cfg.theta_k else 0.0
                
                if kappa[i] > cfg.theta_k:
                    comp = 0.0
                    for j in range(N):
                        if j != i and kappa[j] > cfg.theta_k:
                            period_ratio = min(cfg.skills[i].period,
                                               cfg.skills[j].period) / \
                                           max(cfg.skills[i].period,
                                               cfg.skills[j].period)
                            if period_ratio > 0.3:
                                comp += cfg.beta_comp * kappa[i] * kappa[j]
                    diss = (cfg.alpha_diss
                            * sigmoid(10 * (PE[i] - cfg.diss_threshold))
                            * kappa[i])
                    kappa[i] = (base_k
                                + cfg.eta * kappa[i] * (1.0 - kappa[i])
                                - comp
                                - diss)
                else:
                    kappa[i] = base_k

            kappa[i] = np.clip(kappa[i], 0.0, 1.0)
            L_traj[t, i] = L_i

        # --- 6. Meta-δ update (same as standard) ---
        if cfg.meta_delta and t % cfg.delta_interval == 0 and t > cfg.t_pretrain:
            # Use SOFT gate (L_i) as weights instead of hard lock status
            if cfg.use_soft_gate:
                weights = L_traj[t]  # Use current gate activations as weights
            else:
                weights = kappa * kappa
            w_sum = np.sum(weights)
            if w_sum > 0.01:
                T_dominant = np.average([s.period for s in cfg.skills], weights=weights)
                delta_current += cfg.gamma_delta * (T_dominant - delta_current)
                delta_current = int(np.clip(round(delta_current),
                                            cfg.delta_min, cfg.delta_max))
                delta_current = max(delta_current, 1)

        # --- Record ---
        kappa_traj[t] = kappa
        C_traj[t] = C
        PE_traj[t] = PE
        delta_traj[t] = delta_current
        tau_traj[t] = delta_current

    dt = time.time() - t0_wall

    # --- Analysis ---
    results = _analyze_soft(cfg, kappa_traj, C_traj, L_traj, tau_traj,
                            phase_traj, reward_traj, delta_traj, seed, verbose, dt)

    return results


def _analyze_soft(cfg: SoftGateConfig,
                  kappa_traj: np.ndarray,
                  C_traj: np.ndarray,
                  L_traj: np.ndarray,
                  tau_traj: np.ndarray,
                  phase_traj: np.ndarray,
                  reward_traj: np.ndarray,
                  delta_traj: np.ndarray,
                  seed: int,
                  verbose: bool,
                  dt: float) -> Dict:
    """Analyze soft-gate experiment results."""
    N = cfg.N
    T = cfg.t_total

    # Phase boundaries
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]
    phase_names = ['Pretrain', 'Phase1_A', 'Phase2_AB', 'Phase3_C',
                   'Phase4_D', 'Phase5_E', 'Stabilize', 'Silence']

    # Per-skill summary
    skill_summary = {}
    for i in range(N):
        s = cfg.skills[i]
        k_t = kappa_traj[:, i]
        L_t = L_traj[:, i]
        
        # Find the phase where this skill is active
        active_phase = None
        for p_idx in range(8):
            mask = cfg.skill_mask(p_idx)
            if i < len(mask) and mask[i]:
                active_phase = p_idx
                break
        
        if active_phase is not None:
            p_end = phase_ends[active_phase + 1]
            p_start = phase_ends[active_phase]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_end = float(np.mean(k_t[window]))
            L_at_end = float(np.mean(L_t[window]))
        else:
            k_at_end = float(np.mean(k_t[-500:]))
            L_at_end = float(np.mean(L_t[-500:]))
        
        # Under soft gating, "locked" means gate activation > 0.5
        soft_locked = L_at_end > 0.5
        
        skill_summary[s.name] = {
            'expected': s.expected_crystallize,
            'peak_kappa': float(np.max(k_t)),
            'kappa_at_end': k_at_end,
            'gate_at_end': L_at_end,
            'soft_locked': soft_locked,
            'hard_locked': k_at_end > cfg.theta_k + 0.1,
        }

    # Phase statistics
    phase_stats = {}
    for p_idx, (p_start, p_name) in enumerate(zip(phase_ends[:-1], phase_names)):
        p_end = phase_ends[p_idx + 1]
        k_slice = kappa_traj[p_start:p_end]
        L_slice = L_traj[p_start:p_end]
        tau_slice = tau_traj[p_start:p_end]

        # Total effective gate activation sum
        total_gate_mean = float(np.mean(np.sum(L_slice, axis=1)))
        total_gate_final = float(np.mean(np.sum(L_slice[-200:], axis=1)))

        phase_stats[p_name] = {
            'steps': int(p_end - p_start),
            'total_gate_mean': total_gate_mean,
            'total_gate_final': total_gate_final,
            'tau_mean': float(np.mean(tau_slice)),
            'tau_final': float(tau_slice[-1]),
            'tau_std': float(np.std(tau_slice[-500:])),
            'n_soft_locked': sum(
                1 for i in range(N)
                if np.mean(L_slice[-200:, i]) > 0.5
            ),
        }

    # Critical slowing diagnostic: τ fluctuation near phase transitions
    # Phase boundaries with a window of ±200 steps
    critical_zones = {}
    boundaries = [cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5]
    for b_idx, b in enumerate(boundaries):
        if b >= 200 and b + 200 <= T:
            pre_slice = tau_traj[b-200:b]
            post_slice = tau_traj[b:b+200]
            pre_var = float(np.var(pre_slice))
            post_var = float(np.var(post_slice))
            critical_zones[f'boundary_{b_idx+1}'] = {
                'timestep': b,
                'pre_var': pre_var,
                'post_var': post_var,
                'var_ratio': pre_var / max(post_var, 1e-10),
            }

    # Summary statistics
    n_soft_locked_final = sum(1 for i in range(N)
                              if np.mean(L_traj[-500:, i]) > 0.5)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Soft Gate Experiment — λ_s={cfg.lam_s}, seed={seed}")
        print(f"{'='*60}")
        print(f"Skills: {', '.join(f'{s.name}(T={s.period})' for s in cfg.skills)}")
        print(f"Soft gating: {'ON' if cfg.use_soft_gate else 'OFF'}")
        print(f"λ_s (gate steepness): {cfg.lam_s}")
        print(f"Total steps: {T} in {dt:.1f}s")
        print(f"\nFinal soft-locked: {n_soft_locked_final}/{N}")
        print(f"τ_final: {delta_traj[-1]:.0f}")
        print(f"\nPer-skill (κ_final, L_final, soft_locked):")
        for i in range(N):
            s = cfg.skills[i]
            k_f = float(np.mean(kappa_traj[-200:, i]))
            L_f = float(np.mean(L_traj[-200:, i]))
            flag = '✓' if L_f > 0.5 else '✗'
            print(f"  {s.name:<6} κ={k_f:.3f}  L={L_f:.3f}  {flag}")
        print(f"{'='*60}\n")

    return {
        'skill_summary': skill_summary,
        'phase_stats': phase_stats,
        'critical_zones': critical_zones,
        'n_soft_locked_final': n_soft_locked_final,
        'delta_stats': {
            'final': float(delta_traj[-1]),
            'mean_last_1000': float(np.mean(delta_traj[-1000:])),
            'std_last_1000': float(np.std(delta_traj[-1000:])),
        },
        'trajectories': {
            'kappa': kappa_traj,
            'L': L_traj,
            'delta': delta_traj,
            'tau': tau_traj,
        },
    }


# ============================================================
# Experiment 1: λ sweep — map the phase diagram
# ============================================================

def scan_lambda(seed: int = 42, verbose: bool = True):
    """Scan λ_s from 1 to 100, measuring soft-locked count and τ stability."""
    lam_values = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100]
    results = {}

    for lam_s in lam_values:
        cfg = SoftGateConfig()
        cfg.lam_s = lam_s
        cfg.use_soft_gate = True
        cfg.eliminate_freeze = False
        cfg.meta_delta = True
        cfg.gamma_delta = 0.05

        # Use C-first order for comparability with earlier experiments
        order_names = ['C', 'A', 'B', 'D', 'E']
        from experiment_order import make_order_cfg
        base_cfg = make_order_cfg(order_names, meta_delta=True, gamma_delta=0.05, quick=False)
        cfg.t_pretrain = base_cfg.t_pretrain
        cfg.t_phase1 = base_cfg.t_phase1
        cfg.t_phase2 = base_cfg.t_phase2
        cfg.t_phase3 = base_cfg.t_phase3
        cfg.t_phase4 = base_cfg.t_phase4
        cfg.t_phase5 = base_cfg.t_phase5
        cfg.t_silence = base_cfg.t_silence
        cfg.t_total = base_cfg.t_total
        cfg.skills = [SKILL_POOL[n] for n in order_names]
        cfg.skill_mask = base_cfg.skill_mask

        r = run_soft_gate_experiment(cfg, seed=seed, verbose=False)
        results[lam_s] = {
            'n_soft_locked': r['n_soft_locked_final'],
            'delta_final': r['delta_stats']['final'],
            'delta_std': r['delta_stats']['std_last_1000'],
            'total_gate': r['phase_stats']['Stabilize']['total_gate_final'],
        }

        if verbose:
            print(f"  λ_s={lam_s:3d}: N_locked={results[lam_s]['n_soft_locked']}, "
                  f"δ={results[lam_s]['delta_final']:.0f}±{results[lam_s]['delta_std']:.1f}, "
                  f"ΣL={results[lam_s]['total_gate']:.3f}")

    return results


# ============================================================
# Experiment 2: Critical slowing near the phase boundary
# ============================================================

def run_critical_slow(seed: int = 42, verbose: bool = True):
    """
    Demonstrate critical slowing near λ threshold.
    Run with λ_s = 15 (near transition), track τ autocorrelation.
    """
    cfg = SoftGateConfig()
    cfg.lam_s = 15
    cfg.use_soft_gate = True
    cfg.meta_delta = True
    cfg.gamma_delta = 0.05

    order_names = ['C', 'A', 'B', 'D', 'E']
    from experiment_order import make_order_cfg
    base_cfg = make_order_cfg(order_names, meta_delta=True, gamma_delta=0.05, quick=False)
    for attr in ['t_pretrain', 't_phase1', 't_phase2', 't_phase3',
                 't_phase4', 't_phase5', 't_silence', 't_total']:
        setattr(cfg, attr, getattr(base_cfg, attr))
    cfg.skills = [SKILL_POOL[n] for n in order_names]
    cfg.skill_mask = base_cfg.skill_mask

    r = run_soft_gate_experiment(cfg, seed=seed, verbose=verbose)
    delta_traj = r['trajectories']['delta']

    # Compute τ variance in sliding windows across FULL trajectory
    window_size = 500
    stride = 100
    variances = []
    midpoints = []
    for start in range(0, len(delta_traj) - window_size, stride):
        win = delta_traj[start:start+window_size]
        var = float(np.var(win))
        if var > 0:
            variances.append(var)
            midpoints.append(start + window_size // 2)

    # τ autocorrelation in a sliding window approach
    # Find high-variance regions = critical slowing zones
    r['critical_slowdown'] = {
        'tau_variance_timeline': {
            'windows': midpoints[-50:],  # Last 50 windows
            'variances': variances[-50:],
        },
        'tau_std_full': float(np.std(delta_traj[-5000:])),
        'tau_mean_full': float(np.mean(delta_traj[-5000:])),
    }

    if verbose and len(variances) > 10:
        print(f"\n{'='*60}")
        print("Critical Slowing Diagnostic")
        print(f"{'='*60}")
        # Pick 4 representative windows across the timeline
        n_windows = len(variances)
        indices = [int(n_windows * i / 4) for i in range(4)]
        print("τ variance at representative windows:")
        for idx in indices:
            t_step = midpoints[idx]
            phase = cfg.phase_at(t_step)
            print(f"  t~{t_step:5d} (phase {cfg.phase_name(phase):<12}): "
                  f"var(τ)={variances[idx]:.4f}")
        print(f"τ overall std (last 5000): {r['critical_slowdown']['tau_std_full']:.2f}")
        print(f"τ overall mean (last 5000): {r['critical_slowdown']['tau_mean_full']:.0f}")
        print()

    return r


# ============================================================
# Experiment 3: Eliminate freeze — compare hard vs soft deadlock
# ============================================================

def run_eliminate_freeze(seed: int = 42, verbose: bool = True):
    """
    Compare hard-gate vs soft-gate in a deadlock-prone configuration.
    Start τ₀ = 28 (no skill resonates), observe whether soft gating
    allows escape without any freeze mechanism.
    """
    results = {}
    
    order_names = ['C', 'A', 'B', 'D', 'E']
    
    # Build base config template
    _base = make_order_cfg(order_names, meta_delta=True, gamma_delta=0.05, quick=False)

    # --- Hard gate baseline (ε=noise escape only) ---
    cfg_hard = ExperimentConfig()
    for attr in ['t_pretrain', 't_phase1', 't_phase2', 't_phase3',
                 't_phase4', 't_phase5', 't_silence', 't_total']:
        setattr(cfg_hard, attr, getattr(_base, attr))
    cfg_hard.delta = 28
    cfg_hard.skills = [SKILL_POOL[n] for n in order_names]
    cfg_hard.skill_mask = _base.skill_mask
    
    r_hard = run_experiment(cfg_hard, seed=seed, verbose=False)
    results['hard_gate'] = {
        'n_locked': sum(1 for n in order_names if r_hard['skill_summary'][n]['locked']),
        'delta_final': r_hard['delta_stats']['final'],
        'locked_skills': [n for n in order_names if r_hard['skill_summary'][n]['locked']],
    }

    # --- Soft gate, λ_s=15 ---
    cfg_soft = SoftGateConfig()
    cfg_soft.lam_s = 15
    cfg_soft.use_soft_gate = True
    cfg_soft.meta_delta = True
    cfg_soft.gamma_delta = 0.05
    cfg_soft.delta = 28
    for attr in ['t_pretrain', 't_phase1', 't_phase2', 't_phase3',
                 't_phase4', 't_phase5', 't_silence', 't_total']:
        setattr(cfg_soft, attr, getattr(_base, attr))
    cfg_soft.skills = [SKILL_POOL[n] for n in order_names]
    cfg_soft.skill_mask = _base.skill_mask
    
    r_soft = run_soft_gate_experiment(cfg_soft, seed=seed, verbose=False)
    results['soft_gate_l15'] = {
        'n_soft_locked': r_soft['n_soft_locked_final'],
        'delta_final': r_soft['delta_stats']['final'],
        'total_gate': r_soft['phase_stats']['Stabilize']['total_gate_final'],
    }

    # --- Soft gate, λ_s=5 (very soft) ---
    cfg_soft5 = SoftGateConfig()
    cfg_soft5.lam_s = 5
    cfg_soft5.use_soft_gate = True
    cfg_soft5.meta_delta = True
    cfg_soft5.gamma_delta = 0.05
    cfg_soft5.delta = 28
    for attr in ['t_pretrain', 't_phase1', 't_phase2', 't_phase3',
                 't_phase4', 't_phase5', 't_silence', 't_total']:
        setattr(cfg_soft5, attr, getattr(_base, attr))
    cfg_soft5.skills = [SKILL_POOL[n] for n in order_names]
    cfg_soft5.skill_mask = _base.skill_mask

    r_soft5 = run_soft_gate_experiment(cfg_soft5, seed=seed, verbose=False)
    results['soft_gate_l5'] = {
        'n_soft_locked': r_soft5['n_soft_locked_final'],
        'delta_final': r_soft5['delta_stats']['final'],
        'total_gate': r_soft5['phase_stats']['Stabilize']['total_gate_final'],
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Freeze Elimination Comparison — τ₀=28 (deadlock)")
        print(f"{'='*60}")
        for gate_type, r in results.items():
            print(f"  {gate_type:<15} N_locked={r.get('n_locked', r.get('n_soft_locked', 0))}, "
                  f"δ={r['delta_final']:.0f}")
        print()

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan-lambda', action='store_true', help='λ_s phase diagram')
    parser.add_argument('--critical-slow', action='store_true', help='Critical slowing diagnostic')
    parser.add_argument('--eliminate-freeze', action='store_true', help='Freeze elimination test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not any([args.scan_lambda, args.critical_slow, args.eliminate_freeze]):
        print("Running all soft-gate experiments...")
        args.scan_lambda = args.critical_slow = args.eliminate_freeze = True

    if args.scan_lambda:
        print(f"\n{'='*60}")
        print("Experiment 1: λ_s Phase Diagram Scan")
        print(f"{'='*60}")
        scan_lambda(seed=args.seed, verbose=True)

    if args.critical_slow:
        print(f"\n{'='*60}")
        print("Experiment 2: Critical Slowing Near Phase Boundary")
        print(f"{'='*60}")
        run_critical_slow(seed=args.seed, verbose=True)

    if args.eliminate_freeze:
        print(f"\n{'='*60}")
        print("Experiment 3: Eliminate Freeze — Hard vs Soft Gate")
        print(f"{'='*60}")
        run_eliminate_freeze(seed=args.seed, verbose=True)
