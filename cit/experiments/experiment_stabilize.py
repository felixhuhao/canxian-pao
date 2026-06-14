"""
Meta-δ Stabilization Experiments
=================================
Tests two solutions to the meta-δ self-destruction problem:

Experiment A: Timescale decoupling
  - Make γ_δ much smaller than η_C so δ drift doesn't interfere with C accumulation
  - γ_δ = 0.05 (original), 0.005, 0.0005, 0.00005

Experiment B: δ freeze mechanism
  - When any κ_i crosses θ_κ, freeze δ for T_freeze steps
  - This gives C time to stabilize before δ changes

Test case: C-first order (worst case for meta-δ self-destruction)
"""

import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_order import make_order_cfg, run_order_experiment


# ============================================================
# Experiment A: Timescale decoupling
# ============================================================

def test_gamma_scan(order_name='C-first', quick=True):
    """Scan γ_δ values for a given order."""
    orders = {'C-first': ['C', 'A', 'B', 'D', 'E']}
    skill_names = orders[order_name]

    gammas = [0.05, 0.005, 0.0005, 0.00005]
    results = {}

    print(f"\n{'='*60}")
    print(f"Timescale Decoupling — {order_name}")
    print(f"{'='*60}")

    for gamma in gammas:
        cfg = make_order_cfg(skill_names, meta_delta=True,
                             gamma_delta=gamma, quick=quick)
        r = run_order_experiment(cfg, seed=42, verbose=False)

        lock_info = {}
        for name in skill_names:
            ss = r['skill_summary'][name]
            lock_info[name] = {
                'locked': ss['locked'],
                'peak_k': ss['peak_kappa'],
                'final_k': ss['final_kappa'],
            }

        delta_final = r['delta_stats']['final']

        results[gamma] = {
            'lock_info': lock_info,
            'delta_final': delta_final,
            'delta_min': r['delta_stats']['min'],
            'delta_max': r['delta_stats']['max'],
        }

        n_locked = sum(1 for v in lock_info.values() if v['locked'])
        print(f"  γ_δ={gamma:.5f}: δ={delta_final:.0f} "
              f"[{r['delta_stats']['min']:.0f}-{r['delta_stats']['max']:.0f}] "
              f"N_locked={n_locked} "
              f"locked={[n for n,v in lock_info.items() if v['locked']]}")

    return results


# ============================================================
# Experiment B: δ freeze mechanism
# ============================================================

def test_freeze_scan(order_name='C-first', quick=True):
    """Test different freeze durations."""
    orders = {'C-first': ['C', 'A', 'B', 'D', 'E']}
    skill_names = orders[order_name]

    freeze_times = [0, 100, 500, 1000, 2000]  # steps to freeze δ after κ > θ_κ
    results = {}

    print(f"\n{'='*60}")
    print(f"δ Freeze Mechanism — {order_name}")
    print(f"{'='*60}")

    for freeze_t in freeze_times:
        # Create a custom config with freeze
        from experiment_crystallization import ExperimentConfig

        class FreezeConfig(ExperimentConfig):
            pass

        cfg = FreezeConfig()
        cfg.meta_delta = True
        cfg.gamma_delta = 0.05
        cfg.t_freeze = freeze_t

        # Set skills to C-first order
        from experiment_order import SKILL_POOL
        cfg.skills = [SKILL_POOL[n] for n in skill_names]

        if quick:
            cfg.t_pretrain = 500
            cfg.t_phase1 = 2000
            cfg.t_phase2 = 3500
            cfg.t_phase3 = 5000
            cfg.t_phase4 = 6500
            cfg.t_phase5 = 8000
            cfg.t_silence = 9500
            cfg.t_total = 10500

        # Override skill_mask for order
        def order_mask(phase):
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
        cfg.skill_mask = order_mask  # type: ignore

        # Run with modified experiment that handles freeze
        r = run_with_freeze(cfg, seed=42, verbose=False)

        lock_info = {}
        for name in skill_names:
            ss = r['skill_summary'][name]
            lock_info[name] = {
                'locked': ss['locked'],
                'peak_k': ss['peak_kappa'],
                'final_k': ss['kappa_at_active_end'],
            }

        results[freeze_t] = {
            'lock_info': lock_info,
            'delta_final': r['delta_stats']['final'],
            'delta_min': r['delta_stats']['min'],
            'delta_max': r['delta_stats']['max'],
        }

        n_locked = sum(1 for v in lock_info.values() if v['locked'])
        lock_str = ','.join([n for n, v in lock_info.items() if v['locked']])
        print(f"  freeze={freeze_t:5d}: δ={r['delta_stats']['final']:.0f} "
              f"[{r['delta_stats']['min']:.0f}-{r['delta_stats']['max']:.0f}] "
              f"N_locked={n_locked} [{lock_str}]")

    return results


def run_with_freeze(cfg, seed=42, verbose=False):
    """
    Modified version of run_experiment with δ freeze mechanism.
    When any κ_i > θ_κ, freeze δ for cfg.t_freeze steps.
    """
    # Import and modify the core experiment
    from experiment_crystallization import run_experiment

    # We can't easily hook into run_experiment's inner loop.
    # Instead, we'll replicate the freezing logic by patching the config
    # and running a modified version.

    # For now, let's just run the standard experiment and observe.
    # The freeze mechanism would require modifying the main loop.
    # Let's implement it here.

    N = len(cfg.skills)
    np.random.seed(seed)
    delta = cfg.delta
    freeze_counter = 0
    was_frozen = False

    x_hist = list(np.random.uniform(-0.1, 0.1, cfg.delta + 10))
    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)

    kappa_traj = np.zeros((cfg.t_total, N))
    C_traj = np.zeros((cfg.t_total, N))
    delta_traj = np.zeros(cfg.t_total)

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        mask = cfg.skill_mask(phase)

        # Stimulus
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_i = np.sin(2 * np.pi * t / cfg.skills[i].period)
                S_active += S_i
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        # State update
        x_delayed = x_hist[-delta] if len(x_hist) >= delta else x_hist[0]
        total_kappa = np.mean(kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))
        x_hist.append(x_t)

        # C accumulation
        corr = np.tanh(x_t) * np.tanh(x_delayed)
        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        # PE
        for i in range(N):
            expected = np.sin(2 * np.pi * t / cfg.skills[i].period)
            if mask[i]:
                pe = 0.01 * (1.0 - abs(np.tanh(x_t)))
            else:
                pe = 0.5 + 0.3 * abs(expected) if kappa[i] > cfg.theta_k else 0.0
            PE[i] = (1.0 - cfg.pe_decay) * PE[i] + cfg.pe_decay * pe

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
                diss = cfg.alpha_diss * (1.0 / (1.0 + np.exp(-10*(PE[i]-cfg.diss_threshold)))) * kappa[i]
                kappa[i] = base_k + cfg.eta * kappa[i] * (1.0 - kappa[i]) - comp - diss
            else:
                kappa[i] = base_k
            kappa[i] = np.clip(kappa[i], 0.0, 1.0)

        # ============================================================
        # δ freeze mechanism (phase-based: freeze during active training)
        # ============================================================
        should_freeze = False
        if hasattr(cfg, 't_freeze') and cfg.t_freeze > 0:
            # Freeze δ during ANY active training phase (not just after κ > θ_κ)
            any_active = bool(np.sum(mask)) and phase > 0 and phase < 7
            # Check if current phase's active skill is already locked
            phase_active_skill_locked = False
            for i in range(N):
                if mask[i] and kappa[i] > cfg.theta_k:
                    phase_active_skill_locked = True
                    break

            if any_active and not phase_active_skill_locked:
                # Freeze while skill is being trained but not yet locked
                if freeze_counter < cfg.t_freeze:
                    should_freeze = True
                    freeze_counter += 1
            else:
                freeze_counter = 0

        # Meta-δ update (skipped during freeze)
        if cfg.meta_delta and not should_freeze and t % cfg.delta_interval == 0 and t > cfg.t_pretrain:
            weights = kappa * kappa
            w_sum = np.sum(weights)
            if w_sum > 0.01:
                T_dom = np.average([s.period for s in cfg.skills], weights=weights)
                delta += cfg.gamma_delta * (T_dom - delta)
                delta = int(np.clip(round(delta), cfg.delta_min, cfg.delta_max))
                delta = max(delta, 1)

        delta_traj[t] = delta
        kappa_traj[t] = kappa
        C_traj[t] = C

    # Analysis (simplified)
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
            'success': s.expected_crystallize == (k_at_end > 0.5),
        }

    return {
        'skill_summary': skill_summary,
        'delta_stats': {
            'initial': cfg.delta,
            'final': float(delta_traj[-1]),
            'min': float(np.min(delta_traj)),
            'max': float(np.max(delta_traj)),
        },
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma-scan', action='store_true')
    parser.add_argument('--freeze-scan', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.gamma_scan:
        test_gamma_scan(quick=args.quick)

    if args.freeze_scan:
        test_freeze_scan(quick=args.quick)

    if not args.gamma_scan and not args.freeze_scan:
        # Run both
        print("=== Timescale decoupling (γ_δ scan) ===")
        test_gamma_scan(quick=args.quick)
        print()
        print("=== δ freeze mechanism ===")
        test_freeze_scan(quick=args.quick)
