"""
Meta-Plastic δ: Systematic Characterization
=============================================
Explores the dynamics of self-organizing internal delay δ in delayed adaptive systems.

Key questions:
1. From different initial δ, where does δ converge?
2. Is there multi-stability (hysteresis)?
3. Does δ track the active skill's period through sequential training?
4. What is the "attractor landscape" of δ?

Experiments:
  - δ convergence scan: initial δ ∈ [8, 64], measure final δ, skill success
  - δ tracking: sequential protocol, measure δ trajectory per phase
  - δ inertia: how fast does δ adapt when the environment changes?

Usage:
    python experiment_meta_delta.py --scan           # δ convergence scan
    python experiment_meta_delta.py --tracking       # δ tracking in sequential protocol
    python experiment_meta_delta.py --inertia        # δ inertia measurement
"""

import numpy as np
import time
import json
import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Reuse the core experiment module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef, run_experiment, DEFAULT_SKILLS


# ============================================================
# Experiment 1: δ Convergence Scan
# ============================================================

def delta_convergence_scan(initial_deltas: Optional[List[int]] = None,
                           n_seeds: int = 5,
                           quick: bool = False,
                           verbose: bool = False) -> Dict:
    """
    For each initial δ, run the sequential protocol and measure:
    - Final δ (at end of simulation)
    - δ trajectory
    - Which skills crystallized
    """
    if initial_deltas is None:
        initial_deltas = list(range(8, 66, 2))  # 8, 10, 12, ..., 64

    results = {}

    for delta0 in initial_deltas:
        seed_results = []
        for s in range(n_seeds):
            cfg = ExperimentConfig()
            cfg.delta = delta0
            cfg.meta_delta = True
            cfg.gamma_delta = 0.05

            if quick:
                cfg.t_pretrain = 500
                cfg.t_phase1 = 2000
                cfg.t_phase2 = 3500
                cfg.t_phase3 = 5000
                cfg.t_phase4 = 6500
                cfg.t_phase5 = 8000
                cfg.t_silence = 9500
                cfg.t_total = 10500

            r = run_experiment(cfg, seed=s + 1000, verbose=False)
            seed_results.append(r)

        # Aggregate
        ds = [r['delta_stats']['final'] for r in seed_results]
        skills_locked = {}
        for name in ['A', 'B', 'C', 'D', 'E']:
            locked = sum(1 for r in seed_results
                         if r['skill_summary'][name]['locked'])
            skills_locked[name] = locked / n_seeds

        results[delta0] = {
            'delta_final_mean': float(np.mean(ds)),
            'delta_final_std': float(np.std(ds)),
            'delta_min_seen': float(np.min([r['delta_stats']['min'] for r in seed_results])),
            'delta_max_seen': float(np.max([r['delta_stats']['max'] for r in seed_results])),
            'skill_lock_rates': skills_locked,
            'n_seeds': n_seeds,
        }

        if verbose:
            print(f"  δ₀={delta0:2d} → δ_final={np.mean(ds):5.1f}±{np.std(ds):.1f}  "
                  f"locked: {skills_locked}")

    return results


def print_scan_summary(results: Dict):
    """Print δ convergence scan results as a table."""
    print(f"\n{'='*70}")
    print(f"δ Convergence Scan")
    print(f"{'='*70}")
    print(f"{'δ₀':<6} {'δ_final':<12} {'A':<8} {'B':<8} {'C':<8} {'D':<8} {'E':<8}")
    print('-' * 70)
    for delta0 in sorted(results.keys()):
        r = results[delta0]
        d_mean = r['delta_final_mean']
        d_std = r['delta_final_std']
        a = f"{r['skill_lock_rates']['A']:.0%}"
        b = f"{r['skill_lock_rates']['B']:.0%}"
        c = f"{r['skill_lock_rates']['C']:.0%}"
        d = f"{r['skill_lock_rates']['D']:.0%}"
        e = f"{r['skill_lock_rates']['E']:.0%}"
        print(f"{delta0:<6} {d_mean:5.1f}±{d_std:.1f}   {a:<8} {b:<8} {c:<8} {d:<8} {e:<8}")
    print()

    # Skill convergence plot data
    print("δ₀ → δ_final mapping:")
    for delta0 in sorted(results.keys()):
        r = results[delta0]
        print(f"  {delta0:2d} → {r['delta_final_mean']:5.1f}")


# ============================================================
# Experiment 2: δ Tracking in Sequential Protocol
# ============================================================

def delta_tracking(n_seeds: int = 5, quick: bool = False,
                   verbose: bool = False) -> Dict:
    """
    Run the sequential protocol with meta-δ and track δ across phases.
    Measures: δ at end of each phase, skill lock state at each phase.
    """
    if quick:
        phase_ends = [0, 500, 2000, 3500, 5000, 6500, 8000, 9500, 10500]
    else:
        phase_ends = [0, 2000, 7000, 12000, 17000, 22000, 27000, 32000, 33500]

    phase_names = ['Pretrain', 'Phase1_A', 'Phase2_AB', 'Phase3_C',
                   'Phase4_D', 'Phase5_E', 'Stabilize', 'Silence']

    all_results = []
    for s in range(n_seeds):
        cfg = ExperimentConfig()
        cfg.meta_delta = True
        cfg.gamma_delta = 0.05

        if quick:
            cfg.t_pretrain = 500
            cfg.t_phase1 = 2000
            cfg.t_phase2 = 3500
            cfg.t_phase3 = 5000
            cfg.t_phase4 = 6500
            cfg.t_phase5 = 8000
            cfg.t_silence = 9500
            cfg.t_total = 10500

        r = run_experiment(cfg, seed=s + 1000, verbose=False)
        all_results.append(r)

    # Extract δ at end of each phase
    tracking = {}
    for p_idx, p_name in enumerate(phase_names):
        if p_idx == 0:
            continue  # Skip pretrain
        p_end = phase_ends[p_idx + 1]

        deltas = []
        kappa_A = []
        kappa_B = []
        kappa_C = []
        kappa_D = []
        kappa_E = []

        for r in all_results:
            delta_traj = np.array(r['trajectories']['delta'])
            kappa_traj = np.array(r['trajectories']['kappa'])

            # δ at end of this phase
            window = slice(max(p_end - 200, 0), p_end)
            deltas.append(float(np.mean(delta_traj[window])))

            # κ at end of this phase
            kappa_A.append(float(np.mean(kappa_traj[window, 0])))
            kappa_B.append(float(np.mean(kappa_traj[window, 1])))
            kappa_C.append(float(np.mean(kappa_traj[window, 2])))
            kappa_D.append(float(np.mean(kappa_traj[window, 3])))
            kappa_E.append(float(np.mean(kappa_traj[window, 4])))

        tracking[p_name] = {
            'δ_mean': float(np.mean(deltas)),
            'δ_std': float(np.std(deltas)),
            'κ_A': float(np.mean(kappa_A)),
            'κ_B': float(np.mean(kappa_B)),
            'κ_C': float(np.mean(kappa_C)),
            'κ_D': float(np.mean(kappa_D)),
            'κ_E': float(np.mean(kappa_E)),
            'n_seeds': n_seeds,
        }

    if verbose:
        print(f"\n{'='*70}")
        print(f"δ Tracking in Sequential Protocol")
        print(f"{'='*70}")
        print(f"{'Phase':<15} {'δ':<10} {'κ_A':<10} {'κ_B':<10} {'κ_C':<10} {'κ_D':<10} {'κ_E':<10}")
        print('-' * 70)
        for pname, data in tracking.items():
            print(f"{pname:<15} {data['δ_mean']:<10.1f} "
                  f"{data['κ_A']:<10.4f} {data['κ_B']:<10.4f} "
                  f"{data['κ_C']:<10.4f} {data['κ_D']:<10.4f} "
                  f"{data['κ_E']:<10.4f}")
        print()

    return {
        'tracking': tracking,
        'phase_ends': phase_ends,
        'phase_names': phase_names,
    }


# ============================================================
# Experiment 3: δ Inertia
# ============================================================

def delta_inertia_test(quick: bool = False, verbose: bool = False) -> Dict:
    """
    Test δ inertia by rapidly switching between two skill periods.
    Protocol: A (T=24) → C (T=12) → A (T=24) → C (T=12)
    Measures: how fast δ adapts to each switch.

    Quick mode: 84 steps per sub-phase (rapid switches)
    Full mode: 500 steps per sub-phase
    """
    # Custom configuration with multiple rapid switches
    class InertiaConfig(ExperimentConfig):
        pass

    cfg = InertiaConfig()
    cfg.meta_delta = True

    if quick:
        sub_phase = 500
    else:
        sub_phase = 1000

    cfg.t_pretrain = sub_phase
    cfg.t_phase1 = cfg.t_pretrain + sub_phase    # A
    cfg.t_phase2 = cfg.t_phase1 + sub_phase      # C
    cfg.t_phase3 = cfg.t_phase2 + sub_phase      # A again
    cfg.t_phase4 = cfg.t_phase3 + sub_phase      # C again
    cfg.t_phase5 = cfg.t_phase4 + sub_phase      # A again
    cfg.t_silence = cfg.t_phase5 + sub_phase     # C again
    cfg.t_total = cfg.t_silence + sub_phase      # silence

    # Override skill_mask to only use A and C
    def inertia_mask(phase):
        mask = np.zeros(5, dtype=bool)
        if phase == 1 or phase == 3 or phase == 5:
            mask[0] = True  # A (T=24)
        elif phase == 2 or phase == 4 or phase == 6:
            mask[2] = True  # C (T=12)
        return mask

    # Run with custom mask
    # Need to temporarily modify the config's skill_mask method
    original_mask = cfg.skill_mask
    cfg.skill_mask = inertia_mask  # type: ignore

    r = run_experiment(cfg, seed=42, verbose=False)

    # Restore original
    cfg.skill_mask = original_mask  # type: ignore

    # Analyze δ trajectory relative to phase switches
    delta_traj = np.array(r['trajectories']['delta'])
    kappa_traj = np.array(r['trajectories']['kappa'])
    phase_traj = np.array(r['trajectories']['phase'])

    phases_to_analyze = [
        (1, 'A_1', cfg.t_pretrain, cfg.t_phase1),
        (2, 'C_1', cfg.t_phase1, cfg.t_phase2),
        (3, 'A_2', cfg.t_phase2, cfg.t_phase3),
        (4, 'C_2', cfg.t_phase3, cfg.t_phase4),
        (5, 'A_3', cfg.t_phase4, cfg.t_phase5),
    ]

    inertia_results = []
    for p_idx, p_name, p_start, p_end in phases_to_analyze:
        # δ at start and end of phase
        delta_start = float(np.mean(delta_traj[p_start:min(p_start+50, p_end)]))
        delta_end = float(np.mean(delta_traj[max(p_end-50, p_start):p_end]))
        delta_change = delta_end - delta_start

        # κ at end of phase
        k_end = float(np.mean(kappa_traj[max(p_end-100, p_start):p_end, :], axis=0).mean())

        inertia_results.append({
            'phase': p_name,
            'δ_start': delta_start,
            'δ_end': delta_end,
            'δ_change': delta_change,
        })

    if verbose:
        print(f"\n{'='*60}")
        print("δ Inertia Test — Rapid Switches A(24) ↔ C(12)")
        print(f"{'='*60}")
        print(f"{'Phase':<10} {'Target T':<10} {'δ_start':<10} {'δ_end':<10} {'Δδ':<10}")
        print('-' * 50)
        for ir in inertia_results:
            target = 24 if 'A' in ir['phase'] else 12
            print(f"{ir['phase']:<10} {target:<10} {ir['δ_start']:<10.1f} "
                  f"{ir['δ_end']:<10.1f} {ir['δ_change']:<10.1f}")
        print()

    return {'inertia': inertia_results}


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-plastic δ characterization')
    parser.add_argument('--scan', action='store_true', help='δ convergence scan')
    parser.add_argument('--tracking', action='store_true', help='δ tracking in sequential protocol')
    parser.add_argument('--inertia', action='store_true', help='δ inertia measurement')
    parser.add_argument('--quick', action='store_true', help='Reduced scale')
    parser.add_argument('--n-seeds', type=int, default=5, help='Seeds per condition')
    args = parser.parse_args()

    if not any([args.scan, args.tracking, args.inertia]):
        parser.print_help()
        sys.exit(1)

    if args.scan:
        print(f"δ Convergence Scan (initial δ ∈ [8, 64], {args.n_seeds} seeds each)")
        print(f"Quick mode: {args.quick}")
        results = delta_convergence_scan(
            initial_deltas=list(range(8, 66, 2)),
            n_seeds=args.n_seeds,
            quick=args.quick,
            verbose=True)
        print_scan_summary(results)

    if args.tracking:
        print(f"δ Tracking in Sequential Protocol ({args.n_seeds} seeds)")
        results = delta_tracking(
            n_seeds=args.n_seeds,
            quick=args.quick,
            verbose=True)

    if args.inertia:
        print("δ Inertia Test")
        results = delta_inertia_test(quick=args.quick, verbose=True)
