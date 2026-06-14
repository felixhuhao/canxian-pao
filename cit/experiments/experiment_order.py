"""
Order-Dependent δ Dynamics
===========================
Tests whether the final δ and skill crystallization set depend on the
order in which skills are presented.

Key hypotheses:
  1. δ trajectory is path-dependent (not just attractor-driven)
  2. Some orders produce more crystallized skills than others
  3. The "dead zone" effect (δ₀=28) depends on which skill is presented first

Orders tested:
  Original:   A(24) → B(24,π) → C(12) → D(48) → E(30)
  Reverse:    E(30) → D(48) → C(12) → B(24,π) → A(24)
  D-first:    D(48) → C(12) → A(24) → B(24,π) → E(30)
  C-first:    C(12) → A(24) → B(24,π) → D(48) → E(30)
  E-first:    E(30) → A(24) → B(24,π) → C(12) → D(48)
  Bad-first:  E(30) → D(48) → C(12) → B(24,π) → A(24)

Usage:
    python experiment_order.py
"""

import numpy as np
import time
import os
import sys
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef


# ============================================================
# Define skill sets
# ============================================================

SKILL_POOL = {
    'A': SkillDef('A', 24, 1.0, True),
    'B': SkillDef('B', 24, 1.0, True),
    'C': SkillDef('C', 12, 2.0, True),
    'D': SkillDef('D', 48, 0.5, True),
    'E': SkillDef('E', 30, 0.8, False),
}

# Phase offsets for A/B (π phase shift for competition)
PHASE_OFFSETS = {'A': 0.0, 'B': np.pi, 'C': 0.0, 'D': 0.0, 'E': 0.0}

ORDERS = {
    'Original':   ['A', 'B', 'C', 'D', 'E'],
    'Reverse':    ['E', 'D', 'C', 'B', 'A'],
    'D-first':    ['D', 'C', 'A', 'B', 'E'],
    'C-first':    ['C', 'A', 'B', 'D', 'E'],
    'E-first':    ['E', 'A', 'B', 'C', 'D'],
}


def make_order_cfg(order_names: List[str], meta_delta: bool = True,
                   gamma_delta: float = 0.05, quick: bool = False) -> ExperimentConfig:
    """
    Create an ExperimentConfig that presents skills in the given order.

    Phases:
      0: pretrain
      1: skill[0] active
      2: skill[1] active
      3: skill[2] active
      4: skill[3] active
      5: skill[4] active
      6: stabilize (last skill active)
      7: silence test
    """
    class OrderConfig(ExperimentConfig):
        pass

    cfg = OrderConfig()
    cfg.meta_delta = meta_delta
    cfg.gamma_delta = gamma_delta

    # Replace skills list with the ordered subset
    ordered_skills = [SKILL_POOL[name] for name in order_names]
    cfg.skills = ordered_skills

    if quick:
        cfg.t_pretrain = 500
        cfg.t_phase1 = 2000
        cfg.t_phase2 = 3500
        cfg.t_phase3 = 5000
        cfg.t_phase4 = 6500
        cfg.t_phase5 = 8000
        cfg.t_silence = 9500
        cfg.t_total = 10500

    # Override skill_mask to present skills in order
    original_mask = cfg.skill_mask

    def order_mask(phase: int) -> np.ndarray:
        mask = np.zeros(len(order_names), dtype=bool)
        if phase == 0:
            pass  # Pretrain
        elif 1 <= phase <= 5:
            idx = phase - 1
            if idx < len(order_names):
                mask[idx] = True
        elif phase == 6:
            # Stabilize: last skill active
            mask[len(order_names) - 1] = True
        # Silence (phase 7): no skills
        return mask

    cfg.skill_mask = order_mask  # type: ignore
    return cfg


def run_order_experiment(cfg, seed=42, verbose=False):
    """Run experiment with the given config."""
    from experiment_crystallization import run_experiment
    return run_experiment(cfg, seed=seed, verbose=verbose)


# ============================================================
# Order sweep
# ============================================================

def sweep_orders(orders: Dict[str, List[str]], n_seeds: int = 5,
                 quick: bool = False, verbose: bool = False) -> Dict:
    """Run all orders and compare results."""
    all_results = {}

    for order_name, skill_names in orders.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Order: {order_name} — {' → '.join(skill_names)}")
            print(f"{'='*60}")

        seed_results = []
        for s in range(n_seeds):
            cfg = make_order_cfg(skill_names, meta_delta=True,
                                 gamma_delta=0.05, quick=quick)
            r = run_order_experiment(cfg, seed=s + 1000, verbose=False)
            seed_results.append(r)

        # Aggregate
        N = len(skill_names)
        lock_rates = {}
        for name in skill_names:
            locked = sum(1 for r in seed_results
                         if r['skill_summary'][name]['locked'])
            lock_rates[name] = locked / n_seeds

        # δ stats
        delta_finals = [r['delta_stats']['final'] for r in seed_results]
        delta_trajs = [np.array(r['trajectories']['delta'])
                       for r in seed_results]

        # Number of crystallized skills per seed
        n_crystallized = []
        for r in seed_results:
            n_c = sum(1 for name in skill_names
                      if r['skill_summary'][name]['locked'])
            n_crystallized.append(n_c)

        # Compute δ at end of each phase
        phase_ends = [0]
        cfg = make_order_cfg(skill_names, quick=quick)
        for t_val in [cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                       cfg.t_phase3, cfg.t_phase4, cfg.t_phase5]:
            phase_ends.append(t_val)

        phase_deltas = []
        for pe in phase_ends[1:]:  # skip 0
            window = slice(max(pe - 200, 0), pe)
            pd = [float(np.mean(dt[window])) for dt in delta_trajs]
            phase_deltas.append((float(np.mean(pd)), float(np.std(pd))))

        all_results[order_name] = {
            'skill_names': skill_names,
            'lock_rates': lock_rates,
            'n_crystallized_mean': float(np.mean(n_crystallized)),
            'n_crystallized_std': float(np.std(n_crystallized)),
            'delta_final_mean': float(np.mean(delta_finals)),
            'delta_final_std': float(np.std(delta_finals)),
            'delta_trajectory_mean': [float(np.mean(dt)) for dt in delta_trajs[0][::500]],
            'phase_deltas': phase_deltas,
            'n_seeds': n_seeds,
        }

        if verbose:
            print(f"  δ_final: {all_results[order_name]['delta_final_mean']:.1f} "
                  f"± {all_results[order_name]['delta_final_std']:.1f}")
            print(f"  N_crystallized: {all_results[order_name]['n_crystallized_mean']:.1f} "
                  f"± {all_results[order_name]['n_crystallized_std']:.1f}")
            print(f"  Lock rates: {lock_rates}")

    return all_results


def print_comparison(results: Dict, orders: Dict[str, List[str]]):
    """Print comparison across orders."""
    print(f"\n{'='*80}")
    print(f"Order Comparison — δ and Crystallization")
    print(f"{'='*80}")

    # Header
    all_skills = ['A', 'B', 'C', 'D', 'E']
    header = f"{'Order':<15} {'δ_final':<12} {'N_locked':<10} "
    for s in all_skills:
        header += f"{s+'%':<8}"
    print(header)
    print('-' * 80)

    for order_name in orders:
        r = results[order_name]
        names = r['skill_names']
        line = f"{order_name:<15} {r['delta_final_mean']:<5.1f}±{r['delta_final_std']:<4.1f}  "
        line += f"{r['n_crystallized_mean']:<5.1f}±{r['n_crystallized_std']:<3.1f}  "
        for s in all_skills:
            rate = r['lock_rates'].get(s, -1)
            if rate >= 0:
                line += f"{rate:<8.0%}"
            else:
                line += f"{'—':<8}"
        print(line)
    print()

    # Phase-by-phase δ tracking
    print(f"\n{'='*80}")
    print(f"δ Trajectory by Phase")
    print(f"{'='*80}")

    for order_name in orders:
        r = results[order_name]
        names = r['skill_names']
        print(f"\n{order_name:15} {' → '.join(names)}")
        print(f"  {'Phase':<12} {'Skill':<8} {'δ_mean':<10} {'δ_std':<10}")
        print(f"  {'-'*40}")
        for i, (pd_mean, pd_std) in enumerate(r['phase_deltas']):
            skill = names[i] if i < len(names) else '—'
            print(f"  {'P'+str(i+1):<12} {skill:<8} {pd_mean:<10.1f} {pd_std:<10.1f}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Order-dependent δ dynamics')
    parser.add_argument('--quick', action='store_true', help='Reduced scale')
    parser.add_argument('--n-seeds', type=int, default=5, help='Seeds per order')
    args = parser.parse_args()

    print(f"Order sweep: {len(ORDERS)} orders × {args.n_seeds} seeds")
    if args.quick:
        print("Mode: quick")
    print()

    results = sweep_orders(ORDERS, n_seeds=args.n_seeds,
                           quick=args.quick, verbose=True)
    print_comparison(results, ORDERS)
