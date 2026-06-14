"""
Phase Diagram: δ₀ × γ (initial delay × adaptation rate)
=========================================================
Maps the crystallization landscape of the meta-δ system across two key parameters.

Regions:
  - Crystallization zone: 4-5 skills lock, δ converges smoothly
  - Self-destruction zone: δ moves too fast, correlation destroyed mid-transit
  - Deadlock zone: no skill ever locks, δ frozen in non-resonant state
  - Order-dependent zone: path determines which subset locks

Output: grid of (δ₀, γ_δ) → n_crystallized
"""

import numpy as np
import time
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, run_experiment


def scan_phase_diagram(delta_range=None, gamma_range=None, n_seeds=3, quick=False, verbose=False):
    """Scan the 2D phase diagram: δ₀ × γ_δ."""
    if delta_range is None:
        delta_range = list(range(8, 52, 2))  # 8, 10, ..., 50
    if gamma_range is None:
        gamma_range = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]

    n_delta = len(delta_range)
    n_gamma = len(gamma_range)

    # Result grid: n_crystallized mean and std
    grid_mean = np.zeros((n_delta, n_gamma))
    grid_std = np.zeros((n_delta, n_gamma))
    grid_detail = {}  # (δ₀, γ) → lock_rates

    total = n_delta * n_gamma * n_seeds
    done = 0
    t0 = time.time()

    for di, delta0 in enumerate(delta_range):
        for gi, gamma in enumerate(gamma_range):
            n_cryst = []
            seed_locks = []

            for s in range(n_seeds):
                cfg = ExperimentConfig()
                cfg.delta = delta0
                cfg.meta_delta = True
                cfg.gamma_delta = gamma

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

                # Count locked skills (A, B, C, D — skip E which is expected to fail)
                locked = sum(1 for name in ['A', 'B', 'C', 'D']
                             if r['skill_summary'][name]['locked'])
                n_cryst.append(locked)
                seed_locks.append({
                    name: r['skill_summary'][name]['locked']
                    for name in ['A', 'B', 'C', 'D', 'E']
                })

            grid_mean[di, gi] = float(np.mean(n_cryst))
            grid_std[di, gi] = float(np.std(n_cryst))
            grid_detail[(delta0, gamma)] = {
                'mean': float(np.mean(n_cryst)),
                'lock_rates': {
                    name: sum(1 for sl in seed_locks if sl[name]) / n_seeds
                    for name in ['A', 'B', 'C', 'D', 'E']
                },
                'delta_final_mean': float(np.mean([
                    run_experiment(ExperimentConfig(delta=delta0, meta_delta=True,
                                                    gamma_delta=gamma),
                                   seed=s+1000, verbose=False)['delta_stats']['final']
                    for s in range(n_seeds)
                ])),
            }

            done += 1
            if verbose and (done % 5 == 0 or done == total):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] δ₀={delta0:2d} γ={gamma:.5f} "
                      f"N={grid_mean[di,gi]:.1f}±{grid_std[di,gi]:.1f} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    result = {
        'delta_range': delta_range,
        'gamma_range': gamma_range,
        'grid_mean': grid_mean.tolist(),
        'grid_std': grid_std.tolist(),
        'n_seeds': n_seeds,
        'quick': quick,
    }

    # Text summary
    print(f"\n{'='*70}")
    print(f"Phase Diagram: δ₀ × γ_δ → N_crystallized")
    print(f"{'='*70}")

    # Header
    header = f"{'δ₀':<6}"
    for gamma in gamma_range:
        header += f" {gamma:<8}"
    print(header)
    print('-' * (6 + 9 * n_gamma))

    for di, delta0 in enumerate(delta_range):
        line = f"{delta0:<6}"
        for gi in range(n_gamma):
            val = grid_mean[di, gi]
            if val >= 3.5:
                line += f" {'✓✓':<8}"
            elif val >= 2.5:
                line += f" {'✓':<8}"
            elif val >= 0.5:
                line += f" {'○':<8}"
            else:
                line += f" {'✗':<8}"
        print(line)
    print()

    # Print key zones
    print("Zones:")
    for di, delta0 in enumerate(delta_range):
        for gi, gamma in enumerate(gamma_range):
            val = grid_mean[di, gi]
            key = (delta0, gamma_range[gi])
            det = grid_detail.get(key, {})
            if det:
                locks = det.get('lock_rates', {})
                dead = all(v == 0.0 for v in locks.values())
                sd = (locks.get('A', 0) >= 0.8 and locks.get('C', 0) < 0.3)
                if dead:
                    zone = "DEADLOCK"
                elif sd:
                    zone = "SELF-DESTRUCT"
                else:
                    zone = f"CRYSTAL(N={val:.0f})"

                if val >= 3.0 or val <= 0.5:
                    print(f"  δ₀={delta0:2d} γ={gamma:.5f} => {zone} "
                          f"({['A','B','C','D'][:int(val)] if val > 0 else 'none'})")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', default='phase_diagram_meta.json')
    args = parser.parse_args()

    print("2D Phase Diagram: δ₀ × γ_δ")
    print(f"Quick: {args.quick}")
    print()

    # Use coarser grid for speed
    delta_range = list(range(8, 50, 4))  # 8, 12, 16, ..., 48
    gamma_range = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    result = scan_phase_diagram(
        delta_range=delta_range,
        gamma_range=gamma_range,
        n_seeds=5,
        quick=args.quick,
        verbose=True)

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")
