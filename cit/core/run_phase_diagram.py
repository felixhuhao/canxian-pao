#!/usr/bin/env python3
"""
Run Phase Diagram
=================
Parameter scan for the main-text phase diagram (η_C × η/γ).
Usage:
    python run_phase_diagram.py          # full scan
    python run_phase_diagram.py --quick  # reduced resolution
"""

import argparse
import numpy as np
import model


def run_scan(n_eta_c=31, n_ratio=33, n_seeds=10):
    """Scan η_C and η/γ, returning success rate grid."""
    eta_c_range = np.logspace(-3, 0, n_eta_c)
    ratio_range = np.logspace(-2, 1.5, n_ratio)
    grid = np.zeros((n_eta_c, n_ratio))

    for i, eta_c in enumerate(eta_c_range):
        for j, ratio in enumerate(ratio_range):
            eta = ratio * model.DEFAULT_PARAMS['gamma']
            successes = 0
            for s in range(n_seeds):
                r = model.run(seed=s + 1000, params={
                    'eta_C': eta_c,
                    'eta': eta,
                })
                if r['regime'] == 'S':
                    successes += 1
            grid[i, j] = successes / n_seeds

    return {
        'eta_c': eta_c_range.tolist(),
        'ratio': ratio_range.tolist(),
        'grid': grid.tolist(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    n_eta_c = 10 if args.quick else 31
    n_ratio = 11 if args.quick else 33
    n_seeds = 5 if args.quick else 10

    print(f"Phase diagram scan: {n_eta_c}×{n_ratio}×{n_seeds} = {n_eta_c * n_ratio * n_seeds} sims")
    data = run_scan(n_eta_c, n_ratio, n_seeds)
    print(f"Done. Peak success: {np.max(data['grid']):.1%}")
