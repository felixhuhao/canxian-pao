"""
Ablation Experiments
====================
Verifies that all three conditions (delayed recurrence, delayed self-correlation
accumulation, threshold-gated autocatalytic consolidation) are jointly necessary.

Usage:
    python run_ablation.py          # N=100, saves figure
    python run_ablation.py --quick  # N=10, quick check
"""

import argparse
import numpy as np
import model

CONDITIONS = {
    'full':      {'label': 'Full model',             'params': {}},
    'nodelay':   {'label': 'No delayed recurrence',   'params': {'delta': 1}},
    'nocorr':    {'label': 'No correlation',           'params': {'eta_C': 0.0, 'theta_k': 1.0}},
    'noconsol':  {'label': 'No autocatalytic consol.', 'params': {'eta': 0.0}},
}


def run_condition(name, config, n_seeds):
    """Run ablation for one condition over n_seeds."""
    results = []
    for s in range(n_seeds):
        r = model.run(seed=s + 1000, params=config['params'])
        results.append(r)
    preserved = sum(1 for r in results if r['regime'] == 'S')
    k_sils = [r['k_sil'] for r in results]
    periods = [r['period'] for r in results if r['period'] > 0]
    return {
        'condition': name,
        'label': config['label'],
        'n_preserved': preserved,
        'n_total': n_seeds,
        'rate': preserved / n_seeds,
        'k_sil_mean': float(np.mean(k_sils)),
        'k_sil_std': float(np.std(k_sils)),
        'period_mean': float(np.mean(periods)) if periods else float('nan'),
    }


def print_table(results):
    """Print results as a table."""
    print(f"\n{'Condition':<30} {'Preserved':>10} {'κ_sil':>10} {'Period':>8}")
    print('-' * 60)
    for r in results:
        rate = f"{r['n_preserved']}/{r['n_total']}"
        k = f"{r['k_sil_mean']:.4f}±{r['k_sil_std']:.4f}"
        p = f"{r['period_mean']:.1f}" if not np.isnan(r['period_mean']) else '—'
        print(f"{r['label']:<30} {rate:>10} {k:>10} {p:>8}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run N=10 (quick check)')
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()

    n_seeds = 10 if args.quick else 100
    print(f"\nAblation — N={n_seeds} per condition\n")

    import time
    t0 = time.time()

    all_results = []
    for name, config in CONDITIONS.items():
        t1 = time.time()
        res = run_condition(name, config, n_seeds)
        all_results.append(res)
        dt = time.time() - t1
        print(f"  {config['label']:<28} {res['n_preserved']:>3}/{n_seeds}  ({dt:.1f}s)")

    print_table(all_results)
    print(f"Total: {time.time() - t0:.1f}s")
