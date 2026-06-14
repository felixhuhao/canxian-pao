#!/usr/bin/env python3
"""β_c grid refinement + threshold sensitivity using the full VectorTauModel."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_vector_tau import VectorTauModel

PERIODS = np.array([12.0, 24.0, 24.0, 48.0, 30.0])
M = 3
N = len(PERIODS)
SEEDS = 5
N_STEPS = 8000

params = {
    'eta_C': 0.01, 'gamma': 0.006, 'theta_k': 0.35, 'eta': 0.05,
    'beta_comp': 0.01, 'alpha_diss': 0.03, 'diss_threshold': 0.4,
    'lam': 1.5, 'lam_s': 15.0, 'gamma_tau': 0.05,
    'repulsion_scale': 8.0, 'alpha_W': 0.01, 'lambda_W': 0.005,
    'tau_min': 4, 'tau_max': 64, 'pe_decay': 0.05,
    'alpha': 1.8, 'beta': 2.2, 'sigma': 0.015,
}

def run_scan(repulsion_vals):
    results = {}
    for br in repulsion_vals:
        all_deltas = []
        for seed in range(SEEDS):
            np.random.seed(seed)
            model = VectorTauModel(M=M, N=N, periods=PERIODS, seed=seed)
            x_hist = list(np.random.uniform(-0.1, 0.1, 80))
            offsets = np.zeros(N)
            p = dict(params, repulsion_strength=br)
            for t in range(N_STEPS):
                mask = np.ones(N, dtype=bool) if t > 500 else np.zeros(N, dtype=bool)
                S_active = 0.0
                for i in range(N):
                    if mask[i]: S_active += np.sin(2 * np.pi * t / PERIODS[i])
                n_active = max(int(np.sum(mask)), 1)
                S_active /= n_active
                delta_t = int(np.clip(round(np.mean(model.tau)), 1, 64))
                x_delayed = x_hist[-delta_t] if len(x_hist) >= delta_t else x_hist[0]
                total_kappa = np.mean(model.kappa)
                x_t = ((1.0 - total_kappa) * np.tanh(p['alpha'] * S_active)
                       + total_kappa * np.tanh(p['beta'] * x_delayed)
                       + np.random.normal(0, p['sigma']))
                x_hist.append(x_t)
                model.step(x_t, x_delayed, t, mask, PERIODS, offsets, p)
            tau_final = model.tau
            sorted_t = np.sort(tau_final)
            min_delta = np.min(np.diff(sorted_t))
            all_deltas.append(min_delta)
        results[br] = {'deltas': all_deltas, 'mean': np.mean(all_deltas), 'std': np.std(all_deltas)}
    return results

# ── Coarse scan (original resolution) ──
print("=== Coarse scan (step=0.005) ===")
coarse_vals = np.arange(0, 0.101, 0.005)
coarse = run_scan(coarse_vals)
for br in coarse_vals:
    r = coarse[br]
    print(f"  β={br:.4f}  Δτ_min={r['mean']:.4f}±{r['std']:.4f}  Δτ={np.round(r['deltas'],2)}")

# ── Fine scan around critical region ──
print("\n=== Fine scan (step=0.001) ===")
fine_vals = np.arange(0.015, 0.06, 0.001)
fine = run_scan(fine_vals)
for br in fine_vals:
    r = fine[br]
    print(f"  β={br:.4f}  Δτ_min={r['mean']:.4f}±{r['std']:.4f}")

# ── Threshold sensitivity ──
print("\n=== Threshold Sensitivity ===")
coarse_means = np.array([coarse[b]['mean'] for b in coarse_vals])
fine_means = np.array([fine[b]['mean'] for b in fine_vals])

for label, br_vals, means in [("coarse", coarse_vals, coarse_means), ("fine", fine_vals, fine_means)]:
    print(f"\n  [{label}]")
    for th in [0.3, 0.5, 0.7, 1.0]:
        above = np.where(means > th)[0]
        if len(above) > 0:
            print(f"    threshold={th:.1f}: first β_rep > threshold at {br_vals[above[0]]:.4f}")
        else:
            print(f"    threshold={th:.1f}: never exceeded on this grid")

# ── Exponent stability: different β_c assumptions ──
print("\n=== Exponent Stability ===")
# Use fine grid: take points above threshold
th = 0.5
above_idx = np.where(fine_means > th)[0]
if len(above_idx) > 0:
    beta_c_guess = fine_vals[above_idx[0]]
    print(f"β_c (first > 0.5 on fine grid): {beta_c_guess:.4f}")

    # Build full dataset: combine fine + coarse for points above β_c_guess + 0.01
    all_br = np.sort(np.unique(np.concatenate([coarse_vals, fine_vals])))
    all_means = {}
    for b in all_br:
        if b in fine: all_means[b] = fine[b]['mean']
        elif b in coarse: all_means[b] = coarse[b]['mean']
    all_means_arr = np.array([all_means[b] for b in all_br])

    # Test different β_c values
    for bc in [0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.036]:
        fit_mask = (all_br >= bc + 0.005) & (all_br <= 0.06)
        if np.sum(fit_mask) >= 4:
            x = all_br[fit_mask] - bc
            y = all_means_arr[fit_mask]
            valid = (x > 0.001) & (y > 0.01)
            if np.sum(valid) >= 4:
                coeffs = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
                print(f"  β_c={bc:.3f}: ν={coeffs[0]:.4f}  ({np.sum(valid)} points)")

    # Data truncation: remove furthest points progressively
    fit_mask = (all_br >= beta_c_guess + 0.005) & (all_br <= 0.06)
    x_all = all_br[fit_mask] - beta_c_guess
    y_all = all_means_arr[fit_mask]
    valid = (x_all > 0.001) & (y_all > 0.01)
    x_v, y_v = x_all[valid], y_all[valid]
    order = np.argsort(x_v)[::-1]  # furthest first
    for drop_n in [0, 1, 2, 3]:
        if len(x_v) - drop_n >= 4:
            if drop_n == 0:
                coeffs = np.polyfit(np.log(x_v), np.log(y_v), 1)
            else:
                coeffs = np.polyfit(np.log(x_v[order[:-drop_n]]), np.log(y_v[order[:-drop_n]]), 1)
            print(f"  Drop {drop_n} furthest: ν={coeffs[0]:.4f}  ({len(x_v)-drop_n} points)")

print("\nDone.")
