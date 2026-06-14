#!/usr/bin/env python3
"""
Bifurcation scaling experiment: scan β_rep and measure inter-channel spacing.
Prediction: Δτ_min ∝ (β_rep - β_c)^{1/2} (supercritical pitchfork)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Model parameters (from experiment_vector_tau.py) ──
P = np.array([12, 24, 30, 48])
N_SKILLS = len(P)
M = 3
N_STEPS = 8000
TAU0 = 16.0

ALPHA_RHO = 0.01
GAMMA_TAU = 0.05
BETA_KAPPA = 0.006
ETA_KAPPA = 0.05
THETA = 0.30
LAMBDA_S = 12
LAMBDA = 1.5
SIGMA = 0.015
ALPHA_W = 0.01
LAMBDA_W = 0.005
L_REP = 8.0

def correlation(p, tau):
    return np.cos(2 * np.pi * tau / p)

def run_trial(beta_rep, seed=0):
    rng = np.random.RandomState(seed)
    tau = np.full(M, TAU0, dtype=float)
    W = rng.rand(M, N_SKILLS) * 0.1
    kappa = np.zeros((N_SKILLS, N_STEPS + 1))
    
    for t in range(N_STEPS):
        rho = np.zeros(N_SKILLS)
        L = np.zeros(N_SKILLS)
        for i in range(N_SKILLS):
            rho[i] = correlation(P[i], tau[t % M])
            L[i] = 1.0 / (1.0 + np.exp(-LAMBDA_S * (kappa[i, t] - THETA)))
        
        # Update τ for each channel
        for m in range(M):
            sm = np.exp(W[m]) / np.sum(np.exp(W), axis=0)
            pull = GAMMA_TAU * np.sum(sm * L * (P - tau[m]))
            
            repulsion = 0.0
            for n in range(M):
                if n == m: continue
                Ln = np.max(1.0 / (1.0 + np.exp(-LAMBDA_S * (kappa[:, t] - THETA))))
                rep = beta_rep * np.sign(tau[m] - tau[n]) * np.exp(-(tau[m]-tau[n])**2 / (2*L_REP**2)) * Ln
                repulsion += rep
            
            tau[m] += pull + repulsion
            tau[m] = np.clip(tau[m], 4, 64)
        
        # Update kappa for each skill
        tau_used = tau[int(t % M)]
        for i in range(N_SKILLS):
            r = (1 - ALPHA_RHO) * rho[i] + ALPHA_RHO * correlation(P[i], tau_used)
            drive = np.tanh(LAMBDA * abs(r))
            kappa_i = kappa[i, t] + BETA_KAPPA * (drive - kappa[i, t]) + ETA_KAPPA * kappa[i, t] * (1 - kappa[i, t]) * L[i]
            kappa[i, t+1] = np.clip(kappa_i, 0, 1)
        
        # Update W
        if t % 10 == 0:
            for m in range(M):
                for i in range(N_SKILLS):
                    Cim = np.exp(-4 * abs(tau[m] - P[i]) / P[i])
                    dW = ALPHA_W * (Cim * L[i] - LAMBDA_W * W[m, i])
                    W[m, i] += dW
    
    return tau, kappa[:, -1]

# ── Scan β_rep ──
BETA_REP_VALS = np.linspace(0, 0.5, 15)
SEEDS = 5

results = {b: {'deltas': [], 'nlocks': []} for b in BETA_REP_VALS}

for b in BETA_REP_VALS:
    for s in range(SEEDS):
        tau_final, kappa_final = run_trial(b, seed=s)
        tau_sorted = np.sort(tau_final)
        deltas = np.diff(tau_sorted)
        min_delta = np.min(deltas) if len(deltas) > 0 else 0
        nlock = np.sum(kappa_final > THETA)
        results[b]['deltas'].append(min_delta)
        results[b]['nlocks'].append(nlock)
    avg_d = np.mean(results[b]['deltas'])
    std_d = np.std(results[b]['deltas'])
    avg_n = np.mean(results[b]['nlocks'])
    print(f"β_rep={b:.3f}: Δτ_min={avg_d:.3f}±{std_d:.3f}, N_lock={avg_n:.1f}")

# ── Fit bifurcation scaling ──
means = np.array([np.mean(results[b]['deltas']) for b in BETA_REP_VALS])
stds = np.array([np.std(results[b]['deltas']) for b in BETA_REP_VALS])

# Find critical β_c where Δτ_min starts rising above noise floor (~0.1)
noise_floor = 0.5
above_noise = np.where(means > noise_floor)[0]
if len(above_noise) > 0:
    beta_c_idx = above_noise[0]
    beta_c = BETA_REP_VALS[beta_c_idx]
else:
    beta_c = 0.05

# Fit sqrt scaling: Δτ = a * (β_rep - β_c)^{1/2} for β_rep > β_c
fit_idx = np.where(BETA_REP_VALS >= beta_c + 0.02)[0]
if len(fit_idx) > 3:
    x_fit = BETA_REP_VALS[fit_idx] - beta_c
    y_fit = means[fit_idx]
    # Linear fit on log-log
    coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    fitted_exp = coeffs[0]
    fitted_a = np.exp(coeffs[1])
    print(f"\n--- Bifurcation Fit ---")
    print(f"β_c ≈ {beta_c:.3f}")
    print(f"Fitted exponent: {fitted_exp:.3f} (expected 0.5 for pitchfork)")
    print(f"Fitted a: {fitted_a:.3f}")
else:
    fitted_exp = None
    print("Not enough points for fit")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Bifurcation Analysis: Channel Collapse → Niche Differentiation', fontsize=13, fontweight='bold')

# Left: Δτ_min vs β_rep
ax = axes[0]
ax.errorbar(BETA_REP_VALS, means, yerr=stds, fmt='o-', capsize=3, color='teal', markersize=4)
ax.axvline(beta_c, color='red', ls='--', alpha=0.5, label=f'β_c≈{beta_c:.2f}')
if fitted_exp:
    x_smooth = np.linspace(0.01, max(BETA_REP_VALS - beta_c), 100)
    y_smooth = fitted_a * x_smooth**0.5
    ax.plot(x_smooth + beta_c, y_smooth, 'r-', lw=1.5, alpha=0.6, label=f'sqrt fit (exp={fitted_exp:.2f})')
ax.set_xlabel('Lateral inhibition strength β_rep')
ax.set_ylabel('Min inter-channel spacing Δτ_min')
ax.legend(fontsize=8)
ax.set_title('Channel separation vs. control parameter')

# Right: log-log scaling
ax = axes[1]
if fit_idx.any():
    ax.loglog(BETA_REP_VALS[fit_idx] - beta_c, means[fit_idx], 'o', color='teal', markersize=5)
    if fitted_exp:
        ax.loglog(x_smooth, y_smooth, 'r-', lw=1.5, alpha=0.6, label=f'slope={fitted_exp:.2f}')
    ax.set_xlabel('β_rep - β_c (log)')
    ax.set_ylabel('Δτ_min (log)')
    ax.set_title(f'Scaling: exponent ≈ {fitted_exp:.2f} (pitchfork→0.5)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
out = '/Users/caihengjin/.openclaw/workspace/analysis/code/fig_bifurcation_scaling.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out}")

# ── Save data ──
np.savez('/Users/caihengjin/.openclaw/workspace/analysis/code/bifurcation_data.npz',
         beta_rep=BETA_REP_VALS, delta_means=means, delta_stds=stds,
         beta_c=beta_c, fitted_exp=fitted_exp, fitted_a=fitted_a)
print("Data saved: code/bifurcation_data.npz")
