#!/usr/bin/env python3
"""
Mode A Experiment v3 — Forced Trajectory Through Anti-Phase
============================================================
Instead of letting τ drift naturally (which gets stuck at resonance),
we force τ through the anti-phase zone and observe κ collapse.

Two scenarios:
  1. Cosine correlation + signed drive + η=0  → Mode A collapse
  2. Cosine correlation + abs drive + η=0.05  → Mode B protection (control)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ──
N_SKILLS = 3
P = np.array([12, 24, 48])
T = 40000                           # simulation steps
TAU_START = 9.0                     # start below P=12
TAU_END = 20.0                      # end past anti-phase for P=12 (τ/P=18/12=1.5)
BETA_KAPPA = 0.02
THETA = 0.30
LAMBDA_S = 12
LAMBDA = 1.5

def correlation(P_i, tau):
    return np.cos(2 * np.pi * tau / P_i)

def run_sweep(alpha_rho, eta_kappa, use_abs_rho, label):
    """Force τ through a linear sweep from TAU_START to TAU_END."""
    tau = np.linspace(TAU_START, TAU_END, T)
    rho = np.zeros((N_SKILLS, T))
    kappa = np.zeros((N_SKILLS, T))
    
    for t in range(1, T):
        for i in range(N_SKILLS):
            rho[i, t] = (1 - alpha_rho) * rho[i, t-1] + alpha_rho * correlation(P[i], tau[t])
            
            if use_abs_rho:
                drive = np.tanh(LAMBDA * abs(rho[i, t-1]))
            else:
                drive = np.tanh(LAMBDA * rho[i, t-1])
            
            L = 1.0 / (1.0 + np.exp(-LAMBDA_S * (kappa[i, t-1] - THETA)))
            kappa[i, t] = kappa[i, t-1] + BETA_KAPPA * (drive - kappa[i, t-1]) \
                          + eta_kappa * kappa[i, t-1] * (1 - kappa[i, t-1]) * L
            kappa[i, t] = np.clip(kappa[i, t], 0.0, 1.0)
    
    return tau, rho, kappa

# ── Run ──
print("Running forced τ sweep...")
tau, rho_A, kappa_A = run_sweep(alpha_rho=0.3, eta_kappa=0.0, use_abs_rho=False, label="Mode A")
_, rho_B, kappa_B = run_sweep(alpha_rho=0.01, eta_kappa=0.05, use_abs_rho=True, label="Mode B")
_, rho_C, kappa_C = run_sweep(alpha_rho=0.3, eta_kappa=0.0, use_abs_rho=True, label="No-Mode (abs drive)")

# ── Analysis ──
def find_collapse(kappa_matrix, theta=0.10):
    k_max = np.max(kappa_matrix, axis=0)
    above = np.where(k_max > THETA)[0]
    if len(above) == 0:
        return None, None
    
    # Find the peak
    peak_idx = above[-1]
    
    # Look for sustained drop below theta after peak
    for t in range(peak_idx, len(k_max)):
        if k_max[t] < theta and np.all(k_max[t:min(t+500, len(k_max))] < theta):
            return t, tau[t]
    return None, None

collapse_A, tau_coll_A = find_collapse(kappa_A)
collapse_B, tau_coll_B = find_collapse(kappa_B)
collapse_C, tau_coll_C = find_collapse(kappa_C)

print(f"\n=== Results ===")
print(f"Mode A (cosine+signed+η=0):    collapse={collapse_A}, τ_coll={tau_coll_A:.2f}" if collapse_A else "Mode A (cosine+signed+η=0):    NO COLLAPSE")
print(f"Mode B (standard safe):        collapse={collapse_B}, τ_coll={tau_coll_B:.2f}" if collapse_B else "Mode B (standard safe):        NO COLLAPSE")
print(f"Abs drive (cosine+abs+η=0):    collapse={collapse_C}, τ_coll={tau_coll_C:.2f}" if collapse_C else "Abs drive (cosine+abs+η=0):    NO COLLAPSE")

# Print detailed κ around anti-phase
tau_anti = 18.0  # anti-phase for P=12
idx_anti = np.argmin(np.abs(tau - tau_anti))
print(f"\nκ at anti-phase (τ={tau_anti:.1f}, P=12):")
for name, kmat in [("Mode A", kappa_A), ("Mode B", kappa_B), ("Abs drive", kappa_C)]:
    vals = kmat[:, idx_anti]
    print(f"  {name:12s}: κ_P12={vals[0]:.4f}, κ_P24={vals[1]:.4f}, κ_P48={vals[2]:.4f}")

# ── Plot ──
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
fig.suptitle('Mode A: Forced τ Sweep Through Anti-Phase Zone', fontsize=14, fontweight='bold')

scenarios = [
    ('Mode A: cos+sign+η=0', kappa_A, rho_A),
    ('Mode B: standard safe', kappa_B, rho_B),
    ('Abs drive only: cos+abs+η=0', kappa_C, rho_C),
]
colors = ['b', 'g', 'r']
labels = [f'P={p}' for p in P]

for col, (title, kmat, rmat) in enumerate(scenarios):
    # κ
    ax = axes[0, col]
    for i in range(N_SKILLS):
        ax.plot(tau, kmat[i], colors[i], lw=1.0, label=labels[i])
    ax.axhline(THETA, color='gray', ls='--', alpha=0.5)
    ax.axvline(18, color='r', ls=':', alpha=0.3, label='anti-phase (P=12)')
    ax.set_xlabel('τ')
    ax.set_ylabel('κ')
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_xlim(TAU_START, TAU_END)
    
    # ρ
    ax = axes[1, col]
    for i in range(N_SKILLS):
        ax.plot(tau, rmat[i], colors[i], lw=1.0, label=labels[i])
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.axvline(18, color='r', ls=':', alpha=0.3)
    ax.set_xlabel('τ')
    ax.set_ylabel('ρ')
    ax.set_title(f'ρ(t)')
    ax.legend(fontsize=7)
    ax.set_xlim(TAU_START, TAU_END)
    
    # Drive = R(P, τ)
    ax = axes[2, col]
    tau_vals = np.linspace(TAU_START, TAU_END, 500)
    for i, p in enumerate(P):
        ax.plot(tau_vals, correlation(p, tau_vals), colors[i], lw=1.0, label=labels[i])
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.axvline(18, color='r', ls=':', alpha=0.3, label='anti-phase')
    ax.set_xlabel('τ')
    ax.set_ylabel('R(P, τ)')
    ax.set_title('Correlation function R(P, τ) = cos(2πτ/P)')
    ax.legend(fontsize=7)
    ax.set_xlim(TAU_START, TAU_END)

plt.tight_layout()
out = '/Users/caihengjin/.openclaw/workspace/analysis/code/fig_mode_a.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out}")
