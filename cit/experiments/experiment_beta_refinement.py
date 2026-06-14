#!/usr/bin/env python3
"""Grid refinement + threshold sensitivity for β_c identification."""
import numpy as np

# --- Parameters (matching main model) ---
N_SKILLS = 5
M = 3
P = np.array([12, 24, 24, 48, 30])  # A/B share P=24
ALPHA_RHO = 0.01
GAMMA_TAU = 0.03
BETA_KAPPA = 0.006
ETA_KAPPA = 0.05
THETA = 0.30
LAMBDA_S = 12.0
TAU_MIN, TAU_MAX = 4.0, 64.0
L_REP = 8.0
ALPHA_W = 0.01
LAMBDA_W = 0.005
T_MAX = 5000

def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def run_simulation(beta_rep_val, seed=42):
    _br = beta_rep_val  # local alias
    tau = np.full(M, 24.0, dtype=np.float64)
    rho = np.zeros((M, N_SKILLS), dtype=np.float64)
    kappa = np.zeros((M, N_SKILLS), dtype=np.float64)
    W = np.ones((M, N_SKILLS), dtype=np.float64) * 0.5
    for _ in range(T_MAX):
        # Resonance
        for m in range(M):
            for i in range(N_SKILLS):
                R = np.cos(2 * np.pi * tau[m] / P[i])
                rho[m, i] = (1 - ALPHA_RHO) * rho[m, i] + ALPHA_RHO * R
        # Gate
        L = 1.0 / (1.0 + np.exp(-LAMBDA_S * (kappa - THETA)))
        # Softmax attention
        soft_W = softmax(W)  # M x N
        # Tau update (lateral inhibition)
        L_mean = np.mean(L, axis=1)  # per channel mean gate
        delta_tau = np.zeros(M)
        for m in range(M):
            tau_pull = GAMMA_TAU * np.sum(soft_W[m] * L[m] * (P - tau[m]))
            repulsion = 0.0
            for n in range(M):
                if n == m:
                    continue
                d = tau[m] - tau[n]
                repulsion += (_br if _br > 0 else 0.0) * np.sign(d) * np.exp(-d**2 / (2 * L_REP**2)) * L_mean[n]
            delta_tau[m] = tau_pull + repulsion
        tau += delta_tau
        tau = np.clip(tau, TAU_MIN, TAU_MAX)
        # Kappa update
        eta_abs_rho = ETA_KAPPA * np.tanh(np.abs(rho))
        kappa += BETA_KAPPA * (eta_abs_rho - kappa) + ETA_KAPPA * kappa * (1 - kappa) * L
        kappa = np.clip(kappa, 0, 1)
        # Hebbian affinity
        C = np.exp(-4.0 * np.abs(tau[:, None] - P[None, :]))
        W += ALPHA_W * (C * L - LAMBDA_W * W)
        W = np.clip(W, 0, 5)
        # Early stop if channels clearly differentiated
        if np.ptp(tau) > 5.0 and np.all(np.min(kappa, axis=1) > 0.99):
            break
    delta_min = np.min(np.diff(np.sort(tau))) if M >= 2 else 0.0
    return delta_min, tau.copy()

# --- Grid refinement ---
print("=== Grid Refinement: β_rep around 0.025 ===")

# Coarse grid (original)
coarse_vals = np.arange(0, 0.101, 0.005)
coarse_results = []
for br in coarse_vals:
    dm, tau = run_simulation(br)
    coarse_results.append((br, dm))
    print(f"  β={br:.4f}  Δτ_min={dm:.4f}  τ={np.round(tau,2)}")

# Fine grid around critical region
fine_vals = np.arange(0.015, 0.04, 0.001)
fine_results = []
for br in fine_vals:
    dm, tau = run_simulation(br)
    fine_results.append((br, dm))
    print(f"  β={br:.4f}  Δτ_min={dm:.4f}  τ={np.round(tau,2)}")

# --- Threshold sensitivity ---
print("\n=== Threshold Sensitivity ===")
thresholds = [0.3, 0.5, 0.7]
for th in thresholds:
    # Find first point > threshold on fine grid
    for br, dm in fine_results:
        if dm > th:
            print(f"  threshold={th:.1f}: first β_rep > threshold at {br:.4f} (Δτ_min={dm:.4f})")
            break
    else:
        print(f"  threshold={th:.1f}: none exceeded on fine grid")

# --- Bootstrapped exponent with and without last point ---
from numpy.polynomial import Polynomial

print("\n=== Exponent Stability Under Truncation ===")
# Use fine grid data: β_rep >= first point > 0.5 and <= 0.06
data = [(br, dm) for br, dm in fine_results if dm > 0.5]
# Extend with coarse grid up to 0.06
for br, dm in coarse_results:
    if 0.04 <= br <= 0.06 and br not in [a for a,_ in data]:
        data.append((br, dm))
data.sort(key=lambda x: x[0])

# Full data fit
x_full = np.array([d[0] for d in data])
y_full = np.array([d[1] for d in data])
beta_c_full = 0.025
x_log = np.log(x_full - beta_c_full + 1e-10)
y_log = np.log(y_full + 1e-10)
coeffs_full = Polynomial.fit(x_log, y_log, 1).convert()
nu_full = coeffs_full.coef[1]
print(f"  Full ({len(data)} points): ν={nu_full:.4f}")

# Without last point (truncation robustness)
for drop in [1, 2, 3]:
    x_trunc = x_log[:-drop] if drop < len(x_log) else x_log
    y_trunc = y_log[:-drop] if drop < len(y_log) else y_log
    coeffs = Polynomial.fit(x_trunc, y_trunc, 1).convert()
    print(f"  Drop last {drop}: ν={coeffs.coef[1]:.4f}")

# Different β_c choices
print("\n=== β_c Sensitivity ===")
for bc in [0.022, 0.024, 0.025, 0.026, 0.028]:
    if bc >= min(d[0] for d in data):
        x_bc = np.log(np.array([d[0] for d in data]) - bc + 1e-10)
        y_bc = np.log(np.array([d[1] for d in data]) + 1e-10)
        coeffs = Polynomial.fit(x_bc, y_bc, 1).convert()
        print(f"  β_c={bc:.3f}: ν={coeffs.coef[1]:.4f}")
    else:
        print(f"  β_c={bc:.3f}: invalid (below first data point)")

print("\nDone.")
