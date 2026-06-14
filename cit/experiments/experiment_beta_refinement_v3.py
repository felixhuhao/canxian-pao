#!/usr/bin/env python3
"""β_c grid refinement + threshold sensitivity.
Uses the working simplified model from experiment_bifurcation.py."""
import numpy as np

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
        tau_used = tau[int(t % M)]
        for i in range(N_SKILLS):
            r = (1 - ALPHA_RHO) * rho[i] + ALPHA_RHO * correlation(P[i], tau_used)
            drive = np.tanh(LAMBDA * abs(r))
            kappa_i = kappa[i, t] + BETA_KAPPA * (drive - kappa[i, t]) + ETA_KAPPA * kappa[i, t] * (1 - kappa[i, t]) * L[i]
            kappa[i, t+1] = np.clip(kappa_i, 0, 1)
        if t % 10 == 0:
            for m in range(M):
                for i in range(N_SKILLS):
                    Cim = np.exp(-4 * abs(tau[m] - P[i]) / P[i])
                    dW = ALPHA_W * (Cim * L[i] - LAMBDA_W * W[m, i])
                    W[m, i] += dW
    tau_sorted = np.sort(tau)
    return np.min(np.diff(tau_sorted)) if M >= 2 else 0.0, tau.copy()

SEEDS = 10

# Coarse: 0 to 0.1, step 0.005
print("=== Coarse (step=0.005) ===")
for br in np.arange(0, 0.101, 0.005):
    dm = [run_trial(br, s)[0] for s in range(SEEDS)]
    print(f"  β={br:.4f}  Δτ_min={np.mean(dm):.4f}±{np.std(dm):.4f}")

# Fine: 0 to 0.1, step 0.001
print("\n=== Fine (step=0.001) ===")
fine_br = np.arange(0, 0.101, 0.001)
fine_dm = []
for br in fine_br:
    dm = [run_trial(br, s)[0] for s in range(SEEDS)]
    fine_dm.append(np.mean(dm))
    if br in [v/1000 for v in range(0, 101, 5)]:  # print every 5 steps
        print(f"  β={br:.4f}  Δτ_min={fine_dm[-1]:.4f}±{np.std(dm):.4f}")
fine_dm = np.array(fine_dm)

# Threshold sensitivity
print("\n=== Threshold Sensitivity ===")
for th in [0.3, 0.5, 0.7, 1.0]:
    above = np.where(fine_dm > th)[0]
    if len(above) > 0:
        print(f"  threshold={th:.1f}: first β_rep > threshold at {fine_br[above[0]]:.4f}")
    else:
        print(f"  threshold={th:.1f}: never exceeded on fine grid")

# Exponent stability under different β_c
print("\n=== Exponent Stability vs β_c choice ===")
for bc in np.arange(0.020, 0.041, 0.002):
    fit_mask = (fine_br >= bc + 0.005)
    x = fine_br[fit_mask] - bc
    y = fine_dm[fit_mask]
    valid = (x > 0.001) & (y > 0.01)
    if np.sum(valid) >= 5:
        coeffs = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
        print(f"  β_c={bc:.3f}: ν={coeffs[0]:.4f}  ({np.sum(valid)} points)")
    else:
        print(f"  β_c={bc:.3f}: <5 valid points, skipping")

# Data truncation (remove furthest points)
print("\n=== Exponent Stability vs Truncation ===")
bc = fine_br[np.where(fine_dm > 0.5)[0][0]]
print(f"β_c (Δτ>0.5) = {bc:.4f}")
fit_mask = (fine_br >= bc + 0.005)
x_all = fine_br[fit_mask] - bc
y_all = fine_dm[fit_mask]
valid = (x_all > 0.001) & (y_all > 0.01)
x_v, y_v = x_all[valid], y_all[valid]
order = np.argsort(x_v)[::-1]
for drop_n in [0, 1, 2, 3]:
    if len(x_v) - drop_n < 5: break
    idx = order[:-drop_n] if drop_n > 0 else slice(None)
    coeffs = np.polyfit(np.log(x_v[idx]), np.log(y_v[idx]), 1)
    print(f"  Drop {drop_n} furthest: ν={coeffs[0]:.4f}  ({len(x_v)-drop_n} points)")

print("\nDone.")
