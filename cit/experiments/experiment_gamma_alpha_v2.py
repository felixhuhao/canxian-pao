"""
Experiment: γ_τ-α_ρ Deadlock Phase Diagram (V2)
===============================================
Uses the PROPER scalar τ model (from experiment_crystallization)
to measure deadlock fraction across the (γ_τ, α_ρ) plane.

Prediction: deadlock fraction ~58% everywhere.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '..')
from experiment_crystallization import sigmoid


def measure_1d_deadlock(gamma_tau, alpha_rho, n_tau0=15,
                         n_steps=10000, seed=42):
    """
    Full scalar τ dynamics. Count deadlocked runs.
    Deadlock = N_locked < 5 out of 5 skills.
    """
    periods = np.array([12.0, 24.0, 24.0, 48.0, 30.0])
    N = 5
    tau0_values = np.linspace(4, 60, n_tau0)
    n_dead = 0
    
    for tau0 in tau0_values:
        np.random.seed(seed)
        
        tau = float(tau0)
        rho = np.zeros(N)
        kappa = np.zeros(N)
        L = np.zeros(N)
        
        x_hist = list(np.random.uniform(-0.1, 0.1, 80))
        
        for t in range(n_steps):
            # All skills active
            S_active = (np.sin(2*np.pi*t/12.0) + np.sin(2*np.pi*t/24.0) * 2
                       + np.sin(2*np.pi*t/48.0) + np.sin(2*np.pi*t/30.0)) / 5.0
            
            delta = int(np.clip(round(tau), 1, 64))
            x_delay = x_hist[-delta] if len(x_hist) >= delta else 0.0
            
            x_t = ((1.0 - np.mean(kappa)) * np.tanh(1.8 * S_active)
                   + np.mean(kappa) * np.tanh(2.2 * x_delay)
                   + np.random.normal(0, 0.015))
            x_hist.append(x_t)
            
            kbar = np.mean(kappa)
            
            # ρ update
            for i in range(N):
                corr_i = np.cos(2 * np.pi * tau / periods[i])
                rho[i] += alpha_rho * (corr_i - rho[i])
            
            # τ update (scalar)
            pull = 0.0
            wsum = 0.0
            for i in range(N):
                w = L[i]
                pull += w * (periods[i] - tau)
                wsum += w
            if wsum > 1e-6:
                tau += gamma_tau * pull / wsum
            tau = np.clip(tau, 4, 64)
            
            # κ update
            for i in range(N):
                target = np.tanh(1.5 * abs(rho[i]))
                kappa[i] += 0.006 * (target - kappa[i])
                L[i] = 1.0 / (1.0 + np.exp(-12.0 * (kappa[i] - 0.3)))
                kappa[i] += 0.05 * L[i] * kappa[i] * (1.0 - kappa[i])
                kappa[i] = np.clip(kappa[i], 0.0, 1.0)
        
        L_f = 1.0 / (1.0 + np.exp(-12.0 * (kappa - 0.3)))
        if int(np.sum(L_f > 0.5)) < 5:
            n_dead += 1
    
    return n_dead / n_tau0


if __name__ == '__main__':
    print("=" * 60)
    print("γ_τ-α_ρ Deadlock Phase Diagram (Full Model)")
    print("=" * 60)
    
    gamma_values = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    alpha_values = [0.005, 0.01, 0.03, 0.05, 0.1]
    
    print(f"{'γ_τ':>6} | {'α_ρ':>6} | {'Ratio':>6} | {'Deadlock %':>11}")
    print("-" * 40)
    
    fracs = []
    for gt in gamma_values:
        for ar in alpha_values:
            frac = measure_1d_deadlock(gt, ar, n_tau0=11, n_steps=8000)
            fracs.append(frac)
            ratio = gt / max(ar, 1e-6)
            print(f"{gt:>6.3f} | {ar:>6.3f} | {ratio:>6.2f} | {frac*100:>10.1f}%")
    
    print(f"\nMean deadlock: {np.mean(fracs)*100:.1f}% ± {np.std(fracs)*100:.1f}%")
    if np.std(fracs) < 0.1:
        print("CONCLUSION: Deadlock fraction is constant across (γ_τ, α_ρ). Geometric, not parametric.")
    else:
        print("CONCLUSION: Deadlock fraction varies - may have parametric dependence.")
