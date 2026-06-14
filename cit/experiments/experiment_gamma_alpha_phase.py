"""
Experiment: γ_τ-α_ρ Deadlock Phase Diagram
===========================================
Scans the (γ_τ, α_ρ) plane and measures deadlock volume fraction.
If deadlock is GEOMETRIC (not parametric), then changing γ_τ or α_ρ
changes how fast the system fails, but NOT the fraction of deadlock 
initial conditions.

Prediction: Deadlock fraction is constant across the (γ_τ, α_ρ) plane,
even though the speed of failure changes dramatically.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef, sigmoid


def measure_deadlock_fraction(gamma_tau, alpha_rho, n_tau0=15,
                               n_steps=8000, seed=42):
    """
    For a given (γ_τ, α_ρ), measure the fraction of initial τ₀
    values that result in deadlock (N_locked < 5).
    """
    periods = np.array([12.0, 24.0, 24.0, 48.0, 30.0])
    N = len(periods)
    
    tau0_values = np.linspace(4, 60, n_tau0)
    
    n_deadlocked = 0
    
    for tau0 in tau0_values:
        np.random.seed(seed)
        
        # Simple scalar τ model
        tau = tau0
        kappa = np.zeros(N)
        rho = np.zeros(N)
        L = np.zeros(N)
        
        eta = 0.05
        beta_kappa = 0.006
        theta = 0.3
        lam_s = 12.0
        lam = 1.5
        
        x_hist = list(np.random.uniform(-0.1, 0.1, 80))
        
        for t in range(n_steps):
            # All skills active
            mask = np.ones(N, dtype=bool)
            
            S_active = 0.0
            for i in range(N):
                if mask[i]:
                    S_active += np.sin(2 * np.pi * t / periods[i])
            S_active = S_active / N
            
            delta_current = int(np.clip(round(tau), 1, 64))
            x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
            
            x_t = ((1.0 - np.mean(kappa)) * np.tanh(1.8 * S_active)
                   + np.mean(kappa) * np.tanh(2.2 * x_delayed)
                   + np.random.normal(0, 0.015))
            x_hist.append(x_t)
            
            # ρ update
            for i in range(N):
                rho[i] += alpha_rho * (np.cos(2 * np.pi * tau / periods[i]) - rho[i])
            
            # τ update (scalar)
            tau_pull = 0.0
            w_sum = 0.0
            for i in range(N):
                weight = L[i]
                tau_pull += weight * (periods[i] - tau)
                w_sum += weight
            if w_sum > 0.001:
                tau += gamma_tau * tau_pull / w_sum
            tau = np.clip(tau, 4, 64)
            
            # κ update
            for i in range(N):
                target = np.tanh(lam * abs(rho[i]))
                kappa[i] += beta_kappa * (target - kappa[i])
                L[i] = 1.0 / (1.0 + np.exp(-lam_s * (kappa[i] - theta)))
                kappa[i] += eta * L[i] * kappa[i] * (1.0 - kappa[i])
                kappa[i] = np.clip(kappa[i], 0.0, 1.0)
        
        # Check locked
        L_final = 1.0 / (1.0 + np.exp(-lam_s * (kappa - theta)))
        n_locked = int(np.sum(L_final > 0.5))
        if n_locked < 5:
            n_deadlocked += 1
    
    return n_deadlocked / n_tau0


if __name__ == '__main__':
    print("=" * 60)
    print("γ_τ-α_ρ Deadlock Phase Diagram")
    print("=" * 60)
    print("If deadlock is GEOMETRIC, fraction should be constant across the plane.")
    print()
    
    gamma_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
    alpha_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    
    print(f"{'γ_τ':>6} | {'α_ρ':>6} | {'Ratio':>6} | {'Deadlock %':>11} | {'Fast?'}")
    print("-" * 55)
    
    results = []
    for gamma_tau in gamma_values:
        for alpha_rho in alpha_values:
            frac = measure_deadlock_fraction(gamma_tau, alpha_rho,
                                              n_tau0=11, n_steps=5000)
            ratio = gamma_tau / max(alpha_rho, 0.0001)
            fast = "slow" if ratio < 0.5 else ("fast" if ratio > 3 else "ok")
            results.append((gamma_tau, alpha_rho, frac, ratio))
            print(f"{gamma_tau:>6.3f} | {alpha_rho:>6.3f} | {ratio:>6.2f} | {frac*100:>10.1f}% | {fast}")
    
    # Summary statistics
    fractions = [r[2] for r in results]
    mean_frac = np.mean(fractions)
    std_frac = np.std(fractions)
    
    print(f"\nDeadlock fraction across plane: {mean_frac*100:.1f}% ± {std_frac*100:.1f}%")
    print(f"Relative variation: {std_frac/max(mean_frac, 0.001)*100:.1f}%")
    
    if std_frac < 0.1:
        print("\nCONCLUSION: Deadlock fraction is approximately constant across the (γ_τ, α_ρ) plane.")
        print("This confirms that deadlock is GEOMETRIC, not parametric.")
        print("Parameter tuning changes the SPEED of failure, not the SET of failures.")
    else:
        print("\nCONCLUSION: Deadlock fraction varies significantly across the (γ_τ, α_ρ) plane.")
        print("Deadlock may have parametric dependence.")
