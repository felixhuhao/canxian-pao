"""
Experiment: Coprime Skill Periods
==================================
Tests whether multi-channel τ still differentiates when all skill
periods are pairwise coprime {7, 11, 13, 17, 19}.

Key question: Does the mechanism depend on harmonic relationships 
between periods? Coprime periods eliminate shared divisors.

Prediction: Channels still differentiate. The mechanism is based on
competitive exclusion in τ-space, not on harmonic alignment.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_vector_tau import VectorTauModel, run_vector_tau_experiment
from experiment_continuous_dist import run_continuous_experiment

# Coprime period set
PERIODS_COPRIME = [7.0, 11.0, 13.0, 17.0, 19.0]
SKILL_NAMES = [f'S{i}' for i in range(5)]


def run_coprime_experiment(M=3, tau_init=None, seed=42,
                            n_steps=20000, verbose=True):
    """Run scalar vs vector τ with coprime skill periods."""
    periods = np.array(PERIODS_COPRIME)
    N = len(periods)
    
    params = {
        'eta_C': 0.01,
        'gamma': 0.006,
        'theta_k': 0.35,
        'eta': 0.05,
        'beta_comp': 0.01,
        'alpha_diss': 0.03,
        'diss_threshold': 0.4,
        'lam': 1.5,
        'lam_s': 15.0,
        'gamma_tau': 0.05,
        'repulsion_strength': 0.5,
        'repulsion_scale': 8.0,
        'alpha_W': 0.01,
        'lambda_W': 0.005,
        'tau_min': 4,
        'tau_max': 64,
        'pe_decay': 0.05,
        'alpha': 1.8,
        'beta': 2.2,
        'sigma': 0.015,
    }
    
    np.random.seed(seed)
    
    if tau_init is not None:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed,
                                tau_init=np.array(tau_init, dtype=float))
    else:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed)
    
    x_hist = list(np.random.uniform(-0.1, 0.1, 80))
    offsets = np.zeros(N)
    
    for t in range(n_steps):
        # Progressive schedule
        mask = np.ones(N, dtype=bool) if t > 500 else np.zeros(N, dtype=bool)
        if t > 500:
            phase = min(1.0, t / 10000.0)
            mask = np.random.random(N) < (0.2 + 0.8 * phase)
        
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / periods[i])
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active
        
        delta_current = int(np.clip(round(np.mean(model.tau)), 1, 64))
        x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
        
        total_kappa = np.mean(model.kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(params['alpha'] * S_active)
               + total_kappa * np.tanh(params['beta'] * x_delayed)
               + np.random.normal(0, params['sigma']))
        x_hist.append(x_t)
        
        model.step(x_t, x_delayed, t, mask,
                   np.array(periods), offsets, params)
    
    tau_traj = np.array(model.tau_traj)
    kappa_traj = np.array(model.kappa_traj)
    
    tau_final = tau_traj[-1] if len(tau_traj) > 0 else model.tau
    kappa_final = kappa_traj[-1] if len(kappa_traj) > 0 else model.kappa
    
    lam_s = params['lam_s']
    theta_k = params['theta_k']
    L_final = 1.0 / (1.0 + np.exp(-lam_s * (kappa_final - theta_k)))
    n_locked = int(np.sum(L_final > 0.5))
    
    if M > 1:
        tau_diversity = float(np.mean([abs(tau_final[i] - tau_final[j])
                                        for i in range(M) for j in range(i+1, M)]))
    else:
        tau_diversity = 0.0
    
    if verbose:
        print(f"\n=== Coprime Periods, M={M} ===")
        print(f"Final τ: {np.round(tau_final, 2)}")
        print(f"τ diversity: {tau_diversity:.2f}")
        print(f"N locked: {n_locked}/{N}")
        print(f"Periods: {PERIODS_COPRIME}")
    
    return {
        'M': M,
        'tau_final': tau_final,
        'tau_diversity': tau_diversity,
        'n_locked': n_locked,
        'N': N,
        'kappa_final': kappa_final,
        'tau_traj': tau_traj,
        'kappa_traj': kappa_traj,
        'periods': periods,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Coprime Period Experiment")
    print("=" * 60)
    print(f"Periods: {PERIODS_COPRIME} (all pairwise coprime)")
    print()
    
    results = {}
    for M in [1, 3]:
        r = run_coprime_experiment(M=M, seed=42, n_steps=12000)
        results[M] = r
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for M in [1, 3]:
        r = results[M]
        print(f"M={M}: final τ={np.round(r['tau_final'], 2)}, "
              f"diversity={r['tau_diversity']:.2f}, locked={r['n_locked']}/{r['N']}")
    
    # Key verification
    m3_locked = results[3]['n_locked']
    m1_locked = results[1]['n_locked']
    m3_diversity = results[3]['tau_diversity']
    
    print(f"\nCoprime periods don't prevent niche differentiation:")
    print(f"  Scalar locked: {m1_locked}/5")
    print(f"  Multi-channel locked: {m3_locked}/5")
    print(f"  τ diversity (M=3): {m3_diversity:.2f}")
    print(f"  CONCLUSION: {'Mechanism is robust to coprime periods' if m3_locked > m1_locked and m3_diversity > 2.0 else 'Unexpected result'}")
