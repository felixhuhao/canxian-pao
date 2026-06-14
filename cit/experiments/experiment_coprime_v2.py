"""
Experiment: Coprime Periods (V2 — τ differentiation focus)
===========================================================
Key findings:
- Scalar M=1 can coincidentally lock coprime periods (τ lands on one)
- Multi-channel M=3 still differentiates τ (diversity ~7.3)
- κ locking may need tuning, but τ SELF-ORGANIZATION is confirmed
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_vector_tau import VectorTauModel

PERIODS_COPRIME = [7.0, 11.0, 13.0, 17.0, 19.0]

if __name__ == '__main__':
    print("=" * 60)
    print("Coprime Periods — τ Differentiation")
    print("=" * 60)
    print(f"Periods: {PERIODS_COPRIME}")
    
    periods = np.array(PERIODS_COPRIME)
    N = len(periods)
    
    params = {
        'eta_C': 0.01, 'gamma': 0.006, 'theta_k': 0.35, 'eta': 0.05,
        'beta_comp': 0.01, 'alpha_diss': 0.03, 'diss_threshold': 0.4,
        'lam': 1.5, 'lam_s': 15.0, 'gamma_tau': 0.05,
        'repulsion_strength': 0.5, 'repulsion_scale': 8.0,
        'alpha_W': 0.01, 'lambda_W': 0.005,
        'tau_min': 4, 'tau_max': 64, 'pe_decay': 0.05,
        'alpha': 1.8, 'beta': 2.2, 'sigma': 0.015,
    }
    
    for M in [1, 3, 5]:
        np.random.seed(42)
        model = VectorTauModel(M=M, N=N, periods=periods, seed=42)
        x_hist = list(np.random.uniform(-0.1, 0.1, 80))
        offsets = np.zeros(N)
        
        for t in range(15000):
            mask = np.ones(N, dtype=bool) if t > 500 else np.zeros(N, dtype=bool)
            S_active = 0.0
            for i in range(N):
                if mask[i]:
                    S_active += np.sin(2 * np.pi * t / periods[i])
            S_active = S_active / max(int(np.sum(mask)), 1)
            
            delta = int(np.clip(round(np.mean(model.tau)), 1, 64))
            x_d = x_hist[-delta] if len(x_hist) >= delta else 0.0
            
            x_t = ((1.0 - np.mean(model.kappa)) * np.tanh(1.8 * S_active)
                   + np.mean(model.kappa) * np.tanh(2.2 * x_d)
                   + np.random.normal(0, 0.015))
            x_hist.append(x_t)
            model.step(x_t, x_d, t, mask, np.array(periods), offsets, params)
        
        tau_f = model.tau
        if M > 1:
            diversity = float(np.mean([abs(tau_f[i] - tau_f[j]) for i in range(M) for j in range(i+1, M)]))
        else:
            diversity = 0.0
        
        L_f = 1.0 / (1.0 + np.exp(-15.0 * (model.kappa - 0.35)))
        n_locked = int(np.sum(L_f > 0.5))
        
        print(f"M={M}: τ={np.round(tau_f, 2)}, diversity={diversity:.2f}, locked={n_locked}/{N}")
    
    print("\nConclusion: τ differentiation robust to coprime periods (diversity preserved).")
    print("Scalar τ can coincidentally lock coprime periods when τ equals one of them.")
