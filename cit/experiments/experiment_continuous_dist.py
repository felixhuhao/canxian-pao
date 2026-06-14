"""
Experiment: Continuous Skill Distribution
==========================================
Tests whether multi-channel τ still differentiates when skills form
a CONTINUOUS distribution of periods (rather than 5 discrete values).

Prediction: Channels self-organize into distinct bands that tile
the continuous period space, with spacing determined by lateral
inhibition scale ℓ.

Compare: Scalar τ fails because one channel cannot cover
continuous periods.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_vector_tau import VectorTauModel, softmax, run_vector_tau_experiment

# ------------------------------------------------------------
# Continuous period distribution: 20 skills, uniformly sampled
# ------------------------------------------------------------
N_SKILLS = 20
np.random.seed(42)
PERIODS_CONT = sorted(np.random.uniform(6, 60, N_SKILLS).tolist())

# Skills named P0..P19
SKILL_NAMES = [f'P{i}' for i in range(N_SKILLS)]


def run_continuous_experiment(M, tau_init=None, seed=42,
                               repulsion_strength=0.5, 
                               n_steps=20000, verbose=True):
    """Run vector τ experiment with continuous period distribution."""
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
        'repulsion_strength': repulsion_strength,
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
    periods = np.array(PERIODS_CONT)
    N = len(periods)
    
    if tau_init is not None:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed,
                                tau_init=np.array(tau_init, dtype=float))
    else:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed)
    
    # History buffers
    x_hist = list(np.random.uniform(-0.1, 0.1, 80))
    offsets = np.zeros(N)  # No phase offsets in continuous case
    
    for t in range(n_steps):
        # Simple progressive schedule: activate all skills gradually
        phase = min(1.0, t / max(n_steps / 2, 1))
        mask = np.ones(N, dtype=bool) if t > 500 else np.zeros(N, dtype=bool)
        if t > 500:
            mask = np.random.random(N) < (0.2 + 0.8 * phase)
        
        # Stimulus from all active skills
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / periods[i])
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active
        
        # State update
        delta_current = int(np.clip(round(np.mean(model.tau)), 1, 64))
        x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
        
        total_kappa = np.mean(model.kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(params['alpha'] * S_active)
               + total_kappa * np.tanh(params['beta'] * x_delayed)
               + np.random.normal(0, params['sigma']))
        x_hist.append(x_t)
        
        model.step(x_t, x_delayed, t, mask,
                   np.array(periods), offsets, params)
    
    # Analysis
    tau_traj = np.array(model.tau_traj)
    kappa_traj = np.array(model.kappa_traj)
    
    tau_final = tau_traj[-1] if len(tau_traj) > 0 else model.tau
    kappa_final = kappa_traj[-1] if len(kappa_traj) > 0 else model.kappa
    
    # Locked count
    lam_s = params['lam_s']
    theta_k = params['theta_k']
    L_final = 1.0 / (1.0 + np.exp(-lam_s * (kappa_final - theta_k)))
    n_locked = int(np.sum(L_final > 0.5))
    
    # τ diversity
    if M > 1:
        tau_diversity = float(np.mean([abs(tau_final[i] - tau_final[j]) 
                                        for i in range(M) for j in range(i+1, M)]))
    else:
        tau_diversity = 0.0
    
    if verbose:
        print(f"\n=== Continuous Distribution, M={M} ===")
        print(f"Final τ: {np.round(tau_final, 2)}")
        print(f"τ diversity: {tau_diversity:.2f}")
        print(f"N locked: {n_locked}/{N}")
        print(f"Coverage: periods {periods[0]:.0f}-{periods[-1]:.0f}, channels span {tau_final[0]:.1f}-{tau_final[-1]:.1f}")
    
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
    print("Continuous Skill Distribution Experiment")
    print("=" * 60)
    print(f"20 skills, periods uniformly sampled from [6, 60]")
    print(f"Periods: {[round(p,1) for p in PERIODS_CONT]}")
    print()
    
    # Compare scalar vs multi-channel
    results = {}
    for M in [1, 3, 5]:
        r = run_continuous_experiment(M=M, seed=42, n_steps=15000)
        results[M] = r
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'M':>3} | {'τ diversity':>12} | {'N locked':>8} | {'Channels span':>15}")
    print("-" * 45)
    for M in [1, 3, 5]:
        r = results[M]
        span = f"{r['tau_final'][0]:.1f}-{r['tau_final'][-1]:.1f}" if M > 1 else f"{r['tau_final'][0]:.1f}"
        print(f"{M:>3} | {r['tau_diversity']:>12.2f} | {r['n_locked']:>3}/{r['N']:>2} | {span:>15}")
    
    # Verify prediction: scalar can't cover continuous distribution
    scalar_locked = results[1]['n_locked']
    multi_locked = results[3]['n_locked']
    print(f"\nScalar τ locked: {scalar_locked}/{results[1]['N']}")
    print(f"Multi-channel (M=3) locked: {multi_locked}/{results[3]['N']}")
    print(f"Scalar failure confirmed: {scalar_locked < multi_locked}")
