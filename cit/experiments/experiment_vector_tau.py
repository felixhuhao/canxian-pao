"""
Vector τ Ecological Dynamics
==============================
Implements multi-channel τ with Softmax channel allocation, lateral inhibition,
and ecological niche separation among frequency bands.

Key innovations:
  - M-channel τ vector: τ ∈ ℝ^M (M = 3 channels)
  - Skill-channel affinity matrix W ∈ ℝ^{M×N}: learns which channel "owns" which skill
  - Softmax attention: each skill distributes its resonance across channels
  - Lateral inhibition: channels repel each other to maintain diversity
  - Ecological niche: channels naturally specialize in high/medium/low frequency bands

Architecture:
  τ update for channel m:
    dτ_m/dt = γ_τ · [ Σ_i w_{i,m} · (P_i - τ_m) ] + η · Σ_{n≠m} R(||τ_m - τ_n||)
  
  where w_{i,m} = softmax(W·τ_m) gives channel m's affinity for skill i,
  and R(·) is a repulsion kernel preventing channel collapse.

  W update:
    W_{m,i} += α_W · [ C_{i,m} · κ_i - λ_W · W_{m,i} ]

  Only the winning channel (argmax affinity) for each skill drives τ toward P_i.
  This prevents all channels from converging to the same value.

Predictions:
  1. Channels converge to distinct frequency bands: [~12, ~24, ~48]
  2. N_locked > original scalar τ version (fixing Table 7)
  3. Channel specialization stabilizes over training

Usage:
    python experiment_vector_tau.py --baseline
    python experiment_vector_tau.py --ecological
    python experiment_vector_tau.py --niche-sweep M=2,3,5
"""

import numpy as np
import time
import os
import sys
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import ExperimentConfig, SkillDef, sigmoid
from experiment_order import SKILL_POOL


# ============================================================
# Constants
# ============================================================

N_CHANNELS = 3  # Default number of τ channels
PERIODS = [12.0, 24.0, 24.0, 48.0, 30.0]  # C, A, B, D, E

# Frequency band targets for niche initialization
BAND_INIT = [12.0, 24.0, 48.0]  # High, Medium, Low


# ============================================================
# Vector τ Model
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(np.clip(x_shifted, -50, 50))
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-10)


class VectorTauModel:
    """
    Multi-channel τ dynamics with ecological niche separation.
    
    State:
      tau: np.ndarray (M,) — the M-channel resonance delays
      W: np.ndarray (M, N) — channel-skill affinity matrix
      kappa: np.ndarray (N,) — gate strengths per skill
      C: np.ndarray (N,) — self-correlation per skill
      PE: np.ndarray (N,) — prediction errors
    """
    
    def __init__(self, M: int = 3, N: int = 5,
                 periods: Optional[List[float]] = None,
                 tau_init: Optional[np.ndarray] = None,
                 seed: int = 42):
        self.M = M  # Number of channels
        self.N = N  # Number of skills
        self.periods = np.array(periods) if periods is not None else np.array(PERIODS[:N])
        
        # Initialize τ across frequency bands
        if tau_init is not None:
            self.tau = np.array(tau_init, dtype=float)
        else:
            # Spread across [min_period/2, max_period*1.5]
            p_min = np.min(self.periods)
            p_max = np.max(self.periods)
            if M == 1:
                self.tau = np.array([np.mean(self.periods)])
            else:
                self.tau = np.linspace(p_min * 0.8, p_max * 0.8, M)
        
        # Channel-skill affinity matrix W ∈ ℝ^{M×N}
        # Initialize with band preference: channel m prefers skills with
        # period close to its initial τ
        rng = np.random.RandomState(seed)
        self.W = rng.randn(M, N) * 0.1
        # Add band-preference bias
        for m in range(M):
            for i in range(N):
                diff = abs(self.periods[i] - self.tau[m])
                self.W[m, i] += np.exp(-diff / 10.0) * 2.0
        
        # Initial skill state
        self.kappa = np.zeros(N)
        self.C = np.zeros(N)
        self.PE = np.zeros(N)
        self.L = np.zeros(N)  # Gate activation (continuous)
        
        # History
        self.tau_traj = []
        self.kappa_traj = []
        self.W_traj = []
        self.channel_assignments_traj = []  # Which channel dominates which skill
        
    def compute_resonance(self, tau_m: float, P_i: float,
                          C_i: float, kappa_i: float) -> float:
        """Resonance between channel m and skill i."""
        # Delayed self-correlation based on match between τ and P
        # Similar to CIT's C_i update
        phase_error = abs(tau_m % P_i) / P_i
        resonance = np.exp(-4.0 * min(phase_error, 1.0 - phase_error))
        # Modulate by current κ (locked skills pay more attention)
        return resonance * (0.5 + 0.5 * kappa_i)
    
    def step(self, x_t: float, x_delayed: float,
             t: int,
             mask: np.ndarray,
             skill_periods: np.ndarray,
             skill_offsets: np.ndarray,
             params: Dict) -> Dict:
        """
        One simulation step.
        
        Parameters:
          x_t: current state
          x_delayed: state at τ-delay (for self-correlation)
          t: timestep
          mask: which skills are active
          skill_periods: P_i for each skill
          skill_offsets: phase offset for each skill
          params: dict of hyperparameters
        
        Returns:
          dict with updated state info
        """
        M = self.M
        N = self.N
        
        # --- 1. Compute channel-skill affinities ---
        # For each channel m, compute its affinity for each skill i
        # A_{m,i} = softmax(W_m)[i] · resonance(τ_m, P_i)
        channel_attention = softmax(self.W, axis=1)  # (M, N): each channel attends to skills
        
        # Resonance per (channel, skill) pair
        resonance_matrix = np.zeros((M, N))
        for m in range(M):
            for i in range(N):
                if mask[i] or self.kappa[i] > 0.1:
                    resonance_matrix[m, i] = self.compute_resonance(
                        self.tau[m], skill_periods[i],
                        self.C[i], self.kappa[i])
        
        # Combined channel-skill coupling
        # C_combined[i] = Σ_m A_{m,i} · resonance(τ_m, P_i)
        C_raw = np.sum(channel_attention * resonance_matrix, axis=0)
        
        # --- 2. Update self-correlation C_i ---
        corr = np.tanh(x_t) * np.tanh(x_delayed)
        eta_C = params.get('eta_C', 0.01)
        for i in range(N):
            if mask[i]:
                # Driving by external stimulus
                self.C[i] += eta_C * (corr - self.C[i])
            else:
                # Decay
                self.C[i] -= eta_C * 0.1 * self.C[i]
            # Also modulated by channel allocation
            self.C[i] += eta_C * 0.01 * (C_raw[i] - self.C[i])
            self.C[i] = np.clip(self.C[i], -1.0, 1.0)
        
        # --- 3. Prediction error ---
        pe_decay = params.get('pe_decay', 0.05)
        for i in range(N):
            expected = np.sin(2 * np.pi * t / skill_periods[i] + skill_offsets[i])
            if mask[i]:
                pe_instant = 0.01 * (1.0 - abs(np.tanh(x_t)))
            else:
                pe_instant = 0.5 * self.L[i] + 0.3 * self.L[i] * abs(expected)
            self.PE[i] = (1.0 - pe_decay) * self.PE[i] + pe_decay * pe_instant
        
        # --- 4. κ update (soft-gated by channel allocation) ---
        gamma = params.get('gamma', 0.006)
        theta_k = params.get('theta_k', 0.35)
        eta = params.get('eta', 0.05)
        beta_comp = params.get('beta_comp', 0.01)
        alpha_diss = params.get('alpha_diss', 0.03)
        diss_threshold = params.get('diss_threshold', 0.4)
        lam_s = params.get('lam_s', 15.0)  # Gate steepness
        
        for i in range(N):
            target = np.tanh(params.get('lam', 1.5) * abs(self.C[i]))
            base_k = self.kappa[i] + gamma * (target - self.kappa[i])
            
            # Soft gate
            gate_i = 1.0 / (1.0 + np.exp(-lam_s * (self.kappa[i] - theta_k)))
            self.L[i] = gate_i
            
            # Competition (gated)
            comp = 0.0
            for j in range(N):
                if j != i:
                    gate_j = 1.0 / (1.0 + np.exp(-lam_s * (self.kappa[j] - theta_k)))
                    pr = min(skill_periods[i], skill_periods[j]) / \
                         max(skill_periods[i], skill_periods[j])
                    if pr > 0.3:
                        comp += beta_comp * gate_i * gate_j * self.kappa[i] * self.kappa[j]
            
            # Dissolution (gated)
            diss = (alpha_diss
                    * sigmoid(10 * (self.PE[i] - diss_threshold))
                    * gate_i
                    * self.kappa[i])
            
            # Autocatalytic consolidation (gated)
            self.kappa[i] = (base_k
                             + eta * gate_i * self.kappa[i] * (1.0 - self.kappa[i])
                             - comp
                             - diss)
            self.kappa[i] = np.clip(self.kappa[i], 0.0, 1.0)
        
        # --- 5. τ update with vector dynamics ---
        gamma_tau = params.get('gamma_tau', 0.05)
        repulsion_strength = params.get('repulsion_strength', 0.5)
        repulsion_scale = params.get('repulsion_scale', 8.0)
        
        # Compute winning-channel assignment
        # For each skill i, find which channel has highest affinity
        # channel_attention[m, i] = softmax(W_m)[i]
        # Winner: argmax_m channel_attention[m, i] for each skill
        channel_pulls = np.zeros(M)
        for m in range(M):
            pull = 0.0
            w_sum = 0.0
            for i in range(N):
                # Weighted pull toward skill i's period
                affinity = channel_attention[m, i]
                weight = affinity * self.L[i]  # Gate-weighted
                pull += weight * (skill_periods[i] - self.tau[m])
                w_sum += weight
            if w_sum > 0.001:
                channel_pulls[m] = pull / max(w_sum, 0.001)
        
        # Lateral inhibition: repulsion between channels
        repulsion = np.zeros(M)
        for m in range(M):
            for n in range(M):
                if n != m:
                    delta = self.tau[m] - self.tau[n]
                    dist = abs(delta)
                    if dist < repulsion_scale and dist > 0.1:
                        # Gaussian repulsion kernel
                        rep = repulsion_strength * np.sign(delta) * \
                              np.exp(-dist**2 / (2 * (repulsion_scale/2)**2))
                        repulsion[m] += rep
        
        # τ update
        self.tau += gamma_tau * channel_pulls + repulsion
        self.tau = np.clip(self.tau, 
                           params.get('tau_min', 4),
                           params.get('tau_max', 64))
        
        # --- 6. W update (Hebbian: channel-skill affinity) ---
        alpha_W = params.get('alpha_W', 0.01)
        lambda_W = params.get('lambda_W', 0.005)
        
        for m in range(M):
            for i in range(N):
                # Hebbian update: co-activation of channel m and skill i
                hebb = resonance_matrix[m, i] * self.L[i]
                self.W[m, i] += alpha_W * (hebb - lambda_W * self.W[m, i])
        
        # --- 7. Record ---
        self.tau_traj.append(self.tau.copy())
        self.kappa_traj.append(self.kappa.copy())
        self.W_traj.append(self.W.copy())
        
        # Channel assignments: which skill is "owned" by which channel
        assignments = np.argmax(channel_attention, axis=0)  # (N,): channel index per skill
        self.channel_assignments_traj.append(assignments.copy())
        
        return {
            'tau': self.tau.copy(),
            'kappa': self.kappa.copy(),
            'C': self.C.copy(),
            'L': self.L.copy(),
            'W': self.W.copy(),
            'assignments': assignments.copy(),
            'channel_pulls': channel_pulls,
            'repulsion': repulsion,
        }


# ============================================================
# Full Simulation with Vector τ
# ============================================================

def run_vector_tau_experiment(M: int = 3,
                               seed: int = 42,
                               quick: bool = False,
                               params: Optional[Dict] = None,
                               tau_init: Optional[List[float]] = None,
                               verbose: bool = True) -> Dict:
    """
    Full multi-phase simulation with vector τ dynamics.
    """
    if params is None:
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
    
    # Skills (C-first order)
    order_names = ['C', 'A', 'B', 'D', 'E']
    periods = [12.0, 24.0, 24.0, 48.0, 30.0]
    N = len(order_names)
    
    # Phase config
    if quick:
        t_pretrain = 500
        t_phase1 = 2000
        t_phase2 = 3500
        t_phase3 = 5000
        t_phase4 = 6500
        t_phase5 = 8000
        t_silence = 9500
        t_total = 10500
    else:
        t_pretrain = 2000
        t_phase1 = 7000
        t_phase2 = 12000
        t_phase3 = 17000
        t_phase4 = 22000
        t_phase5 = 27000
        t_silence = 32000
        t_total = 33500
    
    # Phase schedule
    def phase_at(t):
        if t < t_pretrain: return 0
        elif t < t_phase1: return 1
        elif t < t_phase2: return 2
        elif t < t_phase3: return 3
        elif t < t_phase4: return 4
        elif t < t_phase5: return 5
        elif t < t_silence: return 6
        else: return 7
    
    def skill_mask(phase):
        mask = np.zeros(N, dtype=bool)
        if phase == 0: pass
        elif 1 <= phase <= 5:
            idx = phase - 1
            if idx < N: mask[idx] = True
        elif phase == 6:
            mask[N-1] = True
        return mask
    
    # Phase offsets
    offsets = [0.0, 0.0, np.pi, 0.0, 0.0]
    
    # Initialize model
    if tau_init:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed,
                                tau_init=np.array(tau_init, dtype=float))
    else:
        model = VectorTauModel(M=M, N=N, periods=periods, seed=seed)
    
    # History buffers
    x_hist = list(np.random.uniform(-0.1, 0.1, max(60, 24 + 10)))
    
    # For τ update: need a τ-delayed history
    # Vector τ means each channel has its own delay
    # We approximate by using the mean τ for the delay
    delta_current = int(np.mean(model.tau)) if M > 1 else int(model.tau[0])
    delta_current = max(delta_current, 1)
    
    t0 = time.time()
    
    for t in range(t_total):
        phase = phase_at(t)
        mask = skill_mask(phase)
        
        # Stimulus
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_active += np.sin(2 * np.pi * t / periods[i] + offsets[i])
        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active if n_active > 0 else 0.0
        
        # State update with mean τ
        delta_current = int(np.clip(round(np.mean(model.tau)), 1, 64))
        x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
        
        total_kappa = np.mean(model.kappa)
        x_t = ((1.0 - total_kappa) * np.tanh(params['alpha'] * S_active)
               + total_kappa * np.tanh(params['beta'] * x_delayed)
               + np.random.normal(0, params['sigma']))
        x_hist.append(x_t)
        
        # Vector τ step
        state = model.step(x_t, x_delayed, t, mask,
                           np.array(periods), np.array(offsets),
                           params)
    
    dt = time.time() - t0
    
    # --- Analysis ---
    tau_traj = np.array(model.tau_traj)
    kappa_traj = np.array(model.kappa_traj)
    W_traj_arr = np.array(model.W_traj)
    assign_traj = np.array(model.channel_assignments_traj)
    
    # Per-skill summary
    skill_summary = {}
    for i in range(N):
        name = order_names[i]
        k_t = kappa_traj[:, i]
        L_t = model.L if hasattr(model, 'L') else k_t
        
        # Final values
        k_final = float(np.mean(k_t[-500:]))
        L_final = float(np.mean(1.0 / (1.0 + np.exp(-15.0 * (k_t[-500:] - 0.35)))))
        
        # Find active phase
        active_phases = [p for p in range(8) if skill_mask(p)[i]]
        if active_phases:
            p_end = [t_pretrain, t_phase1, t_phase2, t_phase3,
                     t_phase4, t_phase5, t_silence, t_total][active_phases[0] + 1]
            p_start = [0, t_pretrain, t_phase1, t_phase2, t_phase3,
                       t_phase4, t_phase5, t_silence][active_phases[0]]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_active = float(np.mean(k_t[window]))
        else:
            k_at_active = k_final
        
        L_final_val = float(np.mean(L_final))
        skill_summary[name] = {
            'kappa_final': k_final,
            'L_final': L_final_val,
            'locked': L_final_val > 0.5,  # Soft gate: L > 0.5 = locked
            'kappa_at_active_end': k_at_active,
        }
    
    n_locked = sum(1 for s in skill_summary.values() if s['locked'])
    
    # τ final values
    tau_final = tau_traj[-1] if len(tau_traj) > 0 else model.tau
    
    # Channel specialization analysis
    # Which skills are assigned to which channel at the end
    final_assignments = assign_traj[-1] if len(assign_traj) > 0 else np.zeros(N, dtype=int)
    channel_skills = {m: [] for m in range(M)}
    for i in range(N):
        ch = int(final_assignments[i])
        if ch in channel_skills:
            channel_skills[ch].append(order_names[i])
    
    # τ diversity (mean pairwise distance)
    if M > 1:
        pairwise_dists = []
        for m in range(M):
            for n in range(m+1, M):
                pairwise_dists.append(abs(tau_traj[-1, m] - tau_traj[-1, n]))
        tau_diversity = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
    else:
        tau_diversity = 0.0
    
    # Ecological niche: which frequency band does each channel occupy
    tau_sort_idx = np.argsort(tau_final)
    bands = {0: 'high (~12)', 1: 'mid (~24)', 2: 'low (~48)'}
    channel_bands = {}
    for rank, idx in enumerate(tau_sort_idx):
        band_label = bands.get(rank, f'band_{rank}')
        channel_bands[f'channel_{idx}'] = {
            'tau_final': float(tau_final[idx]),
            'band': band_label,
            'skills': channel_skills.get(idx, []),
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Vector τ Experiment — M={M} channels, seed={seed}")
        print(f"{'='*60}")
        print(f"Channels: {', '.join(f'τ_{m}={tau_final[m]:.1f}' for m in range(M))}")
        print(f"τ diversity (mean pairwise distance): {tau_diversity:.1f}")
        print(f"N_locked: {n_locked}/{N}")
        print(f"\nPer-skill status:")
        for i, name in enumerate(order_names):
            s = skill_summary[name]
            ch = int(final_assignments[i])
            flag = '✓' if s['locked'] else '✗'
            print(f"  {name:<8} κ={s['kappa_final']:.3f} L={s['L_final']:.3f} "
                  f"{flag} ch={ch}")
        print(f"\nChannel ecological niches:")
        for ch_name, info in channel_bands.items():
            print(f"  {ch_name}: τ={info['tau_final']:.1f} ({info['band']}), "
                  f"skills={info['skills']}")
        print(f"Total time: {dt:.1f}s")
        print(f"{'='*60}\n")
    
    return {
        'n_locked': n_locked,
        'tau_final': tau_final.tolist(),
        'tau_diversity': tau_diversity,
        'channel_bands': channel_bands,
        'skill_summary': skill_summary,
        'trajectories': {
            'tau': tau_traj,
            'kappa': kappa_traj,
            'assignments': assign_traj,
        },
        'params': params,
    }


# ============================================================
# Experiment 1: Baseline comparison (M=1 vs M=3)
# ============================================================

def baseline_comparison(seed: int = 42, quick: bool = False,
                         verbose: bool = True) -> Dict:
    """Compare M=1 (scalar) vs M=3 (vector) τ."""
    results = {}
    
    for M in [1, 3]:
        results[M] = run_vector_tau_experiment(
            M=M, seed=seed, quick=quick, verbose=verbose)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Baseline Comparison: M=1 vs M=3")
        print(f"{'='*60}")
        print(f"{'M':<5} {'N_locked':<12} {'τ':<30} {'Diversity':<12}")
        print('-' * 60)
        for M in [1, 3]:
            r = results[M]
            tau_str = ', '.join(f'{t:.1f}' for t in r['tau_final'])
            print(f"{M:<5} {r['n_locked']:<12} [{tau_str}]{' ':<10} "
                  f"{r['tau_diversity']:<12.1f}")
    
    return results


# ============================================================
# Experiment 2: Channel number sweep
# ============================================================

def sweep_channels(M_values: List[int] = [1, 2, 3, 4],
                   seed: int = 42, quick: bool = False,
                   verbose: bool = True) -> Dict:
    """Sweep number of channels and measure N_locked + diversity."""
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Channel Number Sweep")
    print(f"{'='*60}")
    print(f"{'M':<5} {'N_locked':<12} {'τ_final':<30} {'Diversity':<12} {'Collapse?':<10}")
    print('-' * 70)
    
    for M in M_values:
        r = run_vector_tau_experiment(M=M, seed=seed, quick=quick, verbose=False)
        results[M] = r
        tau_str = ', '.join(f'{t:.1f}' for t in r['tau_final'])
        collapsed = any(abs(r['tau_final'][m] - r['tau_final'][n]) < 1.0
                        for m in range(M) for n in range(m+1, M))
        print(f"{M:<5} {r['n_locked']:<12} [{tau_str}]{' ':<10} "
              f"{r['tau_diversity']:<12.1f} {'YES!' if collapsed else 'No':<10}")
    
    return results


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if not any([args.baseline, args.sweep]):
        print("Running all vector τ experiments...")
        args.baseline = args.sweep = True
    
    if args.baseline:
        print(f"\n{'='*60}")
        print("Experiment 1: Baseline M=1 vs M=3")
        print(f"{'='*60}")
        baseline_comparison(seed=args.seed, quick=args.quick, verbose=True)
    
    if args.sweep:
        print(f"\n{'='*60}")
        print("Experiment 2: Channel Number Sweep")
        print(f"{'='*60}")
        sweep_channels(M_values=[1, 2, 3, 4], seed=args.seed,
                       quick=args.quick, verbose=True)
