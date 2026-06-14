"""
Figure 3: Vector τ ecological niche differentiation trajectory.
The "cell differentiation" style plot — the paper's flagship figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_vector_tau import run_vector_tau_experiment

# --- Run the experiment ---
params = {
    'eta_C': 0.01, 'gamma': 0.006, 'theta_k': 0.30, 'eta': 0.08,
    'beta_comp': 0.01, 'alpha_diss': 0.03, 'diss_threshold': 0.4,
    'lam': 1.5, 'lam_s': 12.0, 'gamma_tau': 0.03,
    'repulsion_strength': 0.3, 'repulsion_scale': 8.0,
    'alpha_W': 0.01, 'lambda_W': 0.005,
    'tau_min': 4, 'tau_max': 64, 'pe_decay': 0.05, 'alpha': 1.8, 'beta': 2.2, 'sigma': 0.015,
}

# Fix: use quick mode for faster plotting, but we need the trajectory
# The standard run doesn't give us sub-step resolution easily.
# Let me directly use the model for finer granularity.

np.random.seed(42)
from experiment_vector_tau import VectorTauModel

M, N = 3, 5
periods = [12.0, 24.0, 24.0, 48.0, 30.0]
order_names = ['C', 'A', 'B', 'D', 'E']
offsets = [0.0, 0.0, np.pi, 0.0, 0.0]

# Deadlock start
tau_init = np.array([16.0, 16.0, 16.0], dtype=float)
model = VectorTauModel(M=M, N=N, periods=periods, seed=42, tau_init=tau_init)

# Phase config (approximate for quick run)
t_total = 10500  # Quick mode length
phase_ends = [0, 500, 2000, 3500, 5000, 6500, 8000, 9500, 10500]

def phase_at(t):
    for p, end in enumerate(phase_ends):
        if t < end: return p
    return 7

def skill_mask(phase):
    mask = np.zeros(N, dtype=bool)
    if phase == 0: pass
    elif 1 <= phase <= 5:
        idx = phase - 1
        if idx < N: mask[idx] = True
    elif phase == 6:
        mask[N-1] = True
    return mask

# Run simulation
x_hist = list(np.random.uniform(-0.1, 0.1, 60))
tau_history = []
kappa_history = []
assign_history = []
t_history = []

for t in range(t_total):
    phase = phase_at(t)
    mask = skill_mask(phase)
    
    S_active = 0.0
    for i in range(N):
        if mask[i]:
            S_active += np.sin(2 * np.pi * t / periods[i] + offsets[i])
    n_active = max(int(np.sum(mask)), 1)
    S_active = S_active / n_active
    
    delta_current = int(np.clip(round(np.mean(model.tau)), 1, 64))
    x_delayed = x_hist[-delta_current] if len(x_hist) >= delta_current else x_hist[0]
    total_kappa = np.mean(model.kappa)
    x_t = ((1.0 - total_kappa) * np.tanh(1.8 * S_active)
           + total_kappa * np.tanh(2.2 * x_delayed)
           + np.random.normal(0, 0.015))
    x_hist.append(x_t)
    
    state = model.step(x_t, x_delayed, t, mask,
                        np.array(periods), np.array(offsets), params)
    
    if t % 50 == 0:  # Record every 50 steps
        tau_history.append(model.tau.copy())
        kappa_history.append(model.kappa.copy())
        assign_history.append(state['assignments'].copy())
        t_history.append(t)

tau_traj = np.array(tau_history)
kappa_traj = np.array(kappa_history)
assign_traj = np.array(assign_history)

# --- Plot ---
fig = plt.figure(figsize=(14, 10))

# Define phase boundaries for annotation
phase_ends_t = [0, 500, 2000, 3500, 5000, 6500, 8000, 9500, 10500]
phase_labels = ['Pre', 'C', 'A', 'B', 'D', 'E', 'Stab', 'Sil']

# ── Panel A: τ differentiation trajectories (main panel) ──
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30,
                      height_ratios=[1.5, 1.0, 1.0])

ax1 = fig.add_subplot(gs[0, :])
colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
labels = ['Channel 1', 'Channel 2', 'Channel 3']

# Add phase background shading
for p_idx in range(1, 7):
    ax1.axvspan(phase_ends_t[p_idx], phase_ends_t[p_idx+1],
                alpha=0.08, color=f'C{p_idx-1}',
                label=phase_labels[p_idx] if p_idx <= 5 else None)

# Plot each channel
for m in range(M):
    ax1.plot(t_history, tau_traj[:, m], '-', color=colors[m],
             linewidth=2.5, label=labels[m], alpha=0.9)
    # Final value annotation
    ax1.axhline(y=tau_traj[-1, m], color=colors[m], linestyle='--',
                linewidth=0.8, alpha=0.4)

# Annotate final τ values
for m in range(M):
    ax1.annotate(f'τ={tau_traj[-1,m]:.1f}',
                 xy=(t_history[-1], tau_traj[-1, m]),
                 xytext=(t_history[-1] + 200, tau_traj[-1, m] + 1.5),
                 fontsize=10, color=colors[m], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=colors[m], alpha=0.6))

# Add skill period reference lines
for i, name in enumerate(['C', 'A/B', 'D', 'E']):
    p = [12, 24, 48, 30][i]
    ax1.axhline(y=p, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.text(t_history[-1] + 100, p + 0.5, f'$P_{{{name}}}$',
             fontsize=8, color='gray', alpha=0.6)

ax1.set_ylabel('τ (delay steps)', fontsize=12)
ax1.set_title('A. Channel Ecological Niche Differentiation', fontsize=13, fontweight='bold', loc='left')
ax1.set_xlim(0, t_total + 500)
ax1.set_ylim(4, 50)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.15)

# ── Panel B: κ crystallization trajectories ──
ax2 = fig.add_subplot(gs[1, :])
skill_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB']
skill_names = ['C (P=12)', 'A (P=24)', 'B (P=24)', 'D (P=48)', 'E (P=30)']

for i in range(N):
    ax2.plot(t_history, kappa_traj[:, i], '-', color=skill_colors[i],
             linewidth=1.5, label=skill_names[i], alpha=0.8)

# Lock threshold
ax2.axhline(y=0.45, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.text(100, 0.47, 'Lock threshold', fontsize=8, color='red', alpha=0.7)

ax2.set_ylabel('κ (gate strength)', fontsize=12)
ax2.set_title('B. Skill Crystallization Dynamics', fontsize=13, fontweight='bold', loc='left')
ax2.set_xlim(0, t_total + 100)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
ax2.grid(True, alpha=0.15)

# ── Panel C: Ablation comparison (inset) ──
ax3 = fig.add_subplot(gs[2, :])

# Run ablation experiments
abl_configs = [
    ('Full M=3', dict(params), True),
    ('No lateral inhibition', dict(params, repulsion_strength=0.0), False),
]
abl_results = []

for label, p, verbose in abl_configs:
    r = run_vector_tau_experiment(M=3, seed=42, quick=True, params=p, verbose=False)
    tau_str = ', '.join(f'{t:.1f}' for t in r['tau_final'])
    abl_results.append((label, r['n_locked'], tau_str, r['tau_diversity']))

# Bar chart
bar_labels = [r[0] for r in abl_results]
bar_N = [r[1] for r in abl_results]
bar_colors = ['#2ECC71', '#E74C3C']

bars = ax3.bar(range(len(bar_labels)), bar_N, color=bar_colors, width=0.5, edgecolor='black')

# Annotate with τ values
for i, (label, n, tau_str, div) in enumerate(abl_results):
    ax3.text(i, n + 0.1, f'τ=[{tau_str}]\ndiversity={div:.1f}',
             ha='center', fontsize=9, fontweight='bold',
             color=bar_colors[i])

ax3.set_xticks(range(len(bar_labels)))
ax3.set_xticklabels(bar_labels, fontsize=10)
ax3.set_ylabel('N_locked (out of 5)', fontsize=12)
ax3.set_title('C. Ablation: Lateral Inhibition is Critical', fontsize=13, fontweight='bold', loc='left')
ax3.set_ylim(0, 6.5)
ax3.grid(True, alpha=0.15, axis='y')

plt.suptitle('Multi-Channel τ Self-Organizes Into Ecological Niches',
             fontsize=15, fontweight='bold', y=1.01)

save_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(save_dir, 'figure3_niche_differentiation.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ Figure 3 saved: {save_path}")
