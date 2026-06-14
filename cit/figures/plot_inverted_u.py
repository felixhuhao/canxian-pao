"""
Generate inverted-U plot for the paper.
Shows γ_τ vs N_locked, highlighting Mode B dominance.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_crystallization import run_experiment
from experiment_order import SKILL_POOL, make_order_cfg

order_names = ['C', 'A', 'B', 'D', 'E']

def om(phase):
    import numpy as _np
    m = _np.zeros(len(order_names), dtype=bool)
    if phase == 0: pass
    elif 1 <= phase <= 5:
        i = phase - 1
        if i < len(order_names): m[i] = True
    elif phase == 6: m[len(order_names) - 1] = True
    return m

gammas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.3]
skill_end = [7000, 12000, 17000, 22000, 27000]

# Run two conditions
results = {'η=0.008 (weakened, Mode A window)': {}, 'η=0.05 (default, Mode A suppressed)': {}}

for label, eta in [('η=0.008 (weakened, Mode A window)', 0.008), 
                    ('η=0.05 (default, Mode A suppressed)', 0.05)]:
    for gamma in gammas:
        cfg = make_order_cfg(order_names, meta_delta=True, gamma_delta=gamma, quick=False)
        cfg.delta = 24
        cfg.eta = eta
        cfg.eta_C = 0.01
        cfg.skills = [SKILL_POOL[n] for n in order_names]
        cfg.skill_mask = om
        
        r = run_experiment(cfg, seed=42, verbose=False)
        kappa = np.array(r['trajectories']['kappa'])
        
        locked = 0
        for i, name in enumerate(order_names):
            ae = skill_end[i]
            k_at_e = float(np.mean(kappa[max(ae-500,0):ae, i]))
            if k_at_e > 0.45:
                locked += 1
        
        results[label][gamma] = locked

# Fix the figure to be in the analysis directory
save_dir = os.path.dirname(os.path.abspath(__file__))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: full inverted-U attempt
x = [np.log10(g) for g in gammas]
for label, color, marker in [('η=0.008 (weakened, Mode A window)', 'red', 'o'), 
                               ('η=0.05 (default, Mode A suppressed)', 'blue', 's')]:
    y = [results[label].get(g, 0) for g in gammas]
    ax1.plot(x, y, f'-{marker}', color=color, label=label.split('(')[0], markersize=6)
    
ax1.set_xlabel('log₁₀(γ_τ)', fontsize=12)
ax1.set_ylabel('N_locked', fontsize=12)
ax1.set_title('γ_τ Scan: Inverted-U Phase Diagram\n(τ₀=24, C-first, P_C=12)', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{g:.0e}' for g in gammas], rotation=45, ha='right', fontsize=8)
ax1.set_ylim(-0.5, 5.5)

# Annotate Mode B
ax1.annotate('Mode B\n(τ drifts before\nκ accumulates)', xy=(np.log10(0.05), 2),
             xytext=(np.log10(0.1), 3.5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red')

# Right: focus on Mode B zone
gammas_b = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3]
x_b = [np.log10(g) for g in gammas_b]
for label, color, marker in [('η=0.008 (weakened, Mode A window)', 'red', 'o'), 
                               ('η=0.05 (default, Mode A suppressed)', 'blue', 's')]:
    y_b = [results[label].get(g, 0) for g in gammas_b]
    ax2.plot(x_b, y_b, f'-{marker}', color=color, label=label.split('(')[0], markersize=8, linewidth=2)

ax2.axvspan(np.log10(0.03), np.log10(0.3), alpha=0.15, color='red', label='Mode B zone')
ax2.axvspan(np.log10(0.001), np.log10(0.01), alpha=0.15, color='green', label='Stable zone')

ax2.set_xlabel('log₁₀(γ_τ)', fontsize=12)
ax2.set_ylabel('N_locked', fontsize=12)
ax2.set_title('Mode B Dominates the Phase Diagram\n(Green = stable, Red = destructive)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x_b)
ax2.set_xticklabels([f'{g:.0e}' for g in gammas_b], rotation=45, ha='right', fontsize=8)
ax2.set_ylim(-0.5, 5.5)

plt.tight_layout()
save_path = os.path.join(save_dir, 'inverted_u_phasediagram.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

# Print summary table
print()
print("=" * 70)
print("Inverted-U Summary Table")
print("=" * 70)
print(f"{'γ_τ':<12} {'η=0.008 N':<12} {'η=0.05 N':<12} {'Regime':<20}")
print("-" * 56)
for gamma in gammas:
    n1 = results['η=0.008 (weakened, Mode A window)'].get(gamma, 0)
    n2 = results['η=0.05 (default, Mode A suppressed)'].get(gamma, 0)
    if gamma <= 0.01 and n1 == 4:
        regime = 'Stable (τ frozen)'
    elif gamma >= 0.03 and n1 < 4:
        regime = 'Mode B (drift)'
    else:
        regime = 'Transitional'
    print(f"{gamma:<12.6f} {n1:<12} {n2:<12} {regime:<20}")
print()
print("Note: Mode A (left side inverted-U) does not appear in this parameter regime.")
print("Mode A requires η=0.01, α=0.05, γ=0.001-0.005, and a τ trajectory")
print("that forces anti-phase transit during initial crystallization (not pre-crystallized)")
