"""
Inverted-U Phase Diagram: γ_τ scan → N_locked
==============================================
Scans γ_τ from 1e-5 to 1e0 with weakened η (0.008) to expose Mode A.
Starting τ₀=18 (in anti-phase zone between P=12 and P=24).
Uses C-first order: C → A → B → D → E.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_crystallization import ExperimentConfig, SkillDef, run_experiment, sigmoid
from experiment_order import SKILL_POOL, make_order_cfg

# Skill names
order_names = ['C', 'A', 'B', 'D', 'E']

# γ values: log-spaced from 1e-5 to 3e-1
gammas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.3]

# Order mask (reused)
def order_mask(phase):
    mask = np.zeros(len(order_names), dtype=bool)
    if phase == 0: pass
    elif 1 <= phase <= 5:
        idx = phase - 1
        if idx < len(order_names): mask[idx] = True
    elif phase == 6:
        mask[len(order_names) - 1] = True
    return mask

results = {}
print(f"\n{'='*80}")
print(f"Inverted-U: γ_τ scan (η=0.008, τ₀=24, C-first)")
print(f"C target P=12; τ drifts 24→12, passes through anti-phase zone (τ=18)")
print(f"{'='*80}")
print(f"{'γ_τ':<12} {'N_locked':<10} {'κ_C':<10} {'κ_A':<10} {'κ_B':<10} {'κ_D':<10} {'κ_E':<10} {'τ_final':<10}")
print('-' * 80)

for gamma in gammas:
    cfg = make_order_cfg(order_names, meta_delta=True, gamma_delta=gamma, quick=False)
    cfg.delta = 24  # Start at τ₀=24 (resonant with C; 24/12=2.0)
    cfg.eta = 0.008   # Weaken η to expose Mode A
    cfg.eta_C = 0.01  # α_ρ = 0.01 (default)
    cfg.skills = [SKILL_POOL[n] for n in order_names]
    cfg.skill_mask = order_mask
    
    r = run_experiment(cfg, seed=42, verbose=False)
    kappa_finals = {}
    for name in order_names:
        info = r['skill_summary'].get(name, {})
        kappa_finals[name] = info.get('final_kappa', 0.0) if isinstance(info, dict) else 0.0
    locked_count = sum(1 for n in order_names if r['skill_summary'].get(n, {}).get('locked', False))
    tau_final = r['delta_stats']['final']
    results[gamma] = {'N_locked': locked_count, 'kappa': kappa_finals, 'tau_final': tau_final}
    
    print(f"{gamma:<12.6f} {locked_count:<10} "
          f"{kappa_finals.get('C', 0):<10.3f} {kappa_finals.get('A', 0):<10.3f} "
          f"{kappa_finals.get('B', 0):<10.3f} {kappa_finals.get('D', 0):<10.3f} "
          f"{kappa_finals.get('E', 0):<10.3f} {tau_final:<10.0f}")

print()

print(f"\n{'='*80}")
print(f"Reference: γ_τ scan (η=0.05, τ₀=24, C-first) — Mode A should NOT appear")
print(f"{'='*80}")
print(f"{'γ_τ':<12} {'N_locked':<10} {'κ_C':<10} {'κ_A':<10} {'κ_B':<10} {'κ_D':<10} {'κ_E':<10} {'τ_final':<10}")
print('-' * 80)

for gamma in gammas:
    cfg2 = make_order_cfg(order_names, meta_delta=True, gamma_delta=gamma, quick=False)
    cfg2.delta = 24
    cfg2.eta = 0.05
    cfg2.eta_C = 0.01
    cfg2.skills = [SKILL_POOL[n] for n in order_names]
    cfg2.skill_mask = order_mask
    
    r2 = run_experiment(cfg2, seed=42, verbose=False)
    kappa_finals2 = {}
    for name in order_names:
        info = r2['skill_summary'].get(name, {})
        kappa_finals2[name] = info.get('final_kappa', 0.0) if isinstance(info, dict) else 0.0
    locked_count2 = sum(1 for n in order_names if r2['skill_summary'].get(n, {}).get('locked', False))
    tau_final2 = r2['delta_stats']['final']
    
    print(f"{gamma:<12.6f} {locked_count2:<10} "
          f"{kappa_finals2.get('C', 0):<10.3f} {kappa_finals2.get('A', 0):<10.3f} "
          f"{kappa_finals2.get('B', 0):<10.3f} {kappa_finals2.get('D', 0):<10.3f} "
          f"{kappa_finals2.get('E', 0):<10.3f} {tau_final2:<10.0f}")
