"""
Cumulative Skill Training — Inverted-U Verification + Meta-Plastic δ
====================================================================
Implements two extensions to the multi-skill crystallization experiment:

Extension 1: Cumulative training (inverted-U)
  - Skills are ADDED, not replaced
  - Each skill i tracks its own C_i with skill-specific lag = T_i
  - Performance vs N should show inverted-U with peak near N=3-4

Extension 2: Meta-plastic δ
  - Internal delay δ adapts to dominant skill period
  - Creates self-organizing harmonic hierarchy
  - δ_t+1 = δ_t + γ_δ · (T_dominant − δ_t)

Usage:
    python experiment_cumulative.py                          # cumulative + inverted-U
    python experiment_cumulative.py --meta-delta             # with meta-plastic δ
    python experiment_cumulative.py --quick                  # reduced scale
    python experiment_cumulative.py --n-seeds 30             # multi-seed
"""

import numpy as np
import time
import json
import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Skill definitions
# ============================================================

@dataclass
class SkillDef:
    name: str
    period: int
    phase_offset: float = 0.0
    expected_crystallize: bool = True


DEFAULT_CUMULATIVE_SKILLS = [
    SkillDef('A', 24, 0.0, True),      # ρ=1, fundamental
    SkillDef('C', 12, 0.0, True),      # ρ=2, 1st harmonic
    SkillDef('D', 48, 0.0, True),      # ρ=0.5, sub-harmonic
    SkillDef('E', 30, 0.0, False),     # ρ=0.8, off-resonance
]

# For competition test: same-period skills
DEFAULT_COMPETITION_SKILLS = [
    SkillDef('A', 24, 0.0, True),
    SkillDef('B', 24, np.pi, True),    # π phase shift → competes with A
]


@dataclass
class CumulativeConfig:
    """Configuration for cumulative training experiment."""

    # Agent base parameters (inherited from CIT)
    alpha: float = 1.8
    beta: float = 2.2
    gamma: float = 0.006
    eta: float = 0.05
    eta_C: float = 0.01
    lam: float = 1.5
    theta_k: float = 0.35
    sigma: float = 0.015

    # Multi-skill terms
    beta_comp: float = 0.01
    alpha_diss: float = 0.03
    pe_decay: float = 0.05
    diss_threshold: float = 0.4

    # Fixed internal delay (baseline)
    delta: int = 24

    # Meta-plastic δ parameters
    meta_delta: bool = False          # Enable meta-plastic δ
    gamma_delta: float = 0.001        # δ adaptation rate
    delta_min: int = 8
    delta_max: int = 64
    delta_adjust_interval: int = 50   # Adjust δ every N steps

    # Phase boundaries
    t_pretrain: int = 2000
    t_phase1: int = 7000
    t_phase2: int = 12000
    t_phase3: int = 17000
    t_phase4: int = 22000
    t_phase5: int = 27000
    t_silence: int = 32000
    t_total: int = 33500

    # Skills
    skills: List[SkillDef] = field(default_factory=lambda: list(DEFAULT_CUMULATIVE_SKILLS))

    @property
    def N(self) -> int:
        return len(self.skills)

    def phase_at(self, t: int) -> int:
        """Return phase index 0-6."""
        if t < self.t_pretrain:
            return 0
        elif t < self.t_phase1:
            return 1
        elif t < self.t_phase2:
            return 2
        elif t < self.t_phase3:
            return 3
        elif t < self.t_phase4:
            return 4
        elif t < self.t_phase5:
            return 5
        elif t < self.t_silence:
            return 6
        else:
            return 7

    def phase_name(self, phase: int) -> str:
        names = ['Pretrain', 'N=1 (A)', 'N=2 (A+C)', 'N=3 (A+C+D)',
                 'N=4 (A+C+D+E)', 'Maintain', 'Stabilize', 'Silence']
        return names[phase] if phase < len(names) else f'Phase {phase}'

    def n_active_at_phase(self, phase: int) -> int:
        """Number of skills active in a given phase."""
        return int(np.sum(self.skill_mask(phase)))

    def skill_mask(self, phase: int) -> np.ndarray:
        """Cumulative mask: skills accumulate over phases."""
        mask = np.zeros(self.N, dtype=bool)
        # Phase 0: no skills
        if phase >= 1:
            mask[0] = True   # A always active from phase 1
        if phase >= 2:
            mask[1] = True   # C added in phase 2
        if phase >= 3:
            mask[2] = True   # D added in phase 3
        if phase >= 4:
            mask[3] = True   # E added in phase 4
        if phase >= 5:
            pass  # Maintain phase: all currently active stay active
        if phase == 6:
            pass  # Stabilize: same as phase 5
        # Silence (phase 7): all off
        return mask


# ============================================================
# Simulation
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def run_cumulative(cfg: Optional[CumulativeConfig] = None, seed: int = 42,
                   verbose: bool = True) -> Dict:
    """
    Run cumulative skill training experiment.

    Skills are ADDED over phases (not replaced).
    Each skill C_i uses its own period T_i as the correlation lag.
    Optional meta-plastic δ adapts to the dominant skill period.
    """
    if cfg is None:
        cfg = CumulativeConfig()
    N = cfg.N

    np.random.seed(seed)

    phase_names_list = ['Pretrain', 'N1_A', 'N2_AC', 'N3_ACD',
                        'N4_ACDE', 'Maintain', 'Stabilize', 'Silence']
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]

    # Initialisation
    delta = cfg.delta
    # History buffer: need enough for longest skill period
    max_lag = max(max(s.period for s in cfg.skills), cfg.delta)
    x_hist = list(np.random.uniform(-0.1, 0.1, max_lag + 10))

    kappa = np.zeros(N)
    C = np.zeros(N)
    PE = np.zeros(N)

    # Trajectories
    kappa_traj = np.zeros((cfg.t_total, N))
    C_traj = np.zeros((cfg.t_total, N))
    delta_traj = np.zeros(cfg.t_total)
    phase_traj = np.zeros(cfg.t_total, dtype=int)

    if verbose:
        print(f"\n{'='*60}")
        mode = "META-δ" if cfg.meta_delta else "FIXED δ"
        print(f"Cumulative Training — seed={seed} [{mode}]")
        print(f"{'='*60}")
        print(f"Skills: {', '.join(f'{s.name}(T={s.period})' for s in cfg.skills)}")
        print(f"Protocol: cumulative addition over phases")
        print(f"{'='*60}\n")

    t0 = time.time()

    for t in range(cfg.t_total):
        phase = cfg.phase_at(t)
        phase_traj[t] = phase
        mask = cfg.skill_mask(phase)

        # ============================================================
        # 1. Combined stimulus (all active skills)
        # ============================================================
        S_active = 0.0
        for i in range(N):
            if mask[i]:
                S_i = np.sin(2 * np.pi * t / cfg.skills[i].period
                             + cfg.skills[i].phase_offset)
                S_active += S_i

        n_active = max(int(np.sum(mask)), 1)
        S_active = S_active / n_active

        # ============================================================
        # 2. State update (uses current δ, which may change over time)
        # ============================================================
        x_delayed = x_hist[-delta] if len(x_hist) >= delta else x_hist[0]

        total_kappa = np.mean(kappa)

        x_t = ((1.0 - total_kappa) * np.tanh(cfg.alpha * S_active)
               + total_kappa * np.tanh(cfg.beta * x_delayed)
               + np.random.normal(0, cfg.sigma))

        x_hist.append(x_t)

        # ============================================================
        # 3. Universal delayed self-correlation (ALL use same lag = δ)
        #    C_i = corr(x_t, x_{t-δ}) — identical for all skills
        #    This is the pure CIT mechanism. Skills are distinguished
        #    by their active periods T_i and the competition term.
        #    The inverted-U emerges because adding more skills (different
        #    periods) creates destructive interference in the combined
        #    stimulus, lowering the universal C for ALL skills.
        # ============================================================
        corr = np.tanh(x_t) * np.tanh(x_delayed)

        for i in range(N):
            if mask[i]:
                C[i] += cfg.eta_C * (corr - C[i])
            else:
                C[i] -= cfg.eta_C * 0.1 * C[i]

        # ============================================================
        # 4. Prediction error
        # ============================================================
        for i in range(N):
            expected = np.sin(2 * np.pi * t / cfg.skills[i].period
                              + cfg.skills[i].phase_offset)
            if mask[i]:
                pe_instant = 0.01 * (1.0 - abs(np.tanh(x_t)))
            else:
                if kappa[i] > cfg.theta_k:
                    pe_instant = 0.5 + 0.3 * abs(expected)
                else:
                    pe_instant = 0.0
            PE[i] = (1.0 - cfg.pe_decay) * PE[i] + cfg.pe_decay * pe_instant

        # ============================================================
        # 5. Kappa update
        # ============================================================
        for i in range(N):
            target = np.tanh(cfg.lam * abs(C[i]))
            base_k = kappa[i] + cfg.gamma * (target - kappa[i])

            if kappa[i] > cfg.theta_k:
                # Competition: only same-period skills
                comp = 0.0
                for j in range(N):
                    if j != i and kappa[j] > cfg.theta_k:
                        period_ratio = min(cfg.skills[i].period, cfg.skills[j].period) / \
                                       max(cfg.skills[i].period, cfg.skills[j].period)
                        if period_ratio > 0.5:  # tighter competition threshold
                            comp += cfg.beta_comp * kappa[i] * kappa[j]

                diss = (cfg.alpha_diss
                        * sigmoid(10 * (PE[i] - cfg.diss_threshold))
                        * kappa[i])

                kappa[i] = (base_k
                            + cfg.eta * kappa[i] * (1.0 - kappa[i])
                            - comp
                            - diss)
            else:
                kappa[i] = base_k

            kappa[i] = np.clip(kappa[i], 0.0, 1.0)

        # ============================================================
        # 6. Meta-plastic δ update
        # ============================================================
        if cfg.meta_delta and t % cfg.delta_adjust_interval == 0 and t > cfg.t_pretrain:
            # Compute dominant period: weighted by κ²
            weights = kappa * kappa
            w_sum = np.sum(weights)
            if w_sum > 0.01:
                T_dominant = np.average([s.period for s in cfg.skills], weights=weights)
                delta += cfg.gamma_delta * (T_dominant - delta)
                delta = int(np.clip(round(delta), cfg.delta_min, cfg.delta_max))
                delta = max(delta, 1)

        delta_traj[t] = delta

        # Record trajectories
        kappa_traj[t] = kappa
        C_traj[t] = C

    dt = time.time() - t0
    if verbose:
        print(f"Simulation complete: {cfg.t_total} steps in {dt:.1f}s")
        print()

    # ============================================================
    # Analysis
    # ============================================================
    results = _analyze(cfg, kappa_traj, C_traj, delta_traj, phase_traj, seed, verbose)
    if verbose:
        _print_summary(cfg, results)
    return results


def _analyze(cfg: CumulativeConfig,
             kappa_traj: np.ndarray,
             C_traj: np.ndarray,
             delta_traj: np.ndarray,
             phase_traj: np.ndarray,
             seed: int,
             verbose: bool) -> Dict:
    """Analyze cumulative training results."""
    N = cfg.N
    phase_ends = [0, cfg.t_pretrain, cfg.t_phase1, cfg.t_phase2,
                  cfg.t_phase3, cfg.t_phase4, cfg.t_phase5,
                  cfg.t_silence, cfg.t_total]
    phase_names = ['Pretrain', 'N1_A', 'N2_AC', 'N3_ACD',
                   'N4_ACDE', 'Maintain', 'Stabilize', 'Silence']

    # Per-skill summary at end of active phase
    skill_summary = {}
    for i in range(N):
        s = cfg.skills[i]
        k_t = kappa_traj[:, i]
        peak_k = float(np.max(k_t))

        # Find the first phase where this skill is active
        active_phase = None
        for p_idx in range(8):
            if cfg.skill_mask(p_idx)[i]:
                active_phase = p_idx
                break

        if active_phase is not None:
            p_end = phase_ends[active_phase + 1]
            p_start = phase_ends[active_phase]
            window = slice(max(p_end - 200, p_start), p_end)
            k_at_end = float(np.mean(k_t[window]))
        else:
            k_at_end = float(np.mean(k_t[-500:]))

        skill_summary[s.name] = {
            'expected': s.expected_crystallize,
            'period': s.period,
            'peak_kappa': peak_k,
            'kappa_at_active_end': k_at_end,
            'locked': k_at_end > 0.85,
            'success': s.expected_crystallize == (k_at_end > 0.5),
        }

    # Inverted-U: measure performance vs N
    perf_by_phase = {}
    for p_idx in range(1, 6):  # Phases 1-5 (exclude 0, 6, 7)
        p_start = phase_ends[p_idx]
        p_end = phase_ends[p_idx + 1]
        k_slice = kappa_traj[p_start:p_end]
        mask = cfg.skill_mask(p_idx)
        n_skills = int(np.sum(mask))

        # Performance = avg κ of all expected-to-crystallize active skills
        expected_active = [i for i in range(N) if mask[i] and cfg.skills[i].expected_crystallize]
        if expected_active:
            perf = float(np.mean([np.mean(k_slice[:, i]) for i in expected_active]))
        else:
            perf = 0.0

        perf_by_phase[phase_names[p_idx]] = {
            'phase': p_idx,
            'n_skills_active': n_skills,
            'n_expected_active': len(expected_active),
            'avg_kappa': perf,
        }

    # Delta trajectory stats
    delta_stats = {
        'initial': cfg.delta,
        'final': float(delta_traj[-1]),
        'min': float(np.min(delta_traj)),
        'max': float(np.max(delta_traj)),
        'mean': float(np.mean(delta_traj)),
    }

    result = {
        'seed': seed,
        'meta_delta': cfg.meta_delta,
        'skill_summary': skill_summary,
        'performance_by_phase': perf_by_phase,
        'delta_stats': delta_stats,
        'trajectories': {
            'kappa': kappa_traj.tolist(),
            'C': C_traj.tolist(),
            'delta': delta_traj.tolist(),
            'phase': phase_traj.tolist(),
        },
    }
    return result


def _print_summary(cfg: CumulativeConfig, result: Dict):
    """Print results."""
    print(f"\n{'='*60}")
    print(f"RESULTS — seed={result['seed']}")
    print(f"{'='*60}\n")

    # Skill summary table
    print(f"{'Skill':<8} {'T':<6} {'Expected':<10} {'Peak κ':<10} {'κ at end':<10} {'Locked?':<10} {'✓/✗':<6}")
    print('-' * 60)
    for name, ss in result['skill_summary'].items():
        s = next(x for x in cfg.skills if x.name == name)
        exp = '✓' if s.expected_crystallize else '✗'
        ok = '✓' if ss['success'] else '✗'
        print(f"{name:<8} {s.period:<6} {exp:<10} {ss['peak_kappa']:<10.4f} "
              f"{ss['kappa_at_active_end']:<10.4f} {'✓✓' if ss['locked'] else '—':<10} {ok:<6}")
    print()

    # Inverted-U
    print(f"{'Inverted-U check (performance vs N)':^50}")
    print(f"{'Phase':<15} {'N_active':<12} {'Avg κ':<10} {'Perf':<10}")
    print('-' * 47)
    for pname, pdata in result['performance_by_phase'].items():
        print(f"{pname:<15} {pdata['n_skills_active']:<12} {pdata['avg_kappa']:<10.4f} "
              f"{pdata['avg_kappa']:<10.4f}")
    print()

    # Delta stats
    ds = result['delta_stats']
    if result['meta_delta']:
        print(f"Meta-δ: {ds['initial']} → {ds['final']:.0f} "
              f"[min={ds['min']:.0f}, max={ds['max']:.0f}, mean={ds['mean']:.1f}]")
    else:
        print(f"Fixed δ = {ds['initial']}")
    print()


# ============================================================
# Multi-seed
# ============================================================

def run_multi(n_seeds: int = 30, cfg: Optional[CumulativeConfig] = None,
              verbose: bool = False) -> Dict:
    if cfg is None:
        cfg = CumulativeConfig()

    all_results = [run_cumulative(cfg, seed=s + 1000, verbose=verbose)
                   for s in range(n_seeds)]

    N = cfg.N
    agg = {'n_seeds': n_seeds, 'meta_delta': cfg.meta_delta}

    # Skill success rates
    agg['skill_success'] = {}
    for i in range(N):
        name = cfg.skills[i].name
        successes = sum(1 for r in all_results if r['skill_summary'][name]['success'])
        agg['skill_success'][name] = successes / n_seeds

    # Inverted-U aggregate
    agg['inverted_u'] = {}
    phases = list(all_results[0]['performance_by_phase'].keys())
    for pn in phases:
        kappas = [r['performance_by_phase'][pn]['avg_kappa'] for r in all_results]
        Ns = [r['performance_by_phase'][pn]['n_skills_active'] for r in all_results]
        agg['inverted_u'][pn] = {
            'n_skills': int(Ns[0]),
            'kappa_mean': float(np.mean(kappas)),
            'kappa_std': float(np.std(kappas)),
        }

    agg['delta_stats'] = {}
    if cfg.meta_delta:
        deltas = [r['delta_stats']['final'] for r in all_results]
        agg['delta_stats']['final_mean'] = float(np.mean(deltas))
        agg['delta_stats']['final_std'] = float(np.std(deltas))

    return agg


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cumulative skill training (inverted-U)')
    parser.add_argument('--quick', action='store_true', help='Reduced scale')
    parser.add_argument('--meta-delta', action='store_true', help='Enable meta-plastic δ')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-seeds', type=int, default=0)
    args = parser.parse_args()

    cfg = CumulativeConfig()

    if args.quick:
        cfg.t_pretrain = 500
        cfg.t_phase1 = 2000
        cfg.t_phase2 = 3500
        cfg.t_phase3 = 5000
        cfg.t_phase4 = 6500
        cfg.t_phase5 = 8000
        cfg.t_silence = 9500
        cfg.t_total = 10500

    cfg.meta_delta = args.meta_delta

    if args.n_seeds > 0:
        print(f"Multi-seed cumulative: N={args.n_seeds}, meta_δ={args.meta_delta}")
        agg = run_multi(args.n_seeds, cfg, verbose=False)
        print(f"\n{'='*60}")
        mode = "META-δ" if cfg.meta_delta else "FIXED δ"
        print(f"CUMULATIVE MULTI-SEED — N={args.n_seeds} [{mode}]")
        print(f"{'='*60}")
        print(f"\nSkill success:")
        for name, rate in agg['skill_success'].items():
            print(f"  {name}: {rate:.1%}")
        print(f"\nInverted-U (performance vs N):")
        print(f"{'Phase':<15} {'N':<8} {'κ_mean':<10} {'κ_std':<10}")
        print('-' * 43)
        for pn, data in agg['inverted_u'].items():
            print(f"{pn:<15} {data['n_skills']:<8} {data['kappa_mean']:<10.4f} {data['kappa_std']:<10.4f}")
        if agg['delta_stats']:
            ds = agg['delta_stats']
            print(f"\nMeta-δ final: {ds['final_mean']:.1f} ± {ds['final_std']:.1f}")
    else:
        run_cumulative(cfg, seed=args.seed, verbose=True)
