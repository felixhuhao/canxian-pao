# Cognitive Inertia Theorem (CIT)

Simulation code for the **Cognitive Inertia Theorem**: cognitive inertia (persistence of acquired temporal structure after stimulus perturbation) arises from the joint action of delayed self-coupling, delayed self-correlation accumulation, and threshold-gated autocatalytic consolidation in a minimal class of delayed adaptive systems.

**Paper:** *Cognitive Inertia from Structural Hysteresis in Delayed Adaptive Systems*

This project lives at `cit/` in the repository root. Commands below are run from the repository root.

## Structure

```
cit/
├── core/                          # Core model & entry points
│   ├── model.py                   — Model definition (Eqs. 1-6)
│   ├── run_phase_diagram.py       — Phase diagram scans (β×θ, η_C×η/γ)
│   ├── run_ablation.py            — Ablation analysis (3 conditions)
│   └── phase_diagram_meta.py      — Meta-parameter phase diagram utilities
│
├── experiments/                   # All paper experiments
│   ├── experiment_crystallization.py     — κ locking dynamics (Fig 1)
│   ├── experiment_continuous_dist.py     — Inertial drag + coexistence (Fig 2)
│   ├── experiment_coprime.py             — Coprime beat-frequency resonance
│   ├── experiment_coprime_v2.py          — v2: τ differentiation focus
│   ├── experiment_escape.py              — Escape time / MFPT analysis
│   ├── experiment_bifurcation.py         — (β, κ) bifurcation structure
│   ├── experiment_soft_gate.py           — Linear vs tanh gate control
│   ├── experiment_stabilize.py           — Meta-δ stabilization experiments
│   ├── experiment_meta_delta.py          — Perturbation-recovery (CIT vs AFO)
│   ├── experiment_order.py               — Order parameter analysis
│   ├── experiment_noise_sensitivity.py   — Multiplicative noise phase diagram
│   ├── experiment_cumulative.py          — Cumulative skill training
│   ├── experiment_gamma_alpha_phase.py   — γ/α phase diagram
│   ├── experiment_gamma_alpha_v2.py      — γ/α v2
│   ├── experiment_beta_refinement.py     — β refinement scan
│   ├── experiment_beta_refinement_v2.py  — β refinement v2
│   ├── experiment_beta_refinement_v3.py  — β refinement v3
│   ├── experiment_inverted_u.py          — Inverted-U phase diagram
│   ├── experiment_mode_a.py              — 1D Mode A experiment
│   ├── experiment_highdim.py             — High-dimensional deadlock
│   ├── experiment_vector_tau.py          — Vector τ ecological niche dynamics
│   ├── experiment_ai_proof.py            — AI proof-of-concept
│   ├── experiment_ai_proof_v2.py         — AI proof v2
│   ├── modma_pipeline.py                 — EEG cross-frequency coherence (MODMA)
│   └── fig4_eeg_coherence_analysis.py    — EEG Figure 4 analysis
│
├── figures/                       # Figure generation scripts
│   ├── generate_figures.py        — All main paper figures (32K)
│   ├── generate_supp_figures.py   — All supplementary figures
│   ├── plot_figure3.py            — Figure 3: niche differentiation
│   └── plot_inverted_u.py         — Inverted-U figure
│
├── data/                          # Experiment data
│   ├── mode_A_data.npz            — Mode A bifurcation data (11 MB)
│   └── bifurcation_data.npz       — Bifurcation scan data
│
├── paper/                         # Paper LaTeX source
│   ├── CIT_paper.tex              — Main paper wrapper
│   ├── CIT_body.tex               — Paper body
│   ├── CIT_supplementary.tex      — Supplementary material
│   ├── cit_refs.bib               — References
│   └── gen_cit_docx.py            — DOCX generation script
│
└── model_comparison/              # Alternative model baselines
    ├── model.py                   — CIT model reference
    └── experiment_continuous_dist.py — Virtual sensorimotor comparison
```

## Quick Start

```bash
# Core model run
python3 cit/core/model.py

# Ablation analysis (100 seeds per condition)
python3 cit/core/run_ablation.py

# Phase diagram scan (η_C × η/γ, 6830 points)
python3 cit/core/run_phase_diagram.py

# Full figure generation
python3 cit/figures/generate_figures.py
```

## Requirements

- Python 3.11+
- NumPy, SciPy, Matplotlib
- (Optional) python-docx, librosa, mne (for EEG pipeline)

## Model Parameters

| Param | Default | Description |
|-------|---------|-------------|
| α | 1.8 | Stimulus coupling gain |
| β | 2.2 | Self-coupling gain |
| δ | 24 (steps) | Internal delay |
| γ | 0.006 | κ relaxation rate |
| η | 0.05 | Autocatalytic consolidation strength |
| η_C | 0.01 | Correlation accumulation rate |
| λ | 1.5 | Correlation-to-target scaling |
| θ_κ | 0.35 | Autocatalytic activation threshold |
| σ | 0.015 | Noise amplitude |

## Citation

```
Cai, H. & Cai, T. Cognitive Inertia from Structural Hysteresis in Delayed Adaptive Systems
```
