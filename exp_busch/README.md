# exp_busch — Direction 2: the intrinsic-manifold hypothesis for BCI learning

Independent, in-silico test of **Busch, Fincke, Lajoie, Krishnaswamy & Turk-Browne**,
*"Accelerated learning of a noninvasive human brain-computer interface via manifold geometry"*
(bioRxiv 2025.03.29.646109, `../docs/2025.03.29.646109v1.full.pdf`). A separate group from
Cai/RDD — so this is a second, unrelated "geometry helps" framework to probe with the project's
`property ≠ usefulness` lens.

## What the paper claims
- Learn each subject's **intrinsic neural manifold** from fMRI (T-PHATE, a nonlinear diffusion
  embedding), make it real-time-extensible with a **manifold-regularized autoencoder (MRAE)**, then
  take three readout directions via **PCA over the latent**: **IM** = top eigenvector, **WMP** = 2nd,
  **OMP** = 20th (lowest variance).
- A BCI maps the chosen component → avatar heading; participants re-learn control over a session.
  Learning metric = **ΔBrainControl** (staircased control gain, first→last trial).
- **Result:** fast learning for IM, slower-but-real for WMP (both "on-manifold"), **total failure for
  OMP** ("off-manifold"). Proposed mechanism = **neural realignment** = raising the percent variance
  explained (**PEV**) along the trained component. (Realignment predicted learning only in WMP, n.s. in
  IM/OMP.) Nonlinear T-PHATE is needed for *decoding/visualization*; the IM/WMP/OMP directions
  themselves are linear PCs of the latent.

## The two soft spots (both testable purely in silico — no fMRI)
1. **Geometry vs. variance confound.** OMP *is* the lowest-variance direction, and "on-manifold-ness" is
   operationalized as PEV (`CᵀΣC`). A low-variance readout has low SNR and little controllable signal —
   so the result is equally consistent with "on/off-manifold = high/low **controllable, readable signal
   variance**," which needs no nonlinear geometry. Same move as RDD's `diversity ≠ usefulness`.
2. **Nonlinear necessity is asserted, not shown for the *learning* claim.** T-PHATE is required for
   decoding, but the IM/WMP/OMP directions are linear PCs. Does the learning asymmetry need nonlinear
   geometry at all, or would plain PCA reproduce IM > WMP > OMP?

## Planned experiments

### E1 — Variance-vs-geometry deconfound  *(first; pre-registered in `../PREREG_BUSCH_E1.md`)*
Linear–Gaussian "brain": agent learns a policy to drive a BCI readout under a **plasticity/metabolic
prior** (states near the natural distribution are cheap to produce). PEV silently bundles two separable
quantities the single-manifold design can't separate:
- **Readability** — signal the decoder recovers along `C` (∝ `Cᵀ Σ_obs C` / noise).
- **Controllability** — cost to volitionally produce amplitude along `C` (∝ `Cᵀ Σ_act⁻¹ C`).

Busch assume `Σ_obs = Σ_act` (observation manifold = volitional-reachability set), collapsing both onto
PEV. E1 breaks that with diagnostic cells: **D1** = on-manifold (high PEV) but uncontrollable → predicted
to **fail**; **D2** = off-manifold (low PEV) but readable+controllable → predicted to **learn**; plus a
direction sweep where PEV's *partial* effect should vanish once readability+controllability are included.
**Scope:** linear–Gaussian, so E1 does NOT test nonlinear geometry — it asks the prior question of whether
the lever is PEV or the two quantities PEV bundles. Cheap (CPU/GPU, seconds–minutes), held-out seeds.

### E2 — Linear sufficiency (does T-PHATE buy anything for *learning*?)
Same harness, swap a nonlinear (T-PHATE-style diffusion) manifold for plain PCA when defining IM/WMP/OMP.
If the on/off asymmetry survives under linear PCA, nonlinear manifold geometry is **not load-bearing for the
learning claim** (only for decoding/visualization). If it changes the result, nonlinearity carries weight —
and E2 quantifies how much, *beyond* the readability × controllability of E1. Direct follow-on to E1.

### E3 — The "10× harder, not impossible" learnability frontier
The paper's discussion concedes off-manifold mappings are learnable with ~10× training (citing primate
work). Sweep **off-manifold-ness × training budget** to map the learnability frontier: is OMP a hard wall,
or just a steeper slope? Tests whether "failure to learn" is categorical (geometry forbids it) or a
budget artifact (the session was too short for low-controllability directions).

## Why this direction is worth running
Replays the RDD→PAO skeleton on an **independent group's paper**. If E1–E3 land as predicted, the project
can make a stronger *meta*-claim: across two unrelated "geometry/diversity helps" frameworks (Cai/RDD and
Busch/manifold-BCI), the operative variable is the same — **task-aligned, usable signal under a
capacity/plasticity constraint**, not the geometric property each paper foregrounds. E1 also maps onto PAO:
on-manifold = reassociation within the existing repertoire (cheap); off-manifold = generating a new skill
(expensive).

## Result — E1 (N=30, held-out seeds 600–629)
**PEV/manifold-membership is neither necessary nor sufficient; controllability is the lever.** Replicates
on-manifold ≫ off-manifold (IM/WMP ΔControl≈25 vs OMP≈1, g≈+3.3). Then the deconfound breaks the geometric
story: **D1** (high PEV / on-manifold but uncontrollable) **fails**, indistinguishable from OMP (g=−0.03,
p=0.92); **D2** (low PEV / off-manifold but readable+controllable) **learns fully** (g=+3.28 vs OMP). Sweep
regression: controllability β=+0.89 ≫ readSNR β=+0.16; PEV correlates with learning (ρ=0.91, why Busch saw
it) but controllability (ρ=0.99) is the driver. *Deviation:* IM≈WMP (both saturate) not strict IM>WMP>OMP —
Busch's IM>WMP gap is sequential interference, outside E1's scope. Verdict + table: `../FINDINGS.md`
(2026-06-16). Run: `python e1_deconfound.py --mode confirm` → `results/e1_confirm.npz`.

## Status
- **E1: DONE** — pre-registered (`../PREREG_BUSCH_E1.md`, frozen 2026-06-16), confirmed. `property ≠
  usefulness` extends to Busch: the lever is readability × controllability, not geometry.
- E2 (linear vs nonlinear/T-PHATE sufficiency), E3 (learnability frontier): planned, not yet pre-registered.
