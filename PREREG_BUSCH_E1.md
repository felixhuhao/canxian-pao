# Pre-Registration — Busch E1: is on-/off-manifold BCI learning geometry, or readability × controllability?

> **FROZEN 2026-06-16, before the run.** Held-out seeds 600–629 (calibration uses 500–503). Code:
> `exp_busch/e1_deconfound.py` (new). First rung of Direction 2 (independent test of Busch et al.,
> *"Accelerated learning of a noninvasive human BCI via manifold geometry"*, `docs/2025.03.29.646109v1.full.pdf`).

## 1. Claim under test
Busch et al. report that re-learning a perturbed fMRI-BCI mapping is fast when the readout direction is
**on the intrinsic manifold** (IM = top latent eigenvector; WMP = 2nd) and fails when it is **off-manifold**
(OMP = lowest/20th eigenvector). Their learning metric is `ΔBrainControl`; their proposed mechanism is
neural realignment = raising the **percent variance explained (PEV)** along the trained component.

**Confound.** OMP is by construction the *lowest-variance* direction, and "on-manifold-ness" is operationalized
as PEV (`Cᵀ Σ C`). PEV silently bundles two functionally distinct requirements that their single-manifold
design cannot separate:
- **Readability** — how much signal the decoder can recover along `C` (read SNR ∝ `Cᵀ Σ_obs C` / noise).
- **Controllability** — how cheaply the agent can *volitionally* produce amplitude along `C` under a
  plasticity/metabolic prior (cost ∝ `Cᵀ Σ_act⁻¹ C`).

Busch implicitly assume the observation manifold equals the volitional-reachability set (`Σ_obs = Σ_act`),
so both collapse onto PEV and PEV trivially predicts learning. **Hypothesis:** PEV / manifold-membership is a
*proxy*; the operative levers are readability and controllability. If observation-variance and reachability
diverge, the PEV predictor breaks.

**Scope (stated honestly).** E1 uses a linear–Gaussian "brain" (manifold = anisotropic variance subspace), so
it does **not** test whether nonlinear (T-PHATE) geometry adds anything beyond variance — that is E2. E1's
question is narrower and prior: *even granting the variance-subspace picture Busch's own components live in,*
is the learning effect carried by PEV or by the two quantities PEV bundles?

## 2. Model
- Brain/latent dim **D = 20** (matches their 20-D embedding). Intrinsic covariance `Σ_obs` = random orthonormal
  eigenbasis `{u_i}` with geometrically decaying eigenvalues `λ_i = ρ^i` (ρ tuned in calibration; spans IM..OMP).
- Readout component `C` (unit vector) → decoded angle `α = ∠(C·b)` + read noise `σ_read`; effective read
  SNR ∝ `(Cᵀ Σ_obs C)/σ_read²`. **PEV(C) = Cᵀ Σ_obs C / tr(Σ_obs)** (Busch's measure).
- **Agent**: learns a linear policy `θ* → b` (intended angle → brain state) by gradient on per-trial readout
  error **plus** a plasticity penalty on produced amplitude. The minimum-plasticity-cost action achieving
  readout `r` along `C` is `b = r·Σ_act C /(Cᵀ Σ_act C)` with cost `r²/(Cᵀ Σ_act C)`; hence
  **controllability (cheapness) ∝ Cᵀ Σ_act C**, and the agent's optimal gain is `g* = 1/(1+λ_plast·cost)`,
  so low controllability makes the agent *under-drive* the readout. `Σ_act` = volitional reachability cov.
- **Trials**: per trial a random target angle θ* (as in their random goal angles). Per-trial accuracy
  `acc = exp(−err²/2s²)`; **metric = ΔControl = 100·(mean acc, last third − first third)** of the session —
  the construct Busch's ΔBrainControl measures (control gained over a session), session length matched.

> **Amendment 2026-06-16 (before any confirmatory run; calibration only).** The prereg originally specified
> a literal error-vs-running-average staircase (their Eq. 3). Calibration (seeds 500–503) showed this rule
> has a right-skew artifact: per-trial error is right-skewed, so "below the recent average" occurs >50% of
> the time even with no learning, spuriously ramping BrainControl for noise-dominated conditions (OMP). I
> replaced it with the skew-free ΔControl above, which preserves the construct (control gained = sustained
> accuracy improvement). Also corrected the controllability formula to `Cᵀ Σ_act C` (the min-cost
> derivation; the draft's `Cᵀ Σ_act⁻¹ C` was wrong). Held-out seeds 600–629 untouched; predictions below
> unchanged in substance.

Busch's regime is the special case `Σ_act = Σ_obs`. We sweep the two independently to break the bundle.

## 3. Conditions (each = a (readability, controllability) cell on a fixed direction)
Under `Σ_act = Σ_obs` (Busch regime):
- **IM** — top eigvec: high PEV, high readability, low control cost.
- **WMP** — 2nd eigvec: mid/mid/mid.
- **OMP** — bottom eigvec: low PEV, low readability, high control cost.

Deconfound cells (break `Σ_act = Σ_obs`):
- **D1 "on-manifold but uncontrollable"** — direction = `u_1` (**high PEV / high readability**) but `Σ_act`
  made *small* along `u_1` (**high control cost**).
- **D2 "off-manifold but usable"** — direction = `u_D` (**low PEV / low readability-by-default**) but `Σ_act`
  made *large* along `u_D` (**low control cost**) and read noise lowered so it is readable.

Plus a **sweep** of ~12 directions spanning the PEV × controllability plane for the regression (§4, P4).

## 4. Pre-registered predictions (on record)
- **P1 (replicate).** In the Busch regime (`Σ_act=Σ_obs`), eigen-directions give **IM > WMP > OMP** in
  ΔBrainControl (paired Wilcoxon, p<0.05; Hedges g large for IM vs OMP).
- **P2.** **D1 fails** despite high PEV: ΔBrainControl(D1) ≈ OMP, ≪ IM (g large, p<0.05). → high PEV /
  on-manifold is **not sufficient**.
- **P3.** **D2 learns** despite low PEV: ΔBrainControl(D2) ≫ OMP, comparable to WMP/IM. → low PEV /
  off-manifold is **not the cause** of failure.
- **P4.** PEV (Busch's measure) tracks only the *readability* axis (`Cᵀ Σ_obs C`) and is blind to
  *controllability* (`Cᵀ Σ_act C`). In the direction-sweep regression `ΔControl ~ readSNR + controllability`,
  **both** carry positive, non-trivial partial effects (controllability significant → PEV is insufficient),
  even though PEV alone is positively correlated with learning (why Busch observed it). The D1/D2 cells are
  the decisive counterexamples to "PEV predicts learning."

## 5. Interpretation locked in advance
If P1–P4 hold: Busch's "manifold geometry constrains learning" is, at the level their own components occupy,
a **proxy result** — manifold-membership (PEV) predicts learning only because it bundles readability and
volitional controllability, two separable quantities. Dissociating them (high-PEV-but-uncontrollable fails;
low-PEV-but-controllable succeeds) shows the geometric property is not the causal lever. This extends the
project's `property ≠ usefulness` thesis to a **second, independent** framework (Busch, distinct from Cai/RDD):
the operative variable is again *task-aligned, usable signal under a capacity/plasticity constraint*, not the
geometric label. It also sharpens what E2 must test: whether nonlinear (T-PHATE) geometry adds any predictive
power **beyond** readability × controllability.

**Revision rule.** I revise toward Busch if D1 learns, or D2 fails, or PEV remains the dominant partial
predictor in P4 — any of which would show manifold-membership carries causal weight beyond the two bundled
quantities. Calibration (seeds 500–503) only fixes ρ, λ_plast, σ_read, trial counts, and the Σ_act
perturbation magnitude so that P1 reproduces and conditions are non-degenerate; the held-out run (600–629)
is confirmatory and design/metrics are frozen here.
