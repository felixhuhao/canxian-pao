# Pre-Registration — Busch E2: does nonlinear (T-PHATE-style) manifold geometry add anything beyond readability × controllability?

> **FROZEN 2026-06-16, before the confirmatory run** (design pivoted during calibration, see §3 note; no
> held-out data seen). Held-out seeds 700–729 (calibration 750–753).
> Code: `exp_busch/e2_nonlinear.py` (reuses the E1 agent in `e1_deconfound.py`). Second rung of Direction 2,
> follow-on to E1 (`PREREG_BUSCH_E1.md`).

## 1. Claim under test
Busch et al. argue a **nonlinear** diffusion manifold (T-PHATE) is necessary, because a linear approximation
"may not fully capture the intrinsic manifold … and what activity is on vs outside the manifold." But the
IM/WMP/OMP readout directions they actually use are **linear PCs of the latent**. E1 (linear–Gaussian) showed
that, holding the manifold-as-variance-subspace picture, learning is governed by **readability** (`CᵀΣ_obs C/σ²`)
× **controllability** (`CᵀΣ_act C`), not by PEV per se — but E1 imposed the `Σ_obs ≠ Σ_act` split by hand.

E2 asks the question E1 deferred: when data comes from a **genuinely nonlinear** manifold, (a) does Busch's
linear-PCA labelling still correctly identify learnable vs unlearnable directions, and (b) does "nonlinear
manifold membership" add predictive power for learning **beyond** readability × controllability?

## 2. Model (nonlinear manifold; everything else inherited from E1)
- Intrinsic latent `u ∈ R^d` (d=3), `u ~ N(0, Σ_lat)`, anisotropic `Σ_lat = diag(1, .5, .25)`.
- Smooth nonlinear embedding `x = W₂ tanh(gain · W₁ u) ∈ R^D` (D=20, fixed random `W₁,W₂`, seed-fixed so
  geometry is seed-independent as in E1). `gain` = curvature knob (gain→0 ⇒ linear).
- **Global observation covariance** `Σ_obs = Cov(x)` (empirical, what Busch's PCA/decoder see) — defines PEV
  and readability.
- **Local reachable covariance** `Σ_act = J Σ_lat Jᵀ`, `J = ∂x/∂u|₀ = gain·W₂W₁` (what the agent can
  *volitionally* produce by moving along the manifold near its operating point) — defines controllability.
- **Curvature creates the split for free:** because tanh is nonlinear, `Σ_obs` (global) ≠ `Σ_act` (local
  tangent). A high-variance global direction can be locally **normal** to the manifold ⇒ high PEV but
  uncontrollable. This is the E1 dissociation arising *naturally*, and exactly the case a linear PCA mislabels.
- Agent / learning / metric: **identical to E1** (`run_session`): learn gain to drive readout `C` under the
  plasticity prior; metric **ΔControl** = 100·(acc last third − first third). Only `Σ_obs, Σ_act` differ.

> **Design pivot during calibration (2026-06-16, pre-freeze).** Calibration revealed a structural fact that
> reframes E2: on a **single shared** manifold (the data distribution *is* the agent's reachable set),
> high global variance ⟺ the manifold extends there ⟺ locally reachable — so readability (PEV) and
> controllability are **coupled**, and PEV predicts learning *well*. They decouple only when the operating
> point sits in a **curved/saturated** region (parameterized by `offset`, the operating-point distance from
> the manifold's locally-linear center): there, a high-variance direction can be locally normal
> (uncontrollable). So the original "linear PCA always mislabels" hypothesis is false; the honest question is
> *when* PEV is a valid proxy. Predictions below are revised to this mechanism (prereg still DRAFT, not yet
> frozen; no confirmatory data seen).

## 3. Conditions (gain=2.0, operating-point offset=1.5; calibrated)
- **IM_lin / WMP_lin / OMP_lin** — top / 2nd / bottom PC of `Σ_obs` (Busch's linear method).
- **HiPEV_uncontrol** — the least-controllable of the top-3 high-variance PCs: a **genuinely high-PEV**
  component that operating-point curvature has made locally uncontrollable. (On a 3-D manifold only ~3 PCs
  carry real PEV; this typically coincides with WMP_lin.)
- **TAN / NOR** — oracle anchors: top eigvec of `Σ_act` (tangent/controllable) / least-reachable direction.
- **Coupling sweep**: offset ∈ {0, 0.5, 1.0, 1.5, 2.0}; per offset measure, over a random-direction sweep:
  `corr(PEV, controllability)` (coupling), `R²(PEV)` vs `R²(PEV+controllability)` for predicting learning
  (the extra variance controllability explains beyond PEV), and HiPEV_uncontrol vs IM ΔControl.

## 4. Pre-registered predictions (on record)
- **P1.** Tangent/controllable directions learn (TAN, ΔControl high); least-reachable fails (NOR low).
- **P2 — high PEV is not sufficient under curvature:** at the curved operating point, **HiPEV_uncontrol
  fails** (ΔControl ≪ IM_lin) despite high PEV — a high-variance ("on-manifold") component can be unlearnable.
- **P3 — controllability is the lever (nonlinear setting):** in the direction-sweep regression `ΔControl ~
  readSNR + controllability`, controllability carries the dominant standardized β (≫ readSNR), and
  Spearman(controllability, learning) > Spearman(PEV, learning).
- **P4 — PEV is a proxy valid only while coupled:** as `offset` increases, `corr(PEV, controllability)`
  **decreases** and the extra variance controllability explains beyond PEV (`R²(PEV+ctrl) − R²(PEV)`)
  **increases**; HiPEV_uncontrol's ΔControl falls from ≈IM (coupled, offset→0) toward failure (decoupled).
  At offset→0 (locally-linear operating point) PEV predicts learning well — **Busch's single-manifold design
  is sound there.**

## 5. Interpretation locked in advance
If P1–P4 hold, the combined E1+E2 verdict: whether the "manifold hypothesis" works reduces to whether
observation-variance (PEV) and volitional **controllability** coincide. **E1:** if they can differ, PEV is
neither necessary nor sufficient — controllability is the lever. **E2:** a single shared manifold (even a
nonlinear one) *couples* them — data lives where it is reachable — so PEV works, **vindicating Busch within
their design**; the coupling breaks under operating-point curvature, and there controllability still governs.
Therefore nonlinear geometry is **not an independent causal lever**: PEV's validity *is* the obs≈act coupling,
and the lever underneath is always controllability. This both fairly credits Busch (PEV is a fine proxy in
their regime) and pins the mechanism (controllability), consistent with the project's `property ≠ usefulness`
thesis — usefulness tracks *usable* (controllable) signal, not the variance/geometry label. **Revision rule:**
I revise if HiPEV_uncontrol learns at the curved operating point, controllability does **not** out-predict PEV
in the regression, or the coupling does **not** weaken with curvature (which would mean PEV is valid
unconditionally and the obs≈act framing is wrong). Calibration (750–753) fixes gain, offset, σ_read, sample
count and confirms non-degeneracy; held-out run (700–729) is confirmatory and design/metrics are frozen here.
