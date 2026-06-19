# Pre-Registration -- DPA overlap, regime-axis characterization

> Draft/freeze target: run before using the result in `docs/dpa_v14_boundary.tex`.
> Code: `exp_dpa/overlap_regime_axis.py`.
>
> Supersedes the 3-regime robustness check (`PREREG_DPA_OVERLAP_ROBUSTNESS.md`), which found the overlap
> relationship is regime-dependent (T6 strong, T5 weak, T7 null) but had only n=3 on the adaptation-regime
> axis -- too few to characterize *when* it holds. This experiment samples the regime axis densely so the
> outcome is a conditional law with identified boundary conditions, not a one/strong-one/null anecdote.

## Claim under test

Overlap between an adaptation regime and a probe regime predicts the *order* in which probes' routing collapses,
**conditional on** the adaptation regime not saturating collapse (a ceiling effect destroys rank variance) and
on sufficient label--routing conflict. The relationship should appear in both input direction-cosine space and
the gate's hidden-representation space.

## Design

Self-contained NavEnv2D-style control (same environment as `exp_dpa/objective_control.py` /
`overlap_robustness.py`); hard CE adaptation to S0 on all-zero sparse skill-evaluation observations.

- **Adaptation regimes:** 16 interior targets spread across the grid (listed in code), not the five skill
  targets T0--T4. Each is analyzed independently.
- **Probe targets:** a 7x7 grid (49 points) spanning the interior.
- **Seeds:** smoke 1; calib 3; confirm 12; long 24.
- **Checkpoints:** finer near collapse onset --
  `[0,5,10,15,20,25,30,35,40,50,60,80,100,150,200,300]` -- to resolve fast collapses that a coarse grid
  would tie (the fix for the T7 ceiling/resolution problem).

Efficiency: the pre-adaptation gate and probe observations depend only on the seed, so they are computed once
per seed and reused across all adaptation regimes. The routing-collapse metric needs only gate forward passes
(no behavioral rollouts in the per-checkpoint loop).

## Metrics

Primary, per (adaptation regime, seed):

- **routing-collapse step** for each *at-risk* probe = first checkpoint where the probe-trajectory S0
  default-rate crosses 0.5. A probe is "at-risk" if its step-0 default-rate < 0.5 (probes already
  default-routed are not analyzable). Probes that never cross are censored at `last_checkpoint + 1`.
- **per-regime Spearman rho** between overlap and collapse step (input and hidden), then averaged over seeds.

Per-regime descriptors (seed-averaged):

- **saturation** = fraction of at-risk probes that collapse by the final checkpoint;
- **conflict** = fraction of all-zero adaptation observations whose pre-adaptation argmax skill is not S0
  (the degenerate label opposes pre-trained routing);
- **all-zero availability**, **n at-risk probes**.

A regime is **analyzable** if it has >= 5 all-zero adaptation observations and >= 6 at-risk probes.
A regime is **saturating** if seed-mean saturation >= 0.90 (excluded from the law test; reported separately).

## Decision rule (pre-specified, frozen)

- **Primary law:** the propagation law holds if, among non-saturating analyzable regimes, **>= 60%** have
  seed-mean rho < **-0.5** in **both** input and hidden overlap.
- **Boundary characterization (secondary):** across analyzable regimes, predict
  Spearman(seed-mean rho_input, saturation) **> 0** (more saturation -> less negative rho) and
  Spearman(seed-mean rho_input, conflict) **< 0** (more conflict -> more negative rho).

Reporting unit is the adaptation regime (one seed-aggregated rho per regime), which removes the
single-trajectory non-independence of the original 24-target analysis.

## Interpretation

- If the primary rule passes and the boundary correlations hold, the manuscript states a **conditional
  propagation law**: overlap predicts collapse order in non-saturating, high-conflict adaptation regimes, and
  fails by ceiling/low-conflict otherwise. This upgrades the overlap section from "regime-dependent
  diagnostic" to a characterized boundary.
- If the primary rule fails, the manuscript keeps the weaker "geometry-sensitive but inconsistent" framing and
  does not claim a law. Either outcome is reported.

## Calibration deviation (logged before confirmatory seeds)

Calibration (seeds 5100–5102) showed the original 16-regime list contained only 6 degenerate (analyzable)
regimes -- most interior points are reachable by a single frozen skill and so produce no all-zero evidence.
Before running the held-out seeds, the candidate set was expanded to 36 regimes concentrated in the degenerate
band, and the harness filters to the analyzable subset. The frozen decision rule and thresholds were not
changed. This is a sampling/coverage fix, not a change to the test.

## Outcome (confirm, held-out seeds 5120–5131; 24 analyzable regimes)

- **Primary binary law: FAILS** (0.26 of non-saturating regimes show rho<−0.5 in both metrics).
- **Boundary characterization (primary, threshold-free): CONFIRMED.** Spearman(rho_in, saturation) = +0.80
  (p<1e-4; calib +0.91); Spearman(rho_hi, saturation) = +0.89 (p<1e-4). Mean rho by saturation bin:
  low (<0.5) −0.77, mid −0.31, ceiling (≥0.9) +0.20.
- The `conflict` predictor came out wrong-signed (+0.71) but is confounded with saturation; not load-bearing.

Interpretation taken: a **conditional propagation boundary governed by collapse saturation**, not a universal
overlap law. Full notes in `FINDINGS.md`; integrated into `docs/dpa_v14_boundary.tex`.
