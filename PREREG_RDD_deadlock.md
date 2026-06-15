# Pre-Registration — RDD deadlock: is lateral inhibition SPECIAL, or just one symmetry-breaker?

> **FROZEN 2026-06-15, before the N=20 run.** Seeds 0–19. Code: `exp_rdd/rdd_deadlock.py` (reuses
> `rdd_li.py`). Tests RDD's central mechanistic claim directly (escape from geometric deadlock), not
> just via downstream MSE.

## 1. Claim under test
RDD claims a symmetric multi-channel SSM is stuck in a GEOMETRIC DEADLOCK (channels identical →
identical gradients → never differentiate) and that lateral inhibition (Gaussian τ-repulsion) is the
force that escapes it. Two sub-claims:
- **(deadlock real?)** perfectly symmetric init → channels stay collapsed (div≈0, MSE≈single-channel).
- **(LI special?)** LI is *the* escape mechanism (not just any symmetry-breaker), and the escape *helps*.

## 2. Design (symmetric SSM: shared W_in/W_out; channels differ only by τ)
**Regime 1 — PURE symmetry (jitter=0):** can a mechanism escape a true deadlock from nothing? Arms:
`none` (deadlock), `LI` (β=10), `jitter` (τ-init σ=0.1), `dropout` (channel p=0.3), `innoise` (input
σ=0.5), `wdecay` (θ weight-decay 1e-2). Channel-distinguishing mechanisms (LI/jitter/dropout) *could*
escape; symmetric ones (innoise/wdecay) cannot break channel symmetry.
**Regime 2 — SEEDED asymmetry (jitter=0.1 baseline):** given a seed to amplify, is LI special? Arms:
`seed` (jitter only), `LI_seed` (jitter+LI), `drop_seed` (jitter+dropout). Compare test MSE vs `seed`.
**Ceiling:** `asym` (per-channel W_in/W_out). **Severity:** `none` vs `asym` test MSE over M∈{2,3,5,8,12}.
Metric: τ-diversity (escape) + held-out test/OOD MSE. N=20 seeds.

## 3. Sanity (4 seeds) & integrity
Pure: LI div=0.00 (=none, inert), jitter div=0.65 (only mover), dropout/innoise/wdecay div≈0. Seeded:
seed test=5.85, LI_seed div=9.34 but test=6.35 (WORSE, g=+24.8 vs seed), drop_seed≈seed. Severity gap
~2.5→2.27 over M (flat/slightly down). Mechanistic reason logged: repulsion grad ∝(τ_i−τ_j) → 0 at the
symmetric fixed point, so LI is inert there. Design/metrics frozen here; N=20 confirms.

## 4. Pre-registered predictions (on record)
- **LI CANNOT escape a true deadlock:** in the pure regime LI div ≈ 0 ≈ `none` (inert); only `jitter`
  injects asymmetry (div>0). LI amplifies, it does not create.
- **LI is NOT special and NOT beneficial:** in the seeded regime LI_seed drives large diversity (div≫1)
  but does NOT beat `seed` on test MSE (p(help) n.s.; expected significantly worse); `drop_seed` ≈ `seed`.
  The trivial init seed does the symmetry-breaking; LI's amplification adds nothing useful (hurts).
- **Deadlock harm is large but ~M-independent:** gap(none−asym) ≈ 2.3–2.5, not growing with M.

## 5. Interpretation locked in advance
If confirmed: RDD's "LI escapes geometric deadlock" fails twice over — LI is *inert at the actual
deadlock* (gradient vanishes; a fixed-point analogue of the γ-inertness finding) and, where it *can*
act, its task-blind repulsion *worsens* downstream MSE (consistent with `rdd_li`/`rdd_crossover`). The
real symmetry-breaker is a trivial init asymmetry. Combined with the earlier results, the verdict on
RDD's mechanism strengthens: LI maximizes channel distance, which is neither the escape force nor a
performance regularizer. I revise if LI escapes the pure deadlock (div≫0 at jitter=0) or LI_seed beats
seed on MSE.
