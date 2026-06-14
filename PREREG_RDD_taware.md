# Pre-Registration — Does TASK-ALIGNED (coverage) repulsion help downstream, where generic LI didn't?

> **FROZEN 2026-06-14, before the N=20 confirmation run.** Confirms a calibration hint on **held-out
> seeds (100–119)** disjoint from the calibration seed (0). Results → `FINDINGS.md`.

## 1. Claim under test
`rdd_li.py` + `rdd_crossover.py` established that **generic** lateral inhibition never helps downstream
(it is task-blind). The constructive implication: only diversity that is **functionally aligned with
the task** could help. We test a coverage-reward auxiliary loss and ask whether it beats no-LI.

Three repulsion forms (all tested; first two already shown to fail):
- **generic** `Σ exp(−(τ_i−τ_j)²/2ℓ²)` — distance in τ-space (task-blind). *Hurts (prior).*
- **taware-overlap** `Σ_{i<j} Σ_f Ŝ(f) p_i(f) p_j(f)` — penalize spectral overlap. *Fails in calibration
  (minimizes overlap by clustering channels at slow τ; no coverage incentive).*
- **coverage (this test)** reward `Σ_f Ŝ(f) · max_m p_m(f)` — at each powered frequency, some channel
  responds. `p_m(f)` = channel m's normalized frequency response; `Ŝ(f)` = signal power spectrum
  estimated from training data via FFT (not oracle). This is the only form that rewards *covering* the
  signal rather than merely separating channels.

## 2. Design
Same SSM / sine-mixture task. Grid: `{asym, sym}` × `{noLI, genLI(β=10), cov β∈{1,10,50}}` × `M∈{3,8}`.
`asym` = per-channel weights (channels self-separate); `sym` = shared weights + τ-init jitter 0.1
(near-degenerate collapse regime). **N=20 held-out seeds (100–119).**

## 3. Calibration hint (seed 0, why this test exists)
Coverage-reward: in **asym M=8** it *lowered* test MSE (no-LI 3.447 → 3.211 at β=50, and 3.378 at β=10);
in asym M=3 a small drop at β=1 (3.426→3.365). In **sym** it did **not** help (can't break the symmetric
degeneracy — same limitation as the other forms). So the hint is: coverage-reward helps in the
**over-complete, already-differentiated (asym)** regime, *not* the collapse regime.

## 4. Pre-registered decision rule
Primary metric: held-out test MSE (lower better). Paired Wilcoxon + Hedges g.
- **PRIMARY:** `asym M=8, cov β=50` vs `asym M=8 noLI`. **COVERAGE HELPS** iff p(less)<0.05 **and**
  g ≤ −0.5. **ANTI** iff reverse p<0.05 and g ≥ +0.5. **NULL** otherwise.
- **Secondary:** asym M=3 (cov vs noLI); β-monotonicity in asym; sym cells (predict NULL — coverage
  cannot break symmetry). Report the full grid; a cell is starred only at g≤−0.5, p<0.05.
- **Multiple-comparisons honesty:** the calibration selected the asym/over-complete/large-β cell, so the
  PRIMARY is that specific cell on *disjoint* seeds; the rest of the grid is exploratory context.

## 5. Prediction (on record)
I expect a **modest but real COVERAGE-HELPS in asym M=8** (the over-complete regime), **NULL in sym**
(symmetry unbroken), and small/neutral in asym M=3. If it holds, this is the project's **first positive
downstream lead**, and yields a clean principle: *diversity helps downstream only when it is functional
coverage aligned with the task — generic separation (and overlap-penalty) do not, and no form rescues a
symmetric collapse.* If the asym M=8 effect evaporates on held-out seeds, it was calibration noise and
the line closes as fully negative.

## 6. Integrity note
The coverage formulation was the principled fix to an identified mis-specification in taware-overlap
(penalizing overlap ≠ rewarding coverage), not a tweak-until-it-wins. It was calibrated on seed 0;
this confirmation uses **disjoint seeds 100–119**. Grid/metrics/rule frozen here; no further changes.
