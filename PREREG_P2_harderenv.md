# Pre-Registration — P2/Harder-Env: does trigger timing matter with BOCPD in a non-convex env?

> **FROZEN 2026-06-07, before the full run.** Steelman of H1 (the "Option 2" rescue). Shares method
> with `PREREG_P1.md`. Results → `FINDINGS.md`. Amendments dated at bottom.

## 1. Why this test
P1 falsified H1 in the trivial 1D corridor (trigger no better than a competence-matched random
window). Two fair objections: (a) the corridor is convex/trivial so any converged snapshot is optimal
→ timing *can't* matter; (b) the heuristic trigger isn't the paper's BOCPD. This test removes both:
a **harder, non-convex 2D env** where windows genuinely differ in quality (only 33–44% solved by
ep 300), and a **BOCPD** trigger.

## 2. Pre-run facts established during calibration (on record)
- **Env (`env_2d_shift`, 5×5, absolute-coord obs) converges *gradually*** (return ramps 0.3→1.1 over
  ~235 eps), not as a discrete jump. → no clean "phase-transition" event exists to detect.
- **The shipped `BOCPD.update()` firing rule never fires** even on a clean 0→5 step (verified). We use
  the standard robust read instead: **MAP run-length reset**, gated to `return > 1.0`. With that fix
  BOCPD fires ~ep 78–188; the heuristic fires ~ep 176–234. **Both fire before full convergence.**

## 3. Design
- Env: `TwoGate2DShiftEnv`. **P1 = 300** (A→B, default A/B), **P2 = 150** (B→A + **Location-Shift**,
  ε=0.1), **P3 = 150** (A→B, positions restored). `app_thresh = 0.3` (gate live), as in P1.
- Arms (identical mechanism; only crystallisation timing differs), seeded, **N = 30**, paired:
  | Arm | crystallise at |
  |---|---|
  | **BOCPD** | MAP-reset change-point, return>1.0 (the paper-faithful trigger) |
  | **Heuristic** | shipped dual-threshold (reference trigger) |
  | **RandomLate** | uniform random episode in **[100, 250)** (trigger operating regime, signal-ignored) — **decisive control** |
  | **RandomUniform** | uniform random in [0, 300) (weak floor) |
  | **NoSkill** | skill never applied (bias=0) (sanity floor) |
- RandomLate/Uniform episode drawn from RNG `seed+7000` (independent of trajectory). Report each arm's
  crystallisation-episode distribution for transparency (matching check).

## 4. Outcomes (pre-specified)
- **PRIMARY: Q(z)** = skill-only success rate on default A→B (N_val=30). Harder env ⇒ unlikely to saturate.
- **SECONDARY:** Phase-3 reuse (early-20 after restore); Phase-2 lock-in (late-20, shifted regime).

## 5. Statistics
Paired Wilcoxon (one-sided) + Hedges' g + 95% bootstrap CI, BH-FDR across the comparison family. N=30.

## 6. Pre-registered decision rule
**Decisive comparison: BOCPD vs RandomLate, primary Q(z).**
- **H1 SUPPORTED** iff `Q(z)_BOCPD > Q(z)_RandomLate`, p < 0.05 **and** Hedges' g ≥ 0.5.
- **H1 FALSIFIED** iff |g| < 0.3 and p > 0.05 (tie), **or** a significant *reverse* effect
  (RandomLate better) — added explicitly to fix the P1 rule gap.
- **INCONCLUSIVE** otherwise (0.3 ≤ |g| < 0.5).
Secondary outcomes and the Heuristic-vs-RandomLate comparison are reported as support/context.
Sanity: both triggers and RandomLate must beat NoSkill on P3 (else the skill mechanism is inert here).

## 7. Prediction (on record)
Given both triggers fire *before* convergence (premature) and the env converges gradually, I predict
**H1 FALSIFIED again** — RandomLate (a later, more-converged window) ≥ BOCPD on Q(z). If BOCPD instead
wins, that is a genuine, surprising point in favour of principled event-triggering.

---
### Amendments
*(none)*
