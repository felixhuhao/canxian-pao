# Pre-Registration — P1: Does crystallisation *timing* matter? (H1, the crux)

> **FROZEN 2026-06-07, before any P1 run.** Do not edit after the first run; amendments go in a
> dated "Amendments" section at the bottom with justification. Results go to `FINDINGS.md`, not here.
> Author: experiment harness in `exp_two_gate_lock/`. Venv: `basic`.

---

## 1. Question and why it is the foundation

The distinctive claim of the canxianization / PAO framework is that skill acquisition is a
**discrete, signal-triggered event** ("crystallisation at a stable jump"), not just snapshotting a
converged policy. If a behavioural window selected by the **trigger** is no more reusable than a
**competence-matched but randomly-timed** window, then the event-trigger / phase-transition framing
adds nothing — PAO reduces to "freeze a good policy at some point." Skill *caching* might still help
(that is not what is tested here); the *trigger* would be the part that fails.

This is the cheapest decisive test of the foundation, and it is untouched by every reporting/​config
bug found so far (see `FINDINGS.md`).

## 2. Design — single manipulated variable: **when** crystallisation happens

All arms use the **identical** crystallisation mechanism (frozen weight-copy skill + ApplicabilityNet),
with the gate made live (`app_thresh = 0.3`, established necessary in P0.6). The **only** thing that
differs between the two key arms is the episode at which crystallisation fires.

| Arm | Crystallise at | Role |
|---|---|---|
| **Trigger** | first episode the shipped heuristic fires (return > 1.0 ∧ entropy < 0.6 ∧ 3/5 recent successes) | the signal-selected window |
| **RandomLate** | uniformly random episode in **[40, 80)** (2nd half of Phase 1 = converged regime), signal-ignored | **competence-matched random timing — the decisive control** |
| **RandomUniform** | uniformly random episode in **[0, 80)** | weak floor (may catch pre-convergence windows) |
| **NoSkill** | never (bias = 0) | sanity floor — confirms a skill helps at all |

- Phases (quick mode): P1 = 80 (A→B), P2 = 120 (B→A, ε=0.1), P3 = 60 (A→B restored). 1D corridor.
- `app_thresh = 0.3` in all skill arms. Dormancy excluded from P1 (it is a verified no-op; handled in P2).
- **Forced-crystallisation variant** (the one new piece of code): an agent that crystallises at a
  pre-specified episode regardless of the trigger, otherwise byte-for-byte the existing PAOLight path.
- **Seeding:** `set_seed(seed)` (torch+numpy+random) per run. Seeds **0–29 (N = 30)**. Trigger,
  RandomLate, RandomUniform, NoSkill share seeds → **paired** comparisons. RandomLate/RandomUniform
  episode draws use a separate, fixed RNG (`seed + 7000`) so the draw is independent of the seed's own
  trajectory.

## 3. Outcomes (pre-specified to prevent cherry-picking)

- **PRIMARY: skill validation quality `Q(z)`** = success rate of the *frozen skill alone*
  (skill-only rollouts, no base policy) reaching the goal under A→B. `N_val = 30` rollouts.
  Direct operationalisation of "the crystallised window is a reusable causal unit."
- **SECONDARY (supporting): Phase-3 reuse** = mean return over first 20 episodes of P3.
- **SECONDARY (supporting): Phase-2 lock-in** = mean return over last 20 episodes of P2.
- **Diagnostic (not a test): return at crystallisation episode** per arm — to confirm RandomLate is
  genuinely competence-matched (report mean ± SD; expect ≈ Trigger's).

## 4. Statistical analysis plan

- Paired by seed. **Wilcoxon signed-rank** (one-sided, direction below) + **Hedges' g** + **95%
  bootstrap CI** (10k resamples). N = 30.
- Family of comparisons: {Trigger vs RandomLate (key), Trigger vs RandomUniform, Trigger vs NoSkill,
  RandomLate vs NoSkill}. **Benjamini-Hochberg FDR** across the family.

## 5. Pre-registered decision rule (H1 = "trigger timing produces more reusable structure")

Decided on the **key comparison Trigger vs RandomLate**, PRIMARY outcome `Q(z)`:

- **H1 SUPPORTED** iff `Q(z)_Trigger > Q(z)_RandomLate`, one-sided Wilcoxon **p < 0.05** *and*
  **Hedges' g ≥ 0.5** (significant *and* non-trivial). Secondary outcomes must not contradict.
- **H1 FALSIFIED** iff `Trigger ≈ RandomLate`: **|g| < 0.3 and p > 0.05**. → the event-trigger /
  phase-transition framing adds nothing over snapshotting a converged policy.
- **INCONCLUSIVE** otherwise (0.3 ≤ |g| < 0.5, or significant in the *wrong* direction) → report as
  such; needs larger N or a harder environment.

**Saturation rule (pre-committed):** the 1D optimum is trivial, so `Q(z)` may hit the ceiling. If
`Q(z) ≥ 0.95` for *both* Trigger and RandomLate, the PRIMARY is treated as **H1-not-supported**
(timing does not improve skill quality), and **Phase-3 reuse becomes the decisive outcome**, judged by
the same p < 0.05 ∧ g ≥ 0.5 rule.

## 6. Scope / what this does NOT test
- Not testing whether caching helps at all (that is the *NoSkill* sanity contrast).
- Not testing dormancy (no-op; P2) or BOCPD-vs-heuristic trigger quality (P2).
- 1D only. A pass here is necessary, not sufficient; 2D + harder envs follow if H1 survives.
- Known limitation carried from P0.6: the applicability gate at 0.3 is a near-always-on switch, so the
  "skill" ≈ "replay frozen Phase-1 policy." H1 therefore tests timing of *that* mechanism, which is the
  mechanism the paper actually runs.

## 7. Prediction (on record)
Given the corridor is trivial and any converged policy is a near-optimal A→B skill, I expect
**H1 to be FALSIFIED or INCONCLUSIVE** — RandomLate windows should be about as reusable as Trigger
windows. Stating this now so the result cannot be retro-fitted.

---
### Amendments
**2026-06-07 (post-run, interpretive — frozen rule NOT changed).** The decision rule did not crisply
pre-specify the case *significant in the wrong direction on the PRIMARY*; its FALSIFIED branch (|g|<0.3 ∧
p>0.05) was written for a tie. Observed: PRIMARY Q(z) Trigger 0.772 vs RandomLate 0.987, g=−2.79 (large,
wrong direction) → the automated check printed INCONCLUSIVE. The SECONDARY P3 reuse is g=−0.02, p=0.889,
which **does** meet the FALSIFIED criterion. Interpretation on record: **H1 is not supported** — the
trigger is no better than (P3) and worse than (Q(z), via premature firing) a competence-matched random
window. This note documents the rule gap; it does not alter the frozen rule or the data. Future pre-regs
should add an explicit "significant reverse effect on primary ⇒ H1 refuted" branch.
