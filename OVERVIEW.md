# Project Overview — Empirical Evaluation of PAO / Canxianization & Related Geometry Claims

A single-file map of the whole investigation: every pre-registration, its result, and the verdict.
Companion docs: `SYNTHESIS.md` (the unifying principle), `FINDINGS.md` (full lab notebook, newest-first),
`PAO_EXPERIMENT_DIRECTIONS.md` (next PAO rungs), and the per-area READMEs. This file is the index; the
FINDINGS entries are the evidence.

## TL;DR
We stress-tested Hengjin Cai's PAO/Canxianization and RDD frameworks (and, independently, Busch et al.'s
manifold-BCI paper) under a strict pre-registration discipline (freeze hypotheses + held-out seeds before
running). One theme recurs across all three: **a foregrounded structural *property* (crystallization timing,
manifold "health", channel diversity, manifold geometry) is not the causal lever.** The lever is always
**task-aligned, *deployable* signal under a capacity constraint** — coverage (RDD), controllability (Busch),
triggerability (PAO). PAO's distinctive mechanisms are falsified; its one **constructive positive** is that,
reduced to a *competence-gated trigger over pre-mastered skills*, it has real value — specifically under
**partial observability**.

## Master table

| # | Pre-registration | Question | Verdict |
|---|---|---|---|
| **PAO Two-Gate: reproduction & falsification** |
| — | (P0–P0.6, audit) | Does the shipped 1D Two-Gate Lock reproduce? | **No** as shipped (gate never opens; no seed control; dormancy gate dead code). Core 1D effect real only at `app_thresh≈0.3`, where the gate is a dumb always-on switch. Many paper numbers hardcoded/unverifiable. |
| P1 | `PREREG_P1.md` | Does the crystallization *trigger* select specially-reusable windows? | **H1 not supported.** Trigger ties a random-late window on reuse and is *worse* on skill quality (fires early). Caching helps; *timing* does not. |
| P2 | `PREREG_P2_harderenv.md` | Same, in a harder non-convex env with BOCPD. | **Falsified again** (significant reverse). Principled BOCPD fires prematurely; the skill-cache benefit **reverses** — net harmful once the task is non-trivial. |
| P3 | `PREREG_P3_unpredictable.md` | Steelman: multi-skill library, random unsignaled shifts. | **Primary metric fires but is an artifact** (head-start, not competence); plain PPO dominates on total/asymptotic reward. Detection not load-bearing. |
| HM | `PREREG_HM.md` | Is manifold-health `H_M` a valid skill-quality gate? | **Anti-predictive** (ρ=−0.52, partialling out training ρ=−0.32). Both components inverted vs theory. As a gate it selects the *wrong* skills. |
| **RDD path (Cai, lateral inhibition) — CLOSED** |
| RDD-LI | `PREREG_RDD_LI.md` | γ-inertness? does LI-diversity help downstream? | **γ-inertness CONFIRMED** (clean validation). Downstream: **No** — LI raises diversity but worsens MSE monotonically; diversity anti-correlated with usefulness. |
| crossover | `PREREG_RDD_crossover.md` | Any regime (incl. collapse) where LI helps? | **No crossover.** LI hurts in all cells. Mechanism: repulsion is **task-blind** (maximizes channel distance ≠ signal coverage). |
| taware | `PREREG_RDD_taware.md` | Does *task-aligned* diversity help? | **First positive:** a **coverage reward** lowers test+OOD MSE in the over-complete regime. Capacity-gated. Generic LI / overlap-penalty still hurt. |
| capacity | `PREREG_RDD_capacity.md` | Dose-response of the coverage benefit. | **Capacity law:** benefit scales with excess capacity M/K (Spearman +0.94). Coverage *repairs* redundancy harm. |
| R1 | `PREREG_RDD_ladder_r1.md` | Does the coverage *performance* win survive freeze-and-admit? | **No (performance); yes (parsimony).** Frozen units + ridge readout are redundancy-robust. Gate trims library ~6.7 vs 16 units at equal MSE. |
| R1b | `PREREG_RDD_ladder_r1b.md` | What restores the performance benefit? | **Combiner fragility.** Causal law: **coverage-gating's performance payoff ⟺ the combiner is redundancy-fragile.** Robust→none, fragile→scales with capacity. |
| deadlock | `PREREG_RDD_deadlock.md` | Is LI the force that escapes the geometric deadlock? | **No.** LI is **inert at the true deadlock** (gradient ∝ τ_i−τ_j → 0) and **harmful where it can act**. Real symmetry-breaker = trivial init asymmetry. |
| rescue | `PREREG_RDD_rescue.md` | Can LI be salvaged by making repulsion task-aligned? | **No.** Every repulsion form hurts/collapses; only a **coverage/union** objective helps (g=−4.42 vs equally-task-aligned repulsion). The lever is the objective *form*, not task-alignment, not repulsion. |
| **PAO constructive ladder (RDD → PAO's RL regime)** |
| R4 | `PREREG_R4.md` | Does R1b's law transfer to PAO's own additive combiner in RL? | **Yes (capstone).** Coverage/competence-gated admission helps and the gap **grows with capacity** (Spearman +1.0, g up to +10). Only *confidently-wrong* junk corrupts the sum. |
| R5 | `PREREG_R5.md` | Is R4's harm reachable through PAO's *real* channels? | Crystallization **volume harmless**; single mis-fire moderate; **many simultaneous wrong skills → collapse.** PAO's liability is **indiscriminate triggering**, not crystallization. |
| **Direction 2 (Busch et al., manifold-BCI) — CLOSED** |
| E1 | `PREREG_BUSCH_E1.md` | Is on/off-manifold learning geometry, or something else? | **Readability × controllability**, not geometry. PEV is neither necessary nor sufficient: high-PEV-but-uncontrollable fails; low-PEV-but-usable learns. Controllability dominates (β=+0.89). |
| E2 | `PREREG_BUSCH_E2.md` | Does *nonlinear* (T-PHATE) geometry add anything? | **No.** On a shared manifold PEV and controllability are *coupled* (data lives where it's reachable) → PEV works (**Busch vindicated in their regime**); coupling breaks under operating-point curvature, and controllability still governs. Nonlinear geometry is not an independent lever. |
| **PAO trigger (the constructive positive)** |
| trigger | `PREREG_PAO_TRIGGER.md` | Is PAO's value the trigger (state→skill deployment) under partial obs? | **Yes.** Same skills: gated ≫ fire-all at every σ (g=4.6–11.7). And gated beats even a *fair strong monolith* across all σ>0 (g=2.2–4.1), tying at full observability → PAO's niche = **partial observability**. |
| gate | `PREREG_PAO_GATE.md` | Does the advantage survive a gate **learned from reward** (no oracle)? | **In progress** (N=8 running). Smoke test: a REINFORCE bandit gate recovers ≈Bayes (shortfall ≤0.04). |

## The arc, in four movements

**1. PAO's distinctive mechanisms fail (P0–P3, HM).** Crystallization *timing* is a training-duration proxy,
not a phase-transition detector (P1/P2); a BOCPD trigger fires early and hurts; the "fast-recovery" metric is
gameable by a head-start artifact (P3); the manifold-health gate is *anti*-predictive of skill quality (HM).
The shipped code didn't even run the configuration behind the paper's numbers.

**2. RDD's diversity/repulsion fails — but reveals the real lever (RDD path).** γ-inertness replicates
(a genuine validation), but LI-induced diversity *worsens* prediction and is inert at the actual deadlock.
The one thing that helps is a **task-aligned coverage** objective, and only with **spare capacity** — and only
when the **combiner is redundancy-fragile** (R1b). That fragility is exactly PAO's additive skill-combiner.

**3. The law transfers to PAO (R4, R5).** In multi-task RL with PAO's own additive combiner, coverage/
competence-gated skill admission helps and scales with capacity (R4). R5 then localizes the liability: not
crystallization volume, but **indiscriminate triggering** of confidently-wrong skills.

**4. An independent paper, the same lever (Busch E1/E2), and the constructive PAO result (trigger).** Busch's
manifold geometry is a proxy for controllability; nonlinear geometry adds nothing causal. Mapping
controllability → PAO's **triggerability** predicted, and the trigger experiment confirmed: PAO's value lives
entirely in the state→skill trigger, and is real under partial observability.

## The unifying principle (see `SYNTHESIS.md`)
> **A representational property is useful only to the extent it constitutes task-aligned, *deployable* signal
> under the system's capacity constraints. Property ≠ usefulness; usefulness = usable, task-aligned signal ×
> spare capacity.**

Three independent frameworks, one lever: RDD's **coverage**, Busch's **controllability**, PAO's
**triggerability**. Each paper's headline property predicts outcomes only while it stays *coupled* to that
lever; break the coupling (task-blind repulsion; obs≠act / curvature; indiscriminate firing) and the property
fails while the lever still governs.

## Method discipline (applied throughout)
Pre-register a frozen `PREREG_*.md` (hypotheses, predictions on record, revision rule) **before** running;
calibrate on non-held-out seeds; confirm on **held-out** seeds; report effect sizes (Hedges g), paired
Wilcoxon, Spearman; log honest deviations and integrity notes (e.g. E1 metric amendment, E2 design pivot,
R5 prediction refinement) rather than rewriting frozen rules. Commit results; push on request.

## Status
- **PAO mechanisms:** falsified (P1–P3, HM). **RDD path:** closed. **Direction 2 / Busch:** closed.
- **PAO ladder R0→R1→R1b→R4 + R5:** complete. **Trigger:** confirmed (N=8→N=20).
- **Learned gate:** N=8 in progress (then N=20 if it survives).
- **Parked / external:** Dr. Cai's run6 checkpoints (ecological H_M test); whether the RDD coverage/capacity
  law and the PAO trigger result scale to real sequence models / richer environments.
