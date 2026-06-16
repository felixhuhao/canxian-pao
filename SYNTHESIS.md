# Synthesis — what RDD and Busch jointly establish, and what it means for PAO

Two independent literatures, tested under the same pre-registered discipline, converge on one principle.
This doc states it and turns it into a sharpened, falsifiable hypothesis for PAO (the project's end goal).
Detailed evidence: `FINDINGS.md`; experiment code under `exp_rdd/`, `exp_busch/`, `exp_pao_coverage/`.

## 1. The two results, side by side

| | **RDD** (Cai — repulsion-driven differentiation) | **Busch** (manifold-geometry BCI learning) |
|---|---|---|
| Foregrounded property | channel **diversity** via lateral-inhibition repulsion | **manifold geometry** (PEV — variance along a readout) |
| Causal claim | diversity escapes "geometric deadlock" & improves prediction | on-manifold (high-PEV) directions are learnable; off-manifold fail |
| Does the property correlate with the outcome? | yes (LI raises diversity; diversity ~ weakly tracks MSE) | yes (PEV-rank correlates with learning, ρ≈0.9 when coupled) |
| Is the property the **causal lever**? | **No** | **No** |
| What is the lever? | task-aligned **coverage** (fill the signal's spectrum), under **excess capacity** | **controllability** (usable/producible readout), under a plasticity/capacity budget |
| How the property fails when decoupled | task-blind repulsion *worsens* MSE; LI inert at the true deadlock (gradient ∝ τ_i−τ_j → 0); only a coverage objective helps, and coverage ≠ what LI computes | high-PEV-but-uncontrollable directions fail (g=+3.6); PEV predicts only while *coupled* to controllability (corr 0.92→0.52 as curvature decouples them) |
| Where the property "works" | over-complete regime, where coverage and diversity happen to align | single shared manifold, where data lives where it is reachable, so PEV ≈ controllability (**Busch vindicated in their regime**) |

## 2. The unified principle

> **A representational property is useful only to the extent it constitutes task-aligned, *deployable* signal
> under the system's capacity constraints. Property ≠ usefulness; usefulness = usable, task-aligned signal ×
> spare capacity.**

The recurring error in both frameworks is to credit a *structural property* (diversity, geometry) with a
downstream capability, when the property is merely **correlated** with the true lever:

- **Task-alignment / coverage** (RDD): diversity helps only when it *covers the task's structure* — and only
  a coverage objective, not repulsion, produces that. Generic "spread things apart" fights the task gradient.
- **Deployability / controllability** (Busch): variance helps only when the system can *actually produce/use*
  signal along it. Variance the agent cannot volitionally drive is inert, however "on-manifold" it looks.
- **Capacity-gating** (both): the benefit scales with *spare* capacity and is destroyed by redundancy — RDD's
  coverage payoff rises with excess channels (M/K); the PAO ladder (R1b/R4) shows coverage-gating helps a
  *redundancy-fragile* combiner and the gap grows with capacity.

The property predicts the outcome exactly while it stays **coupled** to the lever, and fails the moment the
coupling is broken (task-blind repulsion; observation ≠ volition / operating-point curvature). That coupling —
not the property — is the load-bearing, and usually untested, assumption.

## 3. What this sharpens for PAO

PAO's distinctive mechanisms were already falsified (event-triggered crystallization/BOCPD, manifold-health
gate, dormancy-gated plasticity — `FINDINGS.md`, P1–P3/HM). The controlled ladder R0→R1→R1b→R4 then localized
where any value could live: coverage-gated skill **admission** helps a redundancy-fragile combiner, and R5
isolated the real liability as **indiscriminate skill *triggering*** (many simultaneous wrong skills collapse
the additive policy; confidently-wrong skills corrupt it, weak ones don't).

The unified principle maps directly, and tightens the hypothesis:

- A crystallized skill = added capacity/"channel". Its **volume/timing/diversity is not the lever** (R5: volume
  harmless; and per §2, diversity/property never is).
- Busch's **controllability ≙ PAO's *triggerability***: a skill is usable only if it **fires in the states
  where it is the correct, deployable action**. The *trigger* is PAO's controllability gate.
- RDD's **coverage-under-capacity ≙** skills should **cover the task's niches, gated against redundancy**.

> **Sharpened PAO hypothesis.** PAO has value iff its **trigger** is a competence/coverage gate that deploys a
> skill *only in states where that skill is the controllable, task-aligned action*. Its value is **not** in
> when/whether it crystallizes, nor in skill count or diversity. Its failure mode is indiscriminate triggering
> (R5). Everything reduces to the **state→skill deployment decision**, i.e. controllability of the repertoire.

### The one experiment that would test it
Instrument the **trigger under partial observability** — the regime where the lever actually bites, because
when the state is fully observable any sensible gate works. Setup: a multi-niche task (as in `exp_pao_coverage`)
where the agent receives a **noisy/partial** state cue, so it *cannot perfectly identify which skill applies*.
Compare, on held-out seeds:
1. **Competence/coverage-gated trigger** (fire a skill only when its gate is confident the state is in-niche),
2. **Fire-all / indiscriminate trigger** (PAO's liability, R5),
3. **Strong monolith** (one network, no skills) — the fair baseline.

Pre-registered prediction from the principle: the gated trigger beats fire-all (controllability matters) but
**only matches or beats the monolith when partial observability makes the trigger's coverage decision the
binding constraint** — i.e., PAO's payoff is exactly the *triggerability/controllability* margin, nothing more.
If the gated trigger cannot beat the monolith even here, PAO has no regime-specific value. (User previously
parked the monolith comparison; this is the controllability-lens version of it, and the natural next rung.)

## 4. Status
- RDD path: **closed** (`exp_rdd/README.md`). Direction 2 / Busch: **closed** (E1+E2, `exp_busch/README.md`).
- PAO ladder R0→R1→R1b→R4 + R5: complete (`exp_pao_coverage/`, `FINDINGS.md`).
- **Trigger-under-partial-observability test: DONE** (`exp_pao_coverage/trigger.py`,
  `PREREG_PAO_TRIGGER.md`; N=8 survived → N=20 confirmed, `FINDINGS.md` 2026-06-16). **The hypothesis held:**
  PAO's value localizes entirely to the trigger — same skills, gated ≫ fire-all (g=4.6–11.7); and the gated
  trigger beats even a fair strong monolith across the whole partial-observability range (g=2.2–4.1, p≤1e-4),
  tying only at full observability. This is the project's first **constructive positive** for PAO: not its
  distinctive mechanisms (still falsified), but the reduction *PAO = competence-gated trigger over pre-mastered
  skills* has real value, and specifically in the partial-observability regime. Confirms the §2 principle on a
  third independent framework (RDD coverage, Busch controllability, PAO triggerability — same lever).
- **Learned gate (Priority 1): DONE** — the trigger advantage survives a reward-learned gate (no oracle),
  ~half magnitude, still beating the monolith across σ>0. Not an oracle artifact.
- **De-confound (Priority 2): DONE** — sharpens the PAO result one level further: the lever is
  **factor-then-route, not the modular skill library**. A single shared multitask net under the same gate
  matches K modular skills *exactly* (factored ≡ gated), while both beat a fair end-to-end monolith
  (g=2–3.8). So even within PAO, `property ≠ usefulness`: the *modularity* property is inert; the usable lever
  is the routing factorization (infer context → act with a context-appropriate clean-trained policy). Open:
  Priority 3 (learned/online skills), then the interference regime where modularity *might* re-enter.
