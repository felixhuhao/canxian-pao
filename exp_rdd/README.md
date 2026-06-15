# exp_rdd — Repulsion-Driven Differentiation (RDD), lateral inhibition

Independent test of the **RDD** paper (Hengjin Cai, *"Lateral inhibition enables escape from
geometric deadlock in multi-timescale adaptation"*, `docs/RDD_main.pdf` + supplementary).

This reproduces RDD's **neural-network proof-of-concept** (a multi-channel first-order low-pass
SSM with learned timescales `τ_m = exp(θ_m)` and a Gaussian repulsion loss on `τ`, Table S8) and
**extends it to the question the paper explicitly left open**: does the LI-induced channel diversity
translate into *downstream* performance? It also runs the **γ-inertness** test (the `β_LI` gain
should do nothing without the actual repulsion kernel — RDD Table 2).

## Files
- `rdd_li.py` — model, sine-mixture next-step-prediction task, all conditions, stats. Run:
  ```bash
  python rdd_li.py --seeds 20         # writes results/rdd_li.{log,pkl}
  ```
- `rdd_crossover.py` — follow-up: does LI help in the *collapse* regime (sym vs asym init)?
  ```bash
  python rdd_crossover.py --seeds 20  # writes results/rdd_crossover.{log,pkl}
  ```
- `rdd_deadlock.py` — direct test of RDD's central claim: is LI the force that escapes the geometric
  deadlock, or just one symmetry-breaker? Pure-symmetry vs seeded-asymmetry regimes; LI vs jitter/
  dropout/input-noise/weight-decay; severity vs M.
  ```bash
  python rdd_deadlock.py --seeds 20   # writes results/rdd_deadlock.{log,pkl}
  ```
- `results/` — `.log` + `.pkl` evidence.

Pre-registration: `../PREREG_RDD_LI.md` (frozen before the N=20 run). Verdict: `../FINDINGS.md`
(2026-06-14 entry).

## Result (N=20)
- **γ-inertness — CONFIRMED.** The inert arm (β present, kernel gradient detached) is identical to
  no-LI at every β (diversity range 0.000); real-LI responds to β. RDD's mechanism claim — the
  repulsion *interaction*, not the scalar gain, is operative — replicates in a trained net.
- **Diversity → downstream gains — NO.** LI raises τ-diversity (3.8 → ~10, matching the paper) but
  **monotonically worsens** test MSE (3.40 → 4.41) and OOD MSE; diversity is anti-correlated with
  usefulness (Spearman +0.47). LI is a coverage / anti-collapse tool, not a performance regularizer.
- **No crossover (`rdd_crossover.py`).** Even in a *collapse* regime (symmetric init) built to favour
  LI, LI breaks the degeneracy (diversity 0.7 → ~10) but still **worsens** MSE (5.85 → 6.34), in every
  cell. Mechanism: LI's repulsion is **task-blind** — it maximizes channel distance, not coverage of
  the signal's timescales, so it fights the task gradient. Diversity ≠ functional coverage ≠ usefulness.
- **LI is INERT at the true deadlock (`rdd_deadlock.py`).** Direct test of RDD's central claim. At
  *perfectly* symmetric init, LI gives diversity 0.000 = no-LB baseline — the repulsion gradient
  ∝(τ_i−τ_j) vanishes at τ_i=τ_j, so LI cannot *create* asymmetry, only amplify a pre-existing one. The
  only thing that escapes a true deadlock is a trivial init asymmetry (`jitter`); dropout/input-noise/
  weight-decay are inert. Given a seed (jitter=0.1) where LI *can* act, it drives diversity 0.7→9.1 but
  **worsens** MSE (5.86→6.34, g=+13.7); seed-alone and dropout tie at the best MSE. So RDD's "LI escapes
  geometric deadlock" fails twice: LI is neither the escape force nor a performance regularizer. (Also:
  no τ-mechanism nears the per-channel-weights ceiling 3.51 — the real bottleneck is shared weights, not
  τ-collapse.) Deadlock harm large but M-independent (gap≈2.4; Spearman(M,gap)=−0.40, p=0.51).
- **Coverage reward HELPS (`rdd_taware.py`) — the positive result.** Replacing generic repulsion with a
  *task-aligned* coverage reward (`Σ_f Ŝ(f)·max_m p_m(f)`, signal spectrum estimated from data) lowers
  test + OOD MSE in the **over-complete (M=8)** regime (g −0.78…−1.63, p<0.01; pre-registered, held-out
  seeds). Capacity-gated (little benefit at M=3); cannot rescue a symmetric collapse. Principle:
  **diversity helps downstream only when it is functional coverage aligned with the task AND there is
  spare capacity to deploy it.** Generic LI and overlap-penalty do not help; only the coverage form does.

- **Rescue attempt (`rdd_rescue.py`) — LI cannot be salvaged by task-alignment.** Isolates the single
  ingredient that rescues auxiliary diversity by holding everything fixed but the aux term (M=8 asym,
  held-out seeds 400–419). `spec` (signal-weighted spectral *repulsion*) and `cov` (signal-weighted
  *coverage*) are equally task-aligned; they differ only in pairwise-product vs max-union. Result: every
  repulsion form HURTS — τ-space genLI (g=+17.8), task-aligned spectral spec10 (g=+4.0), and strong spec
  *collapses* the model (div→0.1, MSE→deadlock). **Only coverage helps** (cov10 g=−1.15, p=0.001) and
  beats the equally-task-aligned repulsion decisively (cov vs spec g=−4.42, **p<0.001**). The rescuing
  ingredient is the **objective form (coverage/union), not task-alignment, not repulsion.** This pins the
  `rdd_taware` win to *coverage*, which is not what LI computes.

- **Capacity law (`rdd_capacity.py`).** The coverage benefit **scales with excess capacity**: sweeping
  M against K=5 frequencies, Δ = MSE(noLI)−MSE(cov) rises monotonically with M/K (Spearman +0.94,
  p=0.005) — hurts when under-complete (M<K), helps increasingly when over-complete (g up to −3.4 at
  M/K=3.2). no-LI degrades as M grows (redundancy harm); coverage holds it flat → it *repairs*
  redundancy. Pre-registered, held-out seeds. Principle: **the value of task-aligned diversity scales
  with excess capacity.**

- **Ladder R1 (`rdd_ladder_r1.py`) — transfer toward PAO's mechanism.** Rebuild the ensemble by
  *sequential crystallization of frozen channels* with a coverage admission gate (PAO-style), vs ungated
  admit-all, swept over capacity. **The coverage *performance* benefit does NOT survive** (gated ≈
  ungated; frozen units + ridge readout are robust to redundancy) — R0's win was specific to joint
  co-adaptation. **Parsimony does survive:** the gate keeps ~6.7 units vs 16 at equal MSE. Lesson: R0's
  benefit and PAO's redundant-skill harm are *different mechanisms*; PAO's harm is its fragile RL
  skill-application (P3), so the coverage *performance* payoff for PAO lives in the RL regime, not
  supervised regression. (Controlled RDD→PAO transfer ladder: R1 mechanism; R2 coverage-estimation;
  R3 rich units; R4 RL.)

- **Ladder R1b (`rdd_ladder_r1b.py`) — combiner fragility is the deciding factor.** Same frozen
  sequential admission, but change the combiner: robust ridge (R1) → no performance benefit; **fragile
  averaging (R1b) → coverage-gating helps and scales with capacity** (Spearman(M/K, Δ)=+1.00, p<0.001;
  g up to −3.2). Causal law: **coverage-gating's performance payoff ⟺ the combiner is redundancy-
  fragile.** PAO's skill-application is fragile (additive bias; P3 harm) → predicts coverage-gated skill
  admission helps PAO's performance, not just parsimony.

- **Ladder R4 (`../exp_pao_coverage/`) — the law TRANSFERS to PAO's RL regime (capstone).** Multi-task
  grid RL with PAO's own combiner (skills as ADDITIVE policy-bias, summed over triggered skills). Library
  = base + one mastered skill/niche + `cap` mis-associated (confidently-wrong) junk skills/niche. Result
  (N=8, held-out seeds 100–107): **gating helps and the gap grows with capacity** — ungated collapses
  0.97→0.03 as junk accumulates, gated immune at 0.97, **Spearman(cap, Δ)=+1.00, p=0.000**, g up to +10.
  Added insight: only *confidently-wrong (mis-triggered)* junk corrupts the additive sum, not weak/immature
  junk → PAO's actionable failure is indiscriminate skill **triggering**, fixed by competence-gated
  admission. **Ladder complete: R0→R1→R1b→R4.** (R3 rich-units rung subsumed — R4 already uses network
  policies as units.)

## RDD path — CLOSED
Consistent end-to-end verdict on RDD's repulsion-driven differentiation: **(1)** γ-inert without the
kernel; **(2)** diversity ≠ usefulness (LI worsens MSE); **(3)** LI is inert at the *true* deadlock
(gradient vanishes at τ_i=τ_j) and harmful where it can act; **(4)** LI cannot be rescued by task-
alignment — only a *coverage* objective helps, and coverage is not what LI computes. RDD's mechanism is
neither the deadlock-escape force nor a performance lever. The one durable positive (task-aligned
coverage in the over-complete regime) was carried into the PAO ladder (R0→R1→R1b→R4, `../exp_pao_coverage/`).

Direction 2 (Busch IM/WMP/OMP manifold-alignment) is deferred pending the Busch preprint.
