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
- **Coverage reward HELPS (`rdd_taware.py`) — the positive result.** Replacing generic repulsion with a
  *task-aligned* coverage reward (`Σ_f Ŝ(f)·max_m p_m(f)`, signal spectrum estimated from data) lowers
  test + OOD MSE in the **over-complete (M=8)** regime (g −0.78…−1.63, p<0.01; pre-registered, held-out
  seeds). Capacity-gated (little benefit at M=3); cannot rescue a symmetric collapse. Principle:
  **diversity helps downstream only when it is functional coverage aligned with the task AND there is
  spare capacity to deploy it.** Generic LI and overlap-penalty do not help; only the coverage form does.

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

Direction 2 (Busch IM/WMP/OMP manifold-alignment) is deferred pending the Busch preprint.
