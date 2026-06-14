# Pre-Registration — Ladder R1: does the coverage benefit survive PAO's freeze-and-admit mechanism?

> **FROZEN 2026-06-14, before the N=15 run.** First rung of the controlled RDD→PAO transfer ladder:
> change ONLY the mechanism (joint co-adaptation + aux loss → sequential crystallization of FROZEN
> units + admission gate), keeping the RDD task/units/supervision fixed. Held-out seeds 300–314.

## 1. Claim under test
R0 (`rdd_taware`/`rdd_capacity`) found a task-aligned coverage reward helps downstream and scales with
excess capacity — but via JOINT training + an auxiliary loss. PAO builds a library by SEQUENTIALLY
crystallizing FROZEN units and admitting them through a gate. Does the benefit survive that mechanism?

## 2. Design
Same RDD task (multi-frequency SSM next-step prediction). Build = sequential residual boosting of frozen
low-pass channels; global readout = ridge least squares (refit per admission). Two arms:
- **GATED:** admit a frozen candidate only if it lowers held-out (val) MSE by > EPS (covers something
  new); else reject. (PAO + coverage gate.)
- **UNGATED:** admit every crystallized candidate. (PAO as-is → redundant library.)
Swept over capacity `M_max ∈ {2,3,5,8,12,16}` (K=5). Metric: held-out test MSE; also n_admitted (gated).

## 3. Sanity (1 seed) & integrity
Calibration showed gated ≈ ungated on test MSE (Δ ≤ 0.001 at every M_max) but gated admits fewer
(10 vs 16 at M_max=16). The boosting+ridge readout is robust to redundant frozen units (ridge
down-weights them), unlike R0's joint training. Mechanism/metrics frozen here; N=15 confirms.

## 4. Pre-registered decision rule
- **BENEFIT SURVIVES (performance)** iff gated beats ungated on test MSE AND the gap Δ(ungated−gated)
  grows with M/K: Spearman(M/K, Δ) > 0, p<0.10, and mean Δ over M≥K > 0.
- **DOES NOT SURVIVE (performance)** otherwise (gated ≈ ungated).
- Always report **parsimony**: n_admitted(gated) vs M_max — does gating prune the library at equal MSE?

## 5. Prediction (on record)
**DOES NOT SURVIVE as a performance benefit**, but gating yields **parsimony** (fewer units, equal MSE).
Reason: frozen units + a regularized readout are robust to redundancy (ridge down-weights redundant
channels), so there is no joint-coadaptation overfitting for coverage to prevent. The deeper point this
rung establishes: **R0's coverage benefit and PAO's redundant-skill harm are different mechanisms.**
PAO's harm (P3: redundant/wrong skills applied as policy bias hurt reward) is a property of its
*fragile* RL skill-application, not of a clean readout — so the coverage-gating *performance* benefit
for PAO should be expected in the RL / fragile-combination regime (later rungs), not in supervised
regression. If gated *does* beat ungated here with a growing capacity gap, I revise toward the benefit
being mechanism-robust.
