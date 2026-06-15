# Pre-Registration — Ladder R1b: is coverage-gating's payoff gated by COMBINER fragility?

> **FROZEN 2026-06-15, before the N=15 run.** Held-out seeds 400–414. Isolates the one factor R1 left
> open: the combiner. Robust(ridge) contrast = R1 (`rdd_ladder_r1`, same task, held-out seeds 300–314).

## 1. Claim under test
R1 found coverage-gating gives **no performance benefit** when frozen units are combined by a ROBUST
readout (ridge down-weights redundant units) — only parsimony. Hypothesis: the performance benefit
returns when the combiner is **FRAGILE** (cannot down-weight a bad/redundant unit), which is PAO's
situation (skills applied as additive policy bias / applicability routing; redundant or wrong skills
actively hurt — the P3 finding).

## 2. Design (change ONLY the combiner)
Same RDD task, same sequential frozen admission, swept over capacity `M_max ∈ {2,3,5,8,12,16}` (K=5).
- **Fragile combiner (this run):** each candidate channel trained standalone to predict the full target
  Y; ensemble prediction = **mean** of admitted channels' predictions (fixed equal weights, no
  reweighting — the canonical fragile combination).
- **Robust combiner (contrast = R1):** residual boosting + ridge readout refit per admission.
- **GATED** admits a candidate only if it lowers val MSE of the ensemble; **UNGATED** admits all.
Metric: held-out test MSE; n_admitted.

## 3. Sanity (1 seed) & integrity
Fragile combiner: Δ(ungated−gated) = +0.25 (M=3), +0.55 (M=8), +0.90 (M=16) — gating helps and the gap
grows with capacity (ungated degrades as extra units drag the mean; gated admits ~2 and stays flat).
Mechanism/metrics frozen here; N=15 confirms.

## 4. Pre-registered decision rule
- **GATING HELPS (fragile)** iff gated beats ungated on test MSE AND Δ(ungated−gated) grows with M/K:
  Spearman(M/K, Δ) > 0, p<0.10, mean Δ over M≥K > 0.
- Combined with R1 (robust → no benefit), **CONTINGENCY CONFIRMED** = "gating's performance payoff ⟺
  combiner fragility."

## 5. Prediction (on record)
**Fragile combiner → GATING HELPS, scaling with capacity** (sanity-confirmed). Together with R1's robust
→ no-benefit, this establishes a clean, controlled law: **coverage-gating improves performance exactly
when the unit-combiner is redundancy-fragile.** Because PAO's skill-application is fragile (additive
bias, P3 harm), this predicts coverage-gated skill admission *will* help PAO's task performance (not
just parsimony) — the de-risked basis for taking it into the RL regime next. I revise if the fragile
combiner shows no growing gap.
