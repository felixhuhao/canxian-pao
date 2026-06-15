# Pre-Registration — RDD rescue: can lateral inhibition be salvaged by making it task-aligned?

> **FROZEN 2026-06-15, before the N=20 run.** Held-out seeds 400–419 (calibration used 300–303). Code:
> `exp_rdd/rdd_rescue.py` (reuses `rdd_li.py`, `rdd_taware.py`). The optional final RDD-path rung: a
> constructive *rescue attempt* for LI, designed to isolate the single ingredient that makes auxiliary
> diversity useful.

## 1. Claim under test
Prior RDD results: generic τ-repulsion (`rdd_li`/`crossover`) and the direct deadlock test
(`rdd_deadlock`) find LI inert or harmful; only a task-aligned **coverage reward** helped (`rdd_taware`).
Open question: is LI rescued simply by making its repulsion **task-aligned** (operate in the signal-
weighted spectral metric), or does the *objective form itself* have to change from repulsion to coverage?

## 2. Design (M=8 over-complete, asym; hold everything fixed but the auxiliary term)
- `noLI` — baseline.
- `genLI` — τ-space pairwise repulsion `Σ exp(−(τ_i−τ_j)²/2ℓ²)` (β=10). Task-blind repulsion.
- `spec` — signal-weighted spectral **overlap penalty** `Σ_{i<j}Σ_f Ŝ(f)p_i(f)p_j(f)` (β∈{10,50,100}).
  Task-aligned **repulsion** (push channels apart in the signal's spectrum).
- `cov` — signal-weighted **max-envelope coverage reward** `−Σ_f Ŝ(f)max_m p_m(f)` (β∈{10,50}).
  Task-aligned **coverage** (fill the signal's spectrum).
`spec` and `cov` are equally task-aligned (both signal-weighted spectral); they differ ONLY in
pairwise-product (repulsion) vs max-union (coverage). Metric: held-out test/OOD MSE, τ-diversity,
coverage diagnostic. N=20.

## 3. Sanity (4 seeds) & integrity
noLI 3.580; genLI 4.438 (hurts); spec10 3.944 (worse), spec50/100 6.50 (catastrophic collapse, div→0.17);
cov10/cov50 3.39/3.38 (helps, g≈−2.4/−2.9). KEY: cov vs best spec g=−4.94 (p=0.062 = n=4 Wilcoxon floor).
Design/metrics frozen here; N=20 confirms.

## 4. Pre-registered predictions (on record)
- **Task-alignment does NOT rescue LI:** `spec` (any β) is null-to-harmful vs noLI — mild β ≈ noLI or
  worse, strong β collapses the model. Like `genLI`, it does not help.
- **Coverage DOES help:** `cov` beats noLI on test (and OOD) MSE, g ≤ −0.5, p<0.05.
- **The rescuing ingredient is the OBJECTIVE FORM:** `cov` beats the best `spec` (g<0, p<0.05), even
  though both are equally task-aligned.

## 5. Interpretation locked in advance
If confirmed: LI-qua-repulsion **cannot** be salvaged by task-alignment. Making repulsion spectral and
signal-weighted still fights the task (and over-strength repulsion actively collapses the model); the
useful object is a different one entirely — a **coverage/union** objective that rewards filling the
signal's spectrum, not pushing channels apart. This pins the earlier positive result precisely: the win
in `rdd_taware` came from *coverage*, not from *diverse/repelled* channels. It closes the RDD path with a
consistent verdict — diversity/repulsion (RDD's mechanism) is neither the deadlock-escape force nor a
performance lever; only task-aligned coverage helps, and coverage is not what LI computes. I revise if
`spec` helps or `cov` fails to beat `spec`.
