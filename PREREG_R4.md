# Pre-Registration — Ladder R4: does coverage-gated skill admission help in PAO's RL regime?

> **FROZEN 2026-06-15, before the N=8 run.** Held-out seeds 100–107 (sanity used seed 0). This is the
> capstone rung of the RDD→PAO ladder: transfer R1b's law into the RL regime using **PAO's own
> combiner** (skills applied as ADDITIVE bias to policy logits). Code: `exp_pao_coverage/{grid.py,r4.py}`.

## 1. Claim under test
R1b established: **coverage-gating's performance payoff ⟺ the unit-combiner is redundancy-fragile**
(robust ridge → no benefit; fragile averaging → gating helps, scaling with capacity). PAO's
skill-application is fragile — skills are applied as additive policy bias and a wrong/redundant skill
actively hurts (the P3 finding). Hypothesis: in a multi-task RL setting with PAO's additive combiner,
**coverage/competence-gated skill admission improves task success over ungated admission, and the gap
grows with the amount of redundant junk** (capacity).

## 2. Design
**Env (`grid.py`):** 5×5 grid, start centre, K=4 goals at the cardinal edge-midpoints (up/down/left/right)
— 4 genuinely-distinct niches (validated: a fresh specialist solves only its own niche; no monotone
policy covers two, unlike the degenerate 1D corridor). Obs = [x,y normalised, onehot(task,4)]; potential-
based shaping (provably policy-preserving) makes single-task learning reliable. Success = greedy reach
of the goal.

**Combiner (PAO-faithful, additive):** for eval task k, every library skill tagged to niche k is
TRIGGERED and its action-logits are SUMMED onto the base policy's logits, then argmax. Summing means
junk biases ACCUMULATE → corruption grows with the number of admitted junk skills (redundancy-fragile).

**Library:**
- `base`: round-robin-undertrained policy (mediocre/forgetful — solves ~3/4 niches, needs skills).
- one MASTERED skill per niche (fresh specialist trained to competence, ≥0.9 solo success; up to 4 retries).
- `cap` MIS-ASSOCIATED skills per niche = junk: a skill mastered on a *different* niche but tagged to k
  → a confidently-wrong bias when it fires for task k (models PAO's real failure: a wrong skill fires
  in the wrong context, P3). `cap ∈ {0,1,2,4,8}` is the capacity/redundancy axis.

**Arms (greedy eval, 30 ep/niche):** `noskill` (base alone); `ungated` (base + all niche-matched skills,
1 good + cap junk); `gated` (admit one competent niche-covering skill per niche — competence checked by
solo success ≥0.9, which rejects all mis-associated junk).

## 3. Sanity (1 seed) & integrity
Seed 0, caps {0,1,2,4,8}: ungated mean success 1.00→1.00→0.75→0.25→0.25 (degrades as junk accumulates);
gated 1.00 at every cap (immune); noskill flat 0.75 (floor). Gap (gated−ungated) = 0,0,+0.25,+0.75,+0.75
— grows with capacity. Also observed (v1): *immature/weak* junk does NOT corrupt the additive sum —
only *confidently-wrong* (mis-fired) junk does → PAO's danger is wrong-skill TRIGGERING, not premature
crystallization per se. Design/metrics frozen here; N=8 held-out seeds confirm.

## 4. Pre-registered decision rule
- **GATING HELPS** iff gated > ungated on mean success (paired Wilcoxon, p<0.05, at cap≥2) AND the gap
  Δ=gated−ungated grows with capacity (Spearman(cap, Δ) > 0, p<0.10).
- This **transfers R1b's law into PAO's regime**: with PAO's fragile additive combiner, coverage-gated
  skill admission improves *performance* (not just parsimony).
- **NULL / REFUTED** if gated ≯ ungated or the gap does not grow with capacity.

## 5. Prediction (on record)
**Gating helps, gap grows with capacity** (sanity-confirmed). This completes the controlled RDD→PAO
ladder: the coverage-gating performance law holds in the RL regime *because* PAO applies skills through
a redundancy-fragile additive combiner. The added mechanistic insight on record: the harm comes from
*confidently-wrong* (mis-triggered) skills, not weak/immature ones — so the actionable PAO failure mode
is indiscriminate skill triggering, and the fix is competence-/coverage-gated admission. I revise if the
held-out gap does not grow with capacity or gated does not beat ungated.
