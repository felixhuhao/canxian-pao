# Pre-Registration — PAO trigger under partial observability: is PAO's value the deployment decision?

> **FROZEN 2026-06-16, before the confirmatory run** (monolith baseline calibrated to fairness, see note; no
> held-out data seen). Calibration seeds 900–902; confirmatory **N=8**
> held-out seeds 910–917 (first look). If it survives, a longer N=20 run on a fresh held-out band 920–939.
> Code: `exp_pao_coverage/trigger.py` (reuses `grid.py`, `r4.py`). The constructive PAO rung defined in
> `SYNTHESIS.md` §3.

## 1. Claim under test
The synthesis of the RDD and Busch results (`SYNTHESIS.md`): usefulness = task-aligned **deployable** signal
under capacity. Mapped onto PAO: a crystallized skill is useful only if it is **triggered in the states where
it is the correct action** — *triggerability* is PAO's analog of Busch's controllability. The PAO ladder
(R0→R4) + R5 localized PAO's actionable failure as **indiscriminate triggering** (firing many skills at once
corrupts the additive policy). So the sharpened hypothesis: **PAO's value, if any, lives entirely in the
trigger (the state→skill deployment decision), not in crystallization timing, skill count, or diversity.**
The trigger only *binds* under partial observability — when the agent cannot perfectly read which skill applies.

## 2. Design (reuse the K=4 grid; corrupt the task cue)
- Env: `grid.py`, 4 distinct goal niches. **Partial observability** = the task one-hot in the observation is
  corrupted by Gaussian noise `e ~ N(0, σ²)^K`, drawn **once per episode** (task identity fixed but observed
  noisily); position stays clean. σ sweep ∈ {0, 0.3, 0.6, 1.0, 1.5}.
- **Skill library** (shared by the two PAO arms): K skills, each a `train_mastered` specialist for one niche
  (trained on the CLEAN cue; competence ≥ 0.9). Once selected, a skill acts from the true position with its own
  clean one-hot — i.e. it is a mastered position→action policy for its goal.
- **Arms:**
  1. **Gated trigger** — deploy the single skill `k̂ = argmax(noisy cue)` (Bayes-optimal gate for symmetric
     Gaussian noise — a *favorable* gate for PAO). Run that skill greedily.
  2. **Fire-all (indiscriminate)** — PAO's R5 liability: sum the policy logits of ALL K skills (each fed its own
     clean one-hot + true position), argmax. Cue-blind.
  3. **Strong monolith** — one PPO net trained end-to-end on the noisy multi-niche task **at the same σ**.
     *Made fair during calibration (see note):* capacity-matched (hidden=128, vs the K-network library),
     interleaved large-batch multi-task updates (budget = K×600 eps, matched to the skills), and retry-to-best
     across `MONO_RESTARTS=2` (matching the skills' `RETRY` discipline). The fair non-skill baseline.

> **Calibration note (monolith fairness, 2026-06-16, pre-freeze, no held-out data).** Naive monolith training
> was an unfair strawman: per-episode PPO updates caused catastrophic forgetting (collapsed to 3/4 niches; more
> episodes made it *worse*), and a single 32-hidden net under-matched the K-network library. Fixes, each
> verified on calib seeds 900–902: (a) interleave all niches into each update (multi-task gradient); (b)
> capacity hidden=128; (c) large batch (MONO_EPT=4 eps/task/update) to stop high-variance collapse; (d)
> retry-to-best ×2 to match the skills' retry-to-competence. Result: σ=0 monolith → **1.00** on all 3 calib
> seeds (was 0.75–0.92), σ=0.6 → 0.58. The monolith is now a genuine strong baseline. Design/predictions
> unchanged; held-out seeds 910–917 untouched.
- Metric: greedy **success rate**, averaged over the 4 niches, n eval episodes per (niche, σ). Held-out seeds.

Same skill library underlies arms 1 and 2 — so any gated−fireall gap is *purely the deployment decision*.

## 3. Pre-registered predictions (on record)
- **P1 (core — value is the trigger).** Gated ≫ Fire-all wherever the cue is informative (σ ≤ ~1.0): same
  skills, only the deployment policy differs (paired Wilcoxon p<0.05, large g). Indiscriminate triggering
  squanders a competent library.
- **P2 (shape).** Gated success ≈ cue classification accuracy: ≈1 at σ=0, degrading monotonically with σ;
  Fire-all is flat and low across σ (cue-blind). The gated−fireall gap is **largest at low σ** and shrinks as
  σ→∞ (when no policy can deploy correctly).
- **P3 (decisive — vs monolith).** At matched budget, Gated ≈ Strong monolith (PAO buys the trigger discipline,
  **not** a modularity advantage). We report the per-σ Gated−Monolith curve: any σ-band where Gated > Monolith
  (CI excludes 0) is PAO's regime-specific niche (pre-mastered skills + explicit gate beat end-to-end learning
  under partial obs); if Gated ≤ Monolith at every σ, PAO has **no** value beyond not self-sabotaging.

## 4. Interpretation locked in advance
If P1–P3 hold: PAO's value is **localized entirely to the trigger**. The same skills go from competent (gated)
to useless (fire-all) by changing only the deployment decision — confirming R5's "indiscriminate triggering"
as the liability and the synthesis's triggerability=controllability mapping. And a budget-matched monolith is
competitive, so PAO is not a free lunch over a plain network; its only defensible contribution is a
competence-gated trigger that avoids self-sabotage (and, if P3's band exists, a narrow partial-observability
regime where modular pre-mastered skills + an explicit gate help). This is the constructive complement to the
earlier falsifications: not "PAO's mechanisms are inert," but "here is the *only* place PAO value can live, and
how large it is." **Revision rule:** I revise the hypothesis if Fire-all ≈ Gated (triggering is not the lever),
or if Gated ≫ Monolith broadly (modularity itself, not just the trigger, carries value). N=8 is a first look;
a positive, surviving pattern is confirmed on the fresh N=20 band before any strong claim.
