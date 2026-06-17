# Pre-Registration — PAO online/learned skills: does factor-then-route survive without the clean-skill gift?

> **DRAFT — to be FROZEN before the confirmatory run.** Calibration seeds 1020–1022; confirmatory **N=8**
> held-out seeds 1030–1037; if it survives, **N=20** on fresh band 1040–1059. Code:
> `exp_pao_coverage/online_skills.py` (reuses `trigger.py`, `gate.py`). Priority 3 of
> `PAO_EXPERIMENT_DIRECTIONS.md`.

## 1. Claim under test
All prior PAO-positive results handed the agent **clean, isolated skills** (each specialist mastered on a
single niche with a clean cue). The de-confound (Priority 2) showed the lever is **factor-then-route on a
clean-trained policy** — and crucially that "factored without the clean gift" (a shared multitask net trained
under noise) *is* the vanilla monolith. So the remaining question is whether the advantage survives when the
**skills themselves must be crystallized under the same partial observability** the agent faces — i.e. you do
**not** get to cleanly separate niches during skill-building. If a library learned under noise still beats the
monolith, factor-then-route is genuinely useful for PAO; if not, the constructive positive depended on the
clean-skill gift (isolated, labeled niches) that a monolith never receives.

## 2. Design (no clean gift: crystallize and deploy under the same noise)
Same K=4 grid, σ_train = σ_test = σ ∈ {0.3, 0.6, 1.0}. **Learned skill** k is crystallized under noisy niche
assignment: each training episode is attributed to skill k but the *true* goal is wrong with probability
`c(σ)` = the cue mis-classification rate at σ (estimated by sampling argmax of `e_k + N(0,σ²I_K)`). The skill
always sees its own identity k (clean one-hot) + position, but is rewarded toward the (sometimes wrong) goal —
modeling crystallization from experiences you *believe* are niche k but that noise mislabeled. So library
quality degrades with σ. (At deploy, a skill is invoked exactly as before: fed clean one-hot k + position.)

2×2 decomposition + baselines, all on held-out seeds:

| arm | gate | skills | isolates |
|---|---|---|---|
| **oracle/clean** | Bayes | clean mastered | reference upper bound (= P2 gated) |
| **oracle/learned** | Bayes | learned-under-noise | library-quality loss (gate perfect) |
| **PAO-full** | learned (REINFORCE) | learned-under-noise | the full no-gift PAO condition |
| **monolith** | — | — (end-to-end noisy net) | the honest no-gift baseline |
| fire-all / random-gate | — | learned | floors |

Monolith = fair (hidden=128, interleaved, budget-matched, retry-to-best). Metric: greedy success averaged
over niches, held-out seeds, paired Wilcoxon + Hedges g.

## 3. Pre-registered predictions (on record)
- **P1 — learned skills degrade the library:** oracle/learned < oracle/clean, the gap **growing with σ**
  (more contamination), g<0, p<0.05 at σ≥0.6.
- **P2 — factor-then-route SURVIVES the no-gift setting (the crux):** PAO-full > monolith at low–mid σ
  (σ=0.3, 0.6), g>0, p<0.05; the advantage may narrow or vanish at σ=1.0 as contamination bites. If PAO-full
  ≤ monolith across **all** σ, the constructive positive depended on the clean-skill gift.
- **P3 — gate vs library attribution:** the gap oracle/learned − PAO-full quantifies the *gate-learning* cost
  on top of the *library-quality* cost (oracle/clean − oracle/learned); we report both so the loss is
  attributed to skills vs router.

## 4. Interpretation locked in advance
If P1–P2 hold: PAO's factor-then-route advantage is **not** merely an artifact of being handed clean isolated
skills — a library crystallized under the same partial observability still routes to useful behavior and beats
end-to-end learning in the moderate-noise band, degrading as contamination grows. This is the strongest
no-gift form of the constructive positive and the natural precondition for any scale-up. Combined with P2
(modularity inert) the picture is: the lever is **infer-context-then-act**, and it survives both learned gates
(Priority 1) and learned libraries (here), within a partial-observability operating band.
**Revision rule:** if PAO-full ≤ monolith at every σ, the positive was gift-dependent and must be reframed
before scale-up; if oracle/learned ≈ oracle/clean (no degradation), the contamination model is too weak and
the test is uninformative (re-examine `c(σ)`). Calibration (1020–1022) only confirms non-degeneracy
(learned skills competent at σ=0.3, degraded at σ=1.0; monolith trains) and fixes nothing about the held-out
seeds; predictions frozen here.
