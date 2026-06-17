# PAO Experiment Directions

Context and proposed next rungs for the PAO end goal.

## Context

The original PAO mechanisms have been stress-tested and mostly do not survive controlled tests:
event-triggered crystallization timing does not select specially reusable windows, BOCPD fires too early in
the harder environment, manifold health `H_M` is anti-predictive of skill usefulness, and the Two-Gate
fast-recovery metric can be produced by a head-start artifact.

The constructive result is narrower and sharper. In `exp_pao_coverage/trigger.py`, the same mastered skill
library becomes useful or useless depending only on deployment:

- gated state-to-skill deployment works,
- fire-all deployment collapses,
- the gated arm beats a fair monolith under partial observability and ties it at full observability.

So the live PAO hypothesis should be:

> PAO has value when it is a reusable skill library plus a learned, competence-aware state-to-skill trigger
> that deploys the right skill under partial observability.

This shifts the end goal away from "prove crystallization timing is special" and toward "build and test a
router that turns a skill library into deployable coverage."

## Priority 1: Learned Gate — DONE (2026-06-16)

Result (`PREREG_PAO_GATE.md`, `exp_pao_coverage/gate.py`, N=8→N=20, `FINDINGS.md`): the trigger advantage
**survives without the oracle, at ~half magnitude.** learned ≫ fire-all (g=7.5–15.7); the learned gate
recovers most of Bayes (shortfall ≤0.08, growing with noise); learned > monolith across σ>0 — significant in
the mid-high band (g up to +2.6), marginal at σ=0.3, tie at σ=0. So the PAO positive is **not** mostly an
oracle upper bound. (Arms run: learned, bayes, fire-all, monolith. The `random gate` and `no-skill PPO` arms
listed below were not run; fold them into the next rung.)

Finish notes for the original spec: replace the oracle/Bayes gate from the trigger experiment with a
reward-trained gate.

Core arms:

- oracle gated library,
- learned gate,
- fire-all,
- random gate,
- monolith,
- no-skill PPO.

The decisive question is not whether the learned gate reaches the oracle exactly. It is whether it retains a
meaningful fraction of the oracle advantage and beats the monolith in at least one partial-observability band.
If it cannot, the current PAO positive is mostly an oracle upper bound.

Useful measurements:

- success by cue noise level,
- gate accuracy and calibration,
- false-positive rate for wrong skills,
- regret relative to oracle gate,
- reward gap against monolith.

## Priority 2: Baseline Steelman / De-confound — DONE (2026-06-17)

Result (`PREREG_PAO_DECONFOUND.md`, `exp_pao_coverage/deconfound.py`, N=8→N=20, `FINDINGS.md`): **the lever
is factor-then-route, not the modular skill library.** Ladder vanilla→curriculum→factored→gated(Bayes):
factored ≫ curriculum/vanilla (g=2–3.8, p<1e-4 — the "beats monolith" win is *real*, not a training artifact);
gated ≡ factored byte-identical on all 20 seeds (modularity adds 0 when one shared net masters the library);
clean-training a minor, noise-growing contributor. So the constructive positive is real but **relocated to the
routing factorization** — PAO's modular library is not load-bearing in this regime. Scope: modularity could
still matter under interference (more niches / tighter capacity) → Priority 4. (`random gate`=chance,
`no-skill PPO`=vanilla monolith, both run.)

Original spec follows.

The "gated beats monolith" headline has two asymmetries baked into the current harness. Before mapping
envelopes or scaling environments, test whether the advantage is *real modular-reuse + routing* or a
**baseline/factorization artifact** — this is the biggest threat to the constructive positive and the cheapest
to check (same harness). It is the project's own `property ≠ usefulness` test, now aimed at PAO.

The two asymmetries:

- **Factorization.** The gated arm gets classification (noisy cue → niche) split off from control for free;
  the monolith must learn both jointly, end-to-end. The win might just be "factor context-ID from control,"
  which any architecture can do — not modular skills specifically.
- **Clean-skill pretraining.** Skills are trained on a clean cue; the monolith never gets that affordance.

Steelman baselines (add as arms; include the previously-unrun `random gate` and `no-skill PPO`):

- **Factored monolith** — a classifier (noisy cue → niche) + a *jointly-learned* shared multi-task policy.
  Same factorization as the gated arm, but no separate pre-mastered library. Isolates factorization vs
  modularity.
- **Curriculum monolith** — clean-cue pretrain → noisy fine-tune. Gives the monolith the skills' clean-training
  affordance. Isolates clean-reuse.
- **Random gate** and **no-skill PPO** — floors from the Priority-1 spec.

Decision rule: if gated beats **both** steelman baselines, the constructive positive is robust and genuinely
about modular reuse + routing → proceed to the envelope with confidence. If **either** closes the gap, we have
located the real lever (factorization or clean-curriculum) — the more important finding, and consistent with
the rest of the project. Pairs naturally with Priority 3 (learned-over-learned skills), which removes the
clean-skill asymmetry from the other side.

## Priority 3: Online Skills — DONE (2026-06-17)

Result (`PREREG_PAO_ONLINE.md`, `exp_pao_coverage/online_skills.py`, N=8→N=20, `FINDINGS.md`):
**factor-then-route survives the no-gift setting.** Skills crystallized under the same partial observability
(noisy attribution, contamination c(σ)) + a reward-learned gate beat the fair monolith at every σ, significant
and growing with noise (g=0.21/1.18/1.63, p≤.029). Not gift-dependent. Magnitude ladder across the removed
gifts: clean+oracle (g~2–4) → learned gate (g~0.7–2.6) → learned gate + learned skills (g~0.2–1.6). Cost split
small between library contamination and gate-learning. Caveat: learned skills get retry-to-best (upper bound on
library quality). Combined with Priorities 1–2, the PAO positive is robust and de-confounded.

Original spec follows.

Important decompositions:

- oracle gate over learned skills: tests skill quality while removing gate-learning failure,
- learned gate over oracle skills: tests gate learning while removing skill-quality failure,
- learned gate over learned skills: the full PAO-relevant condition.

This separates "the library is bad" from "the router is bad."

## Priority 4: Gate Learning Envelope

Map the regime where routing works before scaling the environment (do once Priorities 2–3 confirm the effect
is real).

Sweeps:

- cue noise,
- number of task niches,
- library size,
- skill quality,
- skill redundancy,
- gate training budget,
- false-positive and false-negative costs.

This should produce a PAO operating envelope: when does a learned router help, when does it tie, and when
does it collapse?

## Priority 5: Replace `H_M` With Deployable Coverage

Treat `H_M` as failed for skill admission unless a new ecological test rescues it. The admission criterion
should instead ask whether a skill adds task-aligned deployable coverage.

Candidate admission signals:

- validation rollout success,
- marginal reward improvement,
- coverage of an uncovered task niche,
- trigger confidence in the skill's niche,
- calibration of the gate for that skill,
- uncertainty or disagreement against existing skills.

The strongest version would admit a skill only when it improves held-out reward or covers states where the
current library has no reliable action.

## Priority 6: Partial-Observability Scale-Up

Move from the current noisy-cue grid to richer POMDPs where PAO should have a real niche.

Good next environments:

- MiniGrid key-door variants,
- recurring hidden-context tasks,
- delayed or missing task cues,
- cue inferred only from short history,
- context switches that recur but are not directly signaled.

The prediction is specific: PAO should help when a monolith struggles to bind noisy context to reusable
behavior, and the advantage should shrink when the state is fully observable.

## Priority 7: Failure Boundary

Deliberately map collapse instead of treating it as noise.

Stressors:

- wrong-skill cofire count,
- gate false-positive rate,
- library redundancy,
- weak versus confidently wrong skills,
- number of simultaneous active skills,
- stale skills after context drift.

This should turn PAO from a yes/no claim into an operating law: performance is governed by deployable
coverage minus confidently wrong cofire.

## Proposed Ladder

1. ~~Learned gate on the existing partial-observability trigger harness.~~ **DONE** (2026-06-16).
2. ~~Baseline steelman / de-confound (factored + curriculum monolith).~~ **DONE** (2026-06-17): the lever is
   **factor-then-route, not modularity**; the constructive positive is real but relocated to the routing.
3. ~~Learned or crystallized skills, with oracle-vs-learned gate decompositions.~~ **DONE** (2026-06-17):
   factor-then-route survives the no-gift setting (g up to +1.6, growing with noise); not gift-dependent.
4. Gate envelope sweeps to identify the workable regime (incl. the interference regime where modularity might
   re-enter). *Do this next.*
5. Admission based on deployable coverage rather than `H_M`.
6. MiniGrid/POMDP scale-up with recurring hidden contexts.
7. Failure-boundary maps for wrong-skill cofire and stale skills.

The end goal remains PAO, but the evidence (now incl. #2) says the load-bearing object is the **router /
factorization**, not the modular library: PAO should be judged by whether it can learn when and where to
deploy a context-appropriate policy. #2 showed a fair steelmanned baseline does *not* close the gap (the win
is real) but that one shared net matches K modular skills (modularity is not the lever). The next test is
**#3 — learned/online skills**, to remove the remaining clean-skill asymmetry and check whether routing still
pays once the policy itself must be learned in-experiment.
