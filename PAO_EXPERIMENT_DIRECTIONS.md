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

## Priority 1: Learned Gate

Finish the `PREREG_PAO_GATE.md` direction: replace the oracle/Bayes gate from the trigger experiment with a
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

## Priority 2: Gate Learning Envelope

Map the regime where routing works before scaling the environment.

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

## Priority 3: Online Skills

The current constructive trigger result uses clean pre-mastered skills. The next rung should learn or
crystallize skills inside the experiment, then learn the gate over that library.

Important decompositions:

- oracle gate over learned skills: tests skill quality while removing gate-learning failure,
- learned gate over oracle skills: tests gate learning while removing skill-quality failure,
- learned gate over learned skills: the full PAO-relevant condition.

This separates "the library is bad" from "the router is bad."

## Priority 4: Replace `H_M` With Deployable Coverage

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

## Priority 5: Partial-Observability Scale-Up

Move from the current noisy-cue grid to richer POMDPs where PAO should have a real niche.

Good next environments:

- MiniGrid key-door variants,
- recurring hidden-context tasks,
- delayed or missing task cues,
- cue inferred only from short history,
- context switches that recur but are not directly signaled.

The prediction is specific: PAO should help when a monolith struggles to bind noisy context to reusable
behavior, and the advantage should shrink when the state is fully observable.

## Priority 6: Failure Boundary

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

1. Learned gate on the existing partial-observability trigger harness.
2. Gate envelope sweeps to identify the workable regime.
3. Learned or crystallized skills, with oracle-vs-learned gate decompositions.
4. Admission based on deployable coverage rather than `H_M`.
5. MiniGrid/POMDP scale-up with recurring hidden contexts.
6. Failure-boundary maps for wrong-skill cofire and stale skills.

The end goal remains PAO, but the evidence says the load-bearing object is the router: PAO should be judged
by whether it can learn when and where each skill is deployable.
