# Pre-Registration -- DPA objective control: is collapse specific to hard outcome pseudo-labels?

> Draft/freeze target: run before using the result in `docs/dpa_v13.tex`.
> Code: `exp_dpa/objective_control.py`.
>
> This is a boundary test for the Default-Policy Attractor (DPA) paper. The paper's causal mechanism is:
> all skills fail under sparse evaluation, deterministic argmax converts the all-zero vector into label S0, and
> cross-entropy adaptation turns that arbitrary label into default-index drift. A natural reviewer objection is
> that this is not a generic gate-plasticity failure: a reward-gradient objective on the same all-zero evidence
> should have zero or entropy-flattening drift, not a default-index attractor.

## Claim under test

The default-index attractor should appear under **hard CE pseudo-label adaptation** and should not appear under
**reward-gradient adaptation** on the same all-zero sparse skill-evaluation signal.

This scopes the paper correctly:

- if reward-gradient adaptation does not collapse, DPA is a mechanism of hard outcome-pseudo-labeling/self-training
  pipelines, not gate plasticity in general;
- if reward-gradient adaptation also collapses, the paper earns a broader claim about gate plasticity.

## Design

Self-contained NavEnv2D-style control, matching the paper's abstract structure:

- 2D continuous grid, targets T0--T9 as in the manuscript;
- five frozen directional skills tied to T0--T4;
- an MLP gate trained before adaptation to route compositionally from observation `(position, target direction)`;
- T5 adaptation observations collected from the pre-adaptation gated policy.

If a trajectory observation happens to allow one frozen single-skill rollout to pass near T5, it is excluded
from the adaptation batch. The objective-control test is explicitly about the all-zero evidence subset where
the sparse evaluator contains no skill information.

Arms:

- **CE-hard:** all-zero sparse skill returns are converted by deterministic argmax into label S0; update gate with
  cross-entropy.
- **REINFORCE-zero:** sample a skill from the gate for each adaptation observation; reward is the sparse success of
  that single frozen skill on T5. When all rewards are zero, the expected gradient is zero.
- **REINFORCE-positive-baseline:** same as above but with a fixed positive baseline. With all rewards zero, this
  should push down sampled actions symmetrically and tend toward entropy flattening rather than a privileged S0.

Metrics:

- average hard-gated success on T5--T9;
- default-skill rate and mean `p(S0)` on T5 adaptation observations;
- label/reward degeneracy rate: fraction of adaptation observations for which every frozen skill gets zero return.

## Predictions

- **P1:** CE-hard increases default-skill rate and collapses average T5--T9 success.
- **P2:** REINFORCE-zero leaves the gate approximately unchanged when sparse rewards are all zero.
- **P3:** REINFORCE-positive-baseline may flatten gate probabilities, but should not create a privileged S0
  attractor.

## Interpretation

If P1--P3 hold, the manuscript should explicitly claim **default-policy attractors from hard outcome
pseudo-labels**, not generic collapse of reward-trained modular gates. This is not a negative result; it makes
the mechanism sharper and harder to attack.
