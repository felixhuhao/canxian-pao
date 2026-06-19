# Pre-Registration -- DPA overlap robustness

> Draft/freeze target: run before using the result in `docs/dpa_v13.tex`.
> Code: `exp_dpa/overlap_robustness.py`.
>
> This addresses the main fragility in the paper's overlap claim: the original overlap correlation was measured
> against one T5 adaptation trajectory, using input-space direction cosine as a proxy for gate-representation
> overlap.

## Claim under test

Default-margin drift should propagate earlier to regimes whose observations overlap more with the adaptation
regime. If this is a genuine propagation law rather than one T5 trajectory's geometry, the negative
overlap-versus-collapse-step relation should repeat for additional adaptation regimes.

## Design

Self-contained NavEnv2D-style control, using the same environment abstraction as
`exp_dpa/objective_control.py`:

- five frozen directional skills tied to T0--T4;
- an MLP gate pre-trained on T0--T4;
- hard CE adaptation to S0 on all-zero sparse skill-evaluation observations;
- dense probe targets spread over the grid.

Adaptation regimes:

- T5 `(7, 5)`;
- T6 `(5, 3)`;
- T7 `(5, 7)`.

T6 and T7 are the decisive additions: they test whether the overlap law survives beyond the original T5
trajectory.

## Metrics

Primary:

- routing-collapse step for each probe target: first checkpoint where S0 default-rate on that probe's
  pre-adaptation trajectory observations is at least 0.5;
- Spearman correlation between collapse step and overlap. Prediction: negative.

Overlap measures:

- input direction cosine: cosine between mean target-direction components of adaptation and probe observations;
- gate representation cosine: cosine between mean hidden activations of the pre-adaptation gate.

Sensitivity:

- recompute each correlation after removing the two latest-collapsing / non-collapsing probe targets.

Secondary:

- behavioral success collapse under the same adapted gates. This is noisier because arbitrary dense targets can
  be unreachable by a particular frozen-skill composition, so the routing-collapse metric is primary.

## Interpretation

If T6/T7 reproduce the negative relationship, especially in gate-representation space and after the leave-two-out
sensitivity, the overlap result is stronger than a one-trajectory anecdote.

If they do not, the paper should state the original result as a T5-specific geometry diagnostic rather than a
general law.
