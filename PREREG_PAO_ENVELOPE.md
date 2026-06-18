# Pre-Registration — PAO envelope: when does modularity re-enter under capacity pressure?

> **DRAFT / calibration-ready 2026-06-17.** Code: `exp_pao_coverage/envelope.py`. Priority 4 of
> `PAO_EXPERIMENT_DIRECTIONS.md`. This is the first envelope rung after the learned-gate, de-confound, and
> online-skill results.
>
> The current file is meant to be frozen before any confirmatory run. Use `--mode smoke` only for syntax and
> non-degeneracy checks; use `--mode calib` for non-held-out calibration; then freeze this file before
> `--mode confirm` or `--mode long`.

## 1. Claim under test

Priority 2 showed `gated(Bayes) == factored(Bayes)` in the K=4 grid when the shared factored policy has enough
capacity (`hidden=128`). That result relocated PAO's constructive value to **factor-then-route**, not the
modular skill library. The honest remaining possibility is that modularity re-enters under **interference or
capacity pressure**: one shared context-conditioned policy may stop mastering all niches when its hidden size
is squeezed, while K separate specialist policies remain competent.

So this rung asks:

> Is modularity inert only because the shared factored net has enough capacity, or does it remain inert even
> when the shared policy is capacity-limited?

## 2. Design

Same K=4 grid and noisy-cue setup as the trigger/de-confound experiments. The gate is held at the **Bayes
oracle** wherever a gate exists, so the policy representation is isolated.

Capacity axis:

- shared factored/monolith hidden size `H ∈ {8, 16, 32, 64, 128}`,
- modular specialist skills keep the existing `hidden=32` per specialist.

Noise axis:

- `σ ∈ {0.6, 1.0}` for the confirmatory sweep: the band where partial observability matters and monolith
  learning is not near-ceiling.

Arms:

- **gated(Bayes)**: K clean mastered modular skills + Bayes gate.
- **factored(Bayes, H)**: one shared clean-trained multitask net with hidden size H + Bayes gate.
- **monolith(H)**: one end-to-end noisy multitask net with hidden size H.

Metrics:

- mean greedy success over niches,
- modularity gap: `gated(Bayes) - factored(Bayes, H)`,
- factorization gap: `factored(Bayes, H) - monolith(H)`,
- Spearman trend of modularity gap versus H.

## 3. Pre-registered predictions

- **P1 — high-capacity replication:** at `H=128`, `gated(Bayes) ≈ factored(Bayes)` as in Priority 2.
- **P2 — modularity re-enters under capacity pressure:** at low H, `gated(Bayes) > factored(Bayes, H)`, and
  the modularity gap shrinks monotonically as H increases. This is the only remaining pro-modularity regime
  in the current harness.
- **P3 — factorization remains useful:** `factored(Bayes, H) > monolith(H)` in the mid/high-noise band for
  any H where the clean multitask policy is non-degenerate.

## 4. Interpretation locked in advance

If P1-P2 hold: modularity is **capacity-gated**. It is not the general PAO lever, but it becomes useful when a
single shared policy cannot carry all niches without interference. The live PAO law becomes:
**factor-then-route is the primary lever; modularity helps only when shared-policy capacity is binding.**

If `gated ≈ factored` at every H: modularity remains inert in this K=4 harness; the next envelope should move
to more niches/richer interference before spending on modular-library machinery.

If `factored ≈ monolith`: the factorization result does not survive the capacity sweep and the constructive
positive needs rechecking before scale-up.
