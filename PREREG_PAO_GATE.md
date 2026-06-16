# Pre-Registration — PAO learned gate: does the trigger advantage survive without the oracle?

> **FROZEN 2026-06-16, before the confirmatory run.** Confirmatory **N=8** held-out seeds 950–957; if it
> survives, **N=20** on fresh band 960–979. Code: `exp_pao_coverage/gate.py` (reuses `trigger.py`). Follow-on
> to the trigger experiment (`PREREG_PAO_TRIGGER.md`). The monolith baseline is unchanged from `trigger.py`
> (already calibrated to fairness there).
>
> **Calibration note (gate, 2026-06-16, pre-freeze, seed 940 — a non-held-out seed).** Smoke test confirmed
> the REINFORCE bandit gate trains and recovers ≈Bayes: σ=0 learned 1.00 / Bayes 1.00; σ=0.6 0.66 / 0.70;
> σ=1.0 0.44 / 0.44 (shortfall ≤0.04). Gate hyperparams fixed (hidden=32, lr=5e-3, budget = monolith's 2400
> episodes, EMA baseline). Held-out seeds 950–979 untouched; predictions below frozen.

## 1. Claim under test
The trigger experiment used a **Bayes-optimal** gate (`argmax(noisy cue)`) — an oracle that knows the cue
structure and noise model for free. That is an *upper bound* on the trigger. The honest question: **how much
of the trigger advantage survives when the gate is learned from reward**, with no niche labels and the same
experience budget the monolith gets? If most of it survives, the trigger advantage is not an artifact of the
oracle; if it collapses, the oracle was doing the work.

## 2. Design (reuse the trigger harness; swap the gate)
Same K=4 grid, same per-episode Gaussian cue noise σ ∈ {0, 0.3, 0.6, 1.0, 1.5}, same mastered skill library,
same fair monolith (`trigger.py`). New arm:
- **gated-learned** — a small gate net `g(noisy cue) → logits over K skills`, trained by **REINFORCE**
  (contextual bandit, one skill choice per episode): sample task k, present noisy cue, sample skill
  `j ~ softmax(g(cue))`, deploy that mastered skill for the episode, reward = episode success (0/1), update
  the gate with a running-mean baseline. **No niche labels — reward only.** Budget = the monolith's
  (≈ K×600 episodes, matched). Eval deploys `argmax g(cue)`.
- Reference arms (from the trigger experiment): **gated-Bayes** (`argmax(noisy cue)`, the oracle upper
  bound), **fire-all** (R5 liability), **monolith** (fair end-to-end baseline).
Metric: greedy success averaged over the 4 niches, n eval episodes per (niche, σ), held-out seeds.

## 3. Pre-registered predictions (on record)
- **P1 (trigger still beats indiscriminate).** gated-learned ≫ fire-all wherever the cue is informative
  (σ ≤ ~1.0), large g, p<0.05.
- **P2 (most of the oracle gap closes).** gated-learned recovers most of gated-Bayes: the shortfall
  `gated-Bayes − gated-learned` is small at low σ and may widen at high σ (noisy reward → harder bandit). We
  report the per-σ shortfall as *the value of the oracle*.
- **P3 (advantage survives vs monolith).** gated-learned > monolith across the partial-observability range
  (σ>0), tying at σ=0 — i.e. the trigger advantage is **not** an oracle artifact. The decisive, falsifiable
  sub-claim: if gated-learned ≤ monolith at σ where gated-Bayes > monolith, then the oracle was carrying the
  advantage and the constructive PAO result is weaker than it looked.

## 4. Interpretation locked in advance
If P1–P3 hold: the constructive PAO finding stands without the oracle — a gate **learned from reward at the
monolith's budget** recovers most of the Bayes trigger and still beats end-to-end learning under partial
observability, because the K-way skill-selection problem is far easier than learning the full noisy policy.
This strengthens "PAO's value is the trigger": even the *learnable* trigger pays off. The per-σ
gated-Bayes − gated-learned shortfall quantifies what perfect routing is worth on top of a learned router.
**Revision rule:** I revise the constructive claim if gated-learned ≤ monolith in the band where gated-Bayes
beat it (oracle artifact), or if gated-learned ≈ fire-all (a reward-learned trigger cannot be acquired here).
Calibration (940–942) only confirms the bandit gate trains (recovers ≈argmax at σ=0) and fixes the gate's lr,
episode budget, and baseline; held-out seeds untouched; predictions frozen here.
