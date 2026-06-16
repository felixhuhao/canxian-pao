# Pre-Registration — PAO de-confound: is the trigger advantage modularity, or just factor-then-route?

> **DRAFT — to be FROZEN before the confirmatory run.** Calibration seeds 980–982; confirmatory **N=8**
> held-out seeds 990–997; if it survives, **N=20** on fresh band 1000–1019. Code:
> `exp_pao_coverage/deconfound.py` (reuses `trigger.py`, `gate.py`). Priority 2 of
> `PAO_EXPERIMENT_DIRECTIONS.md`. The single biggest threat to the constructive PAO positive.

## 1. Claim under test
The trigger/gate experiments showed a gated skill library beats an end-to-end monolith under partial
observability. But that comparison bakes in two asymmetries: the gated arm gets (a) **clean-cue skill
training** and (b) **factorization** (classify the noisy cue → niche, then act), while the monolith learns
denoising + control jointly, end-to-end. So the win may be *factor-then-route on a clean-trained policy* —
which **one shared network** can also do — rather than the **modular skill library** PAO foregrounds.
This is the project's `property ≠ usefulness` test aimed at PAO's modularity claim.

## 2. Design (decompose the advantage one ingredient at a time)
Same K=4 grid, per-episode Gaussian cue noise σ ∈ {0.3, 0.6, 1.0} (the informative band where the effect
lives; σ=0 ties and σ=1.5 decays, both uninformative for this de-confound). **Gate held at the Bayes oracle**
wherever a gate exists, so the only thing varying is the policy representation:

| arm | clean training | factorization (gate) | modular skills | = |
|---|---|---|---|---|
| **vanilla monolith** | no | no | no | end-to-end noisy net |
| **curriculum monolith** | yes (clean→noisy) | no | no | clean-pretrained end-to-end net |
| **factored (Bayes)** | yes | yes | **no** | Bayes gate + ONE shared clean multitask net |
| **gated (Bayes)** | yes | yes | **yes** | Bayes gate + K clean mastered skills |

Ladder **vanilla → curriculum → factored → gated** adds exactly one ingredient per step (clean training →
factorization → modularity), so each gap attributes cleanly. Monoliths and the clean multitask net are
capacity-matched (hidden=128), budget-matched (≈K×600 eps), and retry-to-best (matching the skills' RETRY).
Floors/context (reported, not in the ladder): **gated-learned** (reward-trained gate, the realistic PAO arm),
**fire-all**, **random gate**, all over the same K skills. Metric: greedy success averaged over niches,
held-out seeds, paired Wilcoxon + Hedges g.

## 3. Pre-registered predictions (on record)
- **P1 — modularity is NOT the lever:** gated(Bayes) ≈ factored(Bayes) (|g| small, no significant gap at
  any σ). K separate mastered skills do **not** beat one shared clean-trained multitask net, same gate.
- **P2 — factorization IS the lever:** factored(Bayes) > curriculum monolith and > vanilla monolith
  (g large, p<0.05) across the band. Factor-then-route beats end-to-end even with clean pretraining.
- **P3 — clean training alone is insufficient:** curriculum ≈ vanilla (clean pretrain without factorization
  does not close the gap), or at most a small partial gain that stays well below factored.

## 4. Interpretation locked in advance
If P1–P3 hold: PAO's **modular skill library is not the load-bearing object** — the lever is the
**factor-then-route structure** (infer context, then act with a clean-trained policy), which a single shared
network achieves equally. This *refines, not refutes,* the constructive positive: PAO has value, but the value
is the routing/factorization, not modularity per se — exactly the project's `property ≠ usefulness` pattern,
now applied to PAO's own central design choice. It also makes the forward claim falsifiable and cheap to
honor: build a router over *a* policy (modular or not), and judge by deployable coverage.

**Alternative outcomes that change the story (revision rule):**
- If **gated(Bayes) ≫ factored(Bayes)** (modularity helps): separate skills genuinely reduce interference vs
  one shared net → PAO's modular library *is* load-bearing; keep it central. I revise toward modularity.
- If **factored ≈ curriculum ≈ vanilla** (no factorization gain): the earlier "beats monolith" result was a
  monolith-training artifact; a fair end-to-end net closes the gap → the constructive positive weakens
  materially and must be re-examined before any scale-up.

Calibration (980–982) only confirms the new arms are non-degenerate (clean multitask net ≈1.0 at σ=0;
curriculum trains; random gate ≈ chance) and fixes the curriculum clean/noisy split; held-out seeds untouched;
predictions frozen here.
