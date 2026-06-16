# exp_pao_coverage — Ladder R4: coverage-gated skill admission in PAO's RL regime

The **capstone rung** of the controlled RDD→PAO ladder (`../exp_rdd/`). It transfers the law established
in supervised regression — *coverage-gating's performance payoff ⟺ the unit-combiner is redundancy-
fragile* (R1/R1b) — into **multi-task RL using PAO's own combiner**: skills applied as ADDITIVE bias to
a base policy's action-logits, summed over all triggered skills.

## Files
- `grid.py` — 5×5 grid env with K=4 cardinal-goal niches (genuinely distinct: a fresh specialist solves
  only its own niche). Potential-based reward shaping (policy-preserving) for reliable single-task
  learning; tunable entropy. Run `python grid.py` for the learnability + non-degeneracy check.
- `r4.py` — the experiment. Base + one mastered skill per niche + `cap` mis-associated junk skills
  (confidently-wrong, mis-triggered) per niche. Additive (summed) combiner. Arms: noskill / ungated /
  gated. Run:
  ```bash
  python r4.py --seeds 8 --start 100 --caps 0 1 2 4 8   # writes results/r4.{log,pkl}
  ```
- `r5.py` — self-critique of R4: is the harm reachable through PAO's *real* channels (no planting)?
  Condition A = natural over-crystallization volume (immature, own-niche snapshots); B-rate = one wrong
  skill co-fires with prob ε; B-count = m wrong skills co-fire. Run:
  ```bash
  python r5.py --seeds 8 --start 200                    # writes results/r5.{log,pkl}
  ```
- `core.py` — the **superseded** 1D-corridor stage-1 (kept for the record). It is task-degenerate: a
  single monotone policy covers two goals, so it has only 2 real niches. `grid.py` replaces it.

Pre-registration: `../PREREG_R4.md` (frozen before the N=8 run, held-out seeds 100–107). Verdict:
`../FINDINGS.md` (2026-06-15 R4 entry).

## Result (N=8, held-out seeds 100–107)
- **GATING HELPS, and the gap grows with capacity — prediction confirmed.** Ungated success collapses as
  junk accumulates (0.97 → 0.03 over cap 0→8); gated is immune (0.97 at every cap); noskill flat (0.562).
  Δ(gated−ungated) grows monotonically, **Spearman(cap, Δ) = +1.000, p=0.000**; Wilcoxon p=0.004 at every
  cap≥1; Hedges g +3.7…+10. R1b's law holds in the regime PAO actually operates in.
- **Added insight:** *immature/weak* junk does not corrupt the additive sum — only *confidently-wrong*
  (mis-triggered) junk does. PAO's actionable failure mode is **indiscriminate skill triggering** (a
  competent-but-wrong skill firing), not premature crystallization per se; the fix is competence-/
  coverage-gated admission. Sharpens the P3 finding.

The ladder is now complete end-to-end: **R0 → R1 → R1b → R4.**

## R5 — is R4's harm reachable through PAO's own channels? (self-critique, N=8, seeds 200–207)
R4 used *planted* confidently-wrong junk. R5 tests the two real channels without planting:
- **Crystallization volume (immature, own-niche): HARMLESS, even mildly helpful** — ungated rises
  0.94→1.00 as more immature snapshots are added; gating them is slightly counter-productive
  (Spearman(cap, Δ)=−0.95). *(Pre-registered Δ≈0; held-out mildly contradicts the exact null but
  reinforces the qualitative claim — recorded as refined.)*
- **One wrong skill mis-firing (rate ε): MODERATE** — 0.94→0.63 over ε 0→1 (Spearman −1.0).
- **m wrong skills firing at once: STEEP COLLAPSE** — 0.94→0.04 over m 0→8 (Spearman −1.0; matches R4).

**Takeaway:** R4's catastrophe needs *many wrong skills firing simultaneously* — not crystallization
volume, nor single mis-fires. PAO's real liability is **indiscriminate fire-all triggering**; the lever
worth engineering is the **router/applicability**, not the crystallization counter. Deflates R4's
apparent strength while sharpening the actionable target. See `../FINDINGS.md` (2026-06-15 R5 entry).

## trigger.py — is PAO's value the trigger? (constructive positive, partial observability)
The synthesis (`../SYNTHESIS.md`) predicts PAO's value lives entirely in the **trigger** (state→skill
deployment) — the controllability/triggerability lens from the Busch work. Pre-registered in
`../PREREG_PAO_TRIGGER.md` (frozen 2026-06-16). The task one-hot in the observation is corrupted by
per-episode Gaussian noise σ (position clean) → partial observability over *which skill applies*. Arms share
one mastered skill library: **gated** (deploy `argmax(noisy cue)`, Bayes gate) / **fire-all** (sum all skills'
logits = R5 liability) / **monolith** (fair end-to-end net: hidden=128, interleaved multi-task, budget-matched,
retry-to-best → solves σ=0). Run:
```bash
python trigger.py --mode confirm   # N=8 first look (910-917)
python trigger.py --mode long      # N=20 confirmation (920-939)
```

### Result (N=8 survived → N=20 confirmed, held-out)
- **P1 — value IS the trigger:** same skills, gated ≫ fire-all at every σ (g=4.6–11.7, p<1e-4); gated
  1.00→0.45, fire-all ≈0.06 (the 4 skills cancel in the additive sum). Confirms R5's indiscriminate-triggering
  liability and the triggerability=controllability mapping.
- **P3 — pro-PAO:** gated beats even the *fair* monolith across all σ>0 (g=2.2–4.1, p≤1e-4), tying only at
  σ=0 (full obs, g=+0.31). PAO's regime-specific niche is **partial observability**: modular pre-mastered
  skills + an explicit gate beat end-to-end learning under a noisy cue; the edge vanishes when the trigger is
  trivial. Caveat: gated gets a Bayes-optimal gate (upper bound) and clean-trained skills — read as "clean
  skills + optimal gate," not modularity magic. Verdict: `../FINDINGS.md` (2026-06-16 trigger entry).
