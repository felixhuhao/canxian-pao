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
