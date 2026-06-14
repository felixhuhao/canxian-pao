# Pre-Registration — Does event-triggered skill REUSE help under unpredictable, recurring shifts?

> **FROZEN 2026-06-14, before the N=30 run.** Mechanism + decision rule frozen after a
> calibration phase (see §6). Results → `FINDINGS.md`. This is the **steelman** test: it gives
> PAO its best honest shot, in the one regime structurally engineered to need what PAO claims.

## 1. Why this test exists
P1/P2 swapped rules on a **fixed, known schedule**, where a clock ("crystallise at ep 40") is
sufficient *by construction* — so the trigger added nothing (falsified). Event-triggered
crystallisation/reuse can only earn its keep where a clock **cannot** know when to act:
**random, unsignaled, recurring** regime changes. This test builds exactly that.

PAO as shipped crystallises **once** (a single frozen sub-policy). The theory ("progressive
assembly") envisions a growing **skill library** reused across regimes. We steelman to that:
`PAOLibrary` — a library of frozen sub-policies + event-triggered re-selection.

## 2. Environment (symmetric, recurring, unsignaled)
1D Two-Gate corridor, task always `A→B`. Regimes = **action-mapping flips**: in a "flipped"
segment the agent's action labels are inverted before reaching the env. This makes the two
regimes (a) **equally learnable** (identical task, relabeled actions) and (b) **maximally
antagonistic** (a skill cached in one regime is exactly right when it recurs, exactly wrong
otherwise). The flip is **unsignaled** — observations are identical; only action semantics
differ — so observations alone cannot reveal the active regime. 8 back-to-back segments,
random lengths ∈ [40,90] episodes, alternating flip flag so each regime recurs ~4×.

## 3. Agent mechanism (frozen)
- **Crystallise (library growth), identical across all library arms:** snapshot the base
  policy as a frozen `ActorCritic` when it is competent + stable + past a debounce, **only
  while not exploiting a cached skill** and **at most once per inter-reset interval** → ~1
  skill per distinct regime. (`cryst_debounce=10`, `max_lib=6`.)
- **Re-select (event-triggered reuse), THE manipulated variable:** on a "reset" signal, enter
  a search — try each library skill (one episode each, exploration off for deterministic
  candidates) plus a no-skill candidate; pick the best, but **fall back to the base policy if
  none clears the competence bar** (so the agent relearns rather than locking onto a bad skill);
  then exploit until the next reset. A **refractory period** (`reset_refractory=30`) prevents
  detection thrash.

## 4. Arms (matched library + search; differ ONLY in the reset trigger)
1. **BOCPD** — reset on detected change-point (`bocpd_hazard=0.01`). *Mechanism under test.*
2. **Random** — reset at random eps, **count-matched per seed** to BOCPD, spaced ≥ refractory.
   Isolates *detecting the change* from *periodically reshuffling*.
3. **Obs** — per-step skill selection by observation confidence, no change-point detection.
   Expected to fail (obs can't reveal the rule).
4. **NoSkill** — plain PPO, no library. The "is there any lead at all?" floor.

Shared: `lr=3e-4`, `entropy_coef=0.02`, exploration `EPS=0.1` on base-policy episodes (0 when a
cached skill drives the episode). N=30 seeds. Schedule is seed-determined and identical across
arms (paired comparison).

## 5. Outcomes, analysis, decision rule
- **Primary metric:** post-recurrence **adaptation speed** `adapt` = mean return over the first
  15 episodes after each swap into a previously-seen regime, averaged over swaps.
- **Secondary:** `recov` = episodes until return first exceeds 0.5 after a recurrence swap
  (lower = faster); `asymp` (last-15 return per recurrence segment); `total` reward; library size.
- Stats: paired Wilcoxon (one-sided) + Hedges' *g* + 95 % bootstrap CI. N=30.

**Pre-registered verdict (on `adapt`):**
- **LEAD FOUND** iff **BOCPD > NoSkill** (p<0.05 and g ≥ 0.5). *The headline question.*
- **Detection load-bearing** iff additionally **BOCPD > Random** (p<0.05 and g ≥ 0.5).
- **NULL** iff BOCPD ≈ NoSkill (|g|<0.3, p>0.05).
- **ANTI** iff BOCPD significantly **worse** than NoSkill (reverse p<0.05 and g ≤ −0.3).
- Cross-check the same four comparisons on `recov` and `total`; report agreement/disagreement.

## 6. Calibration history (pre-freeze) and integrity note
The mechanism was developed against seeds {0,1,2} during calibration. Bugs found and fixed
**before** freezing: (i) action-flip regimes replaced the original A→B/B→A swap, whose two
rules differ in intrinsic difficulty (B→A often never learned) — a confound; (ii) **competence
fallback** — without it the agent locked onto a wrong cached skill and never relearned;
(iii) **refractory** — without it BOCPD over-fired (30+ resets) and thrashed; (iv) **no
exploration noise on cached-skill episodes** — 10 % noise was derailing known-good policies and
corrupting the search. After these fixes the result was **seed-dependent**: seed 0 BOCPD won
large (total +346 vs +188), seeds 1–2 BOCPD lost (totals +34 vs +210; +39 vs +108). The
mechanism is **frozen at this point**; further tuning while watching the BOCPD-vs-NoSkill
outcome would invalidate this pre-registration.

## 7. Prediction (on record)
Given calibration, I expect **INCONCLUSIVE or NULL/ANTI** at N=30, with high variance: the toy
env relearns flips cheaply (~25 episodes), leaving little room for reuse to pay its overhead
(detection lag + search cost + interference). A clean positive would require an env where
relearning is genuinely expensive *and* skills still form — plausibly the large-model / `run6`
regime, which the toy harness cannot provide. The seed-0 win shows the mechanism *can* help when
reuse fires cleanly; the open question is whether that survives across seeds.
