# Two-Gate Lock: Minimal Demonstration of Skill Crystallisation

> A self-contained, laptop-runnable experiment that exhibits **three falsifiable signatures of a structural phase transition in skill acquisition**.

---

## Claim

**Skill acquisition is a discrete phase transition, not a smooth extension of gradient-based learning.**

Three signatures that distinguish this from conventional deep RL:

| # | Signature | What to look for |
|---|-----------|------------------|
| 1 | **Discrete Trigger** | Skill count jumps 0→1 at episode 9–27, with simultaneous entropy collapse (0.69→0.45). No such jump in Flat PPO. |
| 2 | **Structural Inertia** | Under rule swap (Phase 2: A→B → B→A), PAO-light's cached skill **actively resists reversal** — return stays at baseline (−0.88) for all seeds. FlatPPO can partially adapt (+0.29 late Phase 2). |
| 3 | **Reuse Acceleration + Forgetting Immunity** | When the original rule is restored (Phase 3: A→B), PAO-light recovers **instantaneously** at R=1.52. FlatPPO **fails to recover** — its return remains negative throughout Phase 3 (−0.08), exhibiting negative transfer from Phase 2. |

**Taken together, these form a hysteresis loop** — asymmetric adaptation across a symmetric rule change. This is the classic experimental signature of a system with history-dependent structure (phase transitions, magnetic hysteresis, Kuhnian paradigm shifts).

---

## The Hysteresis Loop

```
Phase 1 (A→B)         Phase 2 (B→A)          Phase 3 (A→B)
────────────────────  ─────────────────────  ─────────────────────
PAO:  ────→  skill!   PAO:  ──→  (locked)    PAO:  →  instant 1.52
                      FlatPPO: ──→∼∼ (drift)  FlatPPO: →∼↘→ neg. transfer
```

![Hysteresis plot](results/1d_rule_swap_hysteresis.png)

The shaded regions mark the three phases. Red dashed line = rule swap to B→A. Green dashed line = restore to A→B. Blue stars = skill crystallisation events.

**Key observation:** the rule swaps are symmetric (A→B → B→A → A→B), but PAO-light's adaptation curve is highly asymmetric — it is **slower to reverse** (Phase 2 lock-in) and **faster to recover** (Phase 3 instant). FlatPPO's curve shows the opposite problem: it **partially adapts** to the new rule (Phase 2 drift) but then **cannot un-adapt** when the rule returns (Phase 3 negative transfer). This asymmetry is the hallmark of a system with discrete structural memory.

---

## Experimental Design

### Environment (1D corridor)
```
State:  S . A . . B D G
Index:  0 1 2 3 4 5 6 7
```
- Agent moves LEFT (0) or RIGHT (1)
- Rule **A→B**: visit switch A (index 2), then switch B (index 5) ≤ Δ=6 steps → door D opens → goal G rewards +1
- Rule **B→A**: reversed order (visit B first, then A)
- Shaped rewards: first switch +0.1, second +0.5, goal +1.0, step penalty −0.02

### Agents
- **PAO-light**: PPO + Event-triggered skill crystallisation (heuristic threshold: return > 1.0 AND entropy < 0.6 AND 3/5 recent successes) + state-anchored action bias (cosine similarity matching) + dormancy gate
- **FlatPPO**: Identical PPO backbone, NO skill mechanisms

> **Terminology note:** Full PAO uses Bayesian Online Change-Point Detection (BOCPD) on policy entropy for triggering. This minimal implementation uses a dual-threshold proxy. A standalone BOCPD implementation is included in the codebase for future upgrade.

### Protocol
1. **Phase 1** (episodes 0–79): Rule = A→B, clean environment. PAO-light crystallises the "all-right" skill.
2. **Phase 2** (episodes 80–199): Rule = B→A, +10% ε-greedy exploration noise (forces exploration; PAO-light's cached skill actively fights reversal).
3. **Phase 3** (episodes 200–259): Rule = A→B restored. Clean environment.

---

## Results (10 seeds, quick mode)

| Metric | PAO-light | FlatPPO |
|--------|-----------|---------|
| Skills after Phase 1 | 1.0 (all seeds) | — |
| Skills after Phase 2 | 1.0 (never lost) | — |
| Skills after Phase 3 | 1.0 | — |
| Return Phase 2 (late 20) | **−0.880 ± 0.000** (locked) | **+0.287 ± 0.342** (partially adapted) |
| Return Phase 3 (early 20) | **1.518 ± 0.002** (instant) | **−0.041 ± 0.156** (negative transfer) |
| Return Phase 3 (late 20) | 1.519 ± 0.002 | −0.083 ± 0.091 (worsens) |

**Key finding:** FlatPPO does NOT "re-learn slowly" in Phase 3 — it **experiences negative transfer** from Phase 2. Its return remains negative throughout Phase 3 and even worsens (early: −0.04 → late: −0.08). This means the partial B→A adaptation in Phase 2 actively harms FlatPPO's ability to return to A→B, while PAO-light's cached skill provides **complete immunity to this forgetting**.

**Hysteresis loop confirmed** across 10 seeds:
- Phase 2: PAO → −0.88 vs FlatPPO → +0.29 (PAO slower to reverse: ✓ inertia)
- Phase 3: PAO → 1.52 vs FlatPPO → −0.04 (PAO faster + forgetting-proof: ✓ reuse + immunity)

---

## Falsification Conditions

If any of these hold, the phase-transition hypothesis is weakened:

1. **No discrete trigger**: entropy drop is purely continuous, indistinguishable from FlatPPO's entropy decay
2. **No structural inertia**: Phase 2 adaptation speed of PAO-light ≈ FlatPPO
3. **No reuse advantage**: Phase 3 recovery of PAO-light ≈ FlatPPO

Current status: **all three conditions falsified** — the phase-transition interpretation survives.

---

## Limitations

1. **Minimal 1D corridor**: the policy space is convex (optimal = "always go right"), making PPO's gradient flow unrealistically robust. In a 2D environment with non-convex paths, FlatPPO's performance is expected to degrade further, amplifying the hysteresis signal (predicted 4–8× advantage rather than the ∞× observed here from negative transfer).

2. **Heuristic trigger, not BOCPD**: the current triggering heuristic (return + entropy thresholds) is a proxy for the full Bayesian detection described in the PAO paper. A full BOCPD implementation would provide uncertainty-calibrated crystallisation timing.

3. **Single-shot skill**: PAO-light only crystallises once. Full PAO supports a growing skill library with multiple skills competing via the option framework, as well as active pruning of obsolete skills (PAO Sec. 9.3).

4. **No de-crystallisation**: PAO-light never "melts" its Phase 1 skill in Phase 2. This is consistent with PAO theory (crystallised structures resist contradictory evidence), but a full implementation would require an environment drift detector to trigger evidence-driven melting.

5. **Limited seeds (10, quick mode)**: while sufficient for trend detection, formal hypothesis testing requires the full protocol (original PAO Sec. 8: 400+ episodes, 10 seeds, Wilcoxon signed-rank test with p < 0.05).

---

## Reproduce

```bash
# Requirements: torch, numpy, matplotlib, scipy

# Run quick experiment (10 seeds, ~260 episodes each, ~minutes)
python3 run_rule_swap.py --quick --seeds 0 1 2 3 4 5 6 7 8 9

# Output
results/1d_rule_swap_hysteresis.png   # 4-panel diagnostic plot (300 dpi)
results/rule_swap_1d.pkl              # raw data per seed
```

---

## Code Structure

```
analysis/exp_two_gate_lock/
├── env.py             # 1D corridor (rule-swap aware)
├── env_2d.py          # 2D grid extension (7×5, relative-feature observations)
├── agents.py          # PAOLight + FlatPPO (+ BOCPD util, not used for triggering)
├── run_rule_swap.py   # Rule-swap hysteresis experiment (primary)
├── run_2d.py          # 2D experiment runner (WIP, needs tuning)
├── run.py             # Original reward-corruption experiment
├── make_docx.py       # Generate this report as .docx
└── results/           # Output plots + data + generated report
```

---

## Related

- **Canxianization Theory** (arXiv:submit/7621537, 2026) — the theoretical framework: skill acquisition as information-cost-paid path locking
- **PAO** (arXiv:submit/7621022, 2026) — the engineering architecture: Progressive Assembly Objective with BOCPD triggering, dormancy gating, and manifold health filtering
- Ma et al. (2026, arXiv:2605.08142) — independent empirical verification of diffusive vs. degenerative hallucination classification, consistent with Hallucination Dual Pathology predicted by Canxianization Theory
