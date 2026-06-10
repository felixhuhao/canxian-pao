# Canxianization / PAO — Experiment Plan & Roadmap

> Goal: design and run a **practical, falsifiable experiment** for the canxianization theory,
> starting from the cheapest test of its single load-bearing claim and laddering up to the
> full PAO MVP and an LLM-in-the-loop semantic-communication extension.
>
> Source material: `doc/` (Canxianization, PAO, Qualia papers; 4 working notes; Two-Gate Lock
> draft; full PAO implementation plan). This document is the synthesis and forward plan.
>
> **⚠ Update 2026-06-07:** P0 reproduction FAILED — see `FINDINGS.md`. The headline 1D hysteresis
> did not reproduce (reuse signature reversed: FlatPPO beat PAO-light in Phase 3). Two verified
> causes: (A) global RNG is never seeded; (B) the dormancy gate is a dead no-op. A blocking **P0.5
> (fix seed control)** is inserted before P1, and **P2** is redefined to *actually implement* the
> dormancy gate. The roadmap below is updated accordingly.

---

## 0. The thinking process — why this order

### 0.1 Find the load-bearing claim, test it first and cheapest

The whole framework is a tower:

```
Qualia (consciousness)            ─┐
Semantic communication             │  applications / interpretations
Mechanics of Structural Evolution ─┘
        ▲ all rest on …
PAO engineering (assembly index, dormancy, manifold health, hierarchy)
        ▲ which rests on …
ONE empirical claim:
  "Skill crystallisation is a MEANINGFUL discrete event — a phase transition —
   not just arbitrary policy snapshotting."
```

If crystallising a **randomly-timed** behavioural window is just as reusable as crystallising a
**trigger-detected** window, then "the trigger detects something real" is false, and PAO degrades
to *snapshot-and-filter*. Every layer above (assembly, dormancy, manifold, the consciousness and
semantic-comm stories) loses its foundation. So this is the **命门 / lifeline** — and it is also
the cheapest thing to test. Test it first.

This conclusion was reached **independently twice**: by the PAO implementation plan
(`_PAO工程实现方案.md`, "MVP = P0+P1, test H1") and in our analysis of the papers. That
convergence is itself a signal it's the right starting point.

### 0.2 The gap nobody has closed yet

The **Two-Gate Lock** draft (`TwoGateLock_Paper…docx`) is already built and run — but it
**never ran the random-window control.** It compared the triggered agent (PAO-light) against
FlatPPO / no-dormancy / no-skill. What that actually proves is close to a tautology:
*a frozen policy copy resists change (Phase 2) and is instantly reusable (Phase 3).*
It does **not** prove the **timing** of crystallisation matters.

So the highest-value, lowest-cost move in the entire project is to **add the one missing control
arm** to code that already exists. That is the crux test, runnable on a laptop, now.

### 0.3 Design the control so it can actually fail honestly

A naive random-window control (uniform over all training episodes) is too easy to beat: the
trigger fires *after* the policy stabilises, so a random *early* window distills a half-baked
policy and loses for a boring reason ("later policies are just better"). That would make the
trigger look good without supporting the phase-transition claim.

The honest test gives the random arm a fair shot. We run **two controls**:

- **Weak control — Random-Uniform:** crystallise at a uniformly random episode. (Sanity floor.)
- **Strong control — Random-Late / return-matched:** crystallise at a random episode drawn from
  windows whose policy is already competent (matched return / matched to the empirical
  distribution of trigger episodes). **This is the real H1:** *among already-decent windows, does
  the trigger's specific timing (entropy-collapse / stability signature) add reuse value beyond
  "just pick a good-enough late window"?*

If the trigger beats **Random-Late**, the crystallisation signal captures something real.
If it only beats Random-Uniform, the "phase transition" is really just "train longer."

---

## 1. The crux experiment (Phase 1) in detail

**Reuse the Two-Gate Lock environments and stats; add control arms.** Mirrors the PAO paper's own
Exp-1 design (PAO-AdmitAll vs Random-Window).

### 1.1 Conditions (paired by seed)
| Arm | Crystallise when | Filter | Purpose |
|---|---|---|---|
| `Trigger-NoFilter` | heuristic/BOCPD trigger fires | none | isolates trigger timing |
| `RandomLate-NoFilter` | random episode, return-matched | none | **strong control (real H1)** |
| `RandomUniform-NoFilter` | uniformly random episode | none | weak control / floor |
| `Trigger-Filter` | trigger + Q(z)≥0.4 accept | yes | full pipeline |
| `RandomLate-Filter` | random-late + Q(z)≥0.4 accept | yes | does the filter rescue random? |
| `FlatPPO` | never | — | existing baseline |

**Matching rule:** for each seed, record the trigger episode `t*`. The Random-Late arm crystallises
at an episode drawn to match the cross-seed distribution of `t*` (so the *only* difference is
"signal-selected" vs "time-matched random"). Same crystallisation **budget** (count) in every arm.

### 1.2 Primary outcomes
1. **Skill validation success `Q(z)`** — fraction of skill-only rollouts reaching the goal (already
   implemented in the docx's Appendix B). Direct measure of "is this window a reusable skill?"
2. **Phase-3 reuse acceleration** — instant-recovery return when the original rule is restored.
3. (Secondary) the three docx signatures: entropy discontinuity, Phase-2 inertia, Phase-3 jump.

### 1.3 Hypothesis & falsification
- **H1 (this setting):** `Q(z)` and Phase-3 reuse are higher for `Trigger` than `RandomLate`,
  with a real effect size, across **N ≥ 30** seeds, in 1D; and the strong-lock subpopulation in
  2D Location-Shift shows the same ordering.
- **Falsified if:** `Trigger ≈ RandomLate` (Wilcoxon p > 0.05, |Hedges' g| < 0.3). Then the
  phase-transition interpretation does not hold in the minimal setting → **clean negative result**,
  reported as-is (still valuable; the failure mode informs theory revision).

### 1.4 Stats (match existing rigor)
Paired Wilcoxon signed-rank + Hedges' g + 95% bootstrap CI; N ≥ 30; thresholds **pre-registered
before running**; FDR correction across the arm comparisons.

### 1.5 Run both environments
1D corridor (clean effects) **and** 2D Location-Shift (content-invalidation regime). The
dimension-dependent granularity law predicts the trigger advantage should be sharpest exactly where
content invalidation bites.

---

## 2. Roadmap

Each phase ends on a **go/no-go gate**. A failed gate is a publishable result, not wasted work.

| Phase | Deliverable | Gate (go/no-go) | Effort |
|---|---|---|---|
| **P0 — Reproduce & instrument** | ✅ DONE 2026-06-07 — code synced, `basic` venv set up, run executed | **FAILED**: signatures did not reproduce (`FINDINGS.md`); reuse reversed | done |
| **P0.5 — Fix seed control** | ✅ DONE 2026-06-07 — `set_seed()` added; reproducibility restored | **Revealed both gates are inert** (`FINDINGS.md`): PAO==NoDorm==NoSkill byte-identical; skill applied 0/7106 steps | done |
| **P0.6 — Make a gate actually function** | ✅ DONE 2026-06-07 (`p06_gate_test.py`). Gate opens at `app_thresh=0.3`; **paper's −0.880/1.516 reproduced exactly**; skill gate causally verified vs NoSkill control | **Resolved**: paper real but shipped default (0.7) is wrong; applicability net collapsed (dumb switch); **dormancy still dead** | done |
| **P1 — Random-Window control (CRUX)** | ✅ DONE 2026-06-07 (`p1_crux.py`, N=30, pre-registered) | **H1 NOT SUPPORTED**: trigger tied on reuse (g=−0.02) and *worse* on skill quality (g=−2.79, fires prematurely) vs matched random-late. Caching-helps survives (vs NoSkill g=+0.91). See `FINDINGS.md` | done |
| **P2 — Harder-env + BOCPD steelman (Option 2)** | ✅ DONE 2026-06-08 (`p2_harder.py`, N=30, pre-registered). Corrected the broken BOCPD; non-convex 2D + Location-Shift | **H1 FALSIFIED again** (BOCPD *worse* than random-late, g=−0.82, reverse p<0.001). Skill quality = training amount, not detection. **Caching reverses to net-harmful**: NoSkill beats all skill arms on reuse (g≈−1.1). See `FINDINGS.md` | done |
| **H_M test — does manifold health gate quality?** | ✅ DONE 2026-06-09 (`hm_test.py`, N=80, pre-registered) | **H_M is ANTI-correlated with usefulness** (ρ=−0.52; partial ρ=−0.32 controlling for training; both d_int & V_info inverted vs theory). The cross-cutting geometric criterion is anti-predictive. See `FINDINGS.md`. Ecological confirmation pending run6 checkpoints | done |
| **P3 — PAO MVP (= .md P0+P1)** | Fork DreamerV3; MiniGrid 8×8; G_t + learnability filter + BOCPD + distill + validate; **PAO-AdmitAll vs Random-Window** | **H1 at scale** (proper env, real world model) | 4–5 weeks |
| **P4 — Qwen-1.5B semantic-comm** *(branch)* | Two agents share a small LLM as semantic anchor; "fetch cup A→B" task; **Location-Shift stale-anchor test**; add crystallisation + dormancy gates to the shared semantic cache | Does a dormancy/invalidation gate detect the stale anchor? (puts a real LLM in the loop → justifies "robot brain" framing) | 3–5 weeks |
| **P5 — Full PAO (future)** | .md P2–P4: assembly DAG + 6 thresholds, dormancy + anchoring, manifold health → H2–H5 | per-phase falsification gates | 9–13 weeks |

### Decision logic
```
P0 ✗──► P0.5 (fix seeding) ──► baseline survives? ──no──► 1D effect is an artifact; report it.
                                      │yes
                                      ▼
        P1 (crux) ──► H1 holds? ──no──► STOP. Report clean negative. Theory needs revision.
                          │yes
                          ▼
                P2 (harden) ──► P3 (MVP at scale)
                                     │
                          ┌──────────┴───────────┐
                          ▼                       ▼
                P4 (LLM semantic-comm)    P5 (full PAO H2–H5)
                 [justifies framing]       [completes the theory test]
```

P1 is the gate everything hinges on. P4 and P5 are independent branches taken only after P3.

---

## 3. Key dependencies & risks

| Item | Status / risk | Mitigation |
|---|---|---|
| **Two-Gate Lock source code** | Not in this workspace (`doc/` has only papers/notes). Docx references `analysis/exp_two_gate_lock/` in a "companion repository" we don't have. | Either (a) obtain the repo, or (b) **reimplement PAO-light** — the docx gives full env specs, hyperparameters, and agent definitions; ~1–2 days to rebuild. |
| **Manifold metrics `d_z`/`V_z`** | Noisy at ~20 rollouts; the framework's deepest fragility. Only matters from P3/P5. | Defer; use participation ratio; keep `θ_M` soft; let multiplicative `Γ_M` do selection, not a hard cut. |
| **Discrete-reward training instability** | Gradient-flow blockade + ΔA_total spikes → TD blowups (P3+). | Baseline bootstrapping (η_boot=0.5) per the .md. |
| **H1 genuinely fails** | The real scientific risk. | This is *the point* — a clean yes/no. Negative result is reportable and informs theory. |
| **"Small LLM" framing unsupported** | Neither built artifact has an LLM. | P4 closes this gap deliberately. |

---

## 4. Immediate next actions (P0.5 → P1)

1. ~~Decide code source~~ ✅ original repo obtained, synced to `exp_two_gate_lock/`.
2. ~~Reproduce P0~~ ✅ done — **failed**, see `FINDINGS.md`.
3. **P0.5 — fix seed control (next):** add `torch.manual_seed`/`np.random.seed`/`random.seed`
   keyed off the per-run seed; re-run N=10 (and N=30) to see whether *any* hysteresis survives.
   This is the gate that decides whether the 1D effect is real or an artifact.
4. **Pre-register P1** *(only if P0.5 baseline survives)*: H1 thresholds, the 6 arms, the
   seed-matching rule, and the stats plan to a frozen file *before* running.
5. **Implement the Random-Late / return-matched sampler** (the one genuinely new piece of code).
6. Run N≥30, both environments, evaluate the gate.

---

## 5. Open questions for you

- **Code:** do you have the `exp_two_gate_lock` repo, or should we reimplement PAO-light from the
  draft?
- **Compute:** P0–P2 are laptop-scale. P3 (Dreamer/MiniGrid) wants a GPU — available?
- **Audience/deadline:** is there a target (e.g., the ICSR2026 keynote in note 3.txt) that should
  shape scope and timing?
