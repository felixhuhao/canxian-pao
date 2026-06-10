# Canxianization / PAO — A Controlled, Pre-Registered Re-Examination

This repository is an **independent empirical investigation** of the distinctive claims of the
**Canxianization / Mechanics-of-Structural-Evolution (MSE)** framework and its reinforcement-learning
instantiation, **PAO (Progressive Assembly Objective)**, developed by Hengjin Cai (Wuhan University).

It takes the framework's own minimal experiment (the *Two-Gate Lock*) plus several new, seed-controlled
experiments, and tests — under frozen pre-registrations — whether the mechanisms behave as the theory
claims. **Short version: the three distinctive, load-bearing claims did not survive controlled testing.**

> Full evidence and dated history are in [`FINDINGS.md`](FINDINGS.md). This README is the orientation;
> `FINDINGS.md` is the authoritative record.

---

## The framework under test (one paragraph)

*Canxianization* (坎陷化) proposes that intelligence is the process of paying an information cost to
compress chaotic possibility into **low-dimensional, reusable causal "grooves"** (skills). *PAO*
operationalises this for RL with three mechanisms: **(1) event-triggered crystallisation** — a skill is
"locked" at a detected stable jump (a discrete phase transition, ideally via BOCPD change-point
detection); **(2) effective-assembly / manifold-health filtering** — only skills whose latent trajectory
is a *non-degenerate manifold* (low intrinsic dimension $d_{\text{int}}$ + high information volume
$V_{\text{info}}$, scored by $H_{\mathcal{M}}$) are admitted; **(3) dormancy-gated plasticity** — frozen
skills are protected from being overwritten. The *Two-Gate Lock* is the minimal experiment meant to
demonstrate mechanisms (1) and (3). (Source papers are in `doc/`, gitignored.)

## What this project actually tested, and what it found

Each claim was pre-registered (hypothesis, controls, frozen decision rule) **before** running.

| Distinctive claim | Test | Verdict |
|---|---|---|
| **Crystallisation timing matters** — the trigger detects a *special* moment, not just "trained enough" | `p1_crux.py` (1D), `p2_harder.py` (non-convex 2D + BOCPD) | **Falsified.** Trigger windows are no more reusable than competence-matched **random** windows (1D: tie; harder env: BOCPD significantly *worse*, Hedges' g = −0.82). The trigger tracks *amount of training*, nothing more. |
| **Skill caching helps** | `p1_crux.py`, `p2_harder.py` | **Environment-dependent → net-harmful.** Helps only in a trivial corridor; in a non-trivial env, plain PPO (no skills) **beat every skill arm** on reuse (g ≈ −1.0 to −1.2). |
| **Manifold health $H_{\mathcal{M}}$ gates skill quality** | `hm_test.py` (N=80, pre-registered) | **Anti-correlated.** $H_{\mathcal{M}}$ is *negatively* related to actual skill usefulness (ρ = −0.52; partial ρ = −0.32 controlling for training). Both components are inverted vs. the theory. As an acceptance gate it would select the **wrong** skills. |

Alongside the claims, a code/reporting audit found: **no RNG seeding** anywhere (results irreproducible
as shipped), a **dead dormancy gate** (`dormancy_weights` accepted but never used), a **collapsed
applicability gate** (never opens at the shipped threshold), a **non-functional BOCPD** (never fires,
even on a clean step), and **hardcoded, internally inconsistent paper numbers** with missing 2D data.
The paper's headline result *is* reproducible — but only under a configuration the released code does
not use.

An independent full-scale run by Dr. Cai (the Dreamer-based PAO on MiniGrid, "run6") **converges with
these findings**: skills crystallise but are frequently functionally useless, and the agent barely
solves a trivial gridworld. The decisive open test (below) would close the loop.

## Method / discipline

This investigation was deliberately strict, because the original code was not:

- **Pre-registration** — every test's hypothesis, arms, and pass/fail rule frozen in a `PREREG_*.md`
  file *before* execution (so results cannot be retrofitted).
- **Seed control** — torch + numpy + Python RNG seeded per run; conditions matched on seed.
- **Matched controls** — the right comparison is **PAO-no-skill** (same network allocation, same RNG),
  not vanilla PPO; the trigger is tested against **competence-matched random-window** crystallisation.
- **Honest statistics** — paired Wilcoxon, Hedges' *g*, bootstrap CIs, partial correlations; effect
  sizes reported, not just p-values.
- **On-record predictions** — the analyst's expected outcome was written down before each run.

## Repository layout

```
EXPERIMENT_PLAN.md        Roadmap with a falsification gate at each phase
FINDINGS.md               Dated lab notebook — the authoritative results record
PREREG_P1.md              Frozen pre-registration: the crux H1 test (1D)
PREREG_P2_harderenv.md    Frozen pre-registration: BOCPD + harder env
PREREG_HM.md              Frozen pre-registration: manifold-health discrimination
exp_two_gate_lock/        Code (original Two-Gate Lock + our additions)
  ├─ README.md            Dr. Cai's original experiment readme
  ├─ agents.py            Agents (modified: + PAOForced, PAOBocpd, refactored trigger)
  ├─ env*.py              1D corridor + 2D grid environments
  ├─ run_rule_swap.py     Original 1D runner (modified: + set_seed)
  ├─ p1_crux.py           OURS — random-window control (the H1 crux test)
  ├─ p2_harder.py         OURS — BOCPD + non-convex 2D steelman
  ├─ p06_gate_test.py     OURS — applicability-gate causality test
  ├─ hm_test.py           OURS — manifold-health discrimination test
  ├─ cal_2d.py            OURS — 2D env calibration / BOCPD diagnostics
  └─ results/             .log + .pkl evidence (results_original/ = untouched upstream)
doc/                      Source papers (PDF/DOCX) — gitignored; live on D:\Documents\Research
```

## Reproduce

Requires Python 3.12 with `torch`, `numpy`, `scipy`, `matplotlib`.

```bash
cd exp_two_gate_lock

# Reproduction + the seeding/dead-dormancy demonstration (P0/P0.5)
python run_rule_swap.py --quick --seeds 0 1 2 3 4 5 6 7 8 9 --ablations

# The pre-registered experiments
python p06_gate_test.py    # applicability gate is inert at default, causal at app=0.3
python p1_crux.py          # H1: trigger vs random-window (1D)            -> falsified
python p2_harder.py        # H1 steelman: BOCPD + non-convex 2D           -> falsified
python hm_test.py          # manifold health vs skill usefulness          -> anti-correlated
```

Each script seeds its RNGs and writes `.log` + `.pkl` to `results/`. The corresponding `PREREG_*.md`
states the frozen decision rule.

## Scope & caveats (what this does *not* claim)

- Most tests are in **minimal/small environments** with **proxy measurements** (e.g. manifold health
  over policy-torso activations rather than RSSM latents). The **ecological confirmation** — computing
  $H_{\mathcal{M}}$ vs. reward on Dr. Cai's full-PAO `run6` checkpoints — is **pending** (it should show
  the useless skills having $H_{\mathcal{M}} \ge$ the good one).
- This concerns the **falsifiable RL/engineering claims.** It does **not** test the philosophical core
  of the dialectics paper (finite systems → ineliminable residual → "structural contradiction"), which
  stands largely independently and is more defensible on its own terms.
- The finding is **not** "the framework is worthless." It is: *as implemented and tested, the
  distinctive mechanisms — event-triggered crystallisation, the manifold-health quality gate, and the
  caching benefit — do not do the work the theory attributes to them.*

## Status

Three pre-registered tests complete; all three distinctive claims falsified or inverted. Open:
**(1)** ecological $H_{\mathcal{M}}$ confirmation on `run6` checkpoints; **(2)** a consolidated
negative-results write-up.

## Credit

The Canxianization/MSE theory, PAO framework, and original Two-Gate Lock experiment are the work of
**Hengjin Cai (蔡恒进), Wuhan University**. This repository is an independent investigation built on that
code and those papers; Dr. Cai's own concurrent full-scale results are consistent with its findings.
