# Findings Log — Two-Gate Lock / PAO

Running lab notebook. Newest first. Each entry: what was done, what was found, evidence, implication.

---

## 2026-06-16 — Busch E2: nonlinear geometry is not the lever — PEV works only while coupled to controllability

Follow-on to E1. Same agent/metric, but data now from a **genuinely nonlinear** manifold
`x = W₂ tanh(gain·W₁u)`: `Σ_obs = Cov(x)` (global; PEV/readability) vs `Σ_act = JΣ_latJᵀ` (local reachable
cov at the operating point; controllability). Pre-registered in `PREREG_BUSCH_E2.md` (frozen 2026-06-16;
**design pivoted during calibration**, documented, no held-out data seen). `exp_busch/e2_nonlinear.py`,
held-out seeds 700–729 (calib 750–753). gain=2.0, operating-point offset=1.5.

**Pivot (honest).** The draft hypothesis "linear PCA always mislabels under nonlinearity" is **false** for a
structural reason: on a single shared manifold the data distribution *is* the reachable set, so high variance
⟺ the manifold extends there ⟺ locally reachable — PEV and controllability are **coupled** and PEV predicts
learning fine. They decouple only when the operating point sits in a **curved/saturated** region (offset > 0).

| condition | PEV | control | ΔControl (N=30) |
|---|---|---|---|
| IM_lin (top PC) | 0.627 | 16.15 | 25.2 |
| WMP_lin (2nd PC) = **HiPEV_uncontrol** | **0.237** | 0.28 | **3.2** |
| OMP_lin (bottom PC) | 0.000 | 0.08 | −1.8 |
| TAN (oracle tangent) | 0.453 | 22.55 | 25.1 |
| NOR (least reachable) | 0.012 | 0.00 | −1.2 |

### Verdict: controllability is the lever; PEV is a proxy valid only under obs≈act coupling
- **P1:** TAN > NOR (g=+4.10, p<1e-4) — on/off by controllability.
- **P2 — high PEV is NOT sufficient under curvature:** a bona-fide high-variance "on-manifold" component
  (WMP_lin, PEV=0.237) is locally uncontrollable at the curved operating point and **fails** (ΔControl 3.2
  vs IM 25.2; IM>HiPEV g=+3.60 p<1e-4; HiPEV ≈ OMP, g=+0.55).
- **P3 — controllability is the lever (nonlinear setting):** sweep regression `ΔControl ~ readSNR +
  controllability` gives β_control=+0.91 ≫ β_readSNR=+0.13; Spearman(control)=1.00 vs Spearman(PEV)=0.38.
- **P4 — the coupling sweep (headline):** as offset 0→2, corr(PEV,controllability) **0.92→0.52**, the extra
  variance controllability explains beyond PEV (ΔR²) **0.37→0.60**, and the high-PEV component's ΔControl
  **collapses 24.1→2.3** while IM stays ≈25. **At offset 0 (locally-linear operating point) PEV predicts
  learning — Busch's single-manifold design is sound there.**
- *Note:* the symmetric `LoPEV_tangent` cell did not materialize (ΔControl = OMP) — on a shared manifold there
  is **no** low-variance-but-controllable direction either (low variance ⟹ not reachable). This is additional
  evidence for the coupling thesis, not against it.

### Combined E1+E2 verdict (Direction 2 geometry question — closed)
Whether the "manifold hypothesis" works reduces to whether observation-variance (PEV) equals volitional
**controllability**. **E1:** if they can differ, PEV is neither necessary nor sufficient (controllability
g=+3.3). **E2:** a single shared manifold — even a nonlinear one — *couples* them (data lives where it is
reachable), so PEV works, **vindicating Busch within their design**; the coupling breaks under operating-point
curvature, and there controllability still governs. So **nonlinear geometry (T-PHATE) is not an independent
causal lever** — PEV's validity *is* the obs≈act coupling, and the lever underneath is always controllability.
This both fairly credits Busch (PEV is a fine proxy in their regime) and pins the mechanism, extending the
project's `property ≠ usefulness` thesis: usefulness tracks *usable/controllable* signal, not the
variance/geometry label. **Condition stated honestly:** the decoupling requires the operating point off the
manifold's locally-linear center (resting brain state ≠ variance centroid). Revision rule (not triggered):
would favour Busch if HiPEV_uncontrol learned, controllability failed to out-predict PEV, or coupling did not
weaken with curvature.

---

## 2026-06-16 — Busch E1: on-/off-manifold BCI learning is readability × CONTROLLABILITY, not geometry

Direction 2 opens. Independent in-silico test of **Busch et al.** (*"Accelerated learning of a noninvasive
human BCI via manifold geometry"*, bioRxiv 2025.03.29.646109 — a different group from Cai/RDD). They report
that re-learning a perturbed fMRI-BCI mapping is fast on-manifold (IM = top latent eigenvector, WMP = 2nd)
and **fails** off-manifold (OMP = 20th/lowest), with learning = raising PEV (% variance explained) along the
trained component. Pre-registered in `PREREG_BUSCH_E1.md` (frozen 2026-06-16; metric amended same day,
pre-confirmatory, with rationale). `exp_busch/e1_deconfound.py`, held-out seeds 600–629 (calib 500–503).

**Confound.** OMP *is* the lowest-variance direction; "on-manifold-ness" = PEV (`CᵀΣ_obs C`) silently bundles
two separable quantities their single-manifold design can't pull apart: **readability** (decoder SNR,
`CᵀΣ_obs C/σ²`) and **controllability** (cheapness of *volitionally* producing it, `CᵀΣ_act C`; the
min-plasticity-cost action gives optimal gain `g*=1/(1+λ·cost)`, so low controllability ⇒ under-drive ⇒
failure). Busch implicitly assume `Σ_obs=Σ_act` (observation manifold = reachable set), collapsing both onto
PEV. E1 (linear–Gaussian) breaks that assumption. Metric: ΔControl = 100·(acc last third − first third).

| condition | PEV | readSNR | controllability | ΔControl (N=30) |
|---|---|---|---|---|
| IM (top eigvec) | 0.251 | 222.9 | 5.016 | 25.2 |
| WMP (2nd eigvec) | 0.188 | 167.2 | 3.762 | 24.8 |
| OMP (bottom eigvec) | 0.001 | 0.9 | 0.021 | **1.0** |
| **D1** on-manifold, *uncontrollable* | **0.251** | 222.9 | 0.021 | **0.6** |
| **D2** off-manifold, made *usable* | **0.001** | 222.9 | 5.016 | **25.2** |

### Verdict: PEV/manifold-membership is neither necessary nor sufficient — controllability is the lever
- **P1 (replicate):** on-manifold ≫ off-manifold (IM>OMP g=+3.28, WMP>OMP g=+3.23, both p<1e-4). *Deviation:*
  IM≈WMP (g=+0.16; both high-variance directions saturate the asymptotic metric) rather than the strict
  IM>WMP>OMP — Busch's IM>WMP gap is a sequential-interference effect (WMP always ran after IM), outside
  E1's scope. The load-bearing on/off contrast reproduces.
- **P2 — high PEV is NOT sufficient:** D1 (high PEV / on-manifold, but uncontrollable) **fails**, statistically
  indistinguishable from OMP (D1~OMP g=−0.03, p=0.92; IM>D1 g=+2.98, p<1e-4).
- **P3 — low PEV is NOT the cause:** D2 (low PEV / off-manifold, but readable+controllable) **learns fully**,
  matching the on-manifold conditions (D2>OMP g=+3.28, p<1e-4).
- **P4:** in the direction sweep, controllability dominates (std. β=+0.89) over readSNR (+0.16); PEV is
  positively *correlated* with learning (ρ=0.91 — why Busch observed it) but controllability (ρ=0.99) is the
  true driver, and PEV is blind to it.

### Implication
Busch's "manifold geometry constrains BCI learning" is — at the level their own PCA components occupy — a
**proxy result**: PEV predicts learning only because it bundles readability and volitional controllability,
two separable quantities. Dissociate them and the geometric story breaks (high-PEV-but-uncontrollable fails;
low-PEV-but-usable succeeds). This extends the project's **`property ≠ usefulness`** thesis to a *second,
independent* framework: across Cai/RDD (diversity/repulsion) and Busch (manifold geometry), the operative
variable is the same — task-aligned, *usable* signal under a capacity/plasticity constraint, not the
geometric label each paper foregrounds. Maps onto PAO: on-manifold = reassociation within the existing
repertoire (cheap); off-manifold = generating a new skill (expensive). **Scope:** E1 is linear–Gaussian, so
it does not test whether *nonlinear* (T-PHATE) geometry adds power beyond readability × controllability —
that is E2 (next). Revision rule (not triggered): would favour Busch if D1 learned, D2 failed, or PEV
out-predicted controllability.

---

## 2026-06-15 — RDD rescue: LI cannot be salvaged by task-alignment; the lever is COVERAGE, not repulsion

Pre-registered in `PREREG_RDD_rescue.md` (frozen before run; held-out seeds 400–419, calibration 300–303).
`exp_rdd/rdd_rescue.py`, log `…/results/rdd_rescue.log`. Constructive rescue attempt for LI, designed to
isolate the ONE ingredient that makes auxiliary diversity useful. M=8 over-complete, asym; everything
fixed but the auxiliary term. `spec` and `cov` are equally task-aligned (both signal-weighted spectral);
they differ ONLY in pairwise-product (repulsion) vs max-union (coverage).

| arm | div | test_mse | vs noLI |
|---|---|---|---|
| noLI | 2.69 | 3.506 | — |
| genLI (τ-space repulsion) | 7.57 | 4.441 | HURTS g=+17.8 |
| spec10 (task-aligned spectral *repulsion*) | 4.43 | 4.298 | HURTS g=+4.0 |
| spec50 / spec100 (stronger) | ~0.1 | 6.495 | catastrophic collapse |
| **cov10 (task-aligned *coverage*)** | 2.63 | **3.421** | **HELPS g=−1.15, p=0.001** |
| cov50 | 2.57 | 3.472 | null (g=−0.47, p=0.053) |

**KEY contrast:** best cov (β=10, 3.421) vs best spec (β=10, 4.298): g=−4.42, **p=0.000**.

### Verdict: task-alignment does NOT rescue LI — the objective FORM does
- **Every repulsion form fails.** τ-space repulsion (genLI) and *task-aligned* spectral repulsion (spec)
  both HURT downstream MSE; strong spectral repulsion even *collapses* the model (div→0.1, MSE→deadlock
  level). Making repulsion spectral + signal-weighted is not enough.
- **Only coverage helps**, and it beats the equally-task-aligned repulsion decisively (g=−4.42, p<0.001).
  The rescuing ingredient is **coverage/union (fill the signal's spectrum)**, not repulsion/push-apart,
  and not task-alignment per se.

### Implication — closes the RDD path
This pins the earlier positive result (`rdd_taware`) precisely: the win came from a **coverage** objective,
not from diverse/repelled channels — and coverage is *not* what lateral inhibition computes. Across the
RDD path the verdict is now consistent and complete: **(1)** γ-inert without the kernel (`rdd_li`);
**(2)** diversity ≠ usefulness, LI worsens MSE (`rdd_li`/`rdd_crossover`); **(3)** LI inert at the true
deadlock and harmful where it acts (`rdd_deadlock`); **(4)** LI cannot be rescued by task-alignment — only
a different objective (coverage) helps (`rdd_rescue`). RDD's mechanism (repulsion-driven differentiation)
is neither the deadlock-escape force nor a performance lever. Theme holds: *property (diversity/repulsion)
≠ usefulness (task-aligned coverage)*. **RDD path closed.** (Direction 2, Busch IM/WMP/OMP, remains
parked pending the preprint.)

---

## 2026-06-15 — RDD deadlock: LI is INERT at the true deadlock and is just a (harmful) symmetry-amplifier

Pre-registered in `PREREG_RDD_deadlock.md` (frozen before run; seeds 0–19). `exp_rdd/rdd_deadlock.py`,
log `…/results/rdd_deadlock.log`. Tests RDD's central mechanistic claim *directly* — that lateral
inhibition is the force that escapes the geometric deadlock — rather than via downstream MSE alone.
Symmetric SSM (shared W_in/W_out; channels differ only by τ), two regimes + a ceiling (`asym`).

| arm | div | test_mse | note |
|---|---|---|---|
| asym | 2.69 | 3.510 | ceiling (per-channel weights) |
| none | 0.00 | 5.889 | deadlock baseline |
| **LI** | **0.00** | **5.889** | RDD repulsion — identical to none |
| jitter | 0.70 | 5.856 | τ-init asymmetry (only mover) |
| dropout/innoise/wdecay | ~0 | ~5.89 | inert |
| seed (jitter=0.1) | 0.70 | 5.856 | seed only |
| LI_seed | 9.07 | 6.342 | seed + LI — huge diversity, WORSE MSE (g=+13.7) |
| drop_seed | 0.70 | 5.858 | ≈ seed |

### Verdict: RDD's "LI escapes geometric deadlock" fails twice over
- **LI is INERT at the true deadlock.** At perfectly symmetric init, LI gives div=0.000, MSE identical
  to `none` — because the repulsion gradient ∝ (τ_i−τ_j) vanishes at τ_i=τ_j. LI cannot *create*
  asymmetry; it only *amplifies* a pre-existing one. (A fixed-point analogue of the γ-inertness finding.)
  The only thing that moves a true deadlock is a trivial init asymmetry (`jitter`).
- **Where LI CAN act, it HURTS.** Given a seed (jitter=0.1), LI drives τ-diversity 0.70→9.07 but
  *worsens* test MSE 5.856→6.342 (g=+13.7, p(help)=1.000); `drop_seed` ≈ `seed`. The trivial init seed
  does the symmetry-breaking; LI's task-blind amplification adds only harm (consistent with `rdd_li`,
  `rdd_crossover`).
- **Deadlock harm is large but M-independent** (gap none−asym ≈ 2.34–2.47; Spearman(M,gap)=−0.40,
  p=0.51). Note: no τ-differentiation mechanism in the symmetric architecture approaches the `asym`
  ceiling (3.51) — the binding constraint there is the *shared* W_in/W_out, not τ-collapse, so τ-repulsion
  targets the wrong bottleneck.

### Implication
RDD's mechanistic story does not hold up to a direct test: lateral inhibition is neither the escape force
(inert at the actual deadlock) nor a performance regularizer (it worsens MSE wherever it acts). The
operative symmetry-breaker is a trivial init asymmetry. This consolidates the RDD-path verdict (γ-inert
without the kernel; diversity ≠ usefulness; coverage-reward is the only thing that helped) and reinforces
the project theme *property ≠ usefulness*. RDD path can be closed after the optional task-aligned-LI
rescue (#3).

---

## 2026-06-15 — R5: R4's harm is REACHABLE only via mis-triggering, not crystallization volume

Pre-registered in `PREREG_R5.md` (frozen before run; held-out seeds 200–207). `exp_pao_coverage/r5.py`,
log `…/results/r5.log`. A deliberate self-critique of R4: R4's positive result used *planted*
confidently-wrong junk, so R5 asks whether that harm arises through PAO's two **real** channels without
planting — (A) crystallization volume (many immature, correctly-tagged snapshots) and (B) mis-triggering
(a competent skill firing in the wrong niche; rate ε and simultaneous count m).

**A — natural over-crystallization volume (immature, own-niche):**
| cap | noskill | ungated | gated | Δ(ga−un) |
|---|---|---|---|---|
| 0 | 0.594 | 0.938 | 0.938 | 0.000 |
| 2 | 0.594 | 0.969 | 0.938 | −0.031 |
| 8 | 0.594 | 1.000 | 0.938 | −0.062 |

Spearman(cap, Δ) = −0.95 (p=0.014). **B-rate (one wrong skill, prob ε):** 0.94→0.90→0.86→0.76→0.63 over
ε 0→1 (Spearman −1.0). **B-count (m simultaneous wrong skills, ε=1):** 0.94→0.63→0.25→0.08→0.04 over
m 0→8 (Spearman −1.0; matches R4's cap-8 collapse to ~0.03).

### Verdict
- **A — crystallization volume is HARMLESS (mildly helpful).** Immature *correctly-tagged* snapshots
  never corrupt the additive sum; ungated even *rises* (0.94→1.00) as more are added (weak-but-correct
  bias), so gating them is slightly counter-productive. *Integrity note:* I pre-registered Δ≈0; the
  held-out result mildly **contradicts the exact null** (Δ goes negative) but in the direction that
  **reinforces** the qualitative claim — the crystallization channel needs no gate. Recorded as refined,
  not cleanly confirmed.
- **B-rate — a single mis-fire is MODERATE** (0.63 at ε=1, not catastrophic).
- **B-count — harm SCALES STEEPLY with simultaneous wrong count** (collapse by m≥4). This is the lever
  R4 actually varied (`cap`).

### Implication — deflates R4's strength, sharpens the lesson
R4's catastrophic collapse is reachable **only when many wrong skills fire at once** (additive sum of
confident-wrong biases) — not from crystallization *volume* (harmless), nor from occasional *single*
mis-fires (moderate). So PAO's real liability is **indiscriminate fire-all-admitted triggering of a
junk-laden library**, and competence-gated admission helps *precisely and only* by bounding the fire-set.
The lever worth engineering is PAO's **router / applicability** (how many and which skills fire), not its
crystallization counter — "more skills" is not itself the danger. This both tempers R4's apparent
strength (the harm needed a pessimal many-wrong-fire setup) and gives a concrete, falsifiable target:
fix triggering, not crystallization frequency. Sharpens the P3 finding (wrong skill fires) and the
project theme *property ≠ usefulness*.

---

## 2026-06-15 — Ladder R4: the law TRANSFERS to PAO's RL regime with PAO's own combiner (capstone)

Pre-registered in `PREREG_R4.md` (frozen before run; held-out seeds 100–107, sanity used seed 0).
`exp_pao_coverage/{grid.py,r4.py}`, log `…/results/r4.log`. Final rung of the RDD→PAO ladder: take
R1b's law into **multi-task RL** using **PAO's own combiner** — skills applied as ADDITIVE bias to the
base policy's action-logits, SUMMED over all triggered skills (so junk biases accumulate; this is the
redundancy-fragile combiner, exactly the R1b condition, now operating in an MDP where errors compound).

Env: 5×5 grid, K=4 cardinal-goal niches (genuinely distinct — validated; the 1D corridor of `core.py`
was degenerate, a single monotone policy covered 2 goals). Library = mediocre/forgetful base + one
MASTERED skill per niche + `cap` MIS-ASSOCIATED junk skills per niche (a skill mastered on a *different*
niche, tagged to k → confidently-wrong bias when it fires for k = PAO's P3 failure: wrong skill fires in
wrong context). `cap` = capacity. Arms: `noskill` (base), `ungated` (base + 1 good + cap junk per niche),
`gated` (competence-gated: one niche-covering skill per niche).

| cap | noskill | ungated | gated | Δ(ga−un) | g | p(ga>un) |
|---|---|---|---|---|---|---|
| 0 | 0.562 | 0.969 | 0.969 | +0.000 | 0.00 | 1.000 |
| 1 | 0.562 | 0.562 | 0.969 | +0.406 | +3.73 | 0.004 |
| 2 | 0.562 | 0.312 | 0.969 | +0.656 | +4.44 | 0.004 |
| 4 | 0.562 | 0.094 | 0.969 | +0.875 | +7.47 | 0.004 |
| 8 | 0.562 | 0.031 | 0.969 | +0.938 | +10.03 | 0.004 |

### Verdict: GATING HELPS in the RL regime, and the gap grows with capacity — prediction confirmed
- Ungated **collapses** as junk accumulates (0.97 → 0.03); gated is **immune** (0.97 at every cap);
  noskill flat (0.562, the floor). **Δ(gated−ungated) grows monotonically, Spearman(cap, Δ) = +1.000,
  p=0.000**; paired Wilcoxon p=0.004 (the n=8 floor) at every cap≥1; Hedges g +3.7…+10.
- Pre-registered decision rule met → **R1b's law transfers to PAO's regime**: with PAO's fragile additive
  combiner, coverage/competence-gated skill admission improves *task performance*, not just parsimony.

### Added mechanistic insight (on record before the run)
*Immature/weak* junk does NOT corrupt the additive sum (sanity v1: ungated stayed 1.0) — only
*confidently-wrong* (mis-triggered) junk does, and it compounds over MDP steps. **PAO's actionable
failure mode is indiscriminate skill TRIGGERING (a competent-but-wrong skill firing), not premature
crystallization per se.** The fix is competence-/coverage-gated admission. This sharpens the P3 finding.

### Implication — the controlled ladder is complete
R0 (joint co-adaptation) → R1 (frozen + robust ridge: only parsimony) → R1b (frozen + fragile averaging:
performance, scaling with capacity) → **R4 (RL + PAO's additive combiner: performance, scaling with
capacity).** One factor changed per rung, no confounded RDD→gridworld jump. The clean law
**"coverage-gating's performance payoff ⟺ the combiner is redundancy-fragile"** now holds end-to-end
into the regime PAO actually operates in. This is the project's strongest constructive result *for* a
PAO-style mechanism: a well-defined, falsifiable condition under which coverage-gated skill admission
demonstrably helps — and a matching diagnosis (fragile additive combiner + indiscriminate triggering)
of when PAO's library will instead hurt.

---

## 2026-06-15 — Ladder R1b: gating's performance payoff ⟺ COMBINER FRAGILITY (contingency locked)

Pre-registered in `PREREG_RDD_ladder_r1b.md` (frozen before run; held-out seeds 400–414).
`exp_rdd/rdd_ladder_r1b.py`, log `…/results/rdd_ladder_r1b.log`. Isolates the one factor R1 left open:
the **combiner**. R1 used a robust (ridge) readout and found no performance benefit from coverage-
gating. R1b changes ONLY the combiner to a **fragile** one — each frozen channel predicts the full
target standalone, ensemble = **mean** (fixed equal weights, cannot down-weight a bad/redundant unit),
matching PAO's additive-bias skill-application. Same task, same sequential frozen admission, capacity
sweep. (Robust contrast = R1.)

| M_max | M/K | gated | ungated | Δ(un−ga) | g | p(<) |
|---|---|---|---|---|---|---|
| 2 | 0.4 | 2.495 | 2.784 | +0.289 | −0.46 | 0.009 |
| 5 | 1.0 | 2.250 | 2.711 | +0.461 | −1.03 | 0.000 |
| 8 | 1.6 | 2.114 | 2.646 | +0.532 | −1.46 | 0.000 |
| 12 | 2.4 | 2.017 | 2.658 | +0.641 | −2.70 | 0.000 |
| 16 | 3.2 | 1.984 | 2.665 | +0.680 | −3.18 | 0.000 |

### Verdict: with a fragile combiner, coverage-gating HELPS and scales with capacity
- Gated beats ungated at every capacity; **Δ(ungated−gated) grows monotonically with M/K, Spearman =
  +1.000, p<0.001.** Gated even *improves* with capacity (2.50→1.98, admitting more covering units),
  while ungated stays stuck (~2.66 — extra averaged units drag the mean).
- **Contrast R1 (robust ridge combiner): Spearman(M/K, Δ) = −0.06, no benefit.** Same task, same
  admission, only the combiner differs.

### Implication — the controlled ladder yields a clean causal law
**Coverage-gating's *performance* payoff ⟺ the unit-combiner is redundancy-fragile.** Robust combiner
(ridge) → only parsimony (R1); fragile combiner (fixed-weight averaging / additive bias) → real
performance gain scaling with capacity (R1b); joint co-adaptation → also benefits (R0). Because **PAO's
skill-application is fragile** (skills added as policy bias; redundant/wrong skills actively hurt — the
P3 finding), this controlled chain (R0→R1→R1b) now *predicts* coverage-gated skill admission will
improve PAO's task performance — not merely trim the library. Reached by changing one factor per rung
(joint→frozen, then robust→fragile combiner) instead of one confounded RDD→gridworld jump. Remaining
ladder rungs toward PAO: R3 (rich/network units), R4 (RL control, where PAO's fragile combiner lives).

---

## 2026-06-15 — Ladder R1: coverage PERFORMANCE benefit does NOT survive freeze-and-admit (parsimony does)

Pre-registered in `PREREG_RDD_ladder_r1.md` (frozen before run; held-out seeds 300–314).
`exp_rdd/rdd_ladder_r1.py`, log `…/results/rdd_ladder_r1.log`. First rung of the controlled RDD→PAO
transfer ladder: change ONLY the mechanism — joint co-adaptation + auxiliary loss (R0) → **sequential
crystallization of FROZEN units + a coverage admission gate** (PAO-style) — keeping the RDD task/units/
supervision fixed. Build = residual boosting of frozen low-pass channels; global readout = ridge LS.
Arms: **gated** (admit only if it lowers val MSE > EPS) vs **ungated** (admit all), swept over capacity.

| M_max | M/K | gated | ungated | Δ(un−ga) | gated admits |
|---|---|---|---|---|---|
| 2 | 0.4 | 1.764 | 1.764 | 0.000 | 2.0 |
| 5 | 1.0 | 1.308 | 1.308 | 0.000 | 4.9 |
| 8 | 1.6 | 1.301 | 1.305 | +0.004 | 6.1 |
| 12 | 2.4 | 1.296 | 1.307 | +0.012 | 6.7 |
| 16 | 3.2 | 1.295 | 1.295 | −0.001 | 6.7 |

### Verdict: PERFORMANCE benefit does NOT survive; PARSIMONY benefit does
- **No performance benefit:** gated ≈ ungated at every capacity (Spearman(M/K, Δ) = −0.06, p=0.91).
  The R0 coverage *performance* win does **not** transfer to the freeze-and-admit mechanism.
- **Why:** frozen units + a regularized (ridge) readout are **robust to redundancy** — the readout
  simply down-weights redundant frozen channels. There is no joint-coadaptation overfitting for
  coverage to prevent. (Contrast R0 joint: no-LI MSE *rose* with M, 3.40→3.64.)
- **Parsimony survives:** the gate saturates at ~6.7 admitted units regardless of budget while matching
  ungated's MSE — at M_max=16 that is **6.7 vs 16 units at equal accuracy.** A lean library for free.

### Implication (this is exactly why the controlled ladder mattered)
**R0's coverage benefit and PAO's redundant-skill harm are different mechanisms.** R0's benefit came
from joint-coadaptation fragility; freezing + a good readout removes it, leaving only parsimony. PAO's
harm (P3: redundant/wrong skills *applied as policy bias* actively hurt reward) is a property of its
**fragile RL skill-application**, not of a clean readout. So a single jump RDD→PAO-gridworld would have
conflated these — the ladder localized it on one cheap rung. Consequence for the roadmap: the coverage-
gating *performance* payoff for PAO is expected in the **fragile-combination / RL regime**, not in
supervised regression; the immediately transferable win is **library parsimony** (coverage-gating prunes
the redundant skill library at no accuracy cost — directly addresses the P3 redundant-library problem).

---

## 2026-06-14 — RDD capacity law: coverage benefit scales with excess capacity M/K (CONFIRMED)

Pre-registered in `PREREG_RDD_capacity.md` (frozen before run; held-out seeds 200–219).
`exp_rdd/rdd_capacity.py`, log `…/results/rdd_capacity.log`. Characterizes the positive coverage-reward
result (entry below) by sweeping channels M against K=5 signal frequencies; benefit
`Δ(M) = MSE(noLI) − MSE(cov β=10)`.

| M | M/K | no-LI | cov10 | Δ10 | g | p(<) |
|---|---|---|---|---|---|---|
| 2 | 0.4 | 3.396 | 3.410 | −0.014 | +0.09 | ns |
| 3 | 0.6 | 3.375 | 3.444 | −0.069 | +1.22 | ns |
| 5 | 1.0 | 3.466 | 3.479 | −0.013 | +0.20 | ns |
| 8 | 1.6 | 3.513 | 3.414 | +0.099 | −1.12 | 0.000 |
| 12 | 2.4 | 3.571 | 3.405 | +0.166 | −1.75 | 0.000 |
| 16 | 3.2 | 3.638 | 3.356 | +0.282 | −3.41 | 0.000 |

### Verdict: CAPACITY LAW CONFIRMED
- **Spearman(M/K, Δ10) = +0.943, p=0.005** — the coverage benefit rises monotonically with excess
  capacity. Sign flips at capacity: under-complete (M<K) mean Δ = −0.04 (hurts); over-complete (M≥K)
  mean Δ = +0.13, growing to +0.28 at M/K=3.2 (g=−3.41).
- **no-LI degrades as M grows** (3.40 → 3.64): uncoordinated redundant channels *hurt*. **cov10 stays
  flat/low** (~3.36): the coverage reward converts otherwise-harmful excess capacity into useful
  spectral coverage. So coverage's value is precisely *redundancy repair*.
- Nuance: a *weak* coverage nudge (β=1) helps mildly at all M (Δ1 > 0 everywhere); the *strong* nudge
  (β=10) is what exhibits capacity-scaling (and needs spare capacity, else it over-distorts).

### Implication
Upgrades the positive lead from "a regularizer that helps in one cell" to a **quantitative law**: *the
value of task-aligned diversity scales with excess capacity.* This is exactly RDD's own forward framing
(redundant heads in over-parameterized Transformer/S4/Mamba models) — and gives it a concrete,
pre-registered, held-out-confirmed recipe with a measurable dose-response. Strongest result of the
investigation. Caveat: still the minimal SSM/sine task; the open scaling question is whether the law
holds in a real multi-head sequence model.

---

## 2026-06-14 — RDD coverage reward: FIRST POSITIVE LEAD — task-aligned diversity helps (over-complete)

Pre-registered in `PREREG_RDD_taware.md` (frozen before run; **confirmation on held-out seeds 100–119**,
disjoint from the calibration seed). `exp_rdd/rdd_taware.py`, log `…/results/rdd_taware.log`. Tests the
constructive implication of the two prior RDD entries (generic LI is task-blind → only *task-aligned*
diversity could help): three repulsion forms — **generic** (τ-distance), **taware-overlap** (penalize
spectral overlap), **coverage** (reward covering the signal spectrum, `Σ_f Ŝ(f)·max_m p_m(f)`, with
`Ŝ(f)` estimated from training data by FFT, not oracle). Grid `{asym,sym} × {noLI, genLI, cov β∈{1,10,50}}
× M∈{3,8}`, N=20.

| regime | no-LI | cov β=1 | cov β=10 | cov β=50 | genLI |
|---|---|---|---|---|---|
| **M8 asym (over-complete)** | 3.521 | 3.474 (g−0.86)\* | **3.412 (g−1.63)\*** | 3.447 (g−0.78)\* | 4.417 |
| M3 asym | 3.390 | 3.364 (g−0.35) | 3.450 | 3.533 | 4.405 |
| M8 sym (collapse) | 5.883 | neutral | neutral | neutral | 6.350 |
| M3 sym (collapse) | 5.878 | neutral | neutral | neutral | 6.362 |

(\* = p<0.05 and g≤−0.5, pre-registered "helps" bar. Primary cell asym M=8 β=50: g=−0.78, p=0.005.)

### Verdict: COVERAGE HELPS — confirmed, pre-registered, on held-out seeds (the project's first positive)
- **Task-aligned coverage reward lowers test MSE in the over-complete (M=8) regime** at *all three* β
  (g −0.78 … −1.63, all p<0.01), peak at β=10 (~3% reduction). Benefit carries to **OOD** (4.13→4.03).
- **Capacity-gated:** in M=3 (not over-complete) only the smallest β gives a sub-threshold nudge; larger
  β hurts. The benefit needs spare channels to deploy.
- **Collapse regime (sym): still no help** from any form — no diversity term breaks a symmetric
  degeneracy (the symmetric MSE gradient is degenerate; coverage/repulsion can't pick a useful
  direction). Consistent across all three RDD experiments.
- **Generic LI and overlap-penalty still hurt** everywhere — only the *coverage-rewarding* form helps.

### Implication (a clean, constructive principle)
**Channel diversity improves downstream performance only when it is (i) functional coverage aligned with
the task and (ii) deployed where there is excess capacity.** This turns RDD's negative-for-downstream
result (generic LI doesn't help — our two prior entries) into a positive recipe, and it pins down
exactly *when* the project-wide "property ≠ usefulness" gap closes: task-alignment **plus** spare
capacity. Modest in magnitude (~3%) but robust (held-out, pre-registered, monotone-ish in β, generalises
to OOD). First genuinely positive, actionable result in the investigation. Caveat: minimal SSM /
sine-mixture task; whether the coverage-reward recipe scales to real sequence models (Transformer/S4/
Mamba heads, per RDD's own framing) is the natural next test.

---

## 2026-06-14 — RDD crossover: NO regime where LI helps downstream (repulsion is task-blind)

Pre-registered in `PREREG_RDD_crossover.md` (frozen before run). `exp_rdd/rdd_crossover.py`,
log `…/results/rdd_crossover.log`, data `…/rdd_crossover.pkl`. Follow-up to the RDD-LI entry below:
that test found LI harmful in a *no-collapse* regime; RDD's benefit claim is conditional on **collapse**
("escape through symmetry breaking"), so this constructs the collapse regime and hunts the crossover.
Manipulated variable = init symmetry: **asym** (per-channel `W_in`/`W_out` → channels self-separate)
vs **sym** (shared `W_in`/`W_out` + near-identical τ init with tiny jitter → near-degenerate, LI is the
dominant separating force). Crossed with {no-LI, LI β=10} × M∈{3,8}. N=20.

| regime | no-LI test MSE | LI test MSE | diversity no-LI→LI | verdict |
|---|---|---|---|---|
| M3 asym (no collapse) | 3.397 | 4.407 | 3.80 → 9.63 | LI hurts |
| M3 sym (collapse) | 5.845 | 6.338 | 0.67 → 10.98 | LI hurts |
| M8 asym | 3.510 | 4.417 | 2.69 → 7.55 | LI hurts |
| M8 sym (collapse) | 5.856 | 6.342 | 0.70 → 9.07 | LI hurts |

### Verdict: NO CROSSOVER. LI never helps downstream — not even in the collapse regime built to favour it
- LI HURTS test MSE in **all four** cells (and OOD); pre-registered prediction confirmed.
- Crucially, in the **sym/collapse** regime LI **did act** — it broke the degeneracy (diversity 0.7 →
  ~10) — yet MSE still **worsened** (5.85 → 6.34). So this is not "LI couldn't separate the channels";
  it's "LI separated them and that made prediction worse."
- A perfectly symmetric init is an **LI-proof fixed point** (calibration: div=0 even with LI) — LI needs
  a seed of asymmetry to act on. The jittered sym regime is the meaningful collapse case.

### Mechanism (the actual lead)
**LI's repulsion is task-blind.** It maximizes mutual channel distance, which is *not* coverage of the
signal's relevant timescales. The force that places channels *usefully* is the task (MSE) gradient; LI
fights it. So: in **asym**, channels self-organize to useful timescales and LI drags them off; in
**sym**, channels collapse and LI separates them to *task-agnostic* timescales — worse either way.
Diversity ≠ functional coverage ≠ usefulness. This is a sharper, mechanistic answer to RDD's open
question than the first test alone, and extends the project-wide "property ≠ usefulness" pattern
(cf. H_M, crystallisation, skill diversity) with a causal explanation: the property is produced by a
task-blind force, so it cannot align with task needs. (RDD's *mechanism* claim — γ-inertness — remains
validated; see entry below. Only its downstream-*usefulness* hope fails.)

---

## 2026-06-14 — RDD lateral inhibition: γ-inertness CONFIRMED; diversity does NOT help downstream

Pre-registered in `PREREG_RDD_LI.md` (frozen before the N=20 run). `exp_rdd/rdd_li.py`,
log `exp_rdd/results/rdd_li.log`, data `…/rdd_li.pkl`. New experiment line on the **RDD** paper
(Cai, *"Lateral inhibition enables escape from geometric deadlock"*, `docs/RDD_main.pdf` + supp).
Reproduces RDD's NN proof-of-concept (multi-channel low-pass SSM, learned τ_m=exp(θ_m), Gaussian
repulsion loss, Table S8) and **extends it to RDD's own explicitly-stated open question** — does the
LI-induced diversity translate to downstream gains? Folds in **Direction 1** (γ-inertness) and
**Direction 3** (LI → downstream MSE). N=20 seeds.

| Condition | diversity D | test MSE | OOD MSE |
|---|---|---|---|
| M1 (single channel) | 0.00 | 5.834 | 6.359 |
| **M3, no-LI** | 3.80 | **3.397** | **4.018** |
| M3, LI β=0.1 | 6.29 | 3.559 | 4.145 |
| M3, LI β=1 | 10.56 | 4.131 | 4.730 |
| M3, LI β=10 | 9.63 | 4.407 | 4.995 |
| M3, LI β=30 | 9.58 | 4.430 | 5.018 |
| M3, **inert**-LI (any β) | 3.80 | 3.397 | 4.018 |

### Direction 1 — γ-inertness: CONFIRMED (a clean positive validation of RDD)
The inert-LI arm (β present, repulsion kernel's gradient detached) is **identical to no-LI at every
β** — diversity flat at 3.796, range **0.000** — while real-LI diversity varies with β (range 4.26).
**The scalar gain does nothing without the actual repulsion interaction.** This replicates RDD's
central structural claim (Table 2: no-kernel configs are non-responsive to γ_LI) in a *trained neural
net*, not just the scalar resonance toy. RDD's mechanism claim holds.

### Direction 3 — does diversity translate to downstream performance? NO (monotonically harmful)
- **LI raises diversity but worsens prediction.** test MSE 3.40 (no-LI) → 4.41 (β=10); OOD 4.02 → 5.00.
  Harm is **monotone in β** and present even at the smallest β=0.1 (3.56 > 3.40) — **no sweet spot.**
- **Diversity is anti-correlated with usefulness:** Spearman ρ(D, test MSE) = **+0.466, p<0.001**
  across all LI runs — more separation, worse prediction. (Effect sizes vs no-LI are huge, g≈+18,
  only because the conditions are near-deterministic across seeds; the meaningful figure is the ~30%
  MSE gap and the monotone trend.)
- **The premise barely holds:** without LI the channels already differentiate (D=3.80, not collapsed),
  so LI mostly *forces* extra separation the task doesn't want — pushing τ away from the frequencies
  actually present in the signal. **Coverage and accuracy trade off:** repulsion helps avoid deadlock
  (RDD's regime) but costs fit when natural differentiation already suffices (this regime).

### Implication
This **directly answers RDD's open question** ("does the diversity increase translate to downstream
performance gains?" — RDD Supp §4): in the NN PoC, **it does not — it costs them.** Two-sided result:
RDD's *mechanism* claim (γ-inertness; repulsion is the operative ingredient) is **validated**, but its
*usefulness* hope is **not** — the same "property ≠ usefulness" pattern as the H_M test. LI is a
coverage/anti-collapse tool, not a downstream-performance regularizer here. (Direction 2, Busch
IM/WMP/OMP, deferred — paper not yet available.)

---

## 2026-06-14 — P3 steelman: the "lead" is a metric artifact; plain PPO dominates

Pre-registered in `PREREG_P3_unpredictable.md` (frozen before the N=30 run). `p3_unpredictable.py`,
`agents.py::PAOLibrary*`. Log `results/p3_unpredictable.log`, data `results/p3_unpredictable.pkl`.

**Motivation.** P1/P2 swapped rules on a *fixed, known schedule*, where a clock suffices by
construction — so the trigger trivially added nothing. This test gives PAO its **best honest shot**:
the one regime engineered to need event-triggered crystallisation+reuse — **random, unsignaled,
recurring** shifts (action-mapping flips on the 1D corridor; observations cannot reveal the active
regime). And it **steelmans** PAO beyond the shipped single-skill code to a multi-skill **library**
with change-point-triggered crystallisation *and* re-selection (`PAOLibrary`). Arms (identical
library-building; differ only in the re-selection trigger): **BOCPD** (change-point detection),
**Random** (count-matched random resets), **Obs** (per-step obs-confidence selection, *no* detection),
**NoSkill** (plain PPO). N=30.

| Arm | adapt (primary) | recov ↓ | asymp | total |
|---|---|---|---|---|
| BOCPD | −0.029 | 32.0 | 0.367 | 58.8 |
| Random | −0.180 | 33.5 | 0.411 | 82.2 |
| **Obs** (no detection) | **+0.210** | 32.9 | 0.240 | 72.5 |
| **NoSkill** (plain PPO) | −0.546 | **30.9** | **0.730** | **103.5** |

### Verdict: pre-registered primary says "LEAD FOUND" — and the cross-checks expose it as an artifact
- **Primary metric fires:** BOCPD > NoSkill on `adapt` (first-15-eps after recurrence), g=+1.47,
  p<0.001 → the frozen rule prints **"LEAD FOUND."**
- **But it is not a real benefit.** The cross-checks I pre-registered exactly to catch this all
  contradict it: on **total reward** NoSkill wins (103.5 vs 58.8, g=−0.56); on **asymptotic
  competence** NoSkill wins hugely (0.73 vs 0.37, g=−1.11); on **recovery time** they tie (g=+0.21,
  ns). The `adapt` metric merely rewards *not crashing to the floor in the first 15 episodes*: a
  cached skill blunts the immediate post-swap crash, buying a head-start — paid back with worse
  asymptotic performance and less total reward. Plain PPO crashes, then relearns to a **higher** level.
- **Detection is NOT load-bearing.** BOCPD ≈ Random on `adapt` (g=+0.46, just under the 0.5 bar →
  rule prints "detection load-bearing: NO"), and **Obs — which uses no change-point detection at all —
  scores *best* on the primary metric** (+0.210). So whatever lifts `adapt` is "apply some cached skill
  to soften the crash," which needs none of PAO's distinctive machinery.
- **Net:** on every holistic measure (total, asymptotic, recovery), the skill library is **net-harmful
  vs plain PPO** — consistent with P1/P2. The steelman, in the regime built to favor it, fails.

### Why this matters (methodological lead)
This is a clean, pre-registered demonstration that the **"fast post-swap recovery" style of metric**
— the kind the Two-Gate papers lean on — is **gameable by a head-start artifact** that does not
reflect real competence and does not require the mechanism under test. It explains *how* a published
Two-Gate result can look positive while the mechanism neither generalises nor beats a PPO baseline.
Pre-registering the effect direction + cross-checks (not just the headline comparison) is what caught it.

### Integrity note
The `PAOLibrary` mechanism was developed against seeds {0,1,2} (bugs fixed pre-freeze: action-flip
regimes, competence fallback, detection refractory, no-exploration-on-cached-skill — see PREREG §6),
then **frozen** before the N=30 run. Calibration was seed-fragile (1 big win, 2 losses); the N=30
result resolves it. On-record prediction (inconclusive/null/anti, high variance) was essentially
correct: the primary fires positive but is an artifact; holistically it is ANTI.

---

## 2026-06-09 — H_M test: manifold health is ANTI-correlated with skill usefulness

Pre-registered in `PREREG_HM.md` (frozen before run). `hm_test.py`, controlled replica, N=80 skills
crystallised across episodes {3..65}×seeds{0..9}, app=0.3. Manifold computed over 64-dim policy-torso
activations (internal-rep analog); ground truth = skill-only success Q(z). Log `results/hm_test.log`,
data `results/hm_test.pkl`. **My on-record prediction (H_M invalid) was correct — and the result is
stronger/worse than predicted.**

| vs. usefulness Q | Spearman ρ | p | theory predicts |
|---|---|---|---|
| **H_M (manifold health)** | **−0.516** | <0.0001 | positive (high health → good) |
| d_int (intrinsic dim) | +0.472 | <0.0001 | negative (low dim = compact = good) |
| V_info (info volume) | −0.325 | 0.003 | positive (high volume = good) |
| episode (training amount) | +0.814 | <0.0001 | — |
| **partial ρ(H_M, Q \| episode)** | **−0.319** | 0.004 | positive |

Within-episode (training fixed), every converged stage ep15–65: ρ(H_M,Q) negative (−0.37…−0.62);
mean −0.27.

### Verdict: H_M is anti-predictive — worse than "just a clock"
- $H_{\mathcal{M}}$ is **significantly negatively** correlated with skill usefulness, **even controlling
  for training amount** (partial ρ=−0.32, p=0.004). As an acceptance gate it would **actively select the
  wrong skills.**
- **Both components are inverted vs the theory:** useful skills have *higher* $d_{\text{int}}$ (+0.47,
  theory says compact=good) and *lower* $V_{\text{info}}$ (−0.33, theory says high-volume=good). A good
  skill is a *tight, low-variance, reliable* trajectory; the "high information volume = non-degenerate =
  healthy" intuition is backwards for skills. The $d_{\text{int}}$ result is independent of any
  $V_{\text{info}}$ definition, so the inversion is robust to that choice.
- The frozen rule printed "AMBIGUOUS" only because it bucketed a significant *reverse* effect there
  (same rule-gap as P1; amendment in `PREREG_HM.md`). Substantively: **H_M does not work as a quality
  criterion in this setting; it is anti-predictive.**

### Why this is the deepest result
The manifold-health criterion is the **single most cross-cutting claim** in the whole constellation —
the $\Gamma_M$ acceptance gate in PAO ($A_{\text{eff}}$), the "true quale" condition in the
consciousness paper, and the ρ-U-O viability interval in the dialectics paper. Here it is *anti*-
correlated with the very property it is meant to certify.

### Sharp prediction for Dr. Cai's run6 data (the ecological test)
If H_M is anti-correlated with usefulness, then his **reward-0 skills (seeds 43/46) should have
$H_{\mathcal{M}} \ge$ his best skill (48)** — which would directly explain *why the $\Gamma_M$ filter
admitted the useless skills*. This is the exact correlation to compute once the run6 checkpoints/metrics
are available.

### Caveats
Controlled/minimal env; manifold over policy-torso activations (not RSSM latents); $V_{\text{info}}$ =
total variance (one of several possible operationalisations). The ecological confirmation needs Dr.
Cai's run6 checkpoints (pending). The $d_{\text{int}}$↔Q inversion is robust to the $V_{\text{info}}$
choice; the composite-H_M sign could shift under a very different $V_{\text{info}}$ definition, but it
would have to invert the empirically-observed "tighter = better" relationship to rescue the theory.

---

## 2026-06-08 — P2/Harder-env (BOCPD + non-convex 2D): H1 FALSIFIED again, and caching reverses

Pre-registered in `PREREG_P2_harderenv.md` (frozen before run). `p2_harder.py`, N=30, seeded,
app_thresh=0.3, env=`TwoGate2DShiftEnv` (non-convex, Location-Shift). Log `results/p2_harder.log`,
data `results/p2_harder.pkl`. **My on-record prediction (FALSIFIED) was correct.**

| Arm | Q(z) | P3 reuse | P2 lock-in | cryst_ep |
|---|---|---|---|---|
| BOCPD (corrected trigger) | 0.284 | 0.606 | 0.172 | 118 |
| Heuristic | 0.558 | 0.700 | 0.022 | 249 |
| RandomLate (decisive control) | 0.432 | 0.533 | 0.094 | 173 |
| RandomUniform | 0.358 | 0.460 | 0.097 | 156 |
| **NoSkill (= pure PPO)** | 0.548 | **0.976** | 0.328 | 249 |

### Result 1 — H1 FALSIFIED (worse than 1D: now a significant *reverse* effect)
Decisive comparison BOCPD vs RandomLate, PRIMARY Q(z): 0.284 vs 0.432, **g = −0.82, reverse p < 0.001**.
The principled BOCPD trigger is **significantly worse** than a random-late window, because it fires
**prematurely** (ep 118 vs 173) and crystallises a less-converged skill.

### Result 2 — the trigger detects nothing special; "skill quality" = amount of training
Q(z) is **monotonic in crystallisation episode**, nothing else:
NoSkill/Heur (ep 249 → Q 0.55) > RandomLate (173 → 0.43) > RandomUniform (156 → 0.36) > BOCPD (118 → 0.28).
The Heuristic "beats" RandomLate on Q(z) (g=+0.69, p=0.002) **only because it happens to fire later**
(ep 249), not because it detects a special moment. There is no signal-detection benefit — both triggers
are just noisy proxies for "how long you trained."

### Result 3 (the bombshell) — the skill-caching benefit REVERSES: the mechanism is net HARMFUL
The pre-registered sanity check **failed**: **NoSkill (pure PPO) beats every skill arm on Phase-3 reuse**
— 0.976 vs BOCPD 0.606 (g=−1.02), vs RandomLate 0.533 (g=−1.17). Applying the cached (premature,
30–55%-competent) skill *drags the agent down*. NoSkill also adapts best in Phase-2 (0.328, least locked).
In 1D the caching benefit was real (g=+0.91 *for* the skill); here it is strongly negative.
→ **The 1D "caching helps" result was an artifact of a trivial corridor** where a frozen near-optimal
policy can only help. In a non-trivial env, the always-on frozen skill is a liability.

### Combined verdict across P1 + P2
The distinctive mechanisms do **not** survive controlled testing:
- **Crystallisation timing confers no benefit** (1D tie; harder-env reverse) — the trigger is a
  training-duration proxy, not a phase-transition detector.
- **A principled BOCPD trigger makes it worse** (fires early).
- **The skill-cache benefit is environment-dependent and turns harmful** once the task is non-trivial.
- Pre-conditions for the theory's premise are absent: convergence is **gradual, not discrete**, so there
  is no event to detect; and the shipped BOCPD never fired at all.

### Caveats (honest limits)
- app_thresh=0.3 makes the applicability gate a near-always-on switch (it is collapsed — P0.6). A
  *functioning* applicability gate (apply the skill only when appropriate) might reduce the Phase-3 harm
  in Result 3 — this is the strongest remaining defence and is worth a final check before closing.
- Single skill, no library; dormancy is dead; trigger is corrected-MAP-BOCPD, not the (broken) shipped
  one; env is still a small grid. None of these rescue *timing* (Results 1–2), which is H1 itself.

---

## 2026-06-07 — P1 CRUX (H1): the trigger does NOT select specially-reusable windows

Pre-registered in `PREREG_P1.md` (frozen before running). `p1_crux.py`, N=30, seeded, app_thresh=0.3.
Log: `results/p1_crux.log`, data: `results/p1_crux.pkl`. **My on-record prediction (falsified/inconclusive)
was correct.**

| Arm | Q(z) | P3 reuse | P2 lock-in | cryst_ep | return@cryst |
|---|---|---|---|---|---|
| **Trigger** (signal-selected) | 0.772 | 1.403 ± 0.08 | −0.729 | **21.2** | 1.42 |
| **RandomLate** (matched random, decisive control) | **0.987** | 1.410 ± 0.08 | −0.733 | 59.8 | 1.41 |
| RandomUniform (weak floor) | 0.798 | 1.146 ± 0.16 | −0.548 | 36.5 | 0.69 |
| NoSkill (sanity floor) | 0.772 | 0.698 ± 0.18 | −0.369 | 21.2 | 1.42 |

### Pre-registered decisive comparison: Trigger vs RandomLate
- **PRIMARY Q(z):** 0.772 vs 0.987, **g = −2.79** (huge, **wrong direction** — RandomLate far better),
  p = 1.000 for "Trigger > RandomLate".
- **SECONDARY P3 reuse:** 1.403 vs 1.410, **g = −0.02, p = 0.889** → a clean statistical tie
  (meets the pre-registered FALSIFIED criterion |g|<0.3 ∧ p>0.05).
- **SECONDARY P2 lock-in:** −0.729 vs −0.733, g = +0.01 → identical.

### Verdict: H1 NOT SUPPORTED — the distinctive claim fails
The event-trigger provides **no advantage** over a competence-matched random window:
- On downstream reuse (P3) the two are **indistinguishable**.
- On skill quality (Q(z)) the trigger is **significantly worse** because it **fires prematurely**
  (median ep ≈ 21, return-converged-but-not-settled → Q=0.77) whereas a random *late* window
  (ep ≈ 60) crystallises a fully-converged skill (Q=0.99).

So "crystallisation detects a special phase-transition moment" is refuted in the minimal setting: a
random snapshot of a converged policy is as reusable (and a better skill) than the triggered one.

### What DOES survive (reported in fairness)
- **Skill caching helps:** Trigger and RandomLate both beat NoSkill on P3 reuse by **g ≈ +0.91,
  p < 0.001**. Freezing/reusing a skill genuinely avoids relearning — the framework's *caching* claim
  holds; only the *trigger/timing* claim fails.
- **A weaker "convergence detector" reading of the trigger holds:** Trigger beats RandomUniform on P3
  (1.403 vs 1.146) because it avoids pre-convergence windows. But that is just "wait until the policy
  has converged" (an entropy threshold) — not the theory's distinctive phase-transition detection.

### Honest note on the automated verdict
`p1_crux.py` printed "INCONCLUSIVE" on the PRIMARY because the frozen decision rule bucketed
*significant-in-the-wrong-direction* into INCONCLUSIVE (its FALSIFIED branch required |g|<0.3, written
for the tie case). Substantively the result is stronger than a tie: the trigger is *worse* on Q(z) and
*tied* on reuse. The SECONDARY P3 outcome meets the FALSIFIED criterion outright. I am **not** rewriting
the frozen rule; see the dated amendment in `PREREG_P1.md`. Conclusion either way: **H1 not supported.**

### Caveats (limits of this result)
1D trivial corridor; app_thresh=0.3 makes the applicability gate a near-always-on switch; single skill;
heuristic trigger (not BOCPD). A defender could argue BOCPD or a harder, non-convex environment might
separate trigger from random — that is the legitimate next test if one wants to rescue H1. But the
**burden has shifted**: in the minimal setting the trigger shows no special value, and the headline
papers' framing (event-triggered crystallisation as the key mechanism) is not supported here.

---

## 2026-06-07 — Code/paper integrity audit (beyond the bugs already fixed)

Question: besides the seeding + dead-dormancy + collapsed-gate issues, is anything else wrong?
Answer: yes — five further issues, concentrated in the **paper-generation / reproducibility layer**.

**1. The main paper's numbers are hardcoded literals, not computed from data.**
`make_paper.py` (which generates `TwoGateLock_Paper.docx`) **never calls `pickle.load` (count = 0)**.
The entire 1D results table (lines 260–263) and every stat (Hedges' g, Wilcoxon, bootstrap CIs, 59%)
are typed-in strings, decoupled from any run. (`make_docx.py` *does* load the 1D pkl, so the two
generators can and do disagree.)

**2. The 2D results cannot be verified — the data file does not exist.**
The paper's entire Section 4 (2D Location-Shift: "59% lock-in, N=30, three batches") cites
`results/2d_shift_results.pkl`, which **is not in the repo**. The 2D claims are unreproducible from
the released code.

**3. The FlatPPO baseline takes ≥4 mutually inconsistent values across the project's own files:**
| Source | FlatPPO Phase-2 | FlatPPO Phase-3 early |
|---|---|---|
| `make_paper.py` abstract | **+0.10** | +0.30 |
| `make_paper.py` results table | **−0.155** | 0.300 |
| `make_docx.py` | **−0.16** | — |
| `README.md` | **+0.287** | **−0.041** |
The paper **contradicts itself** (abstract +0.10 vs its own table −0.155). This is the signature of
numbers hand-entered from different runs at different times and never reconciled.

**4. No global RNG seeding in ANY runner except `run_rule_swap.py` (after our P0.5 fix).**
`run.py`, `run_2d.py`, `run_ewc.py`, `run_option_critic.py`, `sweep_rigidity.py` → 0 seeding calls
each. So the 2D experiment, the EWC baseline, and the Option-Critic "0/45 converged" comparison were
all run without reproducible/matched seeds, inheriting the P0.5 problem.

**5. Method-vs-paper mismatch: skills are weight-copied, not distilled.**
Paper + code comments claim "KL-distilled SkillPolicy" via "trajectory-level distillation"; a
`distill_skill`/`SkillPolicy` (cross-entropy) exists — but the run path does
`skill_policy.load_state_dict(self.policy.state_dict())` (a raw copy of the full ActorCritic).
`distill_skill`/`SkillPolicy` are **dead code**. The distillation the paper describes is not what runs.

**6. (Methodological) FlatPPO is a confounded control.** PAO allocates extra networks at
crystallisation → consumes global RNG → diverges from FlatPPO *even when the skill is inert*
(verified: NoSkill app=0.7 == PAO app=0.7 ≠ Flat). So PAO-vs-Flat conflates the mechanism with
RNG-stream divergence. The clean control is **PAO-no-skill** (matched allocation), not FlatPPO.

### Calibrated verdict
- The **core 1D phenomenon is real and reproducible** (P0.6: −0.880/1.516 at app=0.3, causally
  attributable to the skill bias vs a matched NoSkill control). Not fabricated.
- But the **surrounding quantitative claims are unreliable**: hardcoded/inconsistent baselines,
  unverifiable 2D results (missing data), no reproducible seeding outside our fix, and a stated
  method (distillation) that doesn't match the code.
- **None of this touches the actual scientific question** — whether crystallisation *timing* matters
  (H1). That remains untested and is the right next step. Treat all existing quantitative claims
  (especially FlatPPO baselines and the 2D section) as **unverified** until regenerated under our
  seed-controlled, app_thresh=0.3, NoSkill-controlled protocol.

---

## 2026-06-07 — P0.6: mystery resolved — paper IS reproducible, but only at app_thresh≈0.3

**Why the gate never opened.** Instrumented the `ApplicabilityNet` output over 7,106 post-skill steps:
scores collapse to **[0.298, 0.495]** (mean 0.38, max 0.49) — never above 0.5. The classifier cannot
separate positives from negatives because in an 8-state corridor, success and failure trajectories
visit nearly identical states. So "applicability" is ill-posed here; at the default `app_thresh=0.7`
the gate is mathematically unable to open.

**Test (`p06_gate_test.py`, seeded, N=10):** vary `app_thresh` and compare PAO to its own NoSkill control.

| Condition | P2 late | P3 early | skill-apply % |
|---|---|---|---|
| Flat | −0.398 ± 0.254 | 0.557 ± 0.391 | 0.0 |
| PAO app=0.7 (shipped default) | −0.724 ± 0.080 | 1.042 ± 0.287 | **0.0** |
| NoSkill app=0.7 | −0.724 ± 0.080 | 1.042 ± 0.287 | 0.0 |
| **PAO app=0.3** | **−0.880 ± 0.000** | **1.516 ± 0.001** | **52.3** |
| NoSkill app=0.3 | −0.724 ± 0.080 | 1.042 ± 0.287 | 0.0 |

Causal test (PAO vs its own no-skill control): **app=0.7 ⇒ identical (inert); app=0.3 ⇒ divergent (causal).**

### Resolution
1. **The paper's headline numbers are real and reproducible** — `−0.880 ± 0.000` lock-in and `1.516`
   reuse appear **exactly**, but only when the skill gate actually opens (`app_thresh≈0.3`).
2. **The shipped `run_rule_swap.py` uses `app_thresh=0.7` (default) → the gate never opens → it does
   NOT reproduce the paper.** The paper was evidently generated with a low threshold (cf. `sweep_rigidity.py`
   "Soft" = 0.3) or trajectory mode. This is a **configuration mismatch in the released code**, not a
   fabrication: with the right knob, the result is solid and (in the locked regime) deterministic.
3. **The skill gate has a genuine causal effect** when active: vs the matched NoSkill control it produces
   *both* stronger Phase-2 lock-in and stronger Phase-3 reuse — the real hysteresis signature.

### Important caveat — the "intelligent gating" is not doing intelligent work
At `app_thresh=0.3` the collapsed ApplicabilityNet fires as a near-always-on switch, i.e. "always replay
the frozen Phase-1 policy." That trivially resists Phase-2 (it *is* Phase-1 behaviour) and trivially
reuses in Phase-3 (Phase-1 behaviour is correct again). So the **skill-bias mechanism** works, but the
**applicability classifier** (the supposedly smart gate) is non-functional in 1D — it adds no
discrimination. The lock-in is real but somewhat tautological.

### Status of the two gates after P0.6
- **Crystallisation/skill gate:** ✅ causally functional once the threshold lets it open (but the
  applicability classifier itself is collapsed — a dumb switch, not a learned gate).
- **Dormancy gate:** ❌ still dead code (`dormancy_weights` unused) — untouched by this test; remains P2.

### Consequences for the plan
- P0.6 (a)(b) essentially answered for the frozen-policy path: a gate *can* be causal; the released
  default is just wrong. Recommended cleanup: expose `app_thresh` in `run_rule_swap.py` (default 0.3),
  and fix the `n_skills` counter to count `skill_policy`.
- **P1 (random-window CRUX) is now meaningful** — we have a skill mechanism with a real causal effect
  to test trigger-timing against. Open question for P1: since the applicability gate is a dumb always-on
  switch and the skill is just "frozen Phase-1 policy", a random-late window may crystallise an equally
  good Phase-1 policy → H1 (trigger timing matters) is genuinely in doubt. That is exactly what P1 tests.
- Still worth running `sweep_rigidity.py` (seeded) for the Soft/Medium/Hard/Trajectory dose-response as
  confirmation in 2D.

---

## 2026-06-07 — P0.5: seed control fixed → reveals **both gates are functionally inert**

**Setup.** Added `set_seed()` (torch+numpy+random) keyed off each run's seed in `run_rule_swap.py`
(env was already deterministic; `self.rng` unused). Re-ran `--quick --seeds 0..9 --ablations`.
Log: `results/p05_seeded.log`. This makes seeds reproducible *and* matched across conditions
(PAO seed=k and Flat seed=k share the same init).

### Reproducibility is restored — and it exposes that the mechanism does nothing

Seeded aggregate (N=10):

| Metric | PAO-light | FlatPPO |
|---|---|---|
| Phase 2 late | −0.495 ± 0.223 | −0.146 ± 0.310 |
| Phase 3 early | 0.971 ± 0.292 | 0.351 ± 0.380 |
| Phase 3 late | 1.288 ± 0.149 | 0.885 ± 0.323 |

Direction is now "correct" (PAO > Flat in P3) but **non-significant**: P3 Wilcoxon p = 0.19,
g = +0.55; P2 p = 0.34, g = −0.39. So at N=10 under seed control, the effect is weak and NS —
neither the strong original claim nor my first reversed run.

### The decisive finding: the two gates have ZERO causal effect

Verified from `rule_swap_1d.pkl` (per-seed, all 260 episodes):
- **PAO-light == PAO-no-dormancy == PAO-no-skill, byte-identical, every seed.**
  Setting `skill_bias_strength ∈ {0, 1}` and `dormancy_lr_factor ∈ {0.3, 1.0}` changes *nothing*.
- **PAO-light ≠ FlatPPO** — so something separates them, but it is **not** the gates.

Instrumented one seed (counted skill applications during training):
- Skill crystallised at ep 18; **applied 0 times in 7,106 post-crystallisation steps (0.0%).**
- The `ApplicabilityNet` gate (`app_score > 0.7`) **never opens** → the frozen skill is never invoked.

**Therefore:** the entire PAO-vs-FlatPPO difference is an artifact of **RNG-stream divergence** —
allocating + training the (never-used) skill/applicability networks at crystallisation consumes
global RNG, shifting all later action sampling onto a different trajectory. Not a mechanism, noise.

### Why this happens — two code paths, wrong one shipped
- `PAOLight.act()` has two skill modes: **trajectory-matching** (`_use_trajectory=True`,
  nearest-neighbour action replay, radius-based — genuinely rigid) and **frozen-policy + applicability
  gate** (`_use_trajectory=False`, the default).
- `_use_trajectory` is set **only in `sweep_rigidity.py`**. The "primary" runner `run_rule_swap.py`
  never sets it → runs frozen-policy mode → inert gate.
- Corroborating mismatch: README claims **"Skills after Phase 1 = 1.0 (all seeds)"**, but the shipped
  runner yields **`n_skills = 0`** for every seed (frozen-policy skills go to `self.skill_policy`, which
  the `n_skills` counter — pointed at the unused trajectory `SkillCache` — never counts).
  → The shipped primary experiment is **not** the configuration that produced the paper's table.

### Consequences for the plan
1. The frozen-policy "two-gate" config as shipped demonstrates nothing — both gates are inert.
2. **New P0.6 (before P1): make at least one gate actually function.** Two sub-tasks:
   (a) test the **trajectory-matching** path (`sweep_rigidity.py`, `_use_trajectory=True`) — this is the
   path that may produce genuine, mechanism-driven lock-in; (b) fix the frozen-policy path so the
   applicability gate can open (threshold/scaling/training), since that is the PAO-faithful mechanism.
3. P2 (implement dormancy) stands, but is moot until a skill is actually invoked.
4. The CRUX random-window control (P1) is meaningless until a gate has a causal effect — **do P0.6 first.**

---

## 2026-06-07 — P0 reproduction of the 1D Two-Gate Lock **FAILED**

**Setup.** `basic` venv (torch 2.12+cu130, RTX 5070 Ti, but runs on CPU tensors), scipy+matplotlib
installed. Command: `python run_rule_swap.py --quick --seeds 0..9 --ablations` (N=10, quick mode —
identical config to the README's results table). Original `results/` preserved in `results_original/`.
Full log: `results/p0_repro.log`.

### Result: none of the hysteresis signatures reproduced; the reuse signature is REVERSED

| Metric | **PAO-light (repro)** | **FlatPPO (repro)** | Paper/README claim (PAO) |
|---|---|---|---|
| Phase 2 late-20 | −0.337 ± 0.207 | −0.486 ± 0.251 | **−0.880 ± 0.000** |
| Phase 3 early-20 | 0.238 ± 0.376 | 1.059 ± 0.268 | **1.518 ± 0.002** |
| Phase 3 late-20 | 0.328 ± 0.395 | 1.274 ± 0.240 | 1.519 ± 0.002 |

- **Signature 2 (structural inertia, expect PAO < Flat in P2):** NOT observed. PAO −0.337 is *above*
  Flat −0.486. Hedges' g = +0.20 (wrong sign). Wilcoxon p = 0.81.
- **Signature 3 (reuse acceleration, expect PAO > Flat in P3):** **REVERSED.** PAO 0.238 ≪ Flat 1.059.
  Hedges' g = −0.76 (moderate-to-large, wrong direction). Wilcoxon p = 0.81.
- **Signature 1 (discrete trigger / entropy collapse):** qualitatively present — crystallisation fired
  once per seed (e.g. ep 9, 16) and entropy collapsed (~0.67 → ~0.07). This is the only signature that held.

**In this run, FlatPPO is the best Phase-3 performer and PAO-light (both "gates") is the worst.**

### Ablation table (N=10, same run)
| Condition | P2 late | P3 early | P3 late |
|---|---|---|---|
| PAO-light (full) | −0.337 | **0.238** | 0.328 |
| PAO-no-dormancy | −0.315 | 0.776 | 0.819 |
| PAO-no-skill | −0.500 | 0.689 | 1.228 |
| FlatPPO | −0.486 | **1.059** | 1.274 |

### Two verified root causes

**(A) The experiment has no global-RNG seed control.**
`grep` confirms `torch.manual_seed` / `np.random.seed` / `random.seed` appear **nowhere** in the repo.
The `seed` argument only feeds `TwoGateLockEnv(seed=…)` and the explore-ε `RandomState`. Network
initialisation, PPO action sampling, and minibatch shuffling all use the **unseeded global RNG**.
→ The "10 seeds" are *not* reproducible runs; each is a different random network init. The original
`−0.880 ± 0.000` (identical to floating point across all seeds) cannot be produced deterministically by
this code as shipped — it requires the frozen skill to dominate so completely that every run collapses
to the same failure trajectory, which did **not** happen on my draw.

**(B) The dormancy gate is a no-op in 1D.**
`ppo_update(...)` accepts `dormancy_weights` but **never references it** (verified by reading lines
239–279 and by `grep`). `dormancy_lr_factor` (0.3 vs 1.0) is computed, passed in, and discarded.
→ `PAOLight` and `PAONoDormancy` differ only in a value that is thrown away. The paper's ablation
"PAO-no-dormancy ≈ PAO-light ⇒ inertia comes from the skill cache, not plasticity" is **confounded**:
the two conditions are behaviourally identical by construction, and any observed difference (e.g.
P3 0.238 vs 0.776 here) is **pure RNG noise from (A)**, not evidence about dormancy.
→ The "Two-Gate" thesis currently rests on **one** implemented gate.

### Interpretation (fair reading)
Because RNG is uncontrolled, my N=10 rerun is itself a single uncontrolled draw — it is *possible* the
original got a favourable draw and I got an unfavourable one. **But that is precisely the problem:** a
result whose sign flips across uncontrolled random draws is not a robust result. This does not by itself
falsify the underlying idea; it means **the current published evidence is unreliable and must be
re-established under proper seed control before anything is built on top of it.**

### Consequences for the plan
1. **New P0.5 (blocking): fix seed control** — seed torch+numpy+random per run; make the N seeds true
   independent-but-reproducible replicates. Re-run to get an honest baseline (does the effect survive?).
2. **P1 (random-window control) must wait for P0.5** — without seed control the control arms would also
   be noise.
3. **P2 redefined and promoted** — not "wire in BOCPD" but **actually implement the dormancy gate**
   (make `ppo_update` use `dormancy_weights`) *and* seed it, so the second gate can be tested at all.
4. The Q(z) validation gate is also **not called** in the 1D runner (`finish_episode` only sets a
   deferred flag) — so 1D currently admits every triggered skill unconditionally (≈ "AdmitAll").

---
