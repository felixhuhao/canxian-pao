# Pre-Registration — Is there a regime where lateral inhibition HELPS downstream? (the crossover)

> **FROZEN 2026-06-14, before the N=20 run.** Follow-up to `PREREG_RDD_LI.md` / `exp_rdd/rdd_li.py`,
> which found LI harms downstream MSE in a regime where channels self-separate (no collapse). Mechanism
> frozen after calibration (§5). Results → `FINDINGS.md`.

## 1. Claim under test
RDD's benefit claim is **conditional**: lateral inhibition helps when channels would otherwise
**collapse** into the same basin ("escape through symmetry breaking"). `rdd_li.py` tested LI only in a
no-collapse regime (channels self-separate) and found it harmful. This test constructs the **collapse
regime** and asks whether LI flips to helpful there — i.e. is there a **crossover**?

## 2. Design (the manipulated variable = init symmetry)
Same SSM / sine-mixture next-step task as `rdd_li.py`. Two init regimes:
- **asym** — per-channel `W_in`/`W_out` (the seed of asymmetry); channels self-separate → no collapse.
- **sym** — **shared** `W_in`/`W_out` + near-identical τ init with a tiny jitter (`θ += 0.1·N(0,1)`);
  channels are near-degenerate, so LI is the dominant symmetry-breaking force. (The jitter is required:
  a *perfectly* symmetric init is an LI-proof fixed point — calibration confirmed div=0 even with LI —
  and real systems are never perfectly symmetric.)

Crossed with **{no-LI (β=0), LI (β=10)}** and **M ∈ {3, 8}**. N=20 seeds, paired.

## 3. Metrics
Held-out **test MSE** (primary), **OOD MSE**, diversity D = mean pairwise |τ_i−τ_j|, final τ.

## 4. Pre-registered decision rule
Per regime, paired Wilcoxon + Hedges g (LI vs no-LI on test MSE; lower = better):
- **LI HELPS** iff test MSE(LI) < MSE(no-LI): p(less)<0.05 and g ≤ −0.5.
- **LI HURTS** iff p(greater)<0.05 and g ≥ +0.5. **NEUTRAL** otherwise.
- **CROSSOVER CONFIRMED** iff LI HELPS in `sym` **and** does not help (neutral/hurts) in `asym`.
- **NO CROSSOVER** iff LI does not help in `sym` (the regime built to favour it).
Report whether sym-LI separates channels (div ↑) regardless of MSE — to distinguish "LI couldn't act"
from "LI acted but didn't help."

## 5. Calibration (1 seed) & integrity note
Calibration established the design and the honest prediction below: (a) perfectly symmetric init →
div=0 for both no-LI and LI (LI-proof fixed point) → jitter added; (b) with jitter, sym-LI **does**
separate channels (div 0.5→9) but test MSE **worsens** (5.8→6.3) at M=3 and M=8, jitter 0.05 and 0.2;
(c) asym replicates `rdd_li` (LI 3.4→4.4, harmful). The grid/metrics are **frozen** here; N=20 is the
unbiased test. I have already seen LI fail to help in calibration, so this run is confirmatory, not a
search — no further regime tweaking after this point (that would be p-hacking).

## 6. Prediction (on record)
**NO CROSSOVER.** I expect LI to harm (or at best be neutral on) test MSE in **both** regimes. In `sym`
it will break the collapse (div ↑) but place channels at **task-agnostic** timescales — the repulsion
maximizes mutual distance, which is *not* coverage of the signal's relevant frequencies — so MSE will
not improve and likely worsens. Mechanism: the force that places channels usefully is the task
gradient; LI is task-blind and fights it. This would make the finding sharper than `rdd_li` alone:
LI-driven diversity does not translate to downstream gains **even when channels would otherwise
collapse**, because diversity ≠ functional coverage. I would revise toward RDD if sym-LI significantly
beats sym-no-LI on test or OOD MSE.
