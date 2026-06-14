# Pre-Registration — Capacity law: does the coverage-reward benefit grow with M/K?

> **FROZEN 2026-06-14, before the N=20 run.** Characterizes the positive result from
> `PREREG_RDD_taware.md`. Held-out seeds 200–219 (disjoint from prior runs). Results → `FINDINGS.md`.

## 1. Claim under test
`rdd_taware.py` found the task-aligned **coverage reward** helps downstream at M=8 but not M=3 →
hypothesis: the benefit is gated by **excess capacity**. With K=5 signal frequencies, the coverage
benefit `Δ(M) = MSE(noLI) − MSE(cov)` should **increase with M/K** and turn positive around M ≈ K.

## 2. Design
Same SSM / sine task, **asym** init only (coverage cannot rescue a symmetric collapse — established).
Sweep `M ∈ {2,3,5,8,12,16}` (M/K = 0.4 … 3.2). Conditions per M: `noLI`, `cov β=10` (primary, the
confirmed value), `cov β=1` (small-M context). N=20 seeds (200–219). Metric: held-out test MSE.

## 3. Sanity (1 seed, why this test exists)
Δ10 = −0.083 (M=2), −0.038 (M=5), **+0.303 (M=16)** — strong monotone increase with M/K, sign change
near M≈K. Also: noLI MSE rises with M (3.31→3.65 — uncoordinated extra channels hurt), and coverage
rescues it (→3.35 at M=16).

## 4. Pre-registered decision rule
- **CAPACITY LAW CONFIRMED** iff Spearman(M/K, Δ10) > 0 with p<0.05 **and** mean Δ10 over the
  over-complete points (M ≥ K) > 0.
- Report the sign of Δ10 in the under-complete (M<K) vs over-complete (M≥K) groups (predict ≤0 vs >0).
- Per-M "helps" star at g ≤ −0.5, p<0.05 (cov10 vs noLI).

## 5. Prediction (on record)
**CONFIRMED:** Δ10 increases with M/K, ≤0 for M<K, >0 and growing for M>K. Interpretation: a coverage
reward converts otherwise-harmful redundant capacity into useful spectral coverage; with too few
channels there is no spare capacity to deploy, so it only distorts. This sharpens the positive lead
into a quantitative law: *the value of task-aligned diversity scales with excess capacity.*

## 6. Integrity note
Characterization sweep at fixed β=10 (the value confirmed in `rdd_taware`), on **held-out seeds
200–219**. Grid/rule frozen here.
