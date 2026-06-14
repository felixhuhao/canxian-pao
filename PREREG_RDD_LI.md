# Pre-Registration — Does RDD lateral inhibition (diversity) improve DOWNSTREAM performance?

> **FROZEN 2026-06-14, before the N=20 run.** Mechanism/metrics frozen after a 1-seed calibration
> (see §6). Results → `FINDINGS.md`. Covers **Direction 3** (LI → downstream MSE) and **Direction 1**
> (γ-inertness), both fully specified by RDD (`docs/RDD_main.pdf`, `docs/RDD_supplementary.pdf`).
> Direction 2 (Busch IM/WMP/OMP) is deferred — the Busch paper is not yet available.

## 1. Claim under test
RDD (Cai, *"Lateral inhibition enables escape from geometric deadlock"*) shows a Gaussian repulsion
loss on channel timescales raises τ-diversity (3.70 → 9.40 in its NN proof-of-concept), and states
the open question explicitly:

> *"It does not establish superiority on downstream tasks. Future work should assess prediction MSE…
> and whether the diversity increase translates to downstream performance gains."* (RDD Supp. §4)

**Direction 3 (primary):** does LI-driven diversity lower downstream prediction MSE?
**Direction 1 (folded in):** is the β_LI gain **inert** without an actual repulsion kernel (RDD Table 2:
no-kernel configs show zero response to γ_LI)?

## 2. Model & task (RDD Table S8, reproduced exactly)
`M` parallel first-order low-pass channels with learned τ_m = exp(θ_m), identical init τ=15:
`h_m(t+1) = (1−a_m)h_m(t) + a_m W_in^m u(t)`, `a_m = 1/τ_m` (clamped τ≥1.05 for stability);
`ŷ(t) = W_out·concat_m h_m(t) + b`. Task: next-step prediction of multi-frequency sine mixtures,
freqs {0.02,0.05,0.08,0.12,0.18} (0.05 has 3× amplitude), random per-sample phases, seq len 100,
hidden 8/channel, Adam lr 0.01 (0.003 for θ), 200 full-batch steps, 500 train samples.
Loss = MSE(ŷ(t), u(t+1)) + `β_LI · Σ_{i<j} exp(−(τ_i−τ_j)²/2ℓ²)`, ℓ=5.

## 3. Conditions (N=20 seeds, paired)
- `M1` (single channel), `M3_noLI` (β=0).
- `M3_LI_b{0.1,1,10,30}` — real repulsion kernel.
- `M3_inert_b{0.1,1,10,30}` — **γ-inertness control:** same β, but τ detached in the LI term →
  the gain enters the loss value but contributes **zero gradient** (kernel effectively absent).

## 4. Metrics
- **Held-out test MSE** (new random phases) — primary downstream measure.
- **OOD MSE** (held-out frequencies {0.035,0.065,0.10,0.15,0.20}).
- **Diversity D** = mean pairwise |τ_i−τ_j|.
- Train MSE; final τ.

## 5. Pre-registered decision rule
Primary LI arm = `M3_LI_b10` (RDD's reported NN-PoC setting). Paired Wilcoxon + Hedges g + bootstrap CI.
- **LI USEFUL** iff test MSE(`M3_LI_b10`) < MSE(`M3_noLI`): p(less)<0.05 **and** g ≤ −0.5.
- **LI HARMFUL** iff test MSE(`M3_LI_b10`) > MSE(`M3_noLI`): p(greater)<0.05 **and** g ≥ +0.5.
- **NULL** iff |g|<0.3, p>0.05.
- **β-sweep (secondary):** report whether any β lowers MSE vs no-LI (a low-β "sweet spot" is reported
  honestly if present, but does not change the primary verdict). Report monotonicity of D in β.
- **Diversity↔usefulness (secondary):** Spearman ρ(D, test MSE) across all M3-LI runs. ρ>0 ⇒ more
  diversity is associated with *worse* prediction.
- **D1 γ-inertness verdict:** CONFIRMED iff real-LI D varies with β (range ≫ 0) while inert-LI D is
  flat across β (range ≈ 0).

## 6. Calibration (1 seed) & integrity note
Before freezing, 1-seed calibration confirmed: (a) RDD's diversity effect reproduces (no-LI D=3.99,
LI-b10 D=10.10 ≈ paper's 3.70→9.40); (b) γ-inertness reproduces exactly (inert-b10 byte-identical to
no-LI: D=3.99, test MSE 3.43); (c) LI-b10 test MSE 4.39 **> no-LI 3.43** (diversity hurt prediction).
The model/loss/metrics are **frozen** at this point; the N=20 run is the unbiased test. The primary
comparison and decision rule were fixed by design (RDD's own open question), not chosen post-hoc.

## 7. Prediction (on record)
I expect **LI HARMFUL on test MSE** (the repulsion distorts τ away from signal-fitting values),
with the **inert control flat** (γ-inertness confirmed) — i.e. diversity does *not* translate to
downstream gains here, consistent with RDD's caveat and the project's recurring "property ≠
usefulness" pattern. Possible exception: a small benefit at the *smallest* β (mild anti-collapse
without large distortion), which I will report if it appears. I would revise toward LI if `M3_LI_b10`
significantly beats `M3_noLI` on test or OOD MSE.
