# Pre-Registration — Does manifold health H_M discriminate useful from useless skills?

> **FROZEN 2026-06-09, before running.** Controlled replica of the foundation test (the $H_{\mathcal{M}}$
> geometric criterion shared by PAO, Qualia, and the dialectics paper). Results → `FINDINGS.md`.

## 1. Claim under test
The framework's central geometric claim: a *good/reusable* skill has a measurable manifold signature —
low intrinsic dimension $d_{\text{int}}$ + high information volume $V_{\text{info}}$ — captured by
$H_{\mathcal{M}} = \frac{\log(1+D_{\text{world}})\,\log(1+V_{\text{info}})}{\log(1+d_{\text{int}}+\epsilon)}$
(PAO Eq. 2). If true, $H_{\mathcal{M}}$ should predict actual skill usefulness $Q(z)$.

## 2. Design (minimal env, full control)
- Crystallise skills via `PAOForced` at a sweep of episodes → a known spread of skill quality
  (we showed $Q(z)$ rises with crystallisation episode). Episodes **{3,6,10,15,22,32,45,65}** × seeds
  **{0..9}** = **80 skills**. `app_thresh=0.3` (gate live), P1=80, seeded.
- For each crystallised skill measure:
  - **$Q(z)$** = skill-only success rate (N_val=30) — ground-truth usefulness.
  - **Manifold** over the skill's internal representation: collect the 64-dim policy-torso activations
    across 20 skill-only rollouts → point cloud; $d_{\text{int}}$ = participation ratio of covariance
    eigenvalues; $V_{\text{info}}$ = total variance (trace); $D_{\text{world}}=64$; then $H_{\mathcal{M}}$.

## 3. Outcomes & analysis
- **Primary:** Spearman $\rho(H_{\mathcal{M}}, Q)$ across the 80 skills.
- **Decisive:** *partial* Spearman $\rho(H_{\mathcal{M}}, Q \mid \text{episode})$ — does $H_{\mathcal{M}}$
  predict quality **beyond** "how long you trained"? (If $H_{\mathcal{M}}$ only tracks $Q$ through
  training-amount, it adds nothing — the same failure mode as the trigger.)
- Also report $\rho(d_{\text{int}}, Q)$ and $\rho(V_{\text{info}}, Q)$ separately.

## 4. Pre-registered decision rule
- **$H_{\mathcal{M}}$ VALID** iff partial $\rho(H_{\mathcal{M}}, Q \mid \text{episode}) > 0$ with p<0.05
  **and** |ρ| ≥ 0.3 (it carries quality signal beyond a clock).
- **$H_{\mathcal{M}}$ INVALID (just a clock)** iff raw $\rho(H_{\mathcal{M}},Q)$ may be nonzero but the
  **partial** $\rho \approx 0$ (|ρ|<0.15, p>0.05) — $H_{\mathcal{M}}$ adds nothing beyond training-amount.
- **$H_{\mathcal{M}}$ USELESS** iff even raw $\rho(H_{\mathcal{M}},Q)$ is small/ns or wrong-signed.

## 5. Prediction (on record)
I expect $H_{\mathcal{M}}$ to be **INVALID as a quality gate** — at most a clock proxy (partial ρ ≈ 0),
mirroring the trigger result. The strongest sign would be: within a fixed crystallisation episode, seeds
with different $Q$ are *not* separated by $H_{\mathcal{M}}$. I would revise upward if the partial
correlation is real.

## 6. Caveats
Minimal env (not RSSM latents); manifold computed over policy-torso activations as the internal-rep
analog; $d_{\text{int}}/V_{\text{info}}$ are noisy at ~20 rollouts (a known fragility — itself part of
what's being tested). This is the controlled replica; the ecological version needs Dr. Cai's run6
checkpoints.

### Amendment (2026-06-09, post-run, interpretive — frozen rule NOT changed)
The rule did not pre-specify *significant reverse effect* (same gap as PREREG_P1). Observed: partial
ρ(H_M,Q|episode) = **−0.319, p=0.004** — significant and NEGATIVE → the automated check printed
"AMBIGUOUS." On record: this is **stronger than INVALID** — H_M is *anti*-predictive of usefulness (as a
gate it would select worse skills). Future pre-regs should add a "significant reverse ⇒ anti-valid" branch.
