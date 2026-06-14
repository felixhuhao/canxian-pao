# exp_rdd — Repulsion-Driven Differentiation (RDD), lateral inhibition

Independent test of the **RDD** paper (Hengjin Cai, *"Lateral inhibition enables escape from
geometric deadlock in multi-timescale adaptation"*, `docs/RDD_main.pdf` + supplementary).

This reproduces RDD's **neural-network proof-of-concept** (a multi-channel first-order low-pass
SSM with learned timescales `τ_m = exp(θ_m)` and a Gaussian repulsion loss on `τ`, Table S8) and
**extends it to the question the paper explicitly left open**: does the LI-induced channel diversity
translate into *downstream* performance? It also runs the **γ-inertness** test (the `β_LI` gain
should do nothing without the actual repulsion kernel — RDD Table 2).

## Files
- `rdd_li.py` — model, sine-mixture next-step-prediction task, all conditions, stats. Run:
  ```bash
  python rdd_li.py --seeds 20      # writes results/rdd_li.{log,pkl}
  ```
- `results/` — `.log` + `.pkl` evidence.

Pre-registration: `../PREREG_RDD_LI.md` (frozen before the N=20 run). Verdict: `../FINDINGS.md`
(2026-06-14 entry).

## Result (N=20)
- **γ-inertness — CONFIRMED.** The inert arm (β present, kernel gradient detached) is identical to
  no-LI at every β (diversity range 0.000); real-LI responds to β. RDD's mechanism claim — the
  repulsion *interaction*, not the scalar gain, is operative — replicates in a trained net.
- **Diversity → downstream gains — NO.** LI raises τ-diversity (3.8 → ~10, matching the paper) but
  **monotonically worsens** test MSE (3.40 → 4.41) and OOD MSE; diversity is anti-correlated with
  usefulness (Spearman +0.47). LI is a coverage / anti-collapse tool, not a performance regularizer.

Direction 2 (Busch IM/WMP/OMP manifold-alignment) is deferred pending the Busch preprint.
