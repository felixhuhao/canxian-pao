"""
RDD capacity law — does the coverage-reward benefit GROW with excess capacity (M/K)?
====================================================================================
Pre-registered in ../PREREG_RDD_capacity.md.

rdd_taware.py found a task-aligned coverage reward helps downstream in the over-complete
(M=8) regime but not at M=3 — suggesting the benefit is gated by EXCESS CAPACITY. This
sweeps M against the number of signal frequencies K=5 and tests whether the coverage
benefit Δ(M) = MSE(noLI) − MSE(cov) increases with M/K (and changes sign near M≈K).

asym init only (coverage cannot rescue a symmetric collapse — established). Held-out seeds.
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
import rdd_taware as T
from scipy.stats import spearmanr

K = len(R.FREQS)                       # 5 signal frequencies
M_GRID = [2, 3, 5, 8, 12, 16]
COV_BETAS = (1.0, 10.0)                # primary = cov10; cov1 for small-M context


def main(seeds):
    conds = []
    for M in M_GRID:
        conds.append((f"M{M}_noLI", M, "none", 0.0))
        for b in COV_BETAS:
            conds.append((f"M{M}_cov{b:g}", M, "coverage", b))
    data = {name: [] for name, *_ in conds}
    for name, M, lit, beta in conds:
        for s in seeds:
            data[name].append(T.train_one(M, False, lit, beta, s, 0.0))   # asym, no jitter
        print(f"{name:<12} test_mse={np.mean([r['test_mse'] for r in data[name]]):.4f}")

    def col(c, k): return np.array([r[k] for r in data[c]], float)
    print(f"\n{'='*78}\n  RDD CAPACITY LAW (N={len(seeds)}, seeds {seeds[0]}..{seeds[-1]}, K={K})\n{'='*78}")
    print(f"  {'M':>3} {'M/K':>5} {'noLI':>8} {'cov10':>8} {'Δ10':>8} {'g':>7} {'p(<)':>7} {'cov1':>8} {'Δ1':>8}")
    ratios, deltas10 = [], []
    for M in M_GRID:
        no = col(f"M{M}_noLI", 'test_mse'); c10 = col(f"M{M}_cov10", 'test_mse'); c1 = col(f"M{M}_cov1", 'test_mse')
        d10 = no.mean() - c10.mean(); d1 = no.mean() - c1.mean()
        g = R.hedges_g(c10, no); p = R.pw(c10, no, "less")
        ratios.append(M / K); deltas10.append(d10)
        star = "*" if (p < 0.05 and g <= -0.5) else ""
        print(f"  {M:>3} {M/K:>5.1f} {no.mean():>8.4f} {c10.mean():>8.4f} {d10:>+8.4f} {g:>+7.2f} {p:>7.3f}{star:>2} "
              f"{c1.mean():>8.4f} {d1:>+8.4f}")

    rho, prho = spearmanr(ratios, deltas10)
    print(f"\n  TREND: Spearman(M/K, Δ10) = {rho:+.3f} (p={prho:.3f})  [positive => benefit grows with capacity]")
    # sign-change check near M≈K
    under = [d for M, d in zip(M_GRID, deltas10) if M < K]
    over = [d for M, d in zip(M_GRID, deltas10) if M >= K]
    print(f"  mean Δ10 under-complete (M<{K}) = {np.mean(under):+.4f};  over-complete (M>={K}) = {np.mean(over):+.4f}")
    verdict = ("CAPACITY LAW CONFIRMED" if (rho > 0 and prho < 0.05 and np.mean(over) > 0)
               else "NOT CONFIRMED")
    print(f"  -> {verdict}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_capacity.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/rdd_capacity.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(list(range(a.start, a.start + a.n)))
