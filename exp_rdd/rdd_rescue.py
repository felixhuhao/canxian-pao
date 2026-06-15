"""
RDD rescue — can lateral inhibition be SALVAGED by making it task-aligned?
=========================================================================
Pre-registered in ../PREREG_RDD_rescue.md. Reuses rdd_li.py + rdd_taware.py.

The RDD path so far: generic τ-repulsion (rdd_li/crossover) and the direct deadlock test (rdd_deadlock)
all find LI either inert or harmful; only a task-aligned COVERAGE REWARD helped (rdd_taware). This is the
rescue attempt — and it isolates WHICH ingredient does the rescuing by holding everything else fixed:

  noLI   : baseline (no auxiliary term).
  genLI  : τ-space pairwise repulsion        Σ_{i<j} exp(−(τ_i−τ_j)²/2ℓ²)            [task-blind repulsion]
  spec   : signal-weighted spectral OVERLAP  Σ_{i<j} Σ_f Ŝ(f)·p_i(f)·p_j(f)          [task-aligned REPULSION]
  cov    : signal-weighted max-envelope      − Σ_f Ŝ(f)·max_m p_m(f)                 [task-aligned COVERAGE]

`spec` and `cov` are EQUALLY task-aligned (both in signal-weighted spectral space); the only difference
is pairwise-product (repulsion / push-apart) vs max-union (coverage / fill-the-spectrum). If cov helps
and spec does not, the rescuing ingredient is the OBJECTIVE FORM, not task-alignment — i.e. LI-qua-
repulsion cannot be salvaged; you must switch to a coverage objective.
"""
import sys, os, math, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import rdd_li as R
import rdd_taware as T

M = 8                       # over-complete regime where coverage helped
INIT = "asym"              # per-channel weights (the rdd_taware primary cell)
BETA_GEN = 10.0
SPEC_BETAS = (10.0, 50.0, 100.0)
COV_BETAS = (10.0, 50.0)


def spectral_coverage(taus, S_hat):
    """Descriptive: fraction of signal power covered by the channels' max-envelope response."""
    theta = torch.log(torch.tensor(taus, dtype=torch.float32, device=R.DEVICE))
    p = T.channel_response(theta)                       # (M,F)
    return float((S_hat * p.max(dim=0).values).sum())


def run(li_type, beta, seed):
    r = T.train_one(M, INIT == "sym", li_type, beta, seed, theta_jitter=0.0)
    # recompute a coverage diagnostic against this seed's empirical spectrum
    rng = np.random.RandomState(seed)
    S_hat = T.signal_spectrum(R.gen_data(R.N_TRAIN, R.FREQS, R.AMPS, rng)).detach()
    r["cov_metric"] = spectral_coverage(np.array(r["taus"]), S_hat)
    return r


CONDITIONS = [("noLI", "none", 0.0), ("genLI", "generic", BETA_GEN)]
CONDITIONS += [(f"spec{b:g}", "taware", b) for b in SPEC_BETAS]
CONDITIONS += [(f"cov{b:g}", "coverage", b) for b in COV_BETAS]


def main(seeds):
    data = {name: [run(lit, beta, s) for s in seeds] for name, lit, beta in CONDITIONS}
    def col(c, k): return np.array([r[k] for r in data[c]], float)

    print(f"\n{'='*84}\n  RDD RESCUE — M={M} {INIT}, N={len(seeds)} seeds {seeds[0]}..{seeds[-1]}, "
          f"dev={R.DEVICE}\n{'='*84}")
    print(f"  {'arm':<8} {'div':>7} {'cov':>6} {'test_mse':>9} {'ood_mse':>9}")
    for name, *_ in CONDITIONS:
        print(f"  {name:<8} {col(name,'div').mean():7.3f} {col(name,'cov_metric').mean():6.3f} "
              f"{col(name,'test_mse').mean():9.4f} {col(name,'ood_mse').mean():9.4f}")

    no = col("noLI", 'test_mse'); no_o = col("noLI", 'ood_mse')
    print(f"\n  vs noLI (test={no.mean():.4f}):  g<0 & p<.05 => helps; g>0 & p<.05 => hurts")
    for name, *_ in CONDITIONS[1:]:
        x = col(name, 'test_mse'); xo = col(name, 'ood_mse')
        g = R.hedges_g(x, no); ph = R.pw(x, no, "less"); pa = R.pw(x, no, "greater")
        go = R.hedges_g(xo, no_o); pho = R.pw(xo, no_o, "less")
        verdict = "HELPS" if (ph < 0.05 and g <= -0.5) else "HURTS" if (pa < 0.05 and g >= 0.5) else "null"
        print(f"    {name:<8} test {x.mean():.4f} g={g:+.2f} p(<)={ph:.3f} -> {verdict:<5} "
              f"| ood g={go:+.2f} p(<)={pho:.3f}")

    # key contrast: best spec vs best cov (both task-aligned; differ only in objective form)
    bs = min(SPEC_BETAS, key=lambda b: col(f"spec{b:g}", 'test_mse').mean())
    bc = min(COV_BETAS, key=lambda b: col(f"cov{b:g}", 'test_mse').mean())
    sp = col(f"spec{bs:g}", 'test_mse'); cv = col(f"cov{bc:g}", 'test_mse')
    g = R.hedges_g(cv, sp); p = R.pw(cv, sp, "less")
    print(f"\n  KEY: best cov (β={bc:g}, {cv.mean():.4f}) vs best spec (β={bs:g}, {sp.mean():.4f}): "
          f"g={g:+.2f} p(cov<spec)={p:.3f}")
    print("  => if cov HELPS but spec is null/HURTS, the rescuing ingredient is COVERAGE-vs-REPULSION,"
          " not task-alignment.")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results/rdd_rescue.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("\n  Saved results/rdd_rescue.pkl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    main(list(range(a.start, a.start + a.n)))
