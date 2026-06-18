"""
PAO envelope — capacity/interference sweep for modularity.

Pre-registration: ../PREREG_PAO_ENVELOPE.md.

Priority 2 found gated(Bayes) == factored(Bayes) when the shared factored net has hidden=128. This sweep asks
whether modularity re-enters when the shared clean multitask policy is capacity-limited. The Bayes gate is held
fixed, so the only representation change is K specialists vs one shared context-conditioned policy.
"""
import argparse
import os
import numpy as np
from scipy import stats

import grid as G
import r4
import trigger as T
import deconfound as D

SIGMAS = [0.6, 1.0]
HIDDENS = [8, 16, 32, 64, 128]


def _mode_config(mode):
    if mode == "smoke":
        return list(range(1060, 1061)), [8, 128], [0.6]
    if mode == "calib":
        return list(range(1060, 1063)), HIDDENS, SIGMAS
    if mode == "confirm":
        return list(range(1070, 1078)), HIDDENS, SIGMAS
    return list(range(1080, 1100)), HIDDENS, SIGMAS


def run_seed(seed, hiddens, sigmas):
    skills = [r4.train_mastered(k, seed) for k in range(G.K)]
    gated = {s: np.mean([T.roll_gated(skills, k, s, seed) for k in range(G.K)]) for s in sigmas}
    out = {}
    for h in hiddens:
        clean_mt = T.train_monolith(0.0, seed, hidden=h)
        for s in sigmas:
            mono = T.train_monolith(s, seed, hidden=h)
            out[(h, s)] = dict(
                gated_b=gated[s],
                factored=np.mean([D.roll_factored(clean_mt, k, s, seed) for k in range(G.K)]),
                monolith=np.mean([T.roll_monolith(mono, k, s, seed) for k in range(G.K)]),
            )
    return out


def hedges(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return T.hedges_g(a, b)


def wil(a, b, tail):
    try:
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return stats.wilcoxon(a, b, alternative=tail)[1]
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "calib", "confirm", "long"], default="smoke")
    args = ap.parse_args()
    seeds, hiddens, sigmas = _mode_config(args.mode)

    print("=" * 94)
    print(f"  PAO ENVELOPE  mode={args.mode}  N={len(seeds)} seeds {seeds[0]}..{seeds[-1]} "
          f"K={G.K} dev={G.DEVICE}")
    print(f"  sigmas={sigmas}  shared-hidden={hiddens}  modular skills: {G.K} specialists hidden=32")
    print("=" * 94)

    arms = ("gated_b", "factored", "monolith")
    res = {(h, s, a): [] for h in hiddens for s in sigmas for a in arms}
    for sd in seeds:
        o = run_seed(sd, hiddens, sigmas)
        for h in hiddens:
            for s in sigmas:
                for a in arms:
                    res[(h, s, a)].append(o[(h, s)][a])
        print(f"  seed {sd}: " + "  ".join(
            f"H{h}/s{s}:G{o[(h,s)]['gated_b']:.2f}/F{o[(h,s)]['factored']:.2f}/M{o[(h,s)]['monolith']:.2f}"
            for h in hiddens for s in sigmas))

    def col(h, s, a):
        return np.array(res[(h, s, a)])

    print(f"\n  {'H':>5s} {'sigma':>6s} {'gated':>9s} {'factored':>9s} {'monolith':>9s} "
          f"{'G-F':>7s} {'F-M':>7s} {'G>F g':>7s} {'F>M g':>7s}")
    for h in hiddens:
        for s in sigmas:
            Gd, Fc, Mo = col(h, s, "gated_b"), col(h, s, "factored"), col(h, s, "monolith")
            print(f"  {h:5d} {s:6.1f} {Gd.mean():9.2f} {Fc.mean():9.2f} {Mo.mean():9.2f} "
                  f"{(Gd-Fc).mean():+7.2f} {(Fc-Mo).mean():+7.2f} {hedges(Gd,Fc):+7.2f} {hedges(Fc,Mo):+7.2f}")

    print("\n  --- pre-registered tests ---")
    for s in sigmas:
        gaps = np.array([(col(h, s, "gated_b") - col(h, s, "factored")).mean() for h in hiddens])
        if len(np.unique(gaps)) < 2:
            rho, p_rho = float("nan"), float("nan")
        else:
            rho, p_rho = stats.spearmanr(hiddens, gaps)
        print(f"  sigma={s}: modularity gap vs hidden Spearman rho={rho:+.2f} p={p_rho:.4f}")
        for h in hiddens:
            Gd, Fc, Mo = col(h, s, "gated_b"), col(h, s, "factored"), col(h, s, "monolith")
            print(f"    H={h:3d}: G>F gap={(Gd-Fc).mean():+.3f} g={hedges(Gd,Fc):+.2f} p={wil(Gd,Fc,'greater'):.4f}"
                  f" | F>M gap={(Fc-Mo).mean():+.3f} g={hedges(Fc,Mo):+.2f} p={wil(Fc,Mo,'greater'):.4f}")

    os.makedirs("results", exist_ok=True)
    np.savez(f"results/envelope_{args.mode}.npz",
             sigmas=np.array(sigmas), hiddens=np.array(hiddens), seeds=np.array(seeds),
             **{f"{a}_H{h}_s{s}": col(h, s, a) for h in hiddens for s in sigmas for a in arms})
    print(f"\n  saved results/envelope_{args.mode}.npz")


if __name__ == "__main__":
    main()
