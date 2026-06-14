"""Calibration for P3 (NOT the experiment): check BOCPD firing rate, library growth,
and post-swap recovery on a few seeds before freezing PREREG_P3."""
import numpy as np
import p3_unpredictable as P
from agents import PAOLibraryBOCPD, PAONoSkill


def seg_means(rets, boundaries, schedule):
    out = []
    for b in boundaries:
        s = b["ep"]; L = schedule[b["seg"]][1]
        w = rets[s:s + L]
        early = np.mean(w[:P.ADAPT_K]) if w else float("nan")
        late = np.mean(w[-15:]) if w else float("nan")
        out.append((b["seg"], b["regime"], b["recur"], s, L, early, late))
    return out


for seed in [0, 1, 2]:
    sched = P.make_schedule(seed)
    print(f"\n===== seed {seed}  schedule={[(int(f),L) for f,L in sched]}  total={sum(L for _,L in sched)} =====")
    for arm, ctor in [("BOCPD", lambda: PAOLibraryBOCPD(5, 2, cryst_debounce=P.DEBOUNCE, bocpd_hazard=P.HAZARD, **P.KW)),
                      ("NoSkill", lambda: PAONoSkill(5, 2, **P.KW))]:
        P.set_seed(seed)
        ag = ctor()
        rets, bnd = P.run_protocol(ag, seed, sched)
        print(f"\n  --- {arm} ---  resets={getattr(ag,'reset_count',0)} lib={len(getattr(ag,'library',[]))} "
              f"cryst_eps={getattr(ag,'skill_formation_eps',[])}")
        print(f"    adapt={P.adaptation_speed(rets,bnd):+.3f}  recov={P.recovery_time(rets,bnd,sched):5.1f}  "
              f"asymp={P.asymptotic(rets,bnd,sched):+.3f}  total={np.sum(rets):+.1f}")
        for seg, regime, recur, s, L, early, late in seg_means(rets, bnd, sched):
            tag = "RECUR" if recur else "first"
            print(f"    seg{seg} flip={int(regime)} {tag} ep[{s:3d}..{s+L:3d})  early{P.ADAPT_K}={early:+.3f}  late15={late:+.3f}")
