"""Calibration for the 2D-shift harder env: convergence, return scale, trigger firing, timing."""
import sys, os, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from env_2d_shift import TwoGate2DShiftEnv, OBS_DIM, NUM_ACTIONS
from agents import FlatPPO, PAOLight, PAOBocpd

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def run_p1(make, label, seed, n=300):
    set_seed(seed)
    ag = make()
    env = TwoGate2DShiftEnv(rule="A→B", seed=seed, shifted=False)
    t0 = time.time(); rets = []
    for ep in range(n):
        o = env.reset(); d = False
        while not d:
            a = ag.act(o, True); o, r, d, i = env.step(a); ag.step_end(r, d)
        ag.finish_episode()
        rets.append(ag.get_log()["returns"][-1])
    dt = time.time() - t0
    ents = ag.get_log().get("entropies", [])
    fire = ag.get_log().get("skill_episodes", [])
    curve = " ".join(f"{np.mean(rets[i:i+30]):+.2f}" for i in range(0, n, 30))
    print(f"[{label} s{seed}] {dt:.1f}s  R(30-blocks): {curve}")
    print(f"          final R(last30)={np.mean(rets[-30:]):+.3f}  "
          f"entropy last={ents[-1]:.2f}  fire@{fire}  solved%(R>1)={(np.array(rets)>1.0).mean()*100:.0f}")
    return rets

for seed in [0, 1]:
    run_p1(lambda: FlatPPO(OBS_DIM, NUM_ACTIONS, lr=3e-4, entropy_coef=0.03), "Flat   ", seed)
    run_p1(lambda: PAOLight(OBS_DIM, NUM_ACTIONS, lr=3e-4, entropy_coef=0.03, trigger_entropy=1.0), "Heur   ", seed)
    run_p1(lambda: PAOBocpd(OBS_DIM, NUM_ACTIONS, lr=3e-4, entropy_coef=0.03, trigger_entropy=1.0), "BOCPD  ", seed)
    print()
