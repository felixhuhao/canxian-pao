"""
DPA objective control: hard pseudo-label CE vs reward-gradient gate adaptation.

This is a self-contained NavEnv2D-style boundary test for the DPA manuscript. The original NavEnv2D source is
not present in the repo, so this harness recreates the minimal ingredients used by the paper:

  * continuous 2D navigation with manuscript target locations;
  * five frozen directional skills tied to training targets T0--T4;
  * an MLP gate trained to route compositionally from (position, target-direction);
  * T5 adaptation with all-zero sparse single-skill evaluations.

The key comparison is objective-only:
  CE-hard turns all-zero returns into label S0 and should create default-index drift.
  REINFORCE on the same all-zero reward signal should have zero or entropy-flattening drift, not S0 collapse.
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).resolve().parent

GRID = 10.0
START = np.array([5.0, 5.0], dtype=np.float32)
STEP = 0.4
RADIUS = 0.5
MAX_STEPS = 50
K = 5

TARGETS = np.array(
    [
        [1.0, 1.0],
        [9.0, 1.0],
        [1.0, 9.0],
        [9.0, 9.0],
        [3.0, 5.0],
        [7.0, 5.0],
        [5.0, 3.0],
        [5.0, 7.0],
        [2.0, 7.5],
        [7.5, 2.0],
    ],
    dtype=np.float32,
)
TRAIN_REGIMES = list(range(5))
ADAPT_REGIME = 5
EVAL_REGIMES = list(range(5, 10))

PRETRAIN_SAMPLES = 4096
PRETRAIN_EPOCHS = 700
PRETRAIN_LR = 2e-3
ADAPT_LR = 1e-3
BASELINE = 0.2
CHECKPOINTS = [0, 10, 20, 40, 60, 100, 150, 200]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def obs(pos: np.ndarray, target_idx: int) -> np.ndarray:
    direction = unit(TARGETS[target_idx] - pos)
    return np.array([pos[0] / GRID, pos[1] / GRID, direction[0], direction[1]], dtype=np.float32)


def skill_action(skill_idx: int, pos: np.ndarray) -> np.ndarray:
    return unit(TARGETS[skill_idx] - pos)


def oracle_skill(pos: np.ndarray, target_idx: int) -> int:
    desired = unit(TARGETS[target_idx] - pos)
    scores = [float(np.dot(skill_action(k, pos), desired)) for k in range(K)]
    return int(np.argmax(scores))


def rollout_single_skill(skill_idx: int, target_idx: int, start: np.ndarray, return_path: bool = False):
    pos = start.astype(np.float32).copy()
    path = [pos.copy()]
    for _ in range(MAX_STEPS):
        pos = np.clip(pos + STEP * skill_action(skill_idx, pos), 0.0, GRID).astype(np.float32)
        path.append(pos.copy())
        if np.linalg.norm(pos - TARGETS[target_idx]) <= RADIUS:
            return (1.0, path) if return_path else 1.0
    return (0.0, path) if return_path else 0.0


class Gate(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        return self.body(x)


@dataclass
class Trace:
    success: dict[int, list[float]]
    default_rate: list[float]
    p0: list[float]


def make_pretrain_data(seed: int):
    rng = np.random.RandomState(seed * 17 + 1)
    xs, ys = [], []
    for _ in range(PRETRAIN_SAMPLES):
        r = int(rng.choice(TRAIN_REGIMES))
        # Cover the whole state space, not just successful trajectories, so the gate learns direction sectors.
        pos = rng.uniform(0.0, GRID, size=2).astype(np.float32)
        xs.append(obs(pos, r))
        ys.append(oracle_skill(pos, r))
    return (
        torch.as_tensor(np.array(xs), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(ys, dtype=torch.long, device=DEVICE),
    )


def train_initial_gate(seed: int) -> Gate:
    torch.manual_seed(seed * 17 + 2)
    gate = Gate().to(DEVICE)
    x, y = make_pretrain_data(seed)
    opt = torch.optim.Adam(gate.parameters(), lr=PRETRAIN_LR)
    for _ in range(PRETRAIN_EPOCHS):
        loss = F.cross_entropy(gate(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return gate


def gated_rollout(gate: Gate, target_idx: int, start: np.ndarray = START, greedy: bool = True, rng=None):
    pos = start.astype(np.float32).copy()
    path = [pos.copy()]
    choices = []
    for _ in range(MAX_STEPS):
        x = torch.as_tensor(obs(pos, target_idx), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = gate(x)
            if greedy:
                k = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                k = int(dist.sample().item())
        choices.append(k)
        pos = np.clip(pos + STEP * skill_action(k, pos), 0.0, GRID).astype(np.float32)
        path.append(pos.copy())
        if np.linalg.norm(pos - TARGETS[target_idx]) <= RADIUS:
            return 1.0, path, choices
    return 0.0, path, choices


def collect_adapt_obs(gate: Gate, target_idx: int = ADAPT_REGIME):
    succ, path, _ = gated_rollout(gate, target_idx)
    # Use the pre-update trajectory observations. If the initial gate fails, still use its trajectory; the
    # degeneracy diagnostic will report whether sparse single-skill returns are all zero.
    xs = np.array([obs(p, target_idx) for p in path[:-1]], dtype=np.float32)
    starts = np.array(path[:-1], dtype=np.float32)
    return succ, xs, starts


def sparse_skill_rewards(starts: np.ndarray, target_idx: int = ADAPT_REGIME):
    rewards = np.zeros((len(starts), K), dtype=np.float32)
    for i, p in enumerate(starts):
        for k in range(K):
            rewards[i, k] = rollout_single_skill(k, target_idx, p)
    return rewards


def eval_gate(gate: Gate):
    return {r: gated_rollout(gate, r)[0] for r in EVAL_REGIMES}


def default_stats(gate: Gate, adapt_x: np.ndarray):
    x = torch.as_tensor(adapt_x, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p = torch.softmax(gate(x), dim=-1).cpu().numpy()
    return float(np.mean(np.argmax(p, axis=1) == 0)), float(np.mean(p[:, 0]))


def adapt_ce(gate: Gate, adapt_x: np.ndarray, steps: int):
    x = torch.as_tensor(adapt_x, dtype=torch.float32, device=DEVICE)
    y = torch.zeros(len(adapt_x), dtype=torch.long, device=DEVICE)
    opt = torch.optim.Adam(gate.parameters(), lr=ADAPT_LR)
    for _ in range(steps):
        loss = F.cross_entropy(gate(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def adapt_reinforce(gate: Gate, adapt_x: np.ndarray, rewards: np.ndarray, steps: int, baseline: float | None):
    x = torch.as_tensor(adapt_x, dtype=torch.float32, device=DEVICE)
    rtab = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(gate.parameters(), lr=ADAPT_LR)
    for _ in range(steps):
        logits = gate(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        r = rtab[torch.arange(len(adapt_x), device=DEVICE), actions]
        adv = r if baseline is None else (r - baseline)
        # Stop-gradient on rewards/advantages: this is a score-function estimator.
        loss = -(dist.log_prob(actions) * adv.detach()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def run_arm(initial: Gate, arm: str, adapt_x: np.ndarray, rewards: np.ndarray, checkpoints: list[int]) -> Trace:
    gate = copy.deepcopy(initial)
    success, default_rate, p0 = {}, [], []
    prev = 0
    for ckpt in checkpoints:
        delta = ckpt - prev
        if delta:
            if arm == "ce_hard":
                adapt_ce(gate, adapt_x, delta)
            elif arm == "reinforce_zero":
                adapt_reinforce(gate, adapt_x, rewards, delta, baseline=None)
            elif arm == "reinforce_posbase":
                adapt_reinforce(gate, adapt_x, rewards, delta, baseline=BASELINE)
            else:
                raise ValueError(arm)
        ev = eval_gate(gate)
        success[ckpt] = [ev[r] for r in EVAL_REGIMES]
        dr, prob0 = default_stats(gate, adapt_x)
        default_rate.append(dr)
        p0.append(prob0)
        prev = ckpt
    return Trace(success=success, default_rate=default_rate, p0=p0)


def run_seed(seed: int, checkpoints: list[int]):
    initial = train_initial_gate(seed)
    init_t5_success, adapt_x, starts = collect_adapt_obs(initial)
    rewards_all = sparse_skill_rewards(starts)
    allzero_mask = np.max(rewards_all, axis=1) == 0.0
    degenerate = float(np.mean(allzero_mask))
    # The objective-boundary claim concerns the all-zero evidence case. If a few trajectory states happen to
    # let a single frozen skill pass near T5, exclude them from the adaptation batch rather than mixing two
    # mechanisms.
    adapt_x = adapt_x[allzero_mask]
    rewards = rewards_all[allzero_mask]
    if len(adapt_x) == 0:
        raise RuntimeError("No all-zero sparse-evaluation observations collected; boundary test is invalid.")
    arms = {
        "ce_hard": run_arm(initial, "ce_hard", adapt_x, rewards, checkpoints),
        "reinforce_zero": run_arm(initial, "reinforce_zero", adapt_x, rewards, checkpoints),
        "reinforce_posbase": run_arm(initial, "reinforce_posbase", adapt_x, rewards, checkpoints),
    }
    return init_t5_success, degenerate, arms


def mode_config(mode: str):
    if mode == "smoke":
        return [3100], [0, 10, 20, 40]
    if mode == "calib":
        return list(range(3100, 3103)), CHECKPOINTS
    if mode == "confirm":
        return list(range(3120, 3128)), CHECKPOINTS
    return list(range(3140, 3160)), CHECKPOINTS


def summarize(all_runs, checkpoints):
    arms = ["ce_hard", "reinforce_zero", "reinforce_posbase"]
    print(f"\n  {'arm':>18s} {'step':>5s} {'succ':>7s} {'defrate':>8s} {'p0':>7s}")
    for arm in arms:
        for i, ckpt in enumerate(checkpoints):
            succ = np.array([np.mean(run[2][arm].success[ckpt]) for run in all_runs])
            dr = np.array([run[2][arm].default_rate[i] for run in all_runs])
            p0 = np.array([run[2][arm].p0[i] for run in all_runs])
            print(f"  {arm:>18s} {ckpt:5d} {succ.mean():7.2f} {dr.mean():8.2f} {p0.mean():7.2f}")

    print("\n  --- objective-boundary readout (final checkpoint) ---")
    final = checkpoints[-1]
    for arm in arms:
        succ = np.array([np.mean(run[2][arm].success[final]) for run in all_runs])
        dr = np.array([run[2][arm].default_rate[-1] for run in all_runs])
        p0 = np.array([run[2][arm].p0[-1] for run in all_runs])
        print(f"  {arm:>18s}: success={succ.mean():.3f}±{succ.std(ddof=1) if len(succ)>1 else 0:.3f} "
              f"default={dr.mean():.3f} p0={p0.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "calib", "confirm", "long"], default="smoke")
    args = ap.parse_args()
    seeds, checkpoints = mode_config(args.mode)

    print("=" * 96)
    print(f"  DPA OBJECTIVE CONTROL  mode={args.mode}  N={len(seeds)} seeds={seeds[0]}..{seeds[-1]} "
          f"device={DEVICE}")
    print(f"  arms=CE-hard vs REINFORCE-zero vs REINFORCE-positive-baseline  checkpoints={checkpoints}")
    print("=" * 96)

    all_runs = []
    for sd in seeds:
        run = run_seed(sd, checkpoints)
        all_runs.append(run)
        init_succ, deg, arms = run
        msg = [f"seed {sd}: initT5={init_succ:.0f} allzero={deg:.2f}"]
        for arm in arms:
            final_succ = np.mean(arms[arm].success[checkpoints[-1]])
            final_def = arms[arm].default_rate[-1]
            final_p0 = arms[arm].p0[-1]
            msg.append(f"{arm}=succ{final_succ:.2f}/D{final_def:.2f}/p0{final_p0:.2f}")
        print("  " + "  ".join(msg))

    init = np.array([r[0] for r in all_runs])
    deg = np.array([r[1] for r in all_runs])
    print(f"\n  diagnostics: initial T5 success={init.mean():.2f}; all-zero sparse eval rate={deg.mean():.2f}")
    summarize(all_runs, checkpoints)

    out_dir = HERE / "results"
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "seeds": np.array(seeds),
        "checkpoints": np.array(checkpoints),
        "init_t5_success": init,
        "degenerate_rate": deg,
    }
    for arm in ["ce_hard", "reinforce_zero", "reinforce_posbase"]:
        out[f"{arm}_success"] = np.array(
            [[np.mean(run[2][arm].success[c]) for c in checkpoints] for run in all_runs], dtype=np.float32
        )
        out[f"{arm}_default_rate"] = np.array([run[2][arm].default_rate for run in all_runs], dtype=np.float32)
        out[f"{arm}_p0"] = np.array([run[2][arm].p0 for run in all_runs], dtype=np.float32)
    out_path = out_dir / f"objective_control_{args.mode}.npz"
    np.savez(out_path, **out)
    print(f"\n  saved {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}")


if __name__ == "__main__":
    main()
