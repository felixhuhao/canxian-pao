"""
DPA overlap robustness.

This extends the objective-control harness to test whether overlap predicts routing-collapse order for
adaptation regimes beyond T5, and whether the effect appears in the gate's hidden representation rather than
only in input-space direction vectors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import objective_control as oc


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"

ADAPT_REGIMES = {
    "T5": np.array([7.0, 5.0], dtype=np.float32),
    "T6": np.array([5.0, 3.0], dtype=np.float32),
    "T7": np.array([5.0, 7.0], dtype=np.float32),
}

PROBE_TARGETS = np.array(
    [
        [2.0, 2.0],
        [3.0, 2.0],
        [5.0, 2.0],
        [7.0, 2.0],
        [8.0, 2.0],
        [2.0, 3.5],
        [4.0, 3.5],
        [6.0, 3.5],
        [8.0, 3.5],
        [2.8, 2.8],
        [4.6, 4.6],
        [2.0, 5.0],
        [4.0, 5.0],
        [6.0, 5.0],
        [8.0, 5.0],
        [2.0, 6.5],
        [4.0, 6.5],
        [6.0, 6.5],
        [8.0, 6.5],
        [2.0, 8.0],
        [3.5, 8.0],
        [5.0, 8.0],
        [6.5, 8.0],
        [8.0, 8.0],
    ],
    dtype=np.float32,
)

CHECKPOINTS = [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300]
DEFAULT_THRESHOLD = 0.5


@dataclass
class ProbeObs:
    target_id: int
    target: np.ndarray
    x: np.ndarray
    starts: np.ndarray
    init_success: float


def unit(v: np.ndarray) -> np.ndarray:
    return oc.unit(v)


def obs_coord(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    direction = unit(target - pos)
    return np.array([pos[0] / oc.GRID, pos[1] / oc.GRID, direction[0], direction[1]], dtype=np.float32)


def rollout_single_skill_coord(skill_idx: int, target: np.ndarray, start: np.ndarray, return_path: bool = False):
    pos = start.astype(np.float32).copy()
    path = [pos.copy()]
    for _ in range(oc.MAX_STEPS):
        pos = np.clip(pos + oc.STEP * oc.skill_action(skill_idx, pos), 0.0, oc.GRID).astype(np.float32)
        path.append(pos.copy())
        if np.linalg.norm(pos - target) <= oc.RADIUS:
            return (1.0, path) if return_path else 1.0
    return (0.0, path) if return_path else 0.0


def gated_rollout_coord(gate: oc.Gate, target: np.ndarray, start: np.ndarray = oc.START):
    pos = start.astype(np.float32).copy()
    path = [pos.copy()]
    choices = []
    for _ in range(oc.MAX_STEPS):
        x = torch.as_tensor(obs_coord(pos, target), dtype=torch.float32, device=oc.DEVICE).unsqueeze(0)
        with torch.no_grad():
            k = int(torch.argmax(gate(x), dim=-1).item())
        choices.append(k)
        pos = np.clip(pos + oc.STEP * oc.skill_action(k, pos), 0.0, oc.GRID).astype(np.float32)
        path.append(pos.copy())
        if np.linalg.norm(pos - target) <= oc.RADIUS:
            return 1.0, path, choices
    return 0.0, path, choices


def collect_obs_coord(gate: oc.Gate, target: np.ndarray) -> ProbeObs:
    succ, path, _ = gated_rollout_coord(gate, target)
    xs = np.array([obs_coord(p, target) for p in path[:-1]], dtype=np.float32)
    starts = np.array(path[:-1], dtype=np.float32)
    return ProbeObs(-1, target, xs, starts, succ)


def sparse_skill_rewards_coord(starts: np.ndarray, target: np.ndarray):
    rewards = np.zeros((len(starts), oc.K), dtype=np.float32)
    for i, p in enumerate(starts):
        for k in range(oc.K):
            rewards[i, k] = rollout_single_skill_coord(k, target, p)
    return rewards


def hidden_features(gate: oc.Gate, x_np: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(x_np, dtype=torch.float32, device=oc.DEVICE)
    with torch.no_grad():
        h = gate.body[:4](x)
    return h.detach().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-8 or bn < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def default_rate(gate: oc.Gate, x_np: np.ndarray) -> tuple[float, float]:
    x = torch.as_tensor(x_np, dtype=torch.float32, device=oc.DEVICE)
    with torch.no_grad():
        p = torch.softmax(gate(x), dim=-1).cpu().numpy()
    return float(np.mean(np.argmax(p, axis=1) == 0)), float(np.mean(p[:, 0]))


def mode_config(mode: str):
    if mode == "smoke":
        return [4100], ["T5"], CHECKPOINTS[:5]
    if mode == "calib":
        return list(range(4100, 4103)), ["T5", "T6", "T7"], CHECKPOINTS
    if mode == "confirm":
        return list(range(4120, 4125)), ["T5", "T6", "T7"], CHECKPOINTS
    return list(range(4140, 4150)), ["T5", "T6", "T7"], CHECKPOINTS


def summarize_rows(rows: list[dict], checkpoints: list[int]) -> list[dict]:
    final_ckpt = checkpoints[-1]
    out = []
    for regime in sorted({r["adapt_regime"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows if r["adapt_regime"] == regime}):
            subset = [r for r in rows if r["adapt_regime"] == regime and r["seed"] == seed]
            for metric in ["input_overlap", "hidden_overlap"]:
                x = np.array([r[metric] for r in subset], dtype=float)
                y = np.array([r["routing_collapse_step"] for r in subset], dtype=float)
                keep = np.isfinite(x) & np.isfinite(y)
                rho, p = stats.spearmanr(x[keep], y[keep]) if keep.sum() >= 4 else (np.nan, np.nan)
                order = np.argsort(y[keep])
                keep_idx = np.arange(keep.sum())
                if keep.sum() > 4:
                    trimmed = keep_idx[order[:-2]]
                    rho_trim, p_trim = stats.spearmanr(x[keep][trimmed], y[keep][trimmed])
                else:
                    rho_trim, p_trim = np.nan, np.nan
                out.append(
                    {
                        "adapt_regime": regime,
                        "seed": seed,
                        "metric": metric,
                        "rho": float(rho),
                        "p": float(p),
                        "rho_leave2": float(rho_trim),
                        "p_leave2": float(p_trim),
                        "n": int(keep.sum()),
                        "noncollapsed": int(
                            sum(
                                np.isfinite(r["routing_collapse_step"])
                                and r["routing_collapse_step"] > final_ckpt
                                for r in subset
                            )
                        ),
                    }
                )
    return out


def aggregate_summary(rows: list[dict], checkpoints: list[int]) -> list[dict]:
    # Collapse over seeds by taking the median collapse step per adaptation regime and probe target.
    final_ckpt = checkpoints[-1]
    out = []
    for regime in sorted({r["adapt_regime"] for r in rows}):
        subset = [r for r in rows if r["adapt_regime"] == regime]
        probe_ids = sorted({r["probe_id"] for r in subset})
        agg = []
        for pid in probe_ids:
            pr = [r for r in subset if r["probe_id"] == pid]
            steps = np.array([r["routing_collapse_step"] for r in pr], dtype=float)
            finite_steps = steps[np.isfinite(steps)]
            step = float(np.median(finite_steps)) if len(finite_steps) else float("nan")
            agg.append(
                {
                    "probe_id": pid,
                    "input_overlap": float(np.mean([r["input_overlap"] for r in pr])),
                    "hidden_overlap": float(np.mean([r["hidden_overlap"] for r in pr])),
                    "routing_collapse_step": step,
                }
            )
        for metric in ["input_overlap", "hidden_overlap"]:
            x = np.array([a[metric] for a in agg], dtype=float)
            y = np.array([a["routing_collapse_step"] for a in agg], dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            rho, p = stats.spearmanr(x[keep], y[keep]) if keep.sum() >= 4 else (np.nan, np.nan)
            order = np.argsort(y[keep])
            if keep.sum() > 4:
                trimmed = np.arange(keep.sum())[order[:-2]]
                rho_trim, p_trim = stats.spearmanr(x[keep][trimmed], y[keep][trimmed])
            else:
                rho_trim, p_trim = np.nan, np.nan
            out.append(
                {
                    "adapt_regime": regime,
                    "metric": metric,
                    "rho": float(rho),
                    "p": float(p),
                    "rho_leave2": float(rho_trim),
                    "p_leave2": float(p_trim),
                    "n": int(keep.sum()),
                    "noncollapsed": int(
                        sum(
                            np.isfinite(a["routing_collapse_step"])
                            and a["routing_collapse_step"] > final_ckpt
                            for a in agg
                        )
                    ),
                }
            )
    return out


def run_one(seed: int, adapt_name: str, checkpoints: list[int]) -> list[dict]:
    initial = oc.train_initial_gate(seed)
    adapt_target = ADAPT_REGIMES[adapt_name]
    adapt = collect_obs_coord(initial, adapt_target)
    rewards_all = sparse_skill_rewards_coord(adapt.starts, adapt_target)
    allzero = np.max(rewards_all, axis=1) == 0.0
    adapt_x = adapt.x[allzero]
    if len(adapt_x) == 0:
        raise RuntimeError(f"{adapt_name} seed {seed}: no all-zero adaptation observations")

    probe_obs = []
    for i, target in enumerate(PROBE_TARGETS):
        p = collect_obs_coord(initial, target)
        p.target_id = i
        probe_obs.append(p)

    adapt_dir = adapt_x[:, 2:4].mean(axis=0)
    adapt_h = hidden_features(initial, adapt_x).mean(axis=0)
    probe_meta = {}
    for p in probe_obs:
        probe_dir = p.x[:, 2:4].mean(axis=0)
        probe_h = hidden_features(initial, p.x).mean(axis=0)
        probe_meta[p.target_id] = {
            "input_overlap": cosine(adapt_dir, probe_dir),
            "hidden_overlap": cosine(adapt_h, probe_h),
            "init_success": p.init_success,
        }

    gate = copy.deepcopy(initial)
    prev = 0
    first_default = {p.target_id: None for p in probe_obs}
    first_behavior = {p.target_id: None for p in probe_obs}
    initial_default = {}
    initial_p0 = {}
    at_risk = {}
    final_stats = {}
    for ckpt in checkpoints:
        delta = ckpt - prev
        if delta:
            oc.adapt_ce(gate, adapt_x, delta)
        for p in probe_obs:
            dr, p0 = default_rate(gate, p.x)
            succ, _, _ = gated_rollout_coord(gate, p.target)
            if ckpt == 0:
                initial_default[p.target_id] = dr
                initial_p0[p.target_id] = p0
                at_risk[p.target_id] = dr < DEFAULT_THRESHOLD
            if at_risk.get(p.target_id, True) and first_default[p.target_id] is None and dr >= DEFAULT_THRESHOLD:
                first_default[p.target_id] = ckpt
            if first_behavior[p.target_id] is None and p.init_success >= 1.0 and succ <= 0.0:
                first_behavior[p.target_id] = ckpt
            if ckpt == checkpoints[-1]:
                final_stats[p.target_id] = (dr, p0, succ)
        prev = ckpt

    censor_step = checkpoints[-1] + 1
    rows = []
    for p in probe_obs:
        fd = first_default[p.target_id] if first_default[p.target_id] is not None else censor_step
        if not at_risk[p.target_id]:
            fd = np.nan
        fb = first_behavior[p.target_id] if first_behavior[p.target_id] is not None else censor_step
        dr, p0, succ = final_stats[p.target_id]
        rows.append(
            {
                "seed": seed,
                "adapt_regime": adapt_name,
                "adapt_x": float(adapt_target[0]),
                "adapt_y": float(adapt_target[1]),
                "adapt_allzero_rate": float(np.mean(allzero)),
                "probe_id": p.target_id,
                "probe_x": float(p.target[0]),
                "probe_y": float(p.target[1]),
                "input_overlap": probe_meta[p.target_id]["input_overlap"],
                "hidden_overlap": probe_meta[p.target_id]["hidden_overlap"],
                "initial_default_rate": initial_default[p.target_id],
                "initial_p0": initial_p0[p.target_id],
                "routing_at_risk": int(at_risk[p.target_id]),
                "init_success": probe_meta[p.target_id]["init_success"],
                "routing_collapse_step": fd,
                "behavior_collapse_step": int(fb),
                "final_default_rate": dr,
                "final_p0": p0,
                "final_success": succ,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "calib", "confirm", "long"], default="smoke")
    args = ap.parse_args()
    seeds, regimes, checkpoints = mode_config(args.mode)

    print("=" * 100)
    print(f"  DPA OVERLAP ROBUSTNESS  mode={args.mode} device={oc.DEVICE}")
    print(f"  seeds={seeds} regimes={regimes} checkpoints={checkpoints}")
    print("=" * 100)

    rows = []
    for seed in seeds:
        for regime in regimes:
            rr = run_one(seed, regime, checkpoints)
            rows.extend(rr)
            sub = [r for r in rr]
            print(
                f"  seed {seed} {regime}: allzero={np.mean([r['adapt_allzero_rate'] for r in sub]):.2f} "
                f"at-risk={sum(np.isfinite(r['routing_collapse_step']) for r in sub)}/{len(sub)} "
                f"collapsed={sum(np.isfinite(r['routing_collapse_step']) and r['routing_collapse_step'] <= checkpoints[-1] for r in sub)}/{len(sub)}"
            )

    by_seed = summarize_rows(rows, checkpoints)
    by_regime = aggregate_summary(rows, checkpoints)

    print("\n  --- aggregate by adaptation regime, median over seeds ---")
    for s in by_regime:
        print(
            f"  {s['adapt_regime']:>2s} {s['metric']:>14s}: "
            f"rho={s['rho']:+.3f} p={s['p']:.4f} | leave2 rho={s['rho_leave2']:+.3f} "
            f"p={s['p_leave2']:.4f} noncollapsed={s['noncollapsed']}/{s['n']}"
        )

    os.makedirs(RESULT_DIR, exist_ok=True)
    stem = RESULT_DIR / f"overlap_robustness_{args.mode}"
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(stem.with_suffix(".json"), "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "seeds": seeds,
                "regimes": regimes,
                "checkpoints": checkpoints,
                "by_seed": by_seed,
                "by_regime": by_regime,
            },
            f,
            indent=2,
        )
    print(f"\n  saved {stem.with_suffix('.csv')} and {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
