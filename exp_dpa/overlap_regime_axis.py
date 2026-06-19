"""DPA overlap, regime-axis characterization.

Samples the adaptation-regime axis densely (16 regimes) to test whether overlap predicts routing-collapse
order as a *conditional* law -- holding when the adaptation regime does not saturate collapse and has high
label--routing conflict -- rather than a single-trajectory anecdote.

Pre-registration: ../PREREG_DPA_OVERLAP_REGIME_AXIS.md.
Reuses the self-contained NavEnv2D-style environment and coord helpers from objective_control / overlap_robustness.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import objective_control as oc
import overlap_robustness as orb


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"

# Candidate adaptation regimes (not the skill targets T0--T4). Calibration showed most interior points are
# reachable by a single frozen skill (non-degenerate, no all-zero evidence), so we supply a generous candidate
# set concentrated in the degenerate band (the center column and right-center) and let the harness filter to the
# analyzable (degenerate) subset. Deviation from the original 16-regime prereg list is documented in
# PREREG_DPA_OVERLAP_REGIME_AXIS.md (calibration note).
_CANDIDATES = [
    (5.0, 2.0), (5.0, 2.5), (5.0, 3.0), (5.0, 3.5), (5.0, 4.0),
    (5.0, 6.0), (5.0, 6.5), (5.0, 7.0), (5.0, 7.5), (5.0, 8.0),
    (5.5, 3.0), (5.5, 7.0), (6.0, 4.0), (6.0, 5.0), (6.0, 6.0),
    (6.5, 4.0), (6.5, 5.0), (6.5, 6.0), (7.0, 4.0), (7.0, 5.0),
    (7.0, 6.0), (7.5, 4.0), (7.5, 5.0), (7.5, 6.0), (8.0, 4.0),
    (8.0, 5.0), (8.0, 6.0), (4.5, 5.0), (4.0, 5.0), (3.5, 5.0),
    (6.0, 3.5), (6.0, 6.5), (7.0, 4.5), (7.0, 5.5), (8.5, 5.0), (4.0, 4.5),
]
ADAPT_REGIMES = {f"A{i+1}": c for i, c in enumerate(_CANDIDATES)}

# 7x7 interior probe grid.
_PG = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
PROBE_TARGETS = np.array([[x, y] for x in _PG for y in _PG], dtype=np.float32)

CHECKPOINTS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150, 200, 300]

DEFAULT_THRESHOLD = 0.5
SATURATION_THRESHOLD = 0.90
MIN_ALLZERO = 5
MIN_ATRISK = 6
RHO_LAW = -0.5
LAW_FRACTION = 0.60


def precompute_seed(seed: int):
    """Pre-adaptation gate + probe observations depend only on the seed; compute once and reuse."""
    gate = oc.train_initial_gate(seed)
    probes = []
    for i, t in enumerate(PROBE_TARGETS):
        p = orb.collect_obs_coord(gate, np.asarray(t, dtype=np.float32))
        p.target_id = i
        probes.append(p)
    probe_dir = {p.target_id: p.x[:, 2:4].mean(axis=0) for p in probes}
    probe_h = {p.target_id: orb.hidden_features(gate, p.x).mean(axis=0) for p in probes}
    return gate, probes, probe_dir, probe_h


def run_regime(gate, probes, probe_dir, probe_h, name, target, checkpoints):
    target = np.asarray(target, dtype=np.float32)
    adapt = orb.collect_obs_coord(gate, target)
    rewards_all = orb.sparse_skill_rewards_coord(adapt.starts, target)
    allzero = np.max(rewards_all, axis=1) == 0.0
    adapt_x = adapt.x[allzero]
    if len(adapt_x) < MIN_ALLZERO:
        return {"name": name, "analyzable": False, "reason": "insufficient all-zero obs",
                "allzero_rate": float(np.mean(allzero)), "rows": []}

    adapt_dir = adapt_x[:, 2:4].mean(axis=0)
    adapt_h = orb.hidden_features(gate, adapt_x).mean(axis=0)
    # conflict: fraction of all-zero adaptation obs whose pre-adaptation argmax skill is not S0.
    with torch.no_grad():
        pre = gate(torch.as_tensor(adapt_x, dtype=torch.float32, device=oc.DEVICE)).argmax(-1).cpu().numpy()
    conflict = float(np.mean(pre != 0))

    overlaps = {p.target_id: (orb.cosine(adapt_dir, probe_dir[p.target_id]),
                              orb.cosine(adapt_h, probe_h[p.target_id])) for p in probes}

    g = copy.deepcopy(gate)
    prev = 0
    first_default = {p.target_id: None for p in probes}
    at_risk = {}
    for ckpt in checkpoints:
        delta = ckpt - prev
        if delta:
            oc.adapt_ce(g, adapt_x, delta)
        for p in probes:
            dr, _ = orb.default_rate(g, p.x)
            if ckpt == 0:
                at_risk[p.target_id] = dr < DEFAULT_THRESHOLD
            if at_risk.get(p.target_id, True) and first_default[p.target_id] is None and dr >= DEFAULT_THRESHOLD:
                first_default[p.target_id] = ckpt
        prev = ckpt

    censor = checkpoints[-1] + 1
    rows = []
    for p in probes:
        if not at_risk[p.target_id]:
            continue
        fd = first_default[p.target_id] if first_default[p.target_id] is not None else censor
        io, ho = overlaps[p.target_id]
        if not (np.isfinite(io) and np.isfinite(ho)):
            continue
        rows.append({"probe_id": p.target_id, "input_overlap": io, "hidden_overlap": ho,
                     "collapse_step": fd, "collapsed": int(fd <= checkpoints[-1])})

    n_atrisk = len(rows)
    analyzable = n_atrisk >= MIN_ATRISK
    saturation = float(np.mean([r["collapsed"] for r in rows])) if rows else float("nan")
    return {"name": name, "analyzable": analyzable, "allzero_rate": float(np.mean(allzero)),
            "conflict": conflict, "saturation": saturation, "n_atrisk": n_atrisk, "rows": rows}


def regime_rho(rows, metric):
    x = np.array([r[metric] for r in rows], dtype=float)
    y = np.array([r["collapse_step"] for r in rows], dtype=float)
    if len(x) < MIN_ATRISK or np.unique(y).size < 2 or np.unique(x).size < 2:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def mode_config(mode: str):
    names = list(ADAPT_REGIMES)
    if mode == "smoke":
        return [5100], names[:3], CHECKPOINTS[:8]
    if mode == "calib":
        return list(range(5100, 5103)), names, CHECKPOINTS
    if mode == "confirm":
        return list(range(5120, 5132)), names, CHECKPOINTS
    return list(range(5140, 5164)), names, CHECKPOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "calib", "confirm", "long"], default="smoke")
    args = ap.parse_args()
    seeds, names, checkpoints = mode_config(args.mode)

    print("=" * 100)
    print(f"  DPA OVERLAP REGIME-AXIS  mode={args.mode}  N_seeds={len(seeds)} N_regimes={len(names)} "
          f"N_probes={len(PROBE_TARGETS)} device={oc.DEVICE}")
    print(f"  checkpoints={checkpoints}")
    print("=" * 100)

    # raw[(seed, regime)] -> result dict; per_sr rho stored too
    raw_rows = []                                   # flat rows for CSV
    per_regime = {n: {"rho_in": [], "rho_hi": [], "sat": [], "conf": [], "natrisk": [],
                      "allzero": [], "analyzable": []} for n in names}

    for seed in seeds:
        gate, probes, probe_dir, probe_h = precompute_seed(seed)
        for n in names:
            res = run_regime(gate, probes, probe_dir, probe_h, n, ADAPT_REGIMES[n], checkpoints)
            per_regime[n]["allzero"].append(res.get("allzero_rate", float("nan")))
            if not res["analyzable"]:
                per_regime[n]["analyzable"].append(False)
                continue
            rin, _ = regime_rho(res["rows"], "input_overlap")
            rhi, _ = regime_rho(res["rows"], "hidden_overlap")
            per_regime[n]["rho_in"].append(rin)
            per_regime[n]["rho_hi"].append(rhi)
            per_regime[n]["sat"].append(res["saturation"])
            per_regime[n]["conf"].append(res["conflict"])
            per_regime[n]["natrisk"].append(res["n_atrisk"])
            per_regime[n]["analyzable"].append(True)
            for r in res["rows"]:
                raw_rows.append({"seed": seed, "adapt_regime": n, **r})
        print(f"  seed {seed} done")

    # ---- seed-aggregate per regime ----
    summary = []
    for n in names:
        d = per_regime[n]
        analyzable = bool(d["analyzable"]) and any(d["analyzable"]) and len(d["rho_in"]) > 0
        if not analyzable:
            summary.append({"regime": n, "analyzable": False,
                            "allzero": float(np.nanmean(d["allzero"])) if d["allzero"] else float("nan")})
            continue
        summary.append({
            "regime": n, "analyzable": True,
            "rho_in": float(np.nanmean(d["rho_in"])), "rho_hi": float(np.nanmean(d["rho_hi"])),
            "saturation": float(np.nanmean(d["sat"])), "conflict": float(np.nanmean(d["conf"])),
            "n_atrisk": float(np.nanmean(d["natrisk"])), "allzero": float(np.nanmean(d["allzero"])),
            "n_seeds": len(d["rho_in"]),
        })

    ana = [s for s in summary if s.get("analyzable")]
    nonsat = [s for s in ana if s["saturation"] < SATURATION_THRESHOLD]

    print("\n  --- per adaptation regime (seed-averaged) ---")
    print(f"  {'reg':>4s} {'rho_in':>7s} {'rho_hi':>7s} {'sat':>5s} {'conf':>5s} {'natrisk':>7s} {'allzero':>7s} flag")
    for s in summary:
        if not s.get("analyzable"):
            print(f"  {s['regime']:>4s} {'--':>7s} {'--':>7s} {'--':>5s} {'--':>5s} {'--':>7s} "
                  f"{s.get('allzero', float('nan')):7.2f} NOT-ANALYZABLE")
            continue
        flag = "SAT" if s["saturation"] >= SATURATION_THRESHOLD else ""
        print(f"  {s['regime']:>4s} {s['rho_in']:+7.3f} {s['rho_hi']:+7.3f} {s['saturation']:5.2f} "
              f"{s['conflict']:5.2f} {s['n_atrisk']:7.1f} {s['allzero']:7.2f} {flag}")

    # ---- pre-registered decision rule ----
    print("\n  --- pre-registered decision rule ---")
    law = {"n_analyzable": len(ana), "n_nonsaturating": len(nonsat)}
    if nonsat:
        pass_in = [s for s in nonsat if s["rho_in"] < RHO_LAW]
        pass_both = [s for s in nonsat if s["rho_in"] < RHO_LAW and s["rho_hi"] < RHO_LAW]
        frac_both = len(pass_both) / len(nonsat)
        law.update({"frac_rho_lt_thresh_both": frac_both,
                    "law_holds": bool(frac_both >= LAW_FRACTION)})
        print(f"  non-saturating analyzable regimes: {len(nonsat)}")
        print(f"  fraction with rho<{RHO_LAW} in BOTH input & hidden: {frac_both:.2f} "
              f"(threshold {LAW_FRACTION})")
        print(f"  PRIMARY LAW {'HOLDS' if law['law_holds'] else 'FAILS'}")
    else:
        law.update({"frac_rho_lt_thresh_both": float("nan"), "law_holds": False})
        print("  no non-saturating analyzable regimes; primary law cannot be evaluated")

    # ---- boundary characterization ----
    print("\n  --- boundary characterization (across analyzable regimes) ---")
    boundary = {}
    if len(ana) >= 4:
        rin = np.array([s["rho_in"] for s in ana])
        sat = np.array([s["saturation"] for s in ana])
        conf = np.array([s["conflict"] for s in ana])
        for label, v in [("saturation", sat), ("conflict", conf)]:
            if np.unique(v).size >= 2 and np.unique(rin).size >= 2:
                rho, p = stats.spearmanr(rin, v)
            else:
                rho, p = np.nan, np.nan
            boundary[label] = {"rho": float(rho), "p": float(p)}
            print(f"  Spearman(rho_in, {label}) = {rho:+.3f} (p={p:.4f})  "
                  f"[predict {'>0' if label == 'saturation' else '<0'}]")
    else:
        print("  too few analyzable regimes for boundary characterization")

    # ---- save ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    stem = RESULT_DIR / f"overlap_regime_axis_{args.mode}"
    if raw_rows:
        with open(stem.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader()
            w.writerows(raw_rows)
    with open(stem.with_suffix(".json"), "w") as f:
        json.dump({"mode": args.mode, "seeds": seeds, "regimes": names, "checkpoints": checkpoints,
                   "thresholds": {"SATURATION": SATURATION_THRESHOLD, "RHO_LAW": RHO_LAW,
                                  "LAW_FRACTION": LAW_FRACTION, "MIN_ATRISK": MIN_ATRISK,
                                  "MIN_ALLZERO": MIN_ALLZERO},
                   "summary": summary, "decision_rule": law, "boundary": boundary}, f, indent=2)
    print(f"\n  saved {stem.with_suffix('.json')}")


if __name__ == "__main__":
    main()
