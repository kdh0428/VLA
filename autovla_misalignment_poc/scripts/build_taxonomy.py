#!/usr/bin/env python
"""
AutoVLA action taxonomy + scene-conditioned grouping (Plan A).

Why this differs from the ORION taxonomy script: AutoVLA's released RFT checkpoint emits
no chain-of-thought on this split (verified: 2748/2748 produce the identical
"straightforward scenario" stub), so there is no model-stated perception and no observable
reasoning. A per-sample P label of the ORION kind therefore cannot be constructed, and
defining P from a hidden probe would make "is perception decodable in P+ samples?"
circular by construction.

Plan A instead treats perception as a SCENE CONDITION taken from the dataset ground truth
(navsim Annotations): lead-vehicle present, pedestrian present, critical-object side and
motion. Within each condition we compare A+ against A-. That keeps the ORION operational
definitions for A, keeps P grounded in dataset GT, and avoids the circularity.

Outputs
    taxonomy.csv / taxonomy.json      per-sample table
    taxonomy_summary.json             counts, rates, Wilson CIs, per-condition breakdown
    threshold_sensitivity.json        A- rate across the gross-geometry backstop sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, POC_DIR)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def recompute_verdict(r: dict, ade_gross: float, fde_gross: float) -> bool:
    """A+/A- under a different gross-geometry backstop (semantics unchanged)."""
    v = r.get("verdict") or {}
    if v.get("lon_pred") and v.get("lon_gt") and v["lon_pred"] != v["lon_gt"]:
        return False
    if v.get("lat_pred") and v.get("lat_gt") and v["lat_pred"] != v["lat_gt"]:
        return False
    a, f = v.get("ade", float("nan")), v.get("fde", float("nan"))
    if np.isfinite(a) and a > ade_gross:
        return False
    if np.isfinite(f) and f > fde_gross:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs/taxonomy"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    R = [json.loads(l) for l in open(args.records) if l.strip()]
    print(f"loaded {len(R)} records")

    rows = []
    for r in R:
        v = r.get("verdict") or {}
        p = r.get("perception") or {}
        pred_idx = r.get("pred_action_idx") or []
        gt_idx = r.get("gt_action_idx") or []
        n = min(len(pred_idx), len(gt_idx))
        tok_match = float(np.mean([pred_idx[i] == gt_idx[i] for i in range(n)])) if n else float("nan")
        rows.append({
            "token": r["token"], "log_name": r.get("log_name"),
            "instruction": r.get("instruction"),
            "A": bool(v.get("correct")) if v else None,
            "ade": v.get("ade"), "fde": v.get("fde"),
            "reason": v.get("reason"),
            "lon_gt": v.get("lon_gt"), "lon_pred": v.get("lon_pred"),
            "lat_gt": v.get("lat_gt"), "lat_pred": v.get("lat_pred"),
            "sem_gt": v.get("sem_gt"), "sem_pred": v.get("sem_pred"),
            "lead_vehicle": p.get("lead_vehicle"), "lead_distance": p.get("lead_distance"),
            "pedestrian": p.get("pedestrian"),
            "critical_side": p.get("critical_side"), "critical_motion": p.get("critical_motion"),
            "n_objects": p.get("n_objects"), "n_vehicles": p.get("n_vehicles"),
            "cot_present": r.get("cot_present"),
            "token_match_rate": tok_match,
            "hidden_file": r.get("hidden_file"),
        })

    dec = [r for r in rows if r["A"] is not None]
    n_minus = sum(1 for r in dec if not r["A"])
    ci = wilson(n_minus, len(dec))
    ade = np.array([r["ade"] for r in dec if r["ade"] is not None and np.isfinite(r["ade"])])

    # per-condition A- rate (Plan A: perception as a scene condition, from dataset GT)
    conds = {
        "lead_vehicle=True": lambda r: r["lead_vehicle"] is True,
        "lead_vehicle=False": lambda r: r["lead_vehicle"] is False,
        "lead_close(<15m)": lambda r: r["lead_distance"] is not None and r["lead_distance"] <= 15,
        "pedestrian=True": lambda r: r["pedestrian"] is True,
        "pedestrian=False": lambda r: r["pedestrian"] is False,
        "critical_side=center": lambda r: r["critical_side"] == "center",
        "critical_side=left": lambda r: r["critical_side"] == "left",
        "critical_side=right": lambda r: r["critical_side"] == "right",
        "critical_motion=moving": lambda r: r["critical_motion"] == "moving",
        "critical_motion=static": lambda r: r["critical_motion"] == "static",
    }
    per_cond = {}
    for name, fn in conds.items():
        sub = [r for r in dec if fn(r)]
        k = sum(1 for r in sub if not r["A"])
        per_cond[name] = {"n": len(sub), "A-": k,
                          "rate": k / len(sub) if sub else float("nan"),
                          "ci95": wilson(k, len(sub))}

    per_gt_action = {}
    by = defaultdict(list)
    for r in dec:
        by[r["lon_gt"]].append(r)
    for k2, v2 in by.items():
        kk = sum(1 for r in v2 if not r["A"])
        per_gt_action[str(k2)] = {"n": len(v2), "A-": kk, "rate": kk / len(v2),
                                  "ci95": wilson(kk, len(v2))}

    per_log = {}
    byl = defaultdict(list)
    for r in dec:
        byl[r["log_name"]].append(r)
    for k2, v2 in byl.items():
        kk = sum(1 for r in v2 if not r["A"])
        per_log[str(k2)] = {"n": len(v2), "A-": kk, "rate": kk / len(v2)}

    summary = {
        "n_total": len(rows),
        "n_decidable": len(dec),
        "A_minus": {"count": n_minus, "rate": n_minus / len(dec) if dec else 0.0, "ci95": ci},
        "A_plus": len(dec) - n_minus,
        "cot_present": sum(1 for r in rows if r["cot_present"]),
        "n_logs": len({r["log_name"] for r in rows}),
        "failure_reasons": dict(Counter(r["reason"] for r in dec if not r["A"]).most_common(15)),
        "gt_action_dist": dict(Counter(r["lon_gt"] for r in dec).most_common()),
        "pred_action_dist": dict(Counter(r["lon_pred"] for r in dec).most_common()),
        "per_condition_A_minus": per_cond,
        "per_gt_action_A_minus": per_gt_action,
        "per_log_A_minus": per_log,
        "token_match_rate_mean": float(np.nanmean([r["token_match_rate"] for r in rows])),
        "ADE": {"mean": float(ade.mean()), "median": float(np.median(ade)),
                "p75": float(np.percentile(ade, 75)), "p90": float(np.percentile(ade, 90)),
                "max": float(ade.max())} if len(ade) else {},
    }

    sens = {}
    for a_thr in (2.0, 3.0, 4.0, 6.0, 1e9):
        for f_thr in (4.0, 8.0, 1e9):
            k = sum(1 for r in R if not recompute_verdict(r, a_thr, f_thr))
            key = f"ade{'inf' if a_thr > 1e8 else a_thr}_fde{'inf' if f_thr > 1e8 else f_thr}"
            sens[key] = {"A-": k, "rate": k / len(R)}
    # semantics-only (no geometry backstop at all) is the "inf/inf" entry above.

    with open(os.path.join(args.outdir, "taxonomy.json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    cols = ["token", "log_name", "A", "ade", "fde", "reason", "lon_gt", "lon_pred",
            "lat_gt", "lat_pred", "lead_vehicle", "pedestrian", "critical_side",
            "critical_motion", "cot_present", "token_match_rate"]
    with open(os.path.join(args.outdir, "taxonomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.outdir, "taxonomy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(args.outdir, "threshold_sensitivity.json"), "w") as f:
        json.dump(sens, f, indent=2)

    print(f"\nA-: {n_minus}/{len(dec)} = {n_minus/len(dec):.1%} CI95 [{ci[0]:.1%},{ci[1]:.1%}]")
    print(f"CoT present: {summary['cot_present']}/{len(rows)}")
    print(f"logs: {summary['n_logs']}   ADE median {summary['ADE'].get('median'):.3f}")
    print("\nper-condition A- rate:")
    for k2, v2 in per_cond.items():
        if v2["n"]:
            print(f"  {k2:26s} {v2['A-']:4d}/{v2['n']:5d} = {v2['rate']:6.1%} "
                  f"CI[{v2['ci95'][0]:.1%},{v2['ci95'][1]:.1%}]")
    print("\nper GT longitudinal action:")
    for k2, v2 in sorted(per_gt_action.items(), key=lambda x: -x[1]["n"]):
        print(f"  {k2:12s} {v2['A-']:4d}/{v2['n']:5d} = {v2['rate']:6.1%}")
    print("\ntop failure reasons:", list(summary["failure_reasons"].items())[:6])
    print(f"\nwrote -> {args.outdir}")


if __name__ == "__main__":
    main()
