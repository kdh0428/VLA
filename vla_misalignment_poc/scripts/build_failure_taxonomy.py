#!/usr/bin/env python
"""
Build the P/R/A failure taxonomy from the merged inference records.

P (perception)  model's critical-object / traffic-light answers vs the Chat-B2D GT ones,
                compared variable-by-variable with the SAME parser on both sides.
R (reasoning)   model's driving-behaviour explanation vs the GT one, compared as
                normalised intent (semantic), not string match.
A (action)      predicted vs GT trajectory: ADE/FDE plus coarse action semantics plus a
                safety-conditioned check (see analysis/trajectory_metrics.py).

Any variable that cannot be decided is Unknown, and a sample with an Unknown P or R is
reported separately rather than being forced into a +/- bucket.

Outputs
    taxonomy.csv / taxonomy.json     per-sample table
    taxonomy_summary.json            group counts, proportions, Wilson CIs, per-scenario
    threshold_sensitivity.json       group counts across an ADE/FDE threshold sweep
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

from analysis.trajectory_metrics import (ADE_THRESHOLD_M, FDE_THRESHOLD_M,  # noqa: E402
                                         action_correct, action_semantics)


# --------------------------------------------------------------------------------------
# Perception comparison
# --------------------------------------------------------------------------------------

def compare_perception(gt: dict, pred: dict, model_rounds: dict) -> tuple[bool | None, dict]:
    """
    Per-variable agreement. Returns (P, detail).

    P is True only if every *decidable* variable agrees, False if any decidable variable
    disagrees, and None (Unknown) when the model produced no usable critical-object answer
    at all -- in that case we cannot say the perception was right or wrong.
    """
    detail: dict[str, object] = {}

    # If the model emitted no critical-object text, perception is undecidable.
    co = (model_rounds.get("critical_objects") or "").strip()
    if not co:
        return None, {"reason": "no_critical_object_answer"}

    agree: list[bool] = []

    # 1. traffic light.
    #    Compared via `traffic_light_obj` -- derived from the critical-object list on BOTH
    #    sides. ORION's CoT config has no dedicated traffic-light round, so the model can
    #    only express a light through its object list; comparing that against GT's
    #    dedicated round would be an asymmetric test the model cannot pass.
    #    Presence is the primary check (object-derived presence is reliable); the exact
    #    colour is only scored when BOTH sides actually name one, because the colour
    #    stated inside a critical-object clause disagrees with Chat-B2D's dedicated
    #    traffic-light round in ~11% of frames even for GT-vs-GT.
    COLORS = ("red", "green", "yellow")
    g_tl, p_tl = gt.get("traffic_light_obj"), pred.get("traffic_light_obj")
    if g_tl != "unknown" and p_tl != "unknown" and g_tl is not None and p_tl is not None:
        g_present, p_present = (g_tl != "none"), (p_tl != "none")
        ok = (g_present == p_present)
        detail["traffic_light_present"] = {"gt": g_tl, "pred": p_tl, "ok": ok}
        agree.append(ok)
        if g_tl in COLORS and p_tl in COLORS:
            ok_c = (g_tl == p_tl)
            detail["traffic_light_color"] = {"gt": g_tl, "pred": p_tl, "ok": ok_c}
            agree.append(ok_c)

    # 2. leading-vehicle existence
    if gt.get("lead_vehicle") is not None and pred.get("lead_vehicle") is not None:
        ok = bool(gt["lead_vehicle"]) == bool(pred["lead_vehicle"])
        detail["lead_vehicle"] = {"gt": gt["lead_vehicle"], "pred": pred["lead_vehicle"], "ok": ok}
        agree.append(ok)

    # 3. pedestrian hazard
    if gt.get("pedestrian") is not None and pred.get("pedestrian") is not None:
        ok = bool(gt["pedestrian"]) == bool(pred["pedestrian"])
        detail["pedestrian"] = {"gt": gt["pedestrian"], "pred": pred["pedestrian"], "ok": ok}
        agree.append(ok)

    # 4. critical-object side (only when both sides identified one)
    if gt.get("critical_side") and pred.get("critical_side"):
        ok = gt["critical_side"] == pred["critical_side"]
        detail["critical_side"] = {"gt": gt["critical_side"], "pred": pred["critical_side"], "ok": ok}
        agree.append(ok)

    # 5. critical-object motion
    if gt.get("critical_motion") and pred.get("critical_motion"):
        ok = gt["critical_motion"] == pred["critical_motion"]
        detail["critical_motion"] = {"gt": gt["critical_motion"], "pred": pred["critical_motion"], "ok": ok}
        agree.append(ok)

    if not agree:
        return None, {"reason": "no_comparable_variable", **detail}
    P = all(agree)
    detail["n_compared"] = len(agree)
    detail["n_agree"] = sum(agree)
    return P, detail


def compare_reasoning(gt: dict, pred: dict, model_rounds: dict) -> tuple[bool | None, dict]:
    """
    Semantic agreement of the driving decision.

    Longitudinal intent is the primary signal. STOP and DECELERATE are treated as distinct
    but both "slowing"; we require exact category agreement, and additionally accept the
    STOP/DECELERATE pair as agreeing since Chat-B2D uses them near-interchangeably for
    yield situations (recorded in `lenient_pair` so the strict count is recoverable).
    """
    txt = (model_rounds.get("driving_reason") or "").strip()
    if not txt:
        return None, {"reason": "no_reasoning_answer"}
    g, p = gt.get("intent_lon"), pred.get("intent_lon")
    if g is None or p is None:
        return None, {"reason": "unparsed_intent", "gt": g, "pred": p}
    strict = (g == p)
    lenient_pair = ({g, p} == {"STOP", "DECELERATE"})
    detail = {"gt_lon": g, "pred_lon": p, "strict": strict, "lenient_pair": lenient_pair,
              "gt_lat": gt.get("intent_lat"), "pred_lat": pred.get("intent_lat")}
    return (strict or lenient_pair), detail


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct for small counts and proportions near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def group_of(P: bool | None, R: bool | None, A: bool | None) -> str:
    def s(v, ch):
        return "?" if v is None else (f"{ch}+" if v else f"{ch}-")
    return f"{s(P,'P')}{s(R,'R')}{s(A,'A')}"


def classify_all(records: list[dict], ade_thr: float, fde_thr: float) -> list[dict]:
    rows = []
    for r in records:
        pred_t, gt_t = r.get("trajectory_pred"), r.get("trajectory_gt")
        mask = r.get("fut_mask")
        gtp = r.get("gt_perception", {})

        P, pdet = compare_perception(gtp, r.get("model_perception", {}), r.get("model_rounds", {}))
        R, rdet = compare_reasoning(r.get("gt_reasoning", {}), r.get("model_reasoning", {}),
                                    r.get("model_rounds", {}))

        if pred_t and gt_t:
            v = action_correct(pred_t, gt_t, gtp, mask, ade_thr, fde_thr)
            A, ade_v, fde_v = v.correct, v.ade, v.fde
            sem_p, sem_g, a_reason = v.sem_pred, v.sem_gt, v.reason
            safety_req, safe_p, safe_g = v.safety_required, v.safety_ok_pred, v.safety_ok_gt
        else:
            A = None
            ade_v = fde_v = float("nan")
            sem_p = sem_g = a_reason = ""
            safety_req = safe_p = safe_g = None

        rows.append({
            "sample_id": r["sample_id"], "clip_id": r["clip_id"], "frame_idx": r["frame_idx"],
            "scenario": r["scenario"],
            "P": P, "R": R, "A": A, "group": group_of(P, R, A),
            "ADE": ade_v, "FDE": fde_v,
            "high_level_action_gt": sem_g, "high_level_action_pred": sem_p,
            "action_fail_reason": a_reason,
            "safety_required": safety_req, "safety_ok_pred": safe_p, "safety_ok_gt": safe_g,
            "gt_traffic_light": gtp.get("traffic_light"),
            "gt_lead_vehicle": gtp.get("lead_vehicle"),
            "gt_pedestrian": gtp.get("pedestrian"),
            "reasoning_summary": (r.get("model_rounds", {}).get("driving_reason", "") or "")[:300],
            "perception_detail": pdet, "reasoning_detail": rdet,
            "fut_valid_flag": r.get("fut_valid_flag"),
            "hidden_path": r.get("hidden_path"),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    groups = Counter(r["group"] for r in rows)

    def count(pred) -> int:
        return sum(1 for r in rows if pred(r))

    # Decidable denominators: only samples where the relevant labels exist.
    dec_PA = [r for r in rows if r["P"] is not None and r["A"] is not None]
    dec_PRA = [r for r in rows if all(r[k] is not None for k in ("P", "R", "A"))]

    n_PA_minus = sum(1 for r in dec_PA if r["P"] and not r["A"])
    n_PRA_minus = sum(1 for r in dec_PRA if r["P"] and r["R"] and not r["A"])
    n_A_minus = sum(1 for r in dec_PA if not r["A"])
    n_Pminus_Aminus = sum(1 for r in dec_PA if not r["P"] and not r["A"])

    out = {
        "n_total": n,
        "groups": dict(groups.most_common()),
        "n_decidable_PA": len(dec_PA),
        "n_decidable_PRA": len(dec_PRA),
        "P_plus_A_minus": {
            "count": n_PA_minus,
            "rate_of_decidable": n_PA_minus / len(dec_PA) if dec_PA else 0.0,
            "ci95": wilson_ci(n_PA_minus, len(dec_PA)),
            "share_of_all_failures": n_PA_minus / n_A_minus if n_A_minus else 0.0,
        },
        "P_plus_R_plus_A_minus": {
            "count": n_PRA_minus,
            "rate_of_decidable": n_PRA_minus / len(dec_PRA) if dec_PRA else 0.0,
            "ci95": wilson_ci(n_PRA_minus, len(dec_PRA)),
        },
        "n_A_minus": n_A_minus,
        "n_P_minus_A_minus": n_Pminus_Aminus,
        "unknown": {
            "P": count(lambda r: r["P"] is None),
            "R": count(lambda r: r["R"] is None),
            "A": count(lambda r: r["A"] is None),
        },
        "action_fail_reasons": dict(Counter(
            r["action_fail_reason"] for r in rows if r["A"] is False).most_common(15)),
    }

    # Per-scenario P+A- rate, so we can check the phenomenon is not one-scenario-specific.
    per_scen: dict[str, dict] = {}
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for r in dec_PA:
        by_scen[r["scenario"]].append(r)
    for scen, rs in sorted(by_scen.items()):
        k = sum(1 for r in rs if r["P"] and not r["A"])
        per_scen[scen] = {"n": len(rs), "P+A-": k, "rate": k / len(rs),
                          "ci95": wilson_ci(k, len(rs))}
    out["per_scenario_P_plus_A_minus"] = per_scen
    out["n_scenarios_with_P+A-"] = sum(1 for v in per_scen.values() if v["P+A-"] > 0)

    # Safety-conditioned accuracy where a requirement applies.
    saf = [r for r in rows if r["safety_required"] and r["safety_ok_pred"] is not None]
    if saf:
        ok = sum(1 for r in saf if r["safety_ok_pred"])
        out["safety_conditioned"] = {
            "n": len(saf), "satisfied": ok, "rate": ok / len(saf),
            "ci95": wilson_ci(ok, len(saf)),
        }

    valid_ade = [r["ADE"] for r in rows if r["ADE"] == r["ADE"]]
    if valid_ade:
        a = np.array(valid_ade)
        out["ADE_distribution"] = {
            "mean": float(a.mean()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()),
        }
    valid_fde = [r["FDE"] for r in rows if r["FDE"] == r["FDE"]]
    if valid_fde:
        f = np.array(valid_fde)
        out["FDE_distribution"] = {
            "mean": float(f.mean()), "median": float(np.median(f)),
            "p75": float(np.percentile(f, 75)), "p90": float(np.percentile(f, 90)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs", "merged", "records.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(POC_DIR, "outputs", "taxonomy"))
    ap.add_argument("--ade-threshold", type=float, default=ADE_THRESHOLD_M)
    ap.add_argument("--fde-threshold", type=float, default=FDE_THRESHOLD_M)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = []
    with open(args.records) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"loaded {len(records)} records")

    rows = classify_all(records, args.ade_threshold, args.fde_threshold)
    summary = summarise(rows)
    summary["thresholds"] = {"ade": args.ade_threshold, "fde": args.fde_threshold}

    with open(os.path.join(args.outdir, "taxonomy.json"), "w") as f:
        json.dump(rows, f, indent=1, default=str)
    cols = ["sample_id", "clip_id", "frame_idx", "scenario", "P", "R", "A", "group",
            "ADE", "FDE", "high_level_action_gt", "high_level_action_pred",
            "action_fail_reason", "safety_required", "safety_ok_pred",
            "gt_traffic_light", "gt_lead_vehicle", "gt_pedestrian", "reasoning_summary"]
    with open(os.path.join(args.outdir, "taxonomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(args.outdir, "taxonomy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- threshold sensitivity ------------------------------------------------------
    sens = {}
    for a_thr in (0.5, 1.0, 1.5, 2.0, 3.0):
        for f_thr in (1.5, 3.0, 5.0):
            rs = classify_all(records, a_thr, f_thr)
            s = summarise(rs)
            sens[f"ade{a_thr}_fde{f_thr}"] = {
                "P+A-": s["P_plus_A_minus"]["count"],
                "P+A-_rate": s["P_plus_A_minus"]["rate_of_decidable"],
                "P+R+A-": s["P_plus_R_plus_A_minus"]["count"],
                "n_A-": s["n_A_minus"],
                "n_decidable_PA": s["n_decidable_PA"],
            }
    with open(os.path.join(args.outdir, "threshold_sensitivity.json"), "w") as f:
        json.dump(sens, f, indent=2)

    print("\n=== groups ===")
    for g, c in summary["groups"].items():
        print(f"  {g}: {c}")
    pa, pra = summary["P_plus_A_minus"], summary["P_plus_R_plus_A_minus"]
    print(f"\nP+A- : {pa['count']}/{summary['n_decidable_PA']} "
          f"= {pa['rate_of_decidable']:.1%} CI95 "
          f"[{pa['ci95'][0]:.1%},{pa['ci95'][1]:.1%}]  "
          f"({pa['share_of_all_failures']:.1%} of all A- cases)")
    print(f"P+R+A-: {pra['count']}/{summary['n_decidable_PRA']} "
          f"= {pra['rate_of_decidable']:.1%} CI95 "
          f"[{pra['ci95'][0]:.1%},{pra['ci95'][1]:.1%}]")
    print(f"scenarios containing P+A-: {summary['n_scenarios_with_P+A-']}")
    print(f"unknown: {summary['unknown']}")
    if "ADE_distribution" in summary:
        print(f"ADE: {summary['ADE_distribution']}")
    print(f"\nwrote -> {args.outdir}")


if __name__ == "__main__":
    main()
