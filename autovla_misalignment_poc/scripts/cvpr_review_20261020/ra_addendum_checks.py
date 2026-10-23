#!/usr/bin/env python
"""Addendum checks (post_hoc, CPU): CoT exclusion flow, no-op/cache parity summary, 364 vs 365 units, R9 recovery
recomputation and width saturation, SpatialVLA 8-row task table, Table 2 day-vehicle column."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402

T = f"{R.OUT}/tables"


def cot_flow():
    rows = []
    c = {"cot": 0, "runaway": 0, "no_mismatch": 0, "t_gt7": 0, "eligible": 0, "eligibleF": 0, "eligibleS": 0, "n": 0}
    for line in open(f"{R.P1C}/stage1/records.jsonl"):
        x = json.loads(line); p = x["p1c"]; n = x["arms"]["N"]
        c["n"] += 1
        c["cot"] += int(bool(n["cot_present"])); c["runaway"] += int(n["runaway_action_tokens"] > 0)
        c["no_mismatch"] += int(p["exclusion_reason"] == "no_mismatch"); c["t_gt7"] += int(p["exclusion_reason"] == "t_star_gt_7")
        if p["eligible"]:
            c["eligible"] += 1; c["eligibleF"] += int(p["a_eval"] == "A-"); c["eligibleS"] += int(p["a_eval"] == "A+")
    err = sum(1 for _ in open(f"{R.P1C}/stage1/errors.jsonl"))
    rows += [{"cohort": "P1-C stage 1 (64 logs)", "step": "natural plans generated (unit = scene)", "n": c["n"]},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "generation errors (no action within 160 tokens)", "n": err},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "excluded: CoT stub ('complex' after forced stem)", "n": c["cot"]},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "excluded: runaway (>10 action tokens)", "n": c["runaway"]},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "excluded: no first mismatch to GT tokens", "n": c["no_mismatch"]},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "excluded: t* > 7", "n": c["t_gt7"]},
             {"cohort": "P1-C stage 1 (64 logs)", "step": "eligible frame E (F / S)", "n": c["eligible"], "note": f"F {c['eligibleF']} / S {c['eligibleS']}"},
             {"cohort": "P1-C", "step": "sampled for intervention (F census + 4 S per log)", "n": 353},
             {"cohort": "P1-C / B2", "step": "E4 (t* <= 4): eligible / sampled", "n": 312, "note": "1,775 eligible; 100 F + 212 S sampled"}]
    cot, n = 0, 0
    for line in open(f"{R.POC}/outputs/full_extract/records.jsonl"):
        x = json.loads(line); n += 1; cot += int("complex scenario" in x["arms"]["N"]["generated_text"])
    rows += [{"cohort": "development natural evaluation (full_extract)", "step": "natural plans generated (unit = scene)", "n": n},
             {"cohort": "development natural evaluation (full_extract)", "step": "excluded: CoT ('complex scenario')", "n": cot},
             {"cohort": "development natural evaluation (full_extract)", "step": "evaluated (paper: 2,747)", "n": n - cot},
             {"cohort": "development mechanism set", "step": "selected: 52 A- + 156 A+ matched on t*", "n": 208}]
    pd.DataFrame(rows).to_csv(f"{T}/ADD1_cot_exclusion_flow.csv", index=False)
    return rows


def noop():
    a = pd.read_csv(f"{R.POC}/outputs/review_followup3/P1/NOOP_CACHE_AUDIT.csv")
    a["class"] = a.control.map(lambda s: "exact" if s in ("c1", "c2_hook", "c2_token", "c5a", "c5b", "c5d") else "equivalence (bf16 kernel shapes)")
    g = a.groupby(["control", "class"]).agg(rows=("n_rows", "sum"), failed_rows=("n_fail", "sum"), max_abs_logit=("logit_maxabs", "max"),
                                           token_flips=("n_flips", "sum"), suffix_equal_rows=("suffix_equal_rows", "sum")).reset_index()
    g["tolerance"] = "exact: 0; equivalence: T_equiv = 2 x max|dlogit| of c5c_tf, flips explained iff |dlogit| and top-2 margin <= T_equiv"
    g["dtype"] = "bf16 forward, sdpa, TF32 off (PROTOCOL_P1 sec. 5)"
    g["judged"] = "gate_stage0/judge.log 2026-10-14 15:43: H7/H8/H9/P1A PASS, 0 failures"
    g.to_csv(f"{T}/ADD2_noop_cache_parity.csv", index=False)
    return g


def units_364_365():
    g, e = set(), set()
    for line in open(f"{R.P1A}/raw/records.jsonl"):
        r = json.loads(line)
        if r["group"] != "A-":
            continue
        (g if r["family"] == "G" else e).add((r["token"], r["perturbation"]))
    u = pd.read_parquet(f"{R.POC}/outputs/review_followup/R8R9_raw_recovery/run_20261008/units.parquet",
                        columns=["experiment", "unit_id", "scene_id", "group", "perturbation", "condition", "t_star"])
    r9 = u[(u.experiment == "R9") & (u.group == "A-") & (u.condition == "normal")]
    r9ids = set(zip(r9.scene_id, r9.perturbation))
    gs = {s for s, _ in g}; es = {s for s, _ in e}
    out = {"motion_G_family_A-_units": len(g), "motion_G_family_A-_scenes": len(gs),
           "P1A_E_family_A-_units": len(e), "R9_A-_units": len(r9ids), "R9_A-_scenes": r9.scene_id.nunique(),
           "E_family_units_equal_R9_units": e == r9ids, "G_and_R9_share_unit_ids": len({(s, p) for s, p in g} & r9ids),
           "G_scenes_equal_R9_scenes": gs == set(r9.scene_id),
           "G_units_per_scene": pd.Series([s for s, _ in g]).value_counts().value_counts().to_dict(),
           "R9_units_per_scene": r9.scene_id.value_counts().value_counts().to_dict(),
           "explanation": "different perturbation families on the same 52 A- scenes: motion analysis = family G (8 GT-free sectors minus "
                          "sectors with no admissible token), R9 = equal-distance (inherited E) perturbations; not one dropped unit"}
    json.dump(out, open(f"{T}/ADD7_units_364_vs_365.json", "w"), indent=1, default=str)
    # R9 recovery
    cs = pd.read_csv(f"{R.POC}/outputs/review_followup/R8R9_raw_recovery/run_20261008/condition_summary.csv")
    d = cs[(cs.experiment == "R9") & (cs.group == "A-") & (cs.metric == "amplification")].set_index("condition")
    rec = []
    for wcol, lab in (("scene_mean", "scene-mean levels (paper)"), ("unit_mean", "unit-weighted levels")):
        nm, w3, fu = d.loc["normal", wcol], d.loc["win3", wcol], d.loc["gt_history", wcol]
        rec.append({"definition": lab, "normal": nm, "win3": w3, "full_gt": fu, "recovery_pct": 100 * (nm - w3) / (nm - fu)})
    nm, w3, fu = round(100 * d.loc["normal", "scene_mean"], 1), round(100 * d.loc["win3", "scene_mean"], 1), round(100 * d.loc["gt_history", "scene_mean"], 1)
    rec.append({"definition": "from displayed rounded levels (45.1/7.6/3.4)", "normal": nm / 100, "win3": w3 / 100, "full_gt": fu / 100,
                "recovery_pct": 100 * (nm - w3) / (nm - fu)})
    units = u[(u.experiment == "R9") & (u.group == "A-")]
    nunits = units[units.condition == "normal"]
    sat = [{"width": w, "units_saturated_w_ge_8_minus_tstar": int((w >= 8 - nunits.t_star).sum()), "n_units": len(nunits)} for w in range(1, 9)]
    pd.DataFrame(rec).to_csv(f"{T}/ADD7_r9_recovery.csv", index=False)
    pd.DataFrame(sat).to_csv(f"{T}/ADD7_r9_width_saturation.csv", index=False)
    return out, rec, sat


def spatial8():
    f = pd.read_csv(f"{R.AQ}/FINAL_PAPER_NUMBERS.csv")
    c2 = pd.read_csv(f"{T}/C2_spatialvla_task_success.csv")
    rows = []
    for cfg in ("r1e4", "r4e1"):
        for con in ("R-N", "C-F"):
            for task, key in (("google_robot_pick_coke_can", "coke"), ("google_robot_move_near", "move_near")):
                fr = f[f.claim_id == f"SVLA.P0B.P1D.{cfg}.{con}.{key}.physical"].iloc[0]
                cr = c2[(c2.config == cfg) & (c2.contrast == con) & (c2.task == task)].iloc[0]
                rows.append({"task": key, "setting": cfg, "contrast": con,
                             "success_intervention": f"{int(cr.success_intervention)}/{int(cr.n_pairs)}",
                             "success_reference": f"{int(cr.success_reference)}/{int(cr.n_pairs)}",
                             "delta_pp": cr.delta_intervention_minus_reference_pp, "n10/n01": f"{int(cr.n10_only_intervention)}/{int(cr.n01_only_reference)}",
                             "physical_states": int(cr.n_physical_states), "interval_95": f"[{fr.ci_low:.1f}, {fr.ci_high:.1f}]",
                             "interval_method": fr.ci_method, "equivalence_status": "per-task equivalence not claimed (two-task mean only)",
                             "source": "FINAL_PAPER_NUMBERS " + fr.claim_id})
    pd.DataFrame(rows).to_csv(f"{T}/ADD3_spatialvla_8rows.csv", index=False)
    return rows


def table2():
    r = pd.read_csv(f"{R.RF5}/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv")
    r = r[r.family_member]
    p = r.pivot_table(index=["contrast", "estimand"], columns="cluster_axis", values="holm_p").reset_index()
    p2 = r.pivot_table(index=["contrast", "estimand"], columns="cluster_axis", values="p_signflip_exact").reset_index()
    out = p.merge(p2, on=["contrast", "estimand"], suffixes=("_holm", "_raw"))
    th = r[r.cluster_axis == "drive"].set_index(["contrast", "estimand"]).theta
    out["theta_pp"] = [100 * th.loc[(a, b)] for a, b in zip(out.contrast, out.estimand)]
    out.to_csv(f"{T}/ADD7_table2_with_day_vehicle.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(pd.DataFrame(cot_flow()).to_string())
    print(noop().to_string())
    o, rec, sat = units_364_365(); print(json.dumps(o, indent=1, default=str)); print(pd.DataFrame(rec)); print(pd.DataFrame(sat))
    print(pd.DataFrame(spatial8()).drop(columns=["interval_method", "source"]).to_string())
    print(table2().round(4).to_string())
