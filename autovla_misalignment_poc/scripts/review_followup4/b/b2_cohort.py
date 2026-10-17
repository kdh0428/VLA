#!/usr/bin/env python
"""B2 cohort / estimand map (CPU, no intervention outcome read).
P1-C target E = stage-1 eligible scenes (natural plan valid, t* <= 7). B2 primary domain E4 = E with t* <= 4
(anchor a = t*+3 leaves >= 3 generated tokens after the edit). Strata as in P1-C: F = natural A- (census, pi = 1),
S = natural A+ (4 sampled per log by sha order). Domain weights are recomputed per log:
  w_S' = N'_S,log / m'_S,log, with N' = S-eligible scenes of the log with t* <= 4 and m' = sampled S scenes of the log
  with t* <= 4.  Logs with N' > 0 and m' = 0 are a coverage gap (reported; Hajek renormalises over covered logs).
Uses only the natural-pass label (a_eval) and t*, which were fixed before any P1-C intervention."""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

POC = "/root/VLA/autovla_misalignment_poc"
RUN = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008"
ED_DEV = f"{POC}/outputs/equal_distance_perturbation/records.jsonl"


def drive_of(log):
    return log.split("_veh-")[0] + "_veh-" + log.split("_veh-")[1].split("_")[0]


def dayveh_of(log):
    d = log.split(".")
    return f"{d[0]}.{d[1]}.{d[2]}_veh-{log.split('_veh-')[1].split('_')[0]}"


def main():
    out = sys.argv[1]
    tmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    st = {}
    for l in open(f"{RUN}/stage1/records.jsonl"):
        x = json.loads(l)
        p = x["p1c"]
        st[x["token"]] = {"log": x["log_name"], "eligible": p["eligible"], "t_star": p.get("t_star"), "a_eval": p.get("a_eval")}
    sel = {}
    for f in sorted(glob.glob(f"{RUN}/selection/shard_*.json")):
        if f.endswith((".keep_files.json", ".tokens.json")):
            continue
        for s in json.load(open(f))["selected"]:
            sel[s["token"]] = s
    units = {json.loads(l)["token"]: json.loads(l) for l in open(f"{RUN}/units/records.jsonl")}
    E = {t: v for t, v in st.items() if v["eligible"]}
    logs = sorted({v["log"] for v in E.values()})
    # per-log domain counts
    NS, mS, NF = Counter(), Counter(), Counter()
    for t, v in E.items():
        if v["t_star"] is not None and v["t_star"] <= tmax:
            if v["a_eval"] == "A-":
                NF[v["log"]] += 1
            else:
                NS[v["log"]] += 1
    for t, s in sel.items():
        if s["stratum"] == "S" and s["t_star"] <= tmax:
            mS[s["log"]] += 1
    w = {}
    gap = []
    for lg in logs:
        if NS[lg] > 0 and mS[lg] == 0:
            gap.append({"log": lg, "N_S_dom": NS[lg]})
    for t, s in sel.items():
        if s["t_star"] > tmax or t not in units:
            continue
        lg = s["log"]
        if s["stratum"] == "F":
            wt, pi = 1.0, 1.0
        else:
            wt = NS[lg] / mS[lg]; pi = 1 / wt
        w[t] = {"stratum": s["stratum"], "weight": wt, "pi": pi, "weight_P1C_original": s["weight"], "N_S_dom": NS[lg], "m_S_dom": mS[lg],
                "log": lg, "drive": drive_of(lg), "day_vehicle": dayveh_of(lg), "t_star": s["t_star"],
                "n_units": 1 + len(units[t]["alternatives"])}
    den = {
        "E_all_eligible_scenes": len(E),
        "E_F": sum(v["a_eval"] == "A-" for v in E.values()), "E_S": sum(v["a_eval"] != "A-" for v in E.values()),
        f"E{tmax}_scenes": sum(NF.values()) + sum(NS.values()), f"E{tmax}_F": sum(NF.values()), f"E{tmax}_S": sum(NS.values()),
        "sampled_all_scenes": len(sel), "sampled_dom_scenes": len(w),
        "sampled_dom_F_scenes": sum(v["stratum"] == "F" for v in w.values()), "sampled_dom_S_scenes": sum(v["stratum"] == "S" for v in w.values()),
        "sampled_dom_units": sum(v["n_units"] for v in w.values()),
        "sampled_dom_F_units": sum(v["n_units"] for v in w.values() if v["stratum"] == "F"),
        "sampled_dom_S_units": sum(v["n_units"] for v in w.values() if v["stratum"] == "S"),
        "drives_dom": len({v["drive"] for v in w.values()}), "drives_dom_F": len({v["drive"] for v in w.values() if v["stratum"] == "F"}),
        "day_vehicle_dom": len({v["day_vehicle"] for v in w.values()}),
        "logs": len(logs), "coverage_gap_logs": len(gap), "coverage_gap_S_scenes": sum(g["N_S_dom"] for g in gap),
        "sum_weights_S_dom": sum(v["weight"] for v in w.values() if v["stratum"] == "S"),
    }
    dev = [json.loads(l) for l in open(ED_DEV)]
    devd = [r for r in dev if r["t_star"] <= tmax]
    den["dev_scenes"] = len(dev); den[f"dev_scenes_t<={tmax}"] = len(devd)
    den[f"dev_units_t<={tmax}"] = sum(1 + len(r["alternatives"]) for r in devd)
    den[f"dev_Aminus_scenes_t<={tmax}"] = sum(r["group"] == "A-" for r in devd)
    den[f"dev_drives_Aminus_t<={tmax}"] = len({drive_of(r["log"]) for r in devd if r["group"] == "A-"})
    den[f"dev_drives_t<={tmax}"] = len({drive_of(r["log"]) for r in devd})
    json.dump({"rule": __doc__, "t_star_max": tmax, "denominators": den, "coverage_gap": gap, "scene_weights": w}, open(out, "w"), indent=1)
    print(json.dumps(den, indent=1))


if __name__ == "__main__":
    main()
