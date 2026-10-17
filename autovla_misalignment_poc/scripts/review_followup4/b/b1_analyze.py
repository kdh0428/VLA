#!/usr/bin/env python
"""B1 analysis (CPU). Reads B1/run/{records,logit_cmp}.jsonl and writes NUMERICAL_SENSITIVITY.csv + _tables.md.
Outcomes: analyze_action_history.unit_metrics (unchanged): severe = amplification = (not A+) and FDE5 > 3 m; A- = not A+."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, f"{POC}/scripts")
from analyze_action_history import unit_metrics  # noqa: E402

B1 = os.environ.get("B1_OUT", f"{POC}/outputs/review_followup4/B/B1")
RUN = sys.argv[1] if len(sys.argv) > 1 else f"{B1}/run"
ED = {r["token"]: r for r in map(json.loads, open(f"{POC}/outputs/equal_distance_perturbation/records.jsonl"))}
SETTINGS = ["S0r", "SB", "SE", "SF"]


def cp(k, n):
    if n == 0:
        return (None, None)
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return (float(lo), float(hi))


def main():
    recs = {}
    for l in open(f"{RUN}/records.jsonl"):
        x = json.loads(l)
        recs[(x["setting"], x["unit_id"], x["path"], x["row"])] = x          # last record wins (amendment B1-A1)
    for x in recs.values():
        e = ED[x["scene"]]
        poses = np.asarray(x["poses"])
        m = unit_metrics(x["acts"], poses[:, :2].tolist(), [], e["gt"], e["trajectory_gt"], x["t_p"])
        x["severe"], x["Aminus"], x["fde5"] = int(m["amplification"]), int(not m["recovery"]), float(m["fde5"])
        x["X9"] = poses[-1, :2]
    lc = [json.loads(l) for l in open(f"{RUN}/logit_cmp.jsonl")]
    rows_out = []

    # ---------------- 1. row-level numeric variation vs S0
    for s in SETTINGS:
        for path in ("H7", "H8", "P1A", "all"):
            L = [c for c in lc if c["setting"] == s and (path == "all" or c["path"] == path)]
            if not L:
                continue
            n = len(L)
            neq = sum(not c["suffix_equal"] for c in L)
            ep = np.array([c["endpoint_dev_m"] for c in L])
            lm = [c["logit_maxabs"] for c in L if c["logit_maxabs"] is not None]
            fl_sev = fl_am = 0
            dfde = []
            units_changed = set()
            for c in L:
                a = recs.get(("S0", c["unit_id"], c["path"], c["row"]))
                b = recs.get((s, c["unit_id"], c["path"], c["row"]))
                if a is None or b is None:
                    continue
                fl_sev += a["severe"] != b["severe"]
                fl_am += a["Aminus"] != b["Aminus"]
                dfde.append(abs(a["fde5"] - b["fde5"]))
                if not c["suffix_equal"]:
                    units_changed.add(c["unit_id"])
            nu = len({c["unit_id"] for c in L})
            rows_out.append({"section": "row_variation", "setting": s, "path": path, "n_row_runs": n, "n_units": nu,
                             "logit_maxabs_max": max(lm) if lm else None, "logit_maxabs_median": float(np.median(lm)) if lm else None,
                             "suffix_changed": neq, "suffix_changed_rate": neq / n, "suffix_changed_ci95": cp(neq, n),
                             "units_with_any_suffix_change": len(units_changed),
                             "endpoint_dev_mean_m": float(ep.mean()), "endpoint_dev_max_m": float(ep.max()),
                             "endpoint_dev_gt_0.5m": int((ep > 0.5).sum()), "endpoint_dev_gt_2m": int((ep > 2).sum()),
                             "endpoint_dev_mean_given_change_m": float(ep[[not c["suffix_equal"] for c in L]].mean()) if neq else 0.0,
                             "severe_flips": fl_sev, "severe_flip_ci95": cp(fl_sev, n), "Aminus_flips": fl_am, "Aminus_flip_ci95": cp(fl_am, n),
                             "fde5_absdiff_mean_m": float(np.mean(dfde)) if dfde else None, "fde5_absdiff_max_m": float(np.max(dfde)) if dfde else None})

    # ---------------- 2/3. contrasts: reference scale (S0) and stability across settings
    units = sorted({k[1] for k in recs if k[0] == "S0"})

    def get(s, uid, path, row):
        return recs.get((s, uid, path, row))

    def pair(name, s):
        """(cond_rec, ref_rec) per unit for contrast name under setting s."""
        out = {}
        for uid in units:
            if name in ("RN", "FULL"):
                cond = "recent_gt" if name == "RN" else "gt_history"
                if s == "SB":
                    a, b = get("S0", uid, "H8", cond), get("S0", uid, "H8", "normal")       # same rows in the B = 93 batch
                else:
                    a, b = get(s, uid, "H7", cond), get(s, uid, "H7", "normal")
            elif name == "RN_mixed":
                a, b = get("S0", uid, "H7", "recent_gt"), get("S0", uid, "H8", "normal")
            elif name == "RV":
                a, b = get(s, uid, "H8", "reverse@emb"), get(s, uid, "H8", "gt_history")
            elif name == "RV_mixed":
                a, b = get("S0", uid, "H8", "reverse@emb"), get("S0", uid, "H7", "gt_history")
            else:
                cond, ref = {"PERT": ("OC", "U"), "SELF_R": ("OC_refR", "OC"), "SELF_H": ("OC_refH", "OC")}[name]
                a, b = get(s, uid, "P1A", cond), get(s, uid, "P1A", ref)
            if a is not None and b is not None:
                out[uid] = (a, b)
        return out

    contrasts = ["RN", "FULL", "RV", "PERT", "SELF_R", "SELF_H"]
    base = {}
    for name in contrasts:
        P0 = pair(name, "S0")
        d0 = {u: (a["severe"] - b["severe"], a["Aminus"] - b["Aminus"], a["fde5"] - b["fde5"]) for u, (a, b) in P0.items()}
        base[name] = d0
        sh = np.array([float(np.linalg.norm(a["X9"] - b["X9"])) for a, b in P0.values()])
        sfx = sum(a["own"] != b["own"] for a, b in P0.values())
        rows_out.append({"section": "intervention_scale_S0", "contrast": name, "n_units": len(P0),
                         "suffix_differs": sfx, "endpoint_shift_mean_m": float(sh.mean()), "endpoint_shift_median_m": float(np.median(sh)),
                         "endpoint_shift_gt_0.5m": int((sh > 0.5).sum()),
                         "severe_differs": sum(a["severe"] != b["severe"] for a, b in P0.values()),
                         "Aminus_differs": sum(a["Aminus"] != b["Aminus"] for a, b in P0.values()),
                         "mean_d_severe": float(np.mean([v[0] for v in d0.values()])), "mean_d_Aminus": float(np.mean([v[1] for v in d0.values()])),
                         "mean_d_fde5_m": float(np.mean([v[2] for v in d0.values()]))})
    for name in contrasts + ["RN_mixed", "RV_mixed"]:
        for s in (["S0r", "SB", "SE", "SF"] if not name.endswith("_mixed") else ["SB"]):
            Ps = pair(name, s)
            ref_name = name.replace("_mixed", "")
            d0 = base[ref_name]
            ds = {u: (a["severe"] - b["severe"], a["Aminus"] - b["Aminus"], a["fde5"] - b["fde5"]) for u, (a, b) in Ps.items()}
            common = sorted(set(ds) & set(d0))
            if not common:
                continue
            m0 = np.array([d0[u] for u in common]); ms = np.array([ds[u] for u in common])
            mean0, means = m0.mean(0), ms.mean(0)
            delta = means - mean0
            nchg_sev = int((m0[:, 0] != ms[:, 0]).sum()); nchg_am = int((m0[:, 1] != ms[:, 1]).sum())
            mad = float(np.abs(m0[:, 2] - ms[:, 2]).mean())
            mat_a = (np.sign(means[0]) != np.sign(mean0[0]) and abs(mean0[0]) >= 1 / len(common)) or \
                    (np.sign(means[2]) != np.sign(mean0[2]) and abs(mean0[2]) > 0)
            mat_b = abs(delta[0]) > 0.05
            mat_c = abs(mean0[2]) > 0 and abs(delta[2]) > 0.25 * abs(mean0[2])
            rows_out.append({"section": "contrast_stability", "contrast": name, "setting": s, "n_units": len(common),
                             "mean_d_severe_S0": float(mean0[0]), "mean_d_severe_s": float(means[0]), "delta_severe": float(delta[0]),
                             "units_d_severe_changed": nchg_sev,
                             "mean_d_Aminus_S0": float(mean0[1]), "mean_d_Aminus_s": float(means[1]), "delta_Aminus": float(delta[1]),
                             "units_d_Aminus_changed": nchg_am,
                             "mean_d_fde5_S0_m": float(mean0[2]), "mean_d_fde5_s_m": float(means[2]), "delta_fde5_m": float(delta[2]),
                             "rel_delta_fde5": float(delta[2] / mean0[2]) if mean0[2] else None, "mean_abs_unit_change_fde5_m": mad,
                             "material_a_sign": bool(mat_a), "material_b_severe_gt_0.05": bool(mat_b), "material_c_fde5_gt_25pct": bool(mat_c),
                             "material": bool(mat_a or mat_b or mat_c)})
    keys = []
    for r in rows_out:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv
    with open(f"{B1}/NUMERICAL_SENSITIVITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: (json.dumps(v) if isinstance(v, tuple) else v) for k, v in r.items()})
    json.dump(rows_out, open(f"{B1}/numerical_sensitivity_rows.json", "w"), indent=1, default=float)
    for r in rows_out:
        print(json.dumps(r, default=float))


if __name__ == "__main__":
    main()
