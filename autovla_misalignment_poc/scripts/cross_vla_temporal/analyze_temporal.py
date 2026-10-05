#!/usr/bin/env python
"""
Analysis of experiment 33 (outputs/cross_vla_temporal_replication/PROTOCOL.md). CPU only.

Trajectory metrics are those of the AutoVLA experiments: A- = not accepted by the P/R/A rule (analyze_action_history.a_eval),
amplification = A- and FDE5 > 3 m, FDE5 / ADE5 against the logged future. Log-cluster bootstrap 95% CI (2,000, seed 0),
paired McNemar (binary) and Wilcoxon (continuous) over units.

  python analyze_temporal.py <out_dir>        (reads <out_dir>/exp_ab and <out_dir>/exp_cde)
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts")
from analyze_action_history import AMP_FDE, a_eval   # noqa: E402

MAGS, DIRS = (0.2, 0.5), tuple(range(0, 360, 45))
ROWS = ["normal", "win1", "win2", "win3", "win4", "gt_history", "recent_gt", "reverse",
        "near_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched"]


def metrics(traj, gt):
    t, g = np.asarray(traj, float), np.asarray(gt, float)
    e = np.linalg.norm(t - g, axis=1); am = not a_eval(t.tolist(), g.tolist())
    return {"a_minus": float(am), "amp": float(am and e[-1] > AMP_FDE), "fde": float(e[-1]), "ade": float(e.mean())}


def cboot(pairs, reps=2000):
    """pairs: list of (log, value). Mean and log-cluster bootstrap CI."""
    by = defaultdict(list)
    for lg, v in pairs:
        by[lg].append(v)
    logs = sorted(by); rng = random.Random(0)
    bs = sorted(np.mean([x for _ in logs for x in by[rng.choice(logs)]]) for _ in range(reps))
    return {"mean": float(np.mean([v for _, v in pairs])), "ci95": [float(bs[int(0.025 * reps)]), float(bs[int(0.975 * reps) - 1])], "n": len(pairs)}


def paired(units, a, b, key, binary):
    """units: list of dicts {log, a: {key}, b: {key}} -> a - b."""
    pr = [(u["log"], u[a][key], u[b][key]) for u in units]
    d = cboot([(lg, x - y) for lg, x, y in pr])
    if binary:
        g = sum(x > y for _, x, y in pr); l_ = sum(y > x for _, x, y in pr)
        d["mcnemar"] = {"a_only": int(g), "b_only": int(l_), "p": float(binomtest(g, g + l_, 0.5).pvalue) if g + l_ else 1.0}
    else:
        diffs = np.array([x - y for _, x, y in pr])
        d["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return d


def frac(units, row, key, reps=2000):
    by = defaultdict(list)
    for u in units:
        by[u["log"]].append((u["normal"][key], u["gt_history"][key], u[row][key]))
    logs = sorted(by)

    def f(ls):
        v = np.array([x for lg in ls for x in by[lg]]); den = v[:, 0].mean() - v[:, 1].mean()
        return (v[:, 0].mean() - v[:, 2].mean()) / den if abs(den) > 1e-9 else np.nan
    rng = random.Random(0)
    bs = [x for x in (f([rng.choice(logs) for _ in logs]) for _ in range(reps)) if np.isfinite(x)]
    return {"frac": float(f(logs)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None}


def analyze_ab(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    recs = [r for r in recs if all(v["ok"] for v in r["rows"].values())]
    out = {"n_scenes": len(recs), "n_logs": len({r["log"] for r in recs})}
    S = []
    for r in recs:
        gt = r["gt"]; base = np.array(r["rows"]["pert_m0"]["traj"])
        s = {"log": r["log"], "natural": metrics(r["natural"], gt), "natural_rep": metrics(r["rows"]["natural_rep"]["traj"], gt),
             "pert_m0": metrics(base, gt),
             "d_rep": float(np.linalg.norm(np.array(r["rows"]["natural_rep"]["traj"])[-1] - np.array(r["natural"])[-1])),
             "d_m0": float(np.linalg.norm(base[-1] - np.array(r["natural"])[-1]))}
        for mg in MAGS:
            for a in DIRS:
                tr = np.array(r["rows"][f"pert_m{mg}_d{a}"]["traj"]); m = metrics(tr, gt)
                m["r"] = float(np.linalg.norm(tr[-1] - base[-1]) / mg); s[f"m{mg}_d{a}"] = m
        S.append(s)
    out["natural"] = {k: cboot([(s["log"], s["natural"][k]) for s in S]) for k in ("a_minus", "amp", "fde", "ade")}
    out["rerun_variability"] = {
        "natural_vs_rep_amin_discordant": cboot([(s["log"], float(s["natural"]["a_minus"] != s["natural_rep"]["a_minus"])) for s in S]),
        "natural_vs_rep_amp_discordant": cboot([(s["log"], float(s["natural"]["amp"] != s["natural_rep"]["amp"])) for s in S]),
        "natural_vs_m0_amin_discordant": cboot([(s["log"], float(s["natural"]["a_minus"] != s["pert_m0"]["a_minus"])) for s in S]),
        "natural_vs_m0_amp_discordant": cboot([(s["log"], float(s["natural"]["amp"] != s["pert_m0"]["amp"])) for s in S]),
        "final_wp_shift_rep_m": cboot([(s["log"], s["d_rep"]) for s in S]),
        "final_wp_shift_m0_m": cboot([(s["log"], s["d_m0"]) for s in S]),
        "natural_vs_rep_amin": paired([{"log": s["log"], "a": s["natural_rep"], "b": s["natural"]} for s in S], "a", "b", "a_minus", True)}
    # Experiment A
    A = {}
    okA = [s for s in S if s["pert_m0"]["a_minus"] == 0]
    for mg in MAGS:
        keys = [f"m{mg}_d{a}" for a in DIRS]
        A[f"m{mg}"] = {
            "flip_to_a_minus_given_m0_ok": cboot([(s["log"], s[k]["a_minus"]) for s in okA for k in keys]),
            "flip_to_amp_given_m0_ok": cboot([(s["log"], s[k]["amp"]) for s in okA for k in keys]),
            "a_minus_rate": cboot([(s["log"], s[k]["a_minus"]) for s in S for k in keys]),
            "amp_rate": cboot([(s["log"], s[k]["amp"]) for s in S for k in keys]),
            "fde": cboot([(s["log"], s[k]["fde"]) for s in S for k in keys]),
            "r_le_1": cboot([(s["log"], float(s[k]["r"] <= 1)) for s in S for k in keys]),
            "r_ge_3": cboot([(s["log"], float(s[k]["r"] >= 3)) for s in S for k in keys]),
            "r_median": float(np.median([s[k]["r"] for s in S for k in keys])),
            "vs_m0_a_minus": paired([{"log": s["log"], "a": s[k], "b": s["pert_m0"]} for s in S for k in keys], "a", "b", "a_minus", True),
            "vs_m0_amp": paired([{"log": s["log"], "a": s[k], "b": s["pert_m0"]} for s in S for k in keys], "a", "b", "amp", True)}
    A["baseline_flip_natural_ok_to_rep_a_minus"] = cboot([(s["log"], s["natural_rep"]["a_minus"]) for s in S if s["natural"]["a_minus"] == 0])
    out["A"] = A
    # Experiment B
    B = {}
    for mg in MAGS:
        keys = [f"m{mg}_d{a}" for a in DIRS]
        stable = lambda m: m["a_minus"] == 0 and m["r"] <= 1          # noqa: E731
        unstable = lambda m: m["amp"] == 1 or m["r"] >= 3             # noqa: E731
        co = [(s["log"], float(any(stable(s[k]) for k in keys) and any(unstable(s[k]) for k in keys))) for s in S]
        lr = np.array([[math.log(max(s[k]["r"], 1e-3)) for k in keys] for s in S])
        between = float(np.var(lr.mean(1))); within = float(np.mean(lr.var(1)))
        B[f"m{mg}"] = {"coexist_share": cboot(co), "icc_between_share": between / (between + within),
                       "var_between": between, "var_within": within,
                       "mean_log_r_by_direction": {a: float(lr[:, i].mean()) for i, a in enumerate(DIRS)},
                       "amp_by_direction": {a: float(np.mean([s[f"m{mg}_d{a}"]["amp"] for s in S])) for a in DIRS}}
    B["fde_by_magnitude"] = {f"m{mg}": float(np.mean([s[f"m{mg}_d{a}"]["fde"] for s in S for a in DIRS])) for mg in MAGS}
    B["fde_m0"] = float(np.mean([s["pert_m0"]["fde"] for s in S]))
    out["B"] = B
    return out


def analyze_cde(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    U = []
    for r in recs:
        gt = r["gt"]; ref = next(u for u in r["units"] if u["dir"] is None)
        for u in r["units"]:
            if u["dir"] is None or u["parse_fail"] or ref["parse_fail"]:
                continue
            x = {"log": r["log"], "token": r["token"], "dir": u["dir"]}
            for row in ROWS:
                m = metrics(u["rows"][row], gt)
                m["d10"] = float(np.linalg.norm(np.array(u["rows"][row])[-1] - np.array(ref["rows"][row])[-1]))
                x[row] = m
            x["subs"] = {k: v for k, v in u["subs"].items()}
            U.append(x)
    out = {"n_units": len(U), "n_scenes": len({u["token"] for u in U}), "n_logs": len({u["log"] for u in U})}

    def block(us):
        b = {"means": {row: {k: cboot([(u["log"], u[row][k]) for u in us]) for k in ("amp", "a_minus", "fde", "d10")} for row in ROWS}}
        b["vs_normal"] = {row: {k: paired(us, row, "normal", k, k in ("amp", "a_minus")) for k in ("amp", "a_minus", "fde", "d10")} for row in ROWS[1:]}
        b["reverse_vs_gt_history"] = {k: paired(us, "reverse", "gt_history", k, k in ("amp", "a_minus")) for k in ("amp", "a_minus", "fde", "d10")}
        b["reverse_vs_normal"] = b["vs_normal"]["reverse"]
        b["window_frac_of_full"] = {row: {k: frac(us, row, k) for k in ("amp", "d10", "fde")} for row in ("win1", "win2", "win3", "win4", "recent_gt")}
        b["motion"] = {f"{a} - {c}": {k: paired(us, a, c, k, k in ("amp", "a_minus")) for k in ("amp", "fde", "d10")}
                       for a, c in (("dir_wrong_mag_ok", "dir_ok_mag_wrong"), ("near_gt", "recent_gt"), ("random_mag_matched", "dir_ok_mag_wrong"),
                                    ("dir_ok_mag_wrong", "recent_gt"), ("dir_wrong_mag_ok", "recent_gt"))}
        b["substitution_geometry"] = {}
        for row in ("near_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "random_mag_matched"):
            v = [s for u in us for s in u["subs"].get(row, [])]
            b["substitution_geometry"][row] = {k: float(np.nanmean([s[k] for s in v if s[k] is not None])) for k in ("e_own", "d_sub_gt", "dmag", "dphi_deg")} if v else None
        return b
    out["all_units"] = block(U)
    amp_units = [u for u in U if u["normal"]["amp"] == 1]
    out["normal_amplified_units"] = block(amp_units) if len(amp_units) >= 20 else {"n": len(amp_units)}
    out["normal_amplified_units_n"] = len(amp_units)
    return out


def main() -> None:
    root = sys.argv[1]
    res = {}
    if os.path.exists(os.path.join(root, "exp_ab/records.jsonl")):
        res["AB"] = analyze_ab(os.path.join(root, "exp_ab"))
    if os.path.exists(os.path.join(root, "exp_cde/records.jsonl")):
        res["CDE"] = analyze_cde(os.path.join(root, "exp_cde"))
    json.dump(res, open(os.path.join(root, "analysis.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()
