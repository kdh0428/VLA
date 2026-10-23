#!/usr/bin/env python
"""A1-A3 (post_hoc/exploratory): acceptance-rule reconstruction check, metric decomposition, FDE threshold sensitivity.
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python ra_a_metrics.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402
from analyze_action_history import a_eval, CFG  # noqa: E402  (adds /root/VLA/pra_comparison to sys.path)
import pra_labels as L  # noqa: E402
from run_pra_comparison import _pos_to_delta  # noqa: E402

TAUS = [2.0, 3.0, 4.0, 5.0]


def rule(traj, gt):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, 10), gt_delta=_pos_to_delta(gt, 10))
    ok, info = L.label_A(s, CFG)
    return bool(ok), bool(info.get("lon_ok")), bool(info.get("lat_ok")), info


def metrics(traj, gt):
    p, g = np.asarray(traj, float)[:, :2], np.asarray(gt, float)[:, :2]
    e = np.linalg.norm(p - g, axis=1)
    ok, lon, lat, info = rule(traj, gt)
    m = {"ade5": float(e.mean()), "fde5": float(e[-1]), "rule_fail": int(not ok), "lon_fail": int(not lon),
         "lat_fail": int(not lat), "gt_lon": info.get("gt_lon"), "pred_lon": info.get("pred_lon"),
         "gt_lat": info.get("gt_lat"), "pred_lat": info.get("pred_lat")}
    for t in TAUS:
        m[f"fde_gt{int(t)}"] = int(e[-1] > t)
        m[f"severe{int(t)}"] = int((not ok) and e[-1] > t)
    m["state_pass_le3"] = int(ok and e[-1] <= 3); m["state_pass_gt3"] = int(ok and e[-1] > 3)
    m["state_fail_le3"] = int((not ok) and e[-1] <= 3); m["state_fail_gt3"] = int((not ok) and e[-1] > 3)
    return m


MET = (["ade5", "fde5", "rule_fail", "lon_fail", "lat_fail"] + [f"fde_gt{int(t)}" for t in TAUS] +
       [f"severe{int(t)}" for t in TAUS] + ["state_pass_le3", "state_pass_gt3", "state_fail_le3", "state_fail_gt3"])


def rows_ahc(path, units_ref=None, frame="dev"):
    out = []
    for line in open(path):
        r = json.loads(line)
        gt = r["trajectory_gt"]
        for pn, pv in r["perturbations"].items():
            for cond in ("normal", "recent_gt"):
                c = pv["conditions"][cond]
                m = metrics(c["trajectory_pred"], gt)
                out.append({"frame": frame, "scene_id": r["token"], "unit_id": f"{r['token']}|{pn}", "log_id": r["log"],
                            "drive_id": R.drive_of(r["log"]), "group": r.get("group"), "t_star": int(r["t_star"]),
                            "cond": cond, **m})
    return pd.DataFrame(out)


def rows_b2(rec_gt):
    out = []
    uc = pd.read_parquet(f"{R.B2}/p1c_analysis/unit_contrasts.parquet")
    meta = uc[(uc.n_old == 1) & (uc.contrast == "C_GT")].set_index("unit_id")[["scene", "stratum", "weight", "log", "drive", "day_vehicle", "t_star"]]
    for line in open(f"{R.B2}/p1c_main/cells.jsonl"):
        x = json.loads(line)
        if x["n_old"] != 1 or x["cell"] not in ("GT_REF", "GT_NSRC", "NAT_REF", "NAT_NSRC"):
            continue
        g = rec_gt[x["scene"]]
        m = metrics(np.asarray(x["poses"])[:, :2], g)
        mm = meta.loc[x["unit_id"]]
        out.append({"frame": "B2", "scene_id": x["scene"], "unit_id": x["unit_id"], "log_id": mm.log, "drive_id": mm.drive,
                    "day_vehicle": mm.day_vehicle, "stratum": mm.stratum, "weight": mm.weight, "t_star": int(mm.t_star),
                    "cond": x["cell"], "stored_severe": x["severe"], "stored_fde5": x["fde5"], **m})
    return pd.DataFrame(out)


def contrast(df, a, b, key="unit_id"):
    A = df[df.cond == a].set_index(key); B = df[df.cond == b].set_index(key)
    common = A.index.intersection(B.index)
    D = A.loc[common, MET] - B.loc[common, MET]
    keep = [c for c in A.columns if c not in MET and c != "cond"]
    return A.loc[common, keep].join(D), A.loc[common, MET], B.loc[common, MET], len(A), len(B), len(common)


def aggregate(units, wcol=None):
    s = units.groupby("scene_id")[MET].mean()
    if wcol is None:
        w = pd.Series(1.0, index=s.index)
    else:
        w = units.groupby("scene_id")[wcol].first().loc[s.index]
    return (s.mul(w, axis=0).sum() / w.sum()), len(s), units.drive_id.nunique(), float(w.sum())


def main():
    rec_gt = {}
    for line in open(f"{R.P1C}/mech/action_history_causal/records.jsonl"):
        x = json.loads(line); rec_gt[x["token"]] = x["trajectory_gt"]
    # ---------------- P1-C RN
    p1c = rows_ahc(f"{R.P1C}/mech/action_history_causal/records.jsonl", frame="P1C")
    U = pd.read_parquet(f"{R.P1C}/analysis/units.parquet")
    U = U[(U.block == "H7") & U.condition.isin(["normal", "recent_gt"])]
    chk = p1c.merge(U[["unit_id", "condition", "amplification", "fde5", "stratum", "weight", "exclusion_reason"]],
                    left_on=["unit_id", "cond"], right_on=["unit_id", "condition"], how="inner", suffixes=("", "_stored"))
    checks = {"P1C_units_matched": int(len(chk)), "P1C_severe3_eq_stored": int((chk.severe3 == chk.amplification).sum()),
              "P1C_fde_maxabs": float((chk.fde5 - chk.fde5_stored).abs().max()),
              "P1C_records_rows": int(len(p1c)), "P1C_stored_rows": int(len(U))}
    p1c = p1c.merge(U[["unit_id", "condition", "stratum", "weight"]].rename(columns={"condition": "cond"}),
                    on=["unit_id", "cond"], how="inner")
    # ---------------- dev RN (A- stratum = failure-selected)
    dev = rows_ahc(f"{R.DEV_AHC}/records.jsonl", frame="dev")
    dev["stratum"] = dev.group
    # ---------------- B2
    b2 = rows_b2(rec_gt)
    checks["B2_severe3_eq_stored"] = int((b2.severe3 == b2.stored_severe).sum()); checks["B2_rows"] = int(len(b2))
    checks["B2_fde_maxabs"] = float((b2.fde5 - b2.stored_fde5).abs().max())
    allu = pd.concat([dev, p1c, b2], ignore_index=True)
    allu.to_parquet(f"{R.OUT}/data/A_unit_metrics.parquet", index=False)
    specs = [("RN", "recent_gt", "normal", "dev mechanism set A- (failure-selected)", dev[dev.stratum == "A-"], None),
             ("RN", "recent_gt", "normal", "P1-C F", p1c[p1c.stratum == "F"], None),
             ("RN", "recent_gt", "normal", "P1-C E (Hajek)", p1c, "weight"),
             ("C_GT", "GT_NSRC", "GT_REF", "B2 F", b2[b2.stratum == "F"], None),
             ("C_GT", "GT_NSRC", "GT_REF", "B2 E4 (Hajek)", b2, "weight"),
             ("C_NAT", "NAT_NSRC", "NAT_REF", "B2 F", b2[b2.stratum == "F"], None),
             ("C_NAT", "NAT_NSRC", "NAT_REF", "B2 E4 (Hajek)", b2, "weight")]
    rows = []
    for con, a, b, frame, df, w in specs:
        D, A, B, na, nb, nc = contrast(df, a, b)
        D["weight"] = df[df.cond == a].set_index("unit_id").loc[D.index, "weight"] if w else 1.0
        Aw = A.join(D[["scene_id", "drive_id", "weight"]]); Bw = B.join(D[["scene_id", "drive_id", "weight"]])
        est, ns, nd, wsum = aggregate(D, "weight" if w else None)
        ea, _, _, _ = aggregate(Aw, "weight" if w else None)
        eb, _, _, _ = aggregate(Bw, "weight" if w else None)
        for m in MET:
            unit = "m" if m in ("ade5", "fde5") else "pp of units (scene-weighted)"
            sc = 1.0 if unit == "m" else 100.0
            rows.append({"analysis_id": f"A2.{con}.{frame}.{m}", "analysis_status": "post_hoc/exploratory",
                         "contrast": con, "intervention": a, "reference": b, "target_population": frame,
                         "metric": m, "unit": unit, "horizon": "5 s (10 poses x 0.5 s)",
                         "threshold": (m[-1] + " m") if m.startswith(("fde_gt", "severe")) else ("3 m" if m.startswith("state") else "NA"),
                         "n_total_intervention": na, "n_total_reference": nb, "n_complete": nc, "n_scenes": ns, "n_clusters": nd,
                         "cluster_unit": "drive", "weighting_method": "Hajek (stored E/E4 weights)" if w else "equal scene weight",
                         "sum_weights": wsum, "estimate_condition": sc * ea[m], "estimate_reference": sc * eb[m],
                         "delta_intervention_minus_reference": sc * est[m],
                         "interval_low": "NA", "interval_high": "NA",
                         "interval_method/status": "not computed (descriptive decomposition; existing interval policy applies to severe3 only)",
                         "p_raw": "NA", "p_adjusted": "NA", "multiplicity_family": "none (exploratory)",
                         "source_file": "data/A_unit_metrics.parquet"})
    T = pd.DataFrame(rows)
    T.to_csv(f"{R.OUT}/tables/A2_metric_decomposition.csv", index=False)
    T[T.metric.str.startswith(("severe", "fde_gt"))].to_csv(f"{R.OUT}/tables/A3_threshold_sensitivity.csv", index=False)
    # reproduction of original severe3 estimates
    ref = {("RN", "dev mechanism set A- (failure-selected)"): -39.789377, ("RN", "P1-C F"): -34.213908,
           ("RN", "P1-C E (Hajek)"): -4.125463, ("C_GT", "B2 F"): 25.305556, ("C_GT", "B2 E4 (Hajek)"): 3.314256,
           ("C_NAT", "B2 F"): 25.519841, ("C_NAT", "B2 E4 (Hajek)"): 3.976535}
    rep = []
    for (c, f), v in ref.items():
        mine = T[(T.contrast == c) & (T.target_population == f) & (T.metric == "severe3")].delta_intervention_minus_reference.iloc[0]
        rep.append({"contrast": c, "frame": f, "this_reanalysis_pp": mine, "stored_pp": v, "abs_diff_pp": abs(mine - v)})
    pd.DataFrame(rep).to_csv(f"{R.OUT}/tables/A_reproduction_check.csv", index=False)
    # rule failure reasons (GT class vs predicted class) for normal / reference arms
    cls = allu.groupby(["frame", "cond", "gt_lon", "pred_lon"]).size().rename("n").reset_index()
    cls.to_csv(f"{R.OUT}/tables/A1_lon_class_confusion.csv", index=False)
    json.dump(checks, open(f"{R.OUT}/logs/A_checks.json", "w"), indent=1)
    print(json.dumps(checks, indent=1)); print(pd.DataFrame(rep).to_string())
    figure(T)


def figure(T):
    plt = R.mpl()
    frames = ["dev mechanism set A- (failure-selected)", "P1-C F", "P1-C E (Hajek)", "B2 F", "B2 E4 (Hajek)"]
    short = {"dev mechanism set A- (failure-selected)": "dev A−", "P1-C F": "P1-C F", "P1-C E (Hajek)": "P1-C E",
             "B2 F": "B2 F", "B2 E4 (Hajek)": "B2 E4"}
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4))
    keys = [("RN", f) for f in frames[:3]] + [("C_GT", f) for f in frames[3:]] + [("C_NAT", f) for f in frames[3:]]
    lab = [f"{c} {short[f]}" for c, f in keys]
    y = np.arange(len(keys))
    for i, (c, f) in enumerate(keys):
        d = T[(T.contrast == c) & (T.target_population == f)].set_index("metric").delta_intervention_minus_reference
        col = R.PAL[1] if "E" in short[f][-2:] else R.PAL[0]
        axs[0].plot([d["fde5"]], [i], "o", color=col, ms=4)
        for j, t in enumerate([2, 3, 4, 5]):
            axs[1].plot([d[f"severe{t}"]], [i + (j - 1.5) * 0.17], marker="os^D"[j], color=R.PAL[j + 2] if j != 1 else R.INK, ms=3, ls="none",
                        label=f"τ = {t} m" if i == 0 else None)
        axs[2].barh(i, d["rule_fail"], height=0.35, color=R.PAL[6], label="rule failure" if i == 0 else None)
        axs[2].barh(i + 0.36, d["fde_gt3"], height=0.35, color=R.PAL[3], label="FDE5 > 3 m" if i == 0 else None)
    for a in axs:
        a.axvline(0, color=R.INK2, lw=0.6); a.grid(True, axis="x"); a.set_yticks(y); a.invert_yaxis()
    axs[0].set_yticklabels(lab, fontsize=6); axs[1].set_yticklabels([]); axs[2].set_yticklabels([])
    axs[0].set_xlabel("Δ mean FDE5 (m)"); axs[1].set_xlabel("Δ severe_τ (pp)"); axs[2].set_xlabel("Δ component rate (pp)")
    axs[0].set_title("Continuous error", loc="left"); axs[1].set_title("Composite event by τ", loc="left")
    axs[2].set_title("Components", loc="left")
    axs[1].legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    axs[2].legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.text(0.01, -0.13, "Δ = intervention − reference (original contrast sign). F/dev = failure-selected strata; E/E4 = design-weighted "
             "frames (different targets and eligibility). Post-hoc descriptive; no intervals.", fontsize=5.6, color=R.INK2)
    fig.tight_layout()
    R.save(fig, "A_metric_decomposition")


if __name__ == "__main__":
    main()
