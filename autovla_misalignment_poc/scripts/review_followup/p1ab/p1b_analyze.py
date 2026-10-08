#!/usr/bin/env python
"""
P1-B analysis (CPU only). Written before the main run (preregistered with protocol.md).

PRIMARY (total horizon fixed = 10 tokens, same units, same correction count):
  population  A-, common subset t* <= 2 (every window s in 1..4, w in 1..3 fits in context positions <= 8)
  contrast    for w = 1, 2, 3:  amplification(P_gt_s1_w) - amplification(P_gt_s3_w)   (early - late)
  estimand    scene-weighted, log clusters; log block bootstrap 95% CI; log-level sign-flip; Holm m = 3
SECONDARY (reported separately, never pooled with the primary):
  - FDE5 and severe thresholds 2/3/4 m for the same contrasts; full start x length map (s = 1..4, w = 1..3)
  - transient (sliding) windows T_gt_*; self-reference families P_self_* / T_self_* (outcome = deviation vs
    the fixed model reference dev9_ref and severe_ref 2/3/4 m, plus GT-relative amplification)
  - FIXED FREE HORIZON after release (extension of exp 10, 10-token in-distribution): release pose
    rel = t* + s + w; evaluated poses rel+1 .. rel+N; t* <= 1 with N = 2 (s in 1..3), sensitivity t* = 0 with N = 3;
    effect of a window = metric(window) - metric(normal) on the SAME poses; early - late of that effect.
    Re-divergence: among units with GT error <= 1 m at the release pose (stab), share with error > tau at rel+N
    (denominator = stab units, reported per condition).
  - A+ and pooled groups.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402
import p1_stats as ST  # noqa: E402
from analyze_action_history import unit_metrics, a_eval  # noqa: E402

EARLY, LATE = 1, 3
WS = (1, 2, 3)
MARGIN_AMP = 0.10
MARGIN_FDE = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(args.raw, "records.jsonl"))]
    rows, free_rows, curves = [], [], defaultdict(list)
    lock = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], np.asarray(r["trajectory_gt"], float)[:, :2]
        reft = np.asarray(r["ref_traj_xyh"], float)[:, :2]
        lock.append(r["checks"]["output_prefix_locked_all_rows"])
        E, D = {}, {}
        for nm, x in r["rows"].items():
            tr = np.asarray(x["traj_xyh"], float)[:, :2]
            m = unit_metrics(x["action_idx"], tr.tolist(), x["ent"], gt, gtraj.tolist(), t)
            e = np.linalg.norm(tr - gtraj, axis=1); dev = np.linalg.norm(tr - reft, axis=1)
            E[nm], D[nm] = e, dev
            refAm = not a_eval(tr.tolist(), reft.tolist())
            w = x["window"] or {}
            rows.append({
                "model": "AutoVLA", "experiment": "P1B_start_position", "unit_id": r["unit_id"], "log_id": r["log"], "scene_id": r["token"],
                "task_id": None, "episode_id": None, "seed": r["seed"], "group": r["group"], "t_star": t, "perturbation": r["perturbation"],
                "condition": nm, "decode_j": t, "control_t": None,
                "reference_branch": ("gt" if "_gt" in nm else "model_reference" if "_self" in nm else None),
                "window_kind": nm.split("_")[0] if nm[:2] in ("P_", "T_") else nm, "s": w.get("s"), "w": w.get("w"),
                "perturbed_span": f"[{t},{t}]", "corrected_positions": json.dumps(w.get("positions")) if w else None,
                "window_truncated": w.get("truncated"), "release_pose": w.get("release_pose"),
                "perturbation_requested": None, "perturbation_achieved": None,
                "intervention_count": x["n_corrected_forwards"], "corrected_positions_n": x["n_corrected_positions"],
                "corrected_pairs": x["n_corrected_pairs"], "corrected_pairs_effective": x["n_corrected_pairs_effective"],
                "parser_status": "ok", "exclusion_reason": None, "primary_outcome": float(m["amplification"]),
                "amplification": float(m["amplification"]), "recovery": float(m["recovery"]), "fde5": m["fde5"], "ade5": m["ade5"],
                "amp2": float((not m["recovery"]) and e[-1] > 2.0), "amp3": float(m["amplification"]), "amp4": float((not m["recovery"]) and e[-1] > 4.0),
                "dev9_ref": float(dev[-1]), "severe_ref2": float(dev[-1] > 2), "severe_ref3": float(dev[-1] > 3), "severe_ref4": float(dev[-1] > 4),
                "ref_amp3": float(refAm and dev[-1] > 3), "entropy": m["entropy"],
                "err_gt_curve": json.dumps(np.round(e, 4).tolist()), "dev_ref_curve": json.dumps(np.round(dev, 4).tolist()),
            })
            for i in range(C.N_ACT):
                curves[(r["group"], t <= 2, nm, i)].append((r["log"], r["token"], float(e[i]), float(dev[i])))
        # fixed free horizon (secondary)
        for nm, x in r["rows"].items():
            w = x["window"]
            if not w or w["s"] > 3:
                continue
            rel = w["release_pose"]
            for N, cond_t in ((2, t <= 1), (3, t == 0)):
                if not cond_t or rel + N > C.N_ACT - 1:
                    continue
                src = "gt" if "_gt" in nm else "self"
                for tgt, A in (("gt", E), ("ref", D)):
                    fr = {"unit_id": r["unit_id"], "log_id": r["log"], "scene_id": r["token"], "group": r["group"], "t_star": t,
                          "condition": nm, "kind": nm[0], "src": src, "s": w["s"], "w": w["w"], "N": N, "target": tgt,
                          "release_pose": rel, "eval_pose": rel + N,
                          "err_free": float(A[nm][rel + N]), "err_free_normal_same_pose": float(A["normal"][rel + N]),
                          "full_same_pose": float(A[f"full_{src}"][rel + N]),
                          "stab": float(A[nm][rel] <= 1.0), "err_at_release": float(A[nm][rel])}
                    for tau in (1, 2, 3, 4):
                        fr[f"exceed{tau}"] = float(fr["err_free"] > tau)
                        fr[f"exceed{tau}_normal"] = float(fr["err_free_normal_same_pose"] > tau)
                        fr[f"rediverge{tau}"] = float(fr["stab"] and fr["err_free"] > tau) if fr["stab"] else np.nan
                    free_rows.append(fr)
    U = pd.DataFrame(rows)
    U.to_parquet(os.path.join(args.out, "units.parquet"), index=False)
    F = pd.DataFrame(free_rows)
    F.to_parquet(os.path.join(args.out, "free_horizon_units.parquet"), index=False)

    eff, loo = [], []
    seed = ST.SEED
    common = U.t_star <= 2
    # ---- primary
    for w in WS:
        df = ST.paired_frame(U, f"P_gt_s{EARLY}_w{w}", f"P_gt_s{LATE}_w{w}", "amplification", (U.group == "A-") & common)
        if len(df):
            eff.append(ST.effect_row("P1B_primary", f"early(s={EARLY}) - late(s={LATE}), persistent GT, w={w}", "A-", "amplification",
                                     df, seed, margin=MARGIN_AMP, label="two-sided; negative = early corrections leave less amplification"))
            loo += ST.loo_rows("P1B_primary", f"early-late w={w}", "A-", "amplification", df, seed)
        seed += 7
    ST.add_holm(eff, "P1B_primary")
    # ---- secondary: total-horizon-fixed families
    for grp in ("A-", "A+", "all"):
        gm = (U.group == grp) if grp != "all" else pd.Series(True, index=U.index)
        for kind in ("P", "T"):
            for src in ("gt", "self"):
                mets = ["amplification", "amp2", "amp4", "fde5"] if src == "gt" else ["dev9_ref", "severe_ref2", "severe_ref3", "severe_ref4", "amplification"]
                for w in WS:
                    for s_late in (2, 3, 4):
                        for met in mets:
                            if (grp, kind, src, s_late, met) == ("A-", "P", "gt", LATE, "amplification"):
                                continue                   # = primary
                            df = ST.paired_frame(U, f"{kind}_{src}_s{EARLY}_w{w}", f"{kind}_{src}_s{s_late}_w{w}", met, gm & common)
                            if len(df):
                                eff.append(ST.effect_row(f"P1B_secondary_totalH_{kind}_{src}_{grp}",
                                                         f"early(s=1) - late(s={s_late}), w={w}", grp, met, df, seed,
                                                         margin=(MARGIN_FDE if met in ("fde5", "dev9_ref") else MARGIN_AMP)))
                            seed += 7
                # map: each window vs normal
                for s in (1, 2, 3, 4):
                    for w in WS:
                        met = "amplification" if src == "gt" else "dev9_ref"
                        df = ST.paired_frame(U, f"{kind}_{src}_s{s}_w{w}", "normal", met, gm & common)
                        if len(df):
                            eff.append(ST.effect_row(f"P1B_map_{kind}_{src}_{grp}", f"window(s={s},w={w}) - normal", grp, met, df, seed))
                        seed += 7
    # ---- secondary: fixed free horizon
    if len(F):
        for (N, tgt, kind, src, grp), g in F.groupby(["N", "target", "kind", "src", "group"]):
            g = g.copy()
            g["eff"] = g["err_free"] - g["err_free_normal_same_pose"]
            for tau in (2, 3, 4):
                g[f"eff_exceed{tau}"] = g[f"exceed{tau}"] - g[f"exceed{tau}_normal"]
            for w in WS:
                a = g[(g.s == EARLY) & (g.w == w)].set_index("unit_id")
                b = g[(g.s == LATE) & (g.w == w)].set_index("unit_id")
                for met in ("eff", "eff_exceed2", "eff_exceed3", "eff_exceed4"):
                    j = a[["scene_id", "log_id", met]].join(b[[met]], rsuffix="_b", how="inner").dropna()
                    if not len(j):
                        continue
                    df = pd.DataFrame({"log": j.log_id, "scene": j.scene_id, "d": j[met] - j[f"{met}_b"]})
                    df.attrs["dropped_nan"] = 0; df.attrs["missing_pairs"] = int(len(set(a.index) ^ set(b.index)))
                    eff.append(ST.effect_row(f"P1B_secondary_freeH_N{N}_{tgt}_{kind}_{src}_{grp}",
                                             f"early(s=1) - late(s=3) of (window - normal) on same poses, w={w}", grp, met, df, seed,
                                             label=f"release-aligned, N={N} free poses, target={tgt}"))
                    seed += 7
        # re-divergence table with denominators
        red = (F.groupby(["N", "target", "kind", "src", "group", "s", "w"])
                .agg(n_units=("unit_id", "size"), n_stab=("stab", "sum"), rediverge2=("rediverge2", "mean"),
                     rediverge3=("rediverge3", "mean"), mean_err_free=("err_free", "mean")).reset_index())
        red.to_csv(os.path.join(args.out, "free_horizon_rediverge.csv"), index=False)
    pd.DataFrame(eff).to_csv(os.path.join(args.out, "paired_effects.csv"), index=False)
    pd.DataFrame(loo).to_csv(os.path.join(args.out, "loo_logs.csv"), index=False)

    # ---- condition summary (start x length map levels)
    cs = []
    for grp in ("A-", "A+"):
        for sub, smask in (("common_tstar_le2", common), ("all_units", pd.Series(True, index=U.index))):
            G = U[(U.group == grp) & smask]
            for cond, g in G.groupby("condition"):
                for met in ("amplification", "amp2", "amp4", "fde5", "dev9_ref", "severe_ref3"):
                    df = g[["log_id", "scene_id", met]].rename(columns={"log_id": "log", "scene_id": "scene", met: "d"}).dropna()
                    res = ST.log_cluster_effects(df, with_tests=False, b=2000)
                    cs.append({"group": grp, "subset": sub, "condition": cond, "metric": met, "scene_mean": res["scene"]["estimate"],
                               "ci95_lo": res["scene"]["ci95"][0], "ci95_hi": res["scene"]["ci95"][1], "n_units": res["n_units"],
                               "n_scenes": res["n_scenes"], "n_logs": res["n_clusters"],
                               "n_truncated_window": int(sum(bool(v) for v in g["window_truncated"] if v is not None and v == v))})
    pd.DataFrame(cs).to_csv(os.path.join(args.out, "condition_summary.csv"), index=False)
    cv = []
    for (grp, com, nm, i), items in sorted(curves.items()):
        df = pd.DataFrame(items, columns=["log", "scene", "e", "dev"])
        sm = df.groupby(["log", "scene"])[["e", "dev"]].mean()
        cv.append({"group": grp, "common_tstar_le2": com, "condition": nm, "pose": i, "scene_mean_err_gt": float(sm.e.mean()),
                   "scene_mean_dev_ref": float(sm.dev.mean()), "n_scenes": len(sm)})
    pd.DataFrame(cv).to_csv(os.path.join(args.out, "deviation_curves.csv"), index=False)
    tv = {"n_units": len(recs), "output_prefix_locked_all_rows": float(np.mean(lock)) if lock else None}
    json.dump(tv, open(os.path.join(args.out, "trace_verification.json"), "w"), indent=1)
    print(json.dumps(tv)); print(f"[done] units={len(recs)} rows={len(U)} free_rows={len(F)} effects={len(eff)}")


if __name__ == "__main__":
    main()
