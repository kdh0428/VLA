#!/usr/bin/env python
"""
P0-A (review_followup3, 2026-10-14) -- POST-HOC re-analysis of existing data. CPU only, no model inference.
Never writes outside outputs/review_followup3/P0/.

1. Natural selection flow of the dev mechanism set: outputs/full_extract/records.jsonl (arm N, natural fast path,
   navtest shards 0-5) -> 2,747 non-CoT scenes -> first token mismatch -> A-/A+ (dev a_eval rule) -> 52 A- (all) and
   156 A+ (3 per A- at the same t*, random.Random(0) shuffle of the token-sorted pool; exact replica of
   scripts/equal_distance_perturbation.py main()). The replica is checked against equal_distance_perturbation/records.jsonl.
   Counts of scenes / logs / drives / days / vehicles / day-vehicles at every step.
2. Baseline A- prevalence (and severe = A- and FDE5 > 3 m, the dev "amplification" rule applied to the natural plan)
   with cluster-bootstrap CIs at drive (primary), day-vehicle, log level; Wilson (unclustered) for reference.
   Same for the P1-C stage-1 natural pass (5,671 scenes, 64 logs, 24 drives).
3. P1-C denominators per contrast x estimand (F, S, P): scenes, units, logs, drives, informative drives,
   day-vehicles, days, vehicles; harness exclusions; invalid outputs.
4. RN / RV / DM in F, S (= A+ sample), P (Hajek, the original estimand) -- computed with the FROZEN analyze_p1c functions
   (seed 20261008) and asserted equal to decisions.json; S gets sign-flip inversion CIs here (post-hoc; S was a
   secondary bootstrap-only row in the original). Plus an unweighted analysed-sample row (NOT a population estimand).
   Dev cohort (exp 7 RN, R8 RV, exp 32 DM): A-, A+, unweighted 208-scene sample and a t*-matched Hajek reweighting
   (post-hoc) using the known A+ sampling probabilities.
5. Per-condition event counts / evaluated units and raw unit rates for every row.
No prevalence x effect products are formed anywhere.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = os.path.join(POC, "outputs/review_followup3/P0")
RF = os.path.join(POC, "outputs/review_followup")
sys.path.insert(0, os.path.join(POC, "scripts/review_followup2"))
sys.path.insert(0, os.path.join(POC, "scripts/review_followup/p1c"))
sys.path.insert(0, os.path.join(POC, "scripts"))
import stats_core as SC  # noqa: E402
import analyze_p1c as AP  # noqa: E402  (frozen P1-C analysis; imported, not modified)
from analyze_action_history import a_eval  # noqa: E402  (dev A+/A- rule)

SEED = 20261014
B = 10_000
N_ACT = 10
AMP_FDE = 3.0
FULL_EXTRACT = os.path.join(POC, "outputs/full_extract/records.jsonl")
EQD = os.path.join(POC, "outputs/equal_distance_perturbation/records.jsonl")
P1C = os.path.join(RF, "P1C_new_log_replication")
P1C_UNITS = os.path.join(P1C, "run_20261008/analysis/units.parquet")
P1C_DEC = os.path.join(P1C, "run_20261008/analysis/decisions.json")
P1C_STAGE1 = os.path.join(P1C, "run_20261008/stage1/records.jsonl")
P1C_SPLIT = os.path.join(P1C, "design_20261008/split.csv")
DEV_UNITS = os.path.join(RF, "P0B_cluster_stats/run_20261008/units.parquet")
R8_UNITS = os.path.join(RF, "R8R9_raw_recovery/run_20261008/units.parquet")


def ids(log):
    """log segment name -> drive, day, vehicle, day_vehicle (drive = first two fields)."""
    f = log.split("_")
    drive = f[0] + "_" + f[1]
    day = f[0][:10]
    veh = f[1]
    return drive, day, veh, day + "_" + veh


def add_ids(df, logcol="log"):
    x = df[logcol].map(ids)
    df["drive"] = [a[0] for a in x]
    df["day"] = [a[1] for a in x]
    df["vehicle"] = [a[2] for a in x]
    df["day_vehicle"] = [a[3] for a in x]
    return df


def counts(df, label, **extra):
    return {"step": label, "n_scenes": int(len(df)), "n_logs": int(df.log.nunique()) if len(df) else 0,
            "n_drives": int(df.drive.nunique()) if len(df) else 0, "n_days": int(df.day.nunique()) if len(df) else 0,
            "n_vehicles": int(df.vehicle.nunique()) if len(df) else 0,
            "n_day_vehicles": int(df.day_vehicle.nunique()) if len(df) else 0, **extra}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def prevalence(df, col, cohort, denom_label):
    rows = []
    k, n = int(df[col].sum()), len(df)
    for lvl in ("drive", "day_vehicle", "log", "day", "vehicle"):
        g = df.groupby(lvl)[col].agg(["sum", "size"])
        N, D = g["sum"].values.astype(float), g["size"].values.astype(float)
        bs = SC.cluster_bootstrap(N, D, B, SEED)
        ci = SC.pct_ci(bs)
        _, clt = SC.cluster_t(N, D)
        rows.append({"cohort": cohort, "outcome": col, "denominator": denom_label, "events": k, "n_scenes": n,
                     "prevalence": k / n if n else float("nan"), "cluster_level": lvl, "n_clusters": len(g),
                     "n_clusters_with_event": int((g["sum"] > 0).sum()),
                     "boot_ci95_lo": ci[0], "boot_ci95_hi": ci[1], "clt_ci95_lo": clt[0], "clt_ci95_hi": clt[1],
                     "wilson_unclustered_lo": wilson(k, n)[0], "wilson_unclustered_hi": wilson(k, n)[1],
                     "B": B, "seed": SEED, "max_cluster_share_of_events": float(g["sum"].max() / k) if k else float("nan")})
    return rows


# ====================================================================================================== 1. dev flow
def dev_natural():
    rows = []
    for line in open(FULL_EXTRACT):
        if not line.strip():
            continue
        r = json.loads(line)
        n = r["arms"]["N"]
        traj, gtraj = n["trajectory_pred"], r["trajectory_gt"]
        pred, gt = n["pred_action_idx"][:N_ACT], r["gt_action_idx"][:N_ACT]
        valid = len(traj) >= N_ACT and len(pred) >= N_ACT
        t = next((k for k in range(N_ACT) if pred[k] != gt[k]), None) if valid else None
        lab = None
        fde5 = float("nan")
        if valid:
            lab = "A+" if a_eval(traj[:N_ACT], gtraj) else "A-"
            fde5 = float(np.linalg.norm(np.asarray(traj, float)[N_ACT - 1, :2] - np.asarray(gtraj, float)[N_ACT - 1, :2]))
        if n["cot_present"]:
            excl = "cot_present"
        elif n["runaway_action_tokens"]:
            excl = "runaway"
        elif not valid:
            excl = "lt_10_poses"
        elif t is None:
            excl = "no_first_mismatch"
        else:
            excl = None
        rows.append({"token": r["token"], "log": r["log_name"], "cot_present": bool(n["cot_present"]),
                     "runaway": bool(n["runaway_action_tokens"]), "valid10": valid, "t_star": t, "a_label": lab,
                     "fde5": fde5, "excl_selection": excl, "instruction": r.get("instruction"),
                     "map_name": r.get("map_name")})
    df = add_ids(pd.DataFrame(rows))
    df["Aminus"] = (df.a_label == "A-").astype(int)
    df["severe"] = ((df.a_label == "A-") & (df.fde5 > AMP_FDE)).astype(int)
    # exact replica of the equal_distance_perturbation selection
    neg = df[(df.excl_selection.isna()) & (df.a_label == "A-")].sort_values("token")
    pos_by_t = defaultdict(list)
    for r in df[(df.excl_selection.isna()) & (df.a_label == "A+")].itertuples():
        pos_by_t[int(r.t_star)].append(r.token)
    rng = random.Random(0)
    need = Counter()
    for t in neg.t_star:
        need[int(t)] += 3
    picked_plus, pool_size = [], {}
    for t, k in sorted(need.items()):
        pool = sorted(pos_by_t[t])
        rng.shuffle(pool)
        picked_plus += pool[:k]
        pool_size[t] = (len(pos_by_t[t]), k, min(k, len(pos_by_t[t])))
    df["selected"] = None
    df.loc[df.token.isin(set(neg.token)), "selected"] = "A- (all)"
    df.loc[df.token.isin(set(picked_plus)), "selected"] = "A+ (3 per A- at same t*)"
    # check vs stored run
    eq = pd.DataFrame([json.loads(l) for l in open(EQD) if l.strip()])[["token", "group", "t_star", "log"]]
    m = eq.merge(df[["token", "a_label", "t_star", "selected"]], on="token", how="left", suffixes=("_eqd", ""))
    check = {"n_eqd": int(len(eq)), "n_eqd_Aminus": int((eq.group == "A-").sum()), "n_eqd_Aplus": int((eq.group == "A+").sum()),
             "all_eqd_tokens_in_full_extract": bool(m.a_label.notna().all()),
             "group_label_agrees": int((m.group == m.a_label).sum()),
             "t_star_agrees": int((m.t_star_eqd == m.t_star).sum()),
             "replica_selected_equals_eqd_set": bool(set(df[df.selected.notna()].token) == set(eq.token)),
             "n_replica_selected": int(df.selected.notna().sum())}
    return df, pool_size, check


def dev_flow(df, pool_size):
    fl = []
    fl.append(counts(df, "0 full_extract records (arm N, navtest shards 0-5)"))
    nc = df[~df.cot_present]
    fl.append(counts(nc, "1 non-CoT natural scenes (= the '2,747 natural scenes')"))
    fl.append(counts(df[df.cot_present], "1x excluded: CoT stub present"))
    fl.append(counts(nc[nc.runaway | ~nc.valid10], "1x excluded: runaway / fewer than 10 poses"))
    el = nc[~nc.runaway & nc.valid10]
    fl.append(counts(el, "2 valid natural plans"))
    for lab in ("A-", "A+"):
        s = el[el.a_label == lab]
        fl.append(counts(s, f"2{lab} natural {lab} (all t*)"))
    fl.append(counts(el[el.severe == 1], "2A-s natural A- and FDE5 > 3 m (severe)"))
    fl.append(counts(el[(el.a_label == "A-") & (el.severe == 0)], "2A-n natural A- and FDE5 <= 3 m (non-severe)"))
    fl.append(counts(el[el.t_star.isna()], "3x no first token mismatch (not eligible for perturbation)"))
    fl.append(counts(el[el.t_star.isna() & (el.a_label == "A-")], "3x  of which A-"))
    mm = el[el.t_star.notna()]
    fl.append(counts(mm, "3 first token mismatch exists (selection-eligible)"))
    for lab in ("A-", "A+"):
        fl.append(counts(mm[mm.a_label == lab], f"3{lab} mismatch and {lab}"))
    fl.append(counts(mm[mm.severe == 1], "3A-s mismatch and A- and FDE5 > 3 m"))
    sup = set(pool_size)
    fl.append(counts(mm[(mm.a_label == "A+") & mm.t_star.isin(sup)], "4 A+ pool at t* values present among A- (sampling frame)"))
    fl.append(counts(mm[(mm.a_label == "A+") & ~mm.t_star.isin(sup)], "4x A+ at t* values with no A- (inclusion prob. 0)"))
    fl.append(counts(df[df.selected == "A- (all)"], "5 selected A- (all, pi = 1)"))
    fl.append(counts(df[df.selected == "A+ (3 per A- at same t*)"], "5 selected A+ (3 per A-, same t*, Random(0))"))
    fl.append(counts(df[df.selected.notna()], "5 selected total (mechanism set)"))
    return pd.DataFrame(fl)


# ====================================================================================================== 2. P1-C stage 1
def p1c_stage1():
    rows = []
    for l in open(P1C_STAGE1):
        if not l.strip():
            continue
        r = json.loads(l)
        p = r["p1c"]
        rows.append({"token": r["token"], "log": r["log_name"], "t_star": p["t_star"], "a_label": p["a_eval"],
                     "fde5": p["fde5"], "eligible": p["eligible"], "exclusion_reason": p["exclusion_reason"],
                     "instruction": r.get("instruction"), "seed": p.get("seed")})
    df = add_ids(pd.DataFrame(rows))
    df["Aminus"] = (df.a_label == "A-").astype(int)
    df["severe"] = ((df.a_label == "A-") & (df.fde5 > AMP_FDE)).astype(int)
    return df


# ====================================================================================================== 3/4. effects
def unit_counts(u, a, b, metric="amplification"):
    A = u[u.condition == a]
    Bb = u[u.condition == b]
    return {"events_a": int(A[metric].sum()), "n_a": int(len(A)), "events_b": int(Bb[metric].sum()), "n_b": int(len(Bb)),
            "raw_rate_a": float(A[metric].mean()) if len(A) else float("nan"),
            "raw_rate_b": float(Bb[metric].mean()) if len(Bb) else float("nan"),
            "n_invalid_a": int((A.get("parser_status", pd.Series(dtype=str)) != "ok").sum()) if "parser_status" in A else 0,
            "n_invalid_b": int((Bb.get("parser_status", pd.Series(dtype=str)) != "ok").sum()) if "parser_status" in Bb else 0}


def struct_counts(sc, logcol="log"):
    x = add_ids(pd.DataFrame({"log": sc[logcol].values}))
    return {"n_logs": int(x.log.nunique()), "n_drives": int(x.drive.nunique()), "n_day_vehicles": int(x.day_vehicle.nunique()),
            "n_days": int(x.day.nunique()), "n_vehicles": int(x.vehicle.nunique())}


def weighted_level(sc_lvl):
    return float((sc_lvl.w * sc_lvl.d).sum() / sc_lvl.w.sum()) if len(sc_lvl) else float("nan")


P1C_SPECS = [("RN", "H7", "recent_gt", "normal", "Recent-GT - Normal"),
             ("RV", "H8", "reverse@emb", "gt_history", "Reverse(@emb) - Full(GT-history)"),
             ("DM", "H32", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "Direction - Magnitude")]


def p1c_effects():
    ut = pd.read_parquet(P1C_UNITS)
    dec = {(d["contrast"], d["estimand"]): d for d in json.load(open(P1C_DEC))["decisions"]}
    rows, den = [], []
    for cid, blk, a, b, name in P1C_SPECS:
        sc = AP.scene_table(ut, blk, a, b, "amplification")
        la = AP.level_table(ut, blk, a)
        lb = AP.level_table(ut, blk, b)
        ub = ut[(ut.block == blk) & ut.exclusion_reason.isna()]
        excl = ut[(ut.block == blk) & ut.exclusion_reason.notna()].scene_id.nunique()
        for estd, label in (("F", "F = natural A- (failure stratum, all, pi=1)"),
                            ("S", "S = natural A+ (sampled 4/segment, pi = m/N)"),
                            ("P", "P = F u S, Hajek w = 1/pi (original population estimand)"),
                            ("ALL_unweighted", "analysed sample F u S, scenes equally weighted (NOT a population estimand)")):
            if estd == "ALL_unweighted":
                s = sc.copy(); s["w"] = 1.0
                lA, lB = la.copy(), lb.copy(); lA["w"] = 1.0; lB["w"] = 1.0
                us = ub
            else:
                s = AP.estimand_table(sc, estd)
                lA, lB = AP.estimand_table(la, estd), AP.estimand_table(lb, estd)
                us = ub if estd == "P" else ub[ub.stratum == estd]
            r = AP.estimate(s, "scene", AP.SEED, with_ci_inversion=True)
            # swap test: b - a must be exactly -theta, same p
            s_sw = s.copy(); s_sw["d"] = -s_sw["d"]; s_sw["dsum"] = -s_sw["dsum"]
            r_sw = AP.estimate(s_sw, "scene", AP.SEED, boot=False)
            assert abs(r_sw["theta"] + r["theta"]) < 1e-12 and abs(r_sw["p_signflip"] - r["p_signflip"]) < 1e-12
            row = {"cohort": "P1-C new-log replication (64 logs / 24 drives; preregistered b212637+amendments)",
                   "contrast": cid, "contrast_name": name, "block": blk, "cond_a": a, "cond_b": b, "metric": "amplification",
                   "estimand": estd, "estimand_definition": label, "weighting": "scene (Hajek w for P/S; w=1 for F)",
                   "cluster": "drive", **unit_counts(us, a, b), "n_units": r["n_units"], "n_scenes": r["n_scenes"],
                   **struct_counts(s), "n_clusters": r["n_clusters"], "n_informative_clusters": r.get("n_informative_clusters"),
                   "p_min_attainable": 2.0 / 2 ** r.get("n_informative_clusters", r["n_clusters"]),
                   "weighted_level_a": weighted_level(lA), "weighted_level_b": weighted_level(lB),
                   "theta": r["theta"], "boot_ci95_lo": r["boot_ci95"][0], "boot_ci95_hi": r["boot_ci95"][1],
                   "sf_ci95_lo": r["sf_ci95"][0], "sf_ci95_hi": r["sf_ci95"][1], "sf_ci90_lo": r["sf_ci90"][0],
                   "sf_ci90_hi": r["sf_ci90"][1], "clt_ci95_lo": r["clt_ci95"][0], "clt_ci95_hi": r["clt_ci95"][1],
                   "p_signflip_raw": r["p_signflip"], "swap_test_ok": True,
                   "harness_excluded_scenes_in_block": int(excl),
                   "missing_reason": "none (0 harness exclusions, 0 invalid outputs)" if excl == 0 else "harness_exception",
                   "source": f"{P1C_UNITS} via analyze_p1c.scene_table/estimate (seed {AP.SEED}, B {AP.B_BOOT})"}
            if (cid, estd) in dec:
                d = dec[(cid, estd)]
                row.update({"family_id": f"P1C_{estd} {{RN,RV,DM}}", "family_m": 3, "family_alpha": d["family_alpha"],
                            "p_holm": d["holm_p"], "decision_original": d["decision"], "status": "preregistered primary"})
                for k_ours, k_orig in (("theta", "theta"), ("p_signflip_raw", "p_signflip"), ("boot_ci95_lo", "boot_ci95_lo"),
                                       ("boot_ci95_hi", "boot_ci95_hi"), ("sf_ci95_lo", "sf_ci95_lo"), ("sf_ci95_hi", "sf_ci95_hi"),
                                       ("sf_ci90_lo", "sf_ci90_lo"), ("sf_ci90_hi", "sf_ci90_hi")):
                    assert abs(row[k_ours] - d[k_orig]) < 1e-12, (cid, estd, k_ours, row[k_ours], d[k_orig])
                assert (row["n_scenes"], row["n_units"], row["n_clusters"]) == (d["n_scenes"], d["n_units"], d["n_clusters"])
                row["reproduces_decisions_json"] = True
            else:
                row.update({"family_id": "none (secondary / post-hoc)", "family_m": None, "family_alpha": None, "p_holm": None,
                            "decision_original": None,
                            "status": "secondary theta_S in original (bootstrap only); sf CI added post-hoc" if estd == "S"
                            else "post-hoc descriptive (not a population estimand)",
                            "reproduces_decisions_json": None})
            rows.append(row)
            den.append({"contrast": cid, "estimand": estd, "n_scenes": r["n_scenes"], "n_units": r["n_units"],
                        **struct_counts(s), "n_drives_clusters": r["n_clusters"],
                        "n_informative_drives": r.get("n_informative_clusters"),
                        "drives_not_in_estimand": 24 - r["n_clusters"], "harness_excluded_scenes": int(excl),
                        "invalid_units_a": row["n_invalid_a"], "invalid_units_b": row["n_invalid_b"]})
    return pd.DataFrame(rows), pd.DataFrame(den), ut


DEV_SPECS = [("RN", "exp07_action_history_causal", "recent_gt", "normal", "Recent-GT - Normal (exp 7)", "AV-P1"),
             ("RV", "R8", "reverse@emb", "gt_history", "Reverse(@emb) - Full(GT-history) (R8 = exp 8 raw recovery)", "AV-P2"),
             ("DM", "exp32_motion_semantics", "dir_wrong_mag_ok", "dir_ok_mag_wrong", "Direction - Magnitude (exp 32)", "AV-P3")]


def dev_effects(nat, pool_size):
    u = pd.read_parquet(DEV_UNITS)
    r8 = pd.read_parquet(R8_UNITS)
    r8 = r8[(r8.experiment == "R8") & r8.exclusion_reason.isna()]
    tstar = nat.set_index("token").t_star
    # sampling weights of the t*-matched design: A- pi=1; A+ at t: pi = k_t / pool_t
    piw = {t: v[0] / v[2] for t, v in pool_size.items()}
    rows = []
    fam = {}
    for cid, exp, a, b, name, avid in DEV_SPECS:
        src = r8 if exp == "R8" else u[u.experiment == exp]
        src = src[src.exclusion_reason.isna()].copy()
        if "log" not in src:
            src["log"] = src.log_id
        P, rep = SC.pair_rows(src, ["scene_id", "perturbation"], "condition", "amplification", a, b, keep_cols=["log", "group"])
        assert rep["n_unpaired_or_missing"] == 0, rep
        P = add_ids(P)
        P["t_star"] = P.scene_id.map(tstar)
        P["w"] = np.where(P.group == "A-", 1.0, P.t_star.map(piw))
        for estd, label in (("A-", "natural A- (all 52, pi = 1)"), ("A+", "sampled A+ (156; 3 per A- at same t*)"),
                            ("ALL_unweighted", "208-scene mechanism sample, scenes equally weighted (case-enriched, 25% A-; NOT a population estimand)"),
                            ("ALL_tstar_hajek", "post-hoc Hajek reweighting to natural first-mismatch scenes with t* in the A- t* set (A+ w = pool_t / 3 per A- at t)")):
            s = P if estd.startswith("ALL") else P[P.group == estd]
            w = "w" if estd == "ALL_tstar_hajek" else None
            sc = SC.scene_table(s, "drive", "scene_id", w=w, extra=["log", "group"])
            r = SC.analyze(sc, "scene", B, SC.DEFAULT_SEED, ci_inversion=True)
            sc_sw = sc.copy(); sc_sw["dsum"] = -sc_sw["dsum"]; sc_sw["dmean"] = -sc_sw["dmean"]
            r_sw = SC.analyze(sc_sw, "scene", 10, SC.DEFAULT_SEED, ci_inversion=False)
            assert abs(r_sw["theta"] + r["theta"]) < 1e-12 and abs(r_sw["p_signflip"] - r["p_signflip"]) < 1e-12
            us = src[src.scene_id.isin(set(s.scene_id))]
            # weighted condition levels
            lv = {}
            for c, col in ((a, "a"), (b, "b")):
                g = s.groupby("scene_id").agg(v=(col, "mean"), w=("w", "first"))
                ww = g.w if w else 1.0
                lv[col] = float((g.v * ww).sum() / (ww.sum() if w else len(g)))
            row = {"cohort": "dev mechanism set (navtest shards 0-5; 52 A- + 156 A+; exploratory exp 7/8/32)",
                   "contrast": cid, "contrast_name": name, "block": exp, "cond_a": a, "cond_b": b, "metric": "amplification",
                   "estimand": estd, "estimand_definition": label,
                   "weighting": "scene (Hajek t*-matched)" if w else "scene (equal)", "cluster": "drive",
                   **unit_counts(us, a, b), "n_units": r["n_units"], "n_scenes": r["n_scenes"],
                   **struct_counts(sc), "n_clusters": r["n_clusters"], "n_informative_clusters": r["n_informative"],
                   "p_min_attainable": r["p_min_attainable"], "weighted_level_a": lv["a"], "weighted_level_b": lv["b"],
                   "theta": r["theta"], "boot_ci95_lo": r["boot_ci95"][0], "boot_ci95_hi": r["boot_ci95"][1],
                   "sf_ci95_lo": r["sf_ci95"][0], "sf_ci95_hi": r["sf_ci95"][1], "sf_ci90_lo": r["sf_ci90"][0],
                   "sf_ci90_hi": r["sf_ci90"][1], "clt_ci95_lo": r["clt_ci95"][0], "clt_ci95_hi": r["clt_ci95"][1],
                   "p_signflip_raw": r["p_signflip"], "swap_test_ok": True, "harness_excluded_scenes_in_block": 0,
                   "missing_reason": "none (0 unpaired units; exclusion_reason all null)",
                   "source": f"{DEV_UNITS if exp != 'R8' else R8_UNITS} via stats_core (seed {SC.DEFAULT_SEED}, B {B})"}
            if estd == "A-":
                fam[cid] = row
                row.update({"family_id": "AutoVLA primary {AV-P1,AV-P2,AV-P3} (P0-B / review_followup2 A3)", "family_m": 3,
                            "family_alpha": 0.05, "status": "exploratory experiment; drive-clustered post-hoc (review_followup2)"})
            else:
                row.update({"family_id": "none (secondary / post-hoc)", "family_m": None, "family_alpha": None, "p_holm": None,
                            "status": "post-hoc secondary"})
            row["decision_original"] = None
            rows.append(row)
    adj = SC.holm([fam[c]["p_signflip_raw"] for c in ("RN", "RV", "DM")])
    for c, p in zip(("RN", "RV", "DM"), adj):
        fam[c]["p_holm"] = float(p)
    # reproduce review_followup2 A3 drive-level rows
    ref = pd.read_csv(os.path.join(POC, "outputs/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv"))
    return pd.DataFrame(rows), ref


def main():
    os.makedirs(OUT, exist_ok=True)
    nat, pool_size, check = dev_natural()
    flow = dev_flow(nat, pool_size)
    flow.insert(0, "cohort", "dev natural (full_extract arm N)")
    s1 = p1c_stage1()
    fl2 = [counts(s1, "0 stage-1 natural pass, all navtest scenes of the 64 eval logs"),
           counts(s1[s1.exclusion_reason == "invalid_plan"], "0x invalid plan"),
           counts(s1[s1.exclusion_reason == "cot_stub"], "0x CoT stub"),
           counts(s1[s1.exclusion_reason == "runaway"], "0x runaway"),
           counts(s1[s1.a_label == "A-"], "1A- natural A- (all t*)"),
           counts(s1[s1.severe == 1], "1A-s natural A- and FDE5 > 3 m"),
           counts(s1[s1.exclusion_reason == "no_mismatch"], "2x no first mismatch"),
           counts(s1[(s1.exclusion_reason == "no_mismatch") & (s1.a_label == "A-")], "2x  of which A-"),
           counts(s1[s1.exclusion_reason == "t_star_gt_7"], "2x t* in {8,9}"),
           counts(s1[(s1.exclusion_reason == "t_star_gt_7") & (s1.a_label == "A-")], "2x  of which A-"),
           counts(s1[s1.eligible], "3 eligible E (t* <= 7)"),
           counts(s1[s1.eligible & (s1.a_label == "A-")], "3F eligible A- = stratum F (all perturbed)"),
           counts(s1[s1.eligible & (s1.a_label == "A+")], "3S eligible A+ = S frame")]
    sel = []
    import glob
    for f in sorted(glob.glob(os.path.join(P1C, "run_20261008/selection/shard_*.json"))):
        if os.path.basename(f).count(".") == 1:
            sel += json.load(open(f))["selected"]
    sel = add_ids(pd.DataFrame(sel))
    fl2 += [counts(sel[sel.stratum == "F"], "4F selected F"), counts(sel[sel.stratum == "S"], "4S selected S (m = 4 per segment)"),
            counts(sel, "4 selected total (analysed 353)")]
    fl2 = pd.DataFrame(fl2)
    fl2.insert(0, "cohort", "P1-C stage 1 (64 eval logs)")
    flow_all = pd.concat([flow, fl2], ignore_index=True)
    flow_all.to_csv(os.path.join(OUT, "POPULATION_selection_flow.csv"), index=False)

    # t* table for dev selection
    tt = []
    for t, (pool, need, got) in sorted(pool_size.items()):
        tt.append({"t_star": t, "n_Aminus": int(((nat.selected == "A- (all)") & (nat.t_star == t)).sum()),
                   "Aplus_pool": pool, "Aplus_needed": need, "Aplus_selected": got, "pi_Aplus": got / pool,
                   "hajek_weight_Aplus": pool / got})
    pd.DataFrame(tt).to_csv(os.path.join(OUT, "POPULATION_dev_tstar_sampling.csv"), index=False)

    # prevalence
    nc = nat[~nat.cot_present & ~nat.runaway & nat.valid10]
    prev = []
    prev += prevalence(nc, "Aminus", "dev natural (full_extract arm N)", "2,747 non-CoT valid natural plans")
    prev += prevalence(nc, "severe", "dev natural (full_extract arm N)", "2,747 non-CoT valid natural plans")
    mm = nc[nc.t_star.notna()]
    prev += prevalence(mm, "Aminus", "dev natural (full_extract arm N)", "first-mismatch scenes")
    v1 = s1[s1.exclusion_reason.isna() | s1.exclusion_reason.isin(["no_mismatch", "t_star_gt_7"])]
    prev += prevalence(v1, "Aminus", "P1-C stage 1", "valid non-CoT natural plans (64 logs)")
    prev += prevalence(v1, "severe", "P1-C stage 1", "valid non-CoT natural plans (64 logs)")
    el = s1[s1.eligible]
    prev += prevalence(el, "Aminus", "P1-C stage 1", "eligible E (t* <= 7) = P1-C population frame")
    prev = pd.DataFrame(prev)
    prev.to_csv(os.path.join(OUT, "POPULATION_prevalence.csv"), index=False)

    eff_p, den, ut = p1c_effects()
    eff_d, ref = dev_effects(nat, pool_size)
    eff = pd.concat([eff_p, eff_d], ignore_index=True)
    eff.insert(0, "row_id", [f"P0A.{('P1C' if c.startswith('P1-C') else 'DEV')}.{k}.{e}".replace("-", "m").replace("+", "p")
                              for c, k, e in zip(eff.cohort, eff.contrast, eff.estimand)])
    eff.to_csv(os.path.join(OUT, "POPULATION_EFFECTS.csv"), index=False)
    den.to_csv(os.path.join(OUT, "POPULATION_p1c_denominators.csv"), index=False)

    # verification against review_followup2 A3 (dev A-)
    ver = []
    m = {"RN": "AV-P1 Recent-Normal", "RV": "AV-P2 Reverse-Full", "DM": "AV-P3 Direction-Magnitude"}
    for c, name in m.items():
        rr = ref[ref.iloc[:, 0].astype(str).str.startswith(name.split()[0])]
        mine = eff_d[(eff_d.contrast == c) & (eff_d.estimand == "A-")].iloc[0]
        ver.append({"contrast": c, "ours_theta": mine.theta, "ours_p": mine.p_signflip_raw, "ours_holm": mine.p_holm,
                    "ref_row": rr.iloc[0].to_dict() if len(rr) else None})
    meta = {"selection_check": check, "pool_size_by_tstar": {str(k): v for k, v in pool_size.items()}, "seed": SEED, "B": B,
            "a3_reference_rows": ver, "n_dev_natural_records": int(len(nat)), "n_p1c_stage1": int(len(s1))}
    json.dump(meta, open(os.path.join(OUT, "POPULATION_meta.json"), "w"), indent=1, default=str)

    # manifest fragment: one row per natural scene (dev) and per P1-C stage-1 scene
    man = []
    p1u = ut.groupby("scene_id").agg(n_rows=("unit_id", "size"), stratum=("stratum", "first"), weight=("weight", "first"),
                                     n_pert=("perturbation", "nunique"))
    dev_units = pd.read_parquet(DEV_UNITS)
    dvu = dev_units.groupby("scene_id").agg(n_rows=("unit_id", "size"), n_pert=("perturbation", "nunique"))
    for r in nat.itertuples():
        inc = r.selected is not None
        man.append({"dataset": "dev_natural_full_extract", "experiment": "full_extract arm N -> equal_distance/exp6-11/32/R8R9",
                    "model": "AutoVLA_PDMS_89", "task": "navtest open-loop", "unit_id": r.token, "log": r.log, "drive": r.drive,
                    "day": r.day, "vehicle": r.vehicle, "day_vehicle": r.day_vehicle, "scene": r.token,
                    "perturbation": (f"{int(dvu.n_pert.get(r.token, 0))} units (original + alt_k) at t*={int(r.t_star)}" if inc else ""),
                    "initial_physical_state": "logged scene (open-loop; = scene)", "instruction": r.instruction,
                    "rng_seed": "decode seed 0 (T=0.01); A+ sampling Random(0)", "condition_set": "natural N" + ("; mechanism arms" if inc else ""),
                    "n_rows": int(dvu.n_rows.get(r.token, 0)), "included": inc,
                    "missing_reason": (r.excl_selection or ("" if inc else ("A+ not sampled" if r.a_label == "A+" else ""))),
                    "denominator_note": f"label={r.a_label}; t*={r.t_star}; FDE5={r.fde5:.2f}; selected={r.selected}"})
    for r in s1.itertuples():
        inc = r.token in p1u.index
        man.append({"dataset": "P1C_stage1", "experiment": "P1-C (b212637 + amendments)", "model": "AutoVLA_PDMS_89",
                    "task": "navtest open-loop", "unit_id": r.token, "log": r.log, "drive": r.drive, "day": r.day,
                    "vehicle": r.vehicle, "day_vehicle": r.day_vehicle, "scene": r.token,
                    "perturbation": (f"{int(p1u.n_pert[r.token])} units at t*={r.t_star}" if inc else ""),
                    "initial_physical_state": "logged scene (open-loop; = scene)", "instruction": r.instruction,
                    "rng_seed": f"stage-1 seed {r.seed}; S sampling sha256 order (salt P1C-20261008)",
                    "condition_set": "natural" + ("; H7/H8/H9/H32" if inc else ""),
                    "n_rows": int(p1u.n_rows[r.token]) if inc else 0, "included": inc,
                    "missing_reason": (r.exclusion_reason or ("" if inc else "A+ not sampled (S frame)")),
                    "denominator_note": f"label={r.a_label}; t*={r.t_star}; stratum={p1u.stratum[r.token] if inc else ''}; "
                                        f"w={p1u.weight[r.token] if inc else ''}"})
    pd.DataFrame(man).to_csv(os.path.join(OUT, "_frag_manifest_A.csv"), index=False)

    # numbers fragment
    nums = {}
    for r in eff.itertuples():
        nums[r.row_id] = {"value": r.theta, "ci95_boot": [r.boot_ci95_lo, r.boot_ci95_hi], "ci95_sf": [r.sf_ci95_lo, r.sf_ci95_hi],
                          "p_raw": r.p_signflip_raw, "p_holm": r.p_holm, "family": r.family_id, "n_scenes": r.n_scenes,
                          "n_drives": r.n_clusters, "unit": "proportion (x100 = pp)",
                          "source_path": os.path.join(OUT, "POPULATION_EFFECTS.csv"), "source_row": r.row_id,
                          "upstream": r.source}
    for r in prev.itertuples():
        if r.cluster_level in ("drive", "day_vehicle"):
            k = f"P0A.PREV.{'DEV' if r.cohort.startswith('dev') else 'P1C'}.{r.outcome}.{r.denominator.split()[0]}.{r.cluster_level}"
            nums[k] = {"value": r.prevalence, "events": r.events, "n": r.n_scenes, "ci95_boot": [r.boot_ci95_lo, r.boot_ci95_hi],
                       "n_clusters": r.n_clusters, "denominator": r.denominator,
                       "source_path": os.path.join(OUT, "POPULATION_prevalence.csv")}
    json.dump(nums, open(os.path.join(OUT, "_frag_numbers_A.json"), "w"), indent=1, default=float)
    print(json.dumps(check, indent=1))
    print(flow_all.to_string())
    print(prev[prev.cluster_level.isin(["drive", "day_vehicle"])][["cohort", "outcome", "denominator", "events", "n_scenes",
                                                                    "prevalence", "cluster_level", "n_clusters", "boot_ci95_lo",
                                                                    "boot_ci95_hi"]].to_string())
    cols = ["row_id", "estimand", "n_scenes", "n_units", "n_clusters", "n_informative_clusters", "events_a", "n_a", "events_b",
            "n_b", "theta", "boot_ci95_lo", "boot_ci95_hi", "sf_ci95_lo", "sf_ci95_hi", "p_signflip_raw", "p_holm"]
    print(eff[cols].to_string())
    print(den.to_string())
    print(json.dumps(ver, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
