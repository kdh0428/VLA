#!/usr/bin/env python
"""
P1-C analysis (FROZEN before any stage-2 evaluation output exists; protocol §5, §7-§9, analysis.md §6, decision D7d).
CPU only. Implements exactly the preregistered estimands and inference:

UNITS AND OUTCOMES
  unit = (scene, perturbation) with perturbation in {original, alt0..}; per block (H7 action_history_causal,
  H8 prev_action_state_patching, H9 temporal_feedback_window, H32 motion_semantics_ablation) and condition the unit
  outcome is analyze_action_history.unit_metrics (unchanged): amplification = (not A+) and FDE5 > 3.0 m;
  recovery = A+; FDE5. Invalid/parse/runaway output (fewer than 10 actions or no trajectory) -> amplification 0,
  recovery 0, FDE5 missing (dev rule).
  scene effect d_s = mean over the scene's units of (cond_a - cond_b), each contrast inside ONE block (its own Normal).

ESTIMANDS (scene weights w: F -> 1, S -> N_S,seg / m_seg)
  theta_F = sum_F d_s / |F|;  theta_S = sum_S w d / sum_S w;  theta_P = sum_{F u S} w d / sum w  (Hajek);
  Q3 Delta = theta_F - theta_S.
  Primary (scene-weighted); sensitivity: unit-weighted (sum_s w_s sum_u d_u / sum_s w_s n_s) and drive-equal
  (mean over drives of the drive's Hajek ratio).

INFERENCE
  p: drive-level sign-flip on drive totals T_g = sum_{s in g} w_s d_s, statistic |sum_g e_g T_g|, EXACT over all 2^G
     sign patterns (meet-in-the-middle enumeration, G <= 24 -> 1.7e7 patterns), two-sided, p = #{|.| >= obs}/2^G.
     Drives with no scene in the estimand are not clusters. Drive-equal / unit-weighted variants flip their own
     drive summaries (drive ratio / unit totals).
  Holm within family: F = {RN, RV, DM} on theta_F at alpha_F = 0.05; P = {RN, RV, DM} on theta_P at alpha_P = 0.025.
  CIs: drive block bootstrap (B = 10,000, multinomial drive counts, percentile 95% and 90%; F and S scenes of a drive
     move together) and cluster-robust t (ratio linearisation, df = G - 1); test-inversion sign-flip CIs (95%, 90%)
     for the primary contrasts.
  Decision (protocol §9): replicated  <=> Holm p < family alpha AND sign(theta) = predicted sign;
     not replicated, below MME <=> 90% CI inside (-|MME|, +|MME|), using the sign-flip test-inversion 90% CI
     (the inference that governs significance; the bootstrap-90% version is reported alongside);
     otherwise inconclusive ("not detected / not established").
  Sensitivity (all reported): S1 non-navhard-source logs; S2 day-vehicle-disjoint logs; S3 clustering at day-vehicle
     level; S4 without flagged logs (> 5% of the log's eligible scenes failed: stage-1 error or harness exception);
     unit-weighted; drive-equal; common inclusion set (no unit invalid in any condition of the block).
  Secondary: Q3 Delta (RN, RV, DM); windows (normal, win1, win3, win4, gt_history levels in F and S; gap fraction
     f(w) = (normal - win w)/(normal - gt_history), w = 1, 3, 4; checks win4 - gt_history 90% CI within +-5 pp and
     f(3) >= 0.75); FDE5 versions of the primary contrasts; per-drive / per-log effects + forest plot; city strata;
     H32 secondary conditions; OOD rule (mean first-step entropy shift > 1 nat vs Normal) reported, never applied.

  --mask-effects  replaces every effect-bearing value (estimates, CIs, p, decisions, unit outcomes) by "MASKED" in all
                  outputs (pilot / harness test: protocol §6 "pilot effects are not looked at").
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

sys.path.insert(0, C.SCRIPTS)
from analyze_action_history import unit_metrics  # noqa: E402  (dev outcome definition, unchanged)

SEED = 20261008
B_BOOT = 10_000
ALPHA = {"F": 0.05, "P": 0.025}
N_ACT = 10

# (id, block, cond_a, cond_b, predicted sign, MME by family)
PRIMARY = [
    ("RN", "H7", "recent_gt", "normal", -1, {"F": -0.10, "P": -0.02}),
    ("RV", "H8", "reverse@emb", "gt_history", +1, {"F": +0.10, "P": +0.02}),
    ("DM", "H32", "dir_wrong_mag_ok", "dir_ok_mag_wrong", +1, {"F": +0.05, "P": +0.02}),
]
BLOCK_DIR = {"H7": "action_history_causal", "H8": "prev_action_state_patching",
             "H9": "temporal_feedback_window", "H32": "motion_semantics_ablation"}
BLOCK_CONDS = {"H7": ["normal", "recent_gt", "gt_history", "hist_attn_mask", "recent_attn_mask", "hist_emb_neutral",
                      "recent_emb_neutral"],
               "H8": ["normal", "gt_history", "recent_gt", "reverse@emb", "patch_full@emb", "patch_full@L35"],
               "H9": ["normal", "win1", "win2", "win3", "win4", "gt_history"],
               "H32": ["normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"]}
WINDOWS = ["normal", "win1", "win3", "win4", "gt_history"]
MASK = "MASKED"


# ----------------------------------------------------------------------------------------------------------------
# inference primitives (pure numpy; tested on synthetic data in test_analyze_p1c.py)
# ----------------------------------------------------------------------------------------------------------------
def _half_sums(v):
    """all 2^len(v) signed sums of v."""
    s = np.zeros(1)
    for x in v:
        s = np.concatenate([s + x, s - x])
    return s


def signflip_exact(T, tol=1e-12):
    """two-sided exact sign-flip p for H0: sum T = 0 (drive totals T). Meet in the middle: O(2^(G/2) log)."""
    T = np.asarray(T, float)
    G = len(T)
    if G == 0:
        return float("nan")
    obs = abs(T.sum())
    a = _half_sums(T[: G // 2])
    b = np.sort(_half_sums(T[G // 2:]))
    thr = obs - tol * max(1.0, obs)
    # count pairs with |a + b| >= thr  <=>  b >= thr - a  or  b <= -thr - a
    hi = len(b) - np.searchsorted(b, thr - a, side="left")
    lo = np.searchsorted(b, -thr - a, side="right")
    cnt = hi + lo
    if thr <= 0:                      # obs == 0: every pattern counts once (avoid double counting)
        return 1.0
    return float(cnt.sum() / (len(a) * len(b)))


def signflip_ci(N, D, theta, level, span=None, iters=40):
    """test-inversion CI for the ratio sum N / sum D: {delta : p(T_g = N_g - delta D_g) > 1 - level}.
    Bisection on each side of theta (assumes the acceptance region is an interval)."""
    N, D = np.asarray(N, float), np.asarray(D, float)
    a = 1 - level
    if len(N) < 2:
        return [float("nan"), float("nan")]
    span = span or max(1.0, 4 * np.abs(N / np.where(D > 0, D, 1)).max())

    def acc(d):
        return signflip_exact(N - d * D) > a

    out = []
    for direction in (-1, 1):
        lo_, hi_ = theta, theta + direction * span
        if acc(hi_):
            out.append(hi_); continue
        for _ in range(iters):
            mid = 0.5 * (lo_ + hi_)
            if acc(mid):
                lo_ = mid
            else:
                hi_ = mid
        out.append(lo_)
    return [float(out[0]), float(out[1])]


def cluster_t(N, D):
    N, D = np.asarray(N, float), np.asarray(D, float)
    G = len(N)
    if G < 2 or D.sum() <= 0:
        return float("nan"), [float("nan")] * 2
    th = N.sum() / D.sum()
    r = N - th * D
    se = np.sqrt(G / (G - 1) * np.sum(r ** 2)) / D.sum()
    if se == 0:
        return (1.0 if th == 0 else 0.0), [th, th]
    q = tdist.ppf(0.975, G - 1)
    return float(2 * tdist.sf(abs(th / se), G - 1)), [float(th - q * se), float(th + q * se)]


def boot_counts(G, seed=SEED, b=B_BOOT):
    return np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=b).astype(float)


def ratio_boot(Cnt, N, D):
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (Cnt @ N) / (Cnt @ D)
    return x[np.isfinite(x)]


def pct(x, lo, hi):
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------------------------------------------
# estimators on a scene table: columns scene, cluster, w, d, n_units, dsum (sum of unit d)
# ----------------------------------------------------------------------------------------------------------------
def cluster_arrays(sc, weighting):
    """drive summaries (N_g, D_g) so that theta = sum N / sum D (scene, unit) or mean of N/D (drive_equal)."""
    x = pd.DataFrame({"cluster": sc.cluster.values})
    if weighting == "scene":
        x["N"] = (sc.w * sc.d).values; x["D"] = sc.w.values
    elif weighting == "unit":
        x["N"] = (sc.w * sc.dsum).values; x["D"] = (sc.w * sc.n_units).values
    else:
        raise ValueError(weighting)
    g = x.groupby("cluster")[["N", "D"]].sum()
    return g.index.tolist(), g.N.values.astype(float), g.D.values.astype(float)


def estimate(sc, weighting="scene", seed=SEED, with_ci_inversion=False, boot=True):
    """sc: scene table of one estimand (F, S or P). Returns dict with theta, CIs, p values."""
    out = {"n_scenes": int(len(sc)), "n_units": int(sc.n_units.sum()) if len(sc) else 0,
           "n_clusters": int(sc.cluster.nunique()) if len(sc) else 0, "weighting": weighting}
    if len(sc) == 0:
        return out | {"theta": float("nan")}
    if weighting == "drive_equal":
        labs, N, D = cluster_arrays(sc, "scene")
        r = N / D
        G = len(r)
        out["theta"] = float(r.mean())
        out["p_signflip"] = signflip_exact(r)
        if boot:
            Cnt = boot_counts(G, seed)
            bs = Cnt @ r / G
            out["boot_ci95"], out["boot_ci90"] = pct(bs, 2.5, 97.5), pct(bs, 5, 95)
        out["p_cluster_t"], out["clt_ci95"] = cluster_t(r, np.ones(G))
        return out
    labs, N, D = cluster_arrays(sc, weighting)
    G = len(N)
    out["theta"] = float(N.sum() / D.sum())
    out["p_signflip"] = signflip_exact(N)
    out["p_cluster_t"], out["clt_ci95"] = cluster_t(N, D)
    if boot:
        Cnt = boot_counts(G, seed)
        bs = ratio_boot(Cnt, N, D)
        out["boot_ci95"], out["boot_ci90"] = pct(bs, 2.5, 97.5), pct(bs, 5, 95)
    if with_ci_inversion:
        out["sf_ci95"] = signflip_ci(N, D, out["theta"], 0.95)
        out["sf_ci90"] = signflip_ci(N, D, out["theta"], 0.90)
    out["n_informative_clusters"] = int(np.sum(np.abs(N) > 1e-12))
    return out


def estimand_table(sc, which):
    if which == "F":
        return sc[sc.stratum == "F"]
    if which == "S":
        return sc[sc.stratum == "S"]
    return sc


def delta_q3(sc, seed=SEED):
    """Delta = theta_F - theta_S, drive bootstrap with F and S scenes of a drive resampled together; cluster-t by
    linearisation."""
    clusters = sorted(sc.cluster.unique())
    ci = {c: i for i, c in enumerate(clusters)}
    G = len(clusters)
    NF, DF, NS, DS = (np.zeros(G) for _ in range(4))
    for r in sc.itertuples():
        i = ci[r.cluster]
        if r.stratum == "F":
            NF[i] += r.w * r.d; DF[i] += r.w
        else:
            NS[i] += r.w * r.d; DS[i] += r.w
    if DF.sum() == 0 or DS.sum() == 0:
        return {"delta": float("nan")}
    thF, thS = NF.sum() / DF.sum(), NS.sum() / DS.sum()
    Cnt = boot_counts(G, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (Cnt @ NF) / (Cnt @ DF) - (Cnt @ NS) / (Cnt @ DS)
    bs = bs[np.isfinite(bs)]
    z = (NF - thF * DF) / DF.sum() - (NS - thS * DS) / DS.sum()
    se = np.sqrt(G / (G - 1) * np.sum(z ** 2)) if G > 1 else float("nan")
    q = tdist.ppf(0.975, G - 1) if G > 1 else float("nan")
    d = thF - thS
    return {"delta": float(d), "theta_F": float(thF), "theta_S": float(thS), "boot_ci95": pct(bs, 2.5, 97.5),
            "boot_ci90": pct(bs, 5, 95), "clt_ci95": [float(d - q * se), float(d + q * se)],
            "p_cluster_t": float(2 * tdist.sf(abs(d / se), G - 1)) if se and np.isfinite(se) and se > 0 else float("nan"),
            "n_clusters": G}


def decision(holm_p, alpha, theta, sign, ci90, mme):
    if not np.isfinite(theta):
        return "not_estimable"
    if np.isfinite(holm_p) and holm_p < alpha and np.sign(theta) == sign:
        return "replicated"
    if np.all(np.isfinite(ci90)) and -abs(mme) < ci90[0] and ci90[1] < abs(mme):
        return "not_replicated_below_MME"
    return "inconclusive"


# ----------------------------------------------------------------------------------------------------------------
# raw records -> unit table
# ----------------------------------------------------------------------------------------------------------------
def _m(acts, traj, ents, gt, gtraj, t):
    if acts is None or traj is None or len(acts) != N_ACT or len(traj) != N_ACT:
        return {"amplification": 0, "recovery": 0, "fde5": float("nan"), "entropy1": float("nan"), "parser_status": "invalid"}
    m = unit_metrics(acts, traj, ents or [], gt, gtraj, t)
    return {"amplification": int(m["amplification"]), "recovery": int(m["recovery"]), "fde5": m["fde5"],
            "entropy1": m["entropy1"] if m["entropy1"] is not None else float("nan"), "parser_status": "ok"}


def block_rows(block, rec):
    """yield (perturbation, condition, acts, traj, ents, forced_token) from one record of a block."""
    if block == "H7":
        for pname, pr in rec["perturbations"].items():
            for c, v in pr["conditions"].items():
                yield pname, c, v.get("action_idx"), v.get("trajectory_pred"), v.get("entropy_steps"), pr.get("forced_token")
    elif block in ("H8", "H9"):
        for c, v in rec["rows"].items():
            if c in BLOCK_CONDS[block]:
                yield rec["perturbation"], c, v.get("action_idx"), v.get("trajectory_pred"), v.get("ent"), rec.get("forced_token")
    elif block == "H32":
        for c, v in rec["conditions"].items():
            yield rec["perturbation"], c, v.get("action_idx"), v.get("trajectory_pred"), v.get("entropy_steps"), rec.get("forced_token")


def load_run(run, split_path):
    split = pd.read_csv(split_path, dtype=str).set_index("log")
    sel = []
    den = {}
    for f in sorted(glob.glob(os.path.join(run, "selection", "shard_*.json"))):
        if not re.fullmatch(r"shard_\d+\.json", os.path.basename(f)):
            continue
        j = json.load(open(f))
        sel += j["selected"]
        den.update(j["denominators"])
    sel = pd.DataFrame(sel)
    units = {}
    for l in open(os.path.join(run, "units", "records.jsonl")):
        r = json.loads(l)
        units[r["token"]] = r
    city = {}
    s1 = os.path.join(run, "stage1", "records.jsonl")
    if os.path.exists(s1):
        for l in open(s1):
            r = json.loads(l)
            if r["token"] in units:
                city[r["token"]] = r.get("map_name")
    rows, errors = [], defaultdict(set)
    for blk, d in BLOCK_DIR.items():
        p = os.path.join(run, "mech", d, "records.jsonl")
        if not os.path.exists(p):
            continue
        ep = os.path.join(run, "mech", d, "errors.jsonl")
        if os.path.exists(ep):
            for l in open(ep):
                if l.strip():
                    errors[blk].add(json.loads(l)["token"])
        for l in open(p):
            rec = json.loads(l)
            tok = rec["token"]
            u = units.get(tok)
            if u is None:
                continue
            t, gt, gtraj = u["t_star"], u["gt"], u["trajectory_gt"]
            for pname, cond, acts, traj, ents, ftok in block_rows(blk, rec):
                m = _m(acts, traj, ents, gt, gtraj, t)
                rows.append({"block": blk, "scene_id": tok, "perturbation": pname, "condition": cond,
                             "forced_token": ftok, **m})
    raw = pd.DataFrame(rows)
    return split, sel, den, units, raw, errors, city


def build_unit_table(split, sel, units, raw, errors, city):
    """long table with the plan §2 / protocol §11 fields; scenes with a harness exception or incomplete units in a block
    are marked exclusion_reason = 'harness_exception' for that block."""
    if raw.empty:
        return raw
    s = sel.set_index("token")
    raw = raw.copy()
    raw["log_id"] = raw.scene_id.map(s["log"])
    raw["drive_id"] = raw.log_id.map(split["drive"])
    raw["day_vehicle"] = raw.log_id.map(split["day_vehicle"])
    raw["stratum"] = raw.scene_id.map(s["stratum"])
    raw["pi"] = raw.scene_id.map(s["pi"]).astype(float)
    raw["weight"] = raw.scene_id.map(s["weight"]).astype(float)
    raw["t_star"] = raw.scene_id.map(s["t_star"])
    raw["city"] = raw.scene_id.map(city)
    raw["unit_id"] = raw.scene_id + "|" + raw.perturbation
    raw["seed"] = 0
    raw["decode_j"] = None
    raw["model"] = "AutoVLA_PDMS_89"
    raw["task_id"] = None; raw["episode_id"] = None
    raw["control_t"] = raw.t_star
    raw["reference_branch"] = np.where(raw.condition.str.startswith("reverse@"), "normal_row_same_batch", None)
    raw["perturbation_requested"] = raw.forced_token
    raw["perturbation_achieved"] = raw.forced_token
    raw["primary_outcome"] = raw.amplification
    expected = {tok: 1 + len(u["alternatives"]) for tok, u in units.items()}
    raw["exclusion_reason"] = None
    for blk in raw.block.unique():
        b = raw[raw.block == blk]
        got = b.groupby("scene_id").perturbation.nunique()
        bad = {tok for tok, n in got.items() if n != expected.get(tok)} | (errors.get(blk, set()) & set(got.index))
        raw.loc[(raw.block == blk) & raw.scene_id.isin(bad), "exclusion_reason"] = "harness_exception"
    return raw


def scene_table(ut, block, a, b, metric, common_inclusion=False):
    u = ut[(ut.block == block) & ut.exclusion_reason.isna()]
    if common_inclusion:
        bad = set(u[u.parser_status != "ok"].unit_id)
        u = u[~u.unit_id.isin(bad)]
    A = u[u.condition == a].set_index("unit_id")
    Bb = u[u.condition == b].set_index("unit_id")
    j = A[["scene_id", "log_id", "drive_id", "day_vehicle", "stratum", "weight", "city", metric]].join(
        Bb[[metric]], rsuffix="_b", how="inner")
    j["du"] = j[metric].astype(float) - j[f"{metric}_b"].astype(float)
    j = j.dropna(subset=["du"])
    g = j.groupby("scene_id")
    sc = pd.DataFrame({"d": g.du.mean(), "dsum": g.du.sum(), "n_units": g.du.size(),
                       "log": g.log_id.first(), "drive": g.drive_id.first(), "day_vehicle": g.day_vehicle.first(),
                       "stratum": g.stratum.first(), "w": g.weight.first().astype(float), "city": g.city.first()}).reset_index()
    sc = sc.rename(columns={"scene_id": "scene"})
    sc["cluster"] = sc.drive
    return sc


def level_table(ut, block, cond, metric="amplification"):
    """scene-level mean of a condition's outcome (for window levels), same layout as scene_table."""
    u = ut[(ut.block == block) & ut.exclusion_reason.isna() & (ut.condition == cond)]
    g = u.groupby("scene_id")
    sc = pd.DataFrame({"d": g[metric].mean(), "dsum": g[metric].sum(), "n_units": g[metric].size(),
                       "log": g.log_id.first(), "drive": g.drive_id.first(), "day_vehicle": g.day_vehicle.first(),
                       "stratum": g.stratum.first(), "w": g.weight.first().astype(float)}).reset_index()
    sc = sc.rename(columns={"scene_id": "scene"})
    sc["cluster"] = sc.drive
    return sc


# ----------------------------------------------------------------------------------------------------------------
def analyse(ut, split, den, out, mask=False, seed=SEED):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    rows, decisions = [], []
    split_flags = split[["navhard_source", "dayveh_overlaps_poc"]]

    # flagged logs (S4): > 5% of the log's eligible scenes failed (stage-1 error/missing or harness exception)
    flagged = []
    for log, d in den.items():
        nerr = sum(v for k, v in d["not_eligible_by_reason"].items() if k in ("stage1_error", "stage1_missing"))
        exc = ut[(ut.log_id == log) & (ut.exclusion_reason == "harness_exception")].scene_id.nunique() if len(ut) else 0
        if (nerr + exc) > 0.05 * max(1, d["eligible"] + nerr):
            flagged.append(log)

    def subset(sc, name):
        if name == "S1_non_navhard":
            keep = set(split_flags.index[split_flags.navhard_source == "False"])
            return sc[sc.log.isin(keep)]
        if name == "S2_dayveh_disjoint":
            keep = set(split_flags.index[split_flags.dayveh_overlaps_poc == "False"])
            return sc[sc.log.isin(keep)]
        if name == "S3_dayveh_cluster":
            x = sc.copy(); x["cluster"] = x.day_vehicle; return x
        if name == "S4_no_flagged_logs":
            return sc[~sc.log.isin(flagged)]
        return sc

    def add(kind, cid, block, a, b, metric, estd, variant, weighting, r, extra=None):
        row = {"kind": kind, "contrast": cid, "block": block, "cond_a": a, "cond_b": b, "metric": metric,
               "estimand": estd, "variant": variant, "weighting": weighting}
        for k, v in r.items():
            if isinstance(v, list):
                row[f"{k}_lo"], row[f"{k}_hi"] = v
            else:
                row[k] = v
        row.update(extra or {})
        rows.append(row)
        return row

    # ---- primary families -------------------------------------------------------------------------------------
    prim = {}
    for cid, blk, a, b, sign, mme in PRIMARY:
        if blk not in set(ut.block):
            continue
        sc = scene_table(ut, blk, a, b, "amplification")
        for estd in ("F", "P", "S"):
            r = estimate(estimand_table(sc, estd), "scene", seed, with_ci_inversion=estd in ("F", "P"))
            prim[(cid, estd)] = add("primary" if estd != "S" else "secondary_theta_S", cid, blk, a, b, "amplification",
                                    estd, "main", "scene", r)
            for wt in ("unit", "drive_equal"):
                add("sensitivity_weighting", cid, blk, a, b, "amplification", estd, "main", wt,
                    estimate(estimand_table(sc, estd), wt, seed))
            for sv in ("S1_non_navhard", "S2_dayveh_disjoint", "S3_dayveh_cluster", "S4_no_flagged_logs"):
                add("sensitivity_subset", cid, blk, a, b, "amplification", estd, sv, "scene",
                    estimate(estimand_table(subset(sc, sv), estd), "scene", seed))
        sci = scene_table(ut, blk, a, b, "amplification", common_inclusion=True)
        for estd in ("F", "P"):
            add("sensitivity_common_inclusion", cid, blk, a, b, "amplification", estd, "common_inclusion", "scene",
                estimate(estimand_table(sci, estd), "scene", seed))
        # FDE5 version
        scf = scene_table(ut, blk, a, b, "fde5")
        for estd in ("F", "P", "S"):
            add("secondary_fde5", cid, blk, a, b, "fde5", estd, "main", "scene", estimate(estimand_table(scf, estd), "scene", seed))
        # recovery version
        scr = scene_table(ut, blk, a, b, "recovery")
        for estd in ("F", "P", "S"):
            add("secondary_recovery", cid, blk, a, b, "recovery", estd, "main", "scene", estimate(estimand_table(scr, estd), "scene", seed))
        # Q3
        q = delta_q3(sc, seed)
        add("secondary_Q3_delta", cid, blk, a, b, "amplification", "F-S", "main", "scene", q)
        # per drive / per log / city
        for level in ("drive", "log", "city"):
            for (key, estd), g in sc.groupby([level, "stratum"]):
                th = float((g.w * g.d).sum() / g.w.sum())
                add(f"per_{level}", cid, blk, a, b, "amplification", estd, str(key), "scene",
                    {"theta": th, "n_scenes": int(len(g)), "n_units": int(g.n_units.sum())})
        for key, g in sc.groupby("city"):
            add("per_city", cid, blk, a, b, "amplification", "P", str(key), "scene", estimate(g, "scene", seed, boot=False))

    # Holm + decisions
    for fam in ("F", "P"):
        keys = [(cid, fam) for cid, *_ in PRIMARY if (cid, fam) in prim]
        if not keys:
            continue
        adj = holm([prim[k]["p_signflip"] for k in keys])
        for k, pa in zip(keys, adj):
            cid = k[0]
            spec = next(p for p in PRIMARY if p[0] == cid)
            r = prim[k]
            r["holm_p"] = float(pa); r["family_alpha"] = ALPHA[fam]; r["mme"] = spec[5][fam]; r["predicted_sign"] = spec[4]
            r["decision"] = decision(pa, ALPHA[fam], r["theta"], spec[4], [r.get("sf_ci90_lo"), r.get("sf_ci90_hi")], spec[5][fam])
            r["decision_with_bootstrap_ci90"] = decision(pa, ALPHA[fam], r["theta"], spec[4],
                                                          [r.get("boot_ci90_lo"), r.get("boot_ci90_hi")], spec[5][fam])
            decisions.append({k2: r.get(k2) for k2 in ("contrast", "estimand", "theta", "p_signflip", "holm_p", "family_alpha",
                                                        "predicted_sign", "mme", "sf_ci90_lo", "sf_ci90_hi", "sf_ci95_lo",
                                                        "sf_ci95_hi", "boot_ci95_lo", "boot_ci95_hi", "decision",
                                                        "decision_with_bootstrap_ci90", "n_clusters", "n_scenes", "n_units")})
    # Holm inside the sensitivity variants (reference only)
    df = pd.DataFrame(rows)

    # ---- windows (H9) -----------------------------------------------------------------------------------------
    if "H9" in set(ut.block):
        for estd in ("F", "S"):
            for c in WINDOWS:
                add("secondary_window_level", c, "H9", c, None, "amplification", estd, "main", "scene",
                    estimate(estimand_table(level_table(ut, "H9", c), estd), "scene", seed))
            gap = scene_table(ut, "H9", "normal", "gt_history", "amplification")
            for w in (1, 3, 4):
                win = scene_table(ut, "H9", "normal", f"win{w}", "amplification")
                m = win.merge(gap[["scene", "d"]], on="scene", suffixes=("", "_gap"))
                m = estimand_table(m, estd)
                if len(m) == 0:
                    continue
                cl = sorted(m.cluster.unique()); G = len(cl); ci = {c_: i for i, c_ in enumerate(cl)}
                Nw, Ng = np.zeros(G), np.zeros(G)
                for r in m.itertuples():
                    Nw[ci[r.cluster]] += r.w * r.d; Ng[ci[r.cluster]] += r.w * r.d_gap
                f = Nw.sum() / Ng.sum() if Ng.sum() != 0 else float("nan")
                Cnt = boot_counts(G, seed)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bs = (Cnt @ Nw) / (Cnt @ Ng)
                bs = bs[np.isfinite(bs)]
                extra = {"check_f3_ge_0.75": bool(f >= 0.75)} if w == 3 else {}
                add("secondary_window_gap_fraction", f"f({w})", "H9", "normal-win%d" % w, "normal-gt_history",
                    "amplification", estd, "main", "scene",
                    {"theta": float(f), "boot_ci95": pct(bs, 2.5, 97.5), "boot_ci90": pct(bs, 5, 95), "n_clusters": G,
                     "n_scenes": int(len(m))}, extra)
            sc = scene_table(ut, "H9", "win4", "gt_history", "amplification")
            r = estimate(estimand_table(sc, estd), "scene", seed, with_ci_inversion=True)
            bci, sci_ = r.get("boot_ci90"), r.get("sf_ci90")
            eq = {"equiv_margin": 0.05,
                  "equivalent_boot90": bool(-0.05 < bci[0] and bci[1] < 0.05) if bci else None,
                  "equivalent_sf90": bool(-0.05 < sci_[0] and sci_[1] < 0.05) if sci_ else None}
            add("secondary_window_equivalence", "win4-gt_history", "H9", "win4", "gt_history", "amplification", estd, "main",
                "scene", r, eq)

    # ---- H32 secondary conditions, H7/H8 other contrasts vs normal; OOD entropy rule ------------------------------
    for blk in ("H7", "H8", "H32"):
        if blk not in set(ut.block):
            continue
        for c in BLOCK_CONDS[blk]:
            if c == "normal":
                continue
            sc = scene_table(ut, blk, c, "normal", "amplification")
            for estd in ("F", "S", "P"):
                add("secondary_vs_normal", f"{c}-normal", blk, c, "normal", "amplification", estd, "main", "scene",
                    estimate(estimand_table(sc, estd), "scene", seed))
            se = scene_table(ut, blk, c, "normal", "entropy1")
            if len(se):
                dh = float((se.w * se.d).sum() / se.w.sum())
                add("ood_entropy_rule", f"{c}-normal", blk, c, "normal", "entropy1", "P", "main", "scene",
                    {"theta": dh, "ood_flag_gt_1nat": bool(dh > 1.0), "n_scenes": int(len(se))}, {"applied": False})

    df = pd.DataFrame(rows)
    cs = []
    for (blk, cond), g in ut[ut.exclusion_reason.isna()].groupby(["block", "condition"]):
        for estd in ("F", "S"):
            gg = g[g.stratum == estd]
            cs.append({"block": blk, "condition": cond, "stratum": estd, "n_units": len(gg), "n_scenes": gg.scene_id.nunique(),
                       "n_invalid": int((gg.parser_status != "ok").sum()),
                       "amplification_unit_mean": gg.amplification.mean(), "recovery_unit_mean": gg.recovery.mean(),
                       "fde5_unit_mean": gg.fde5.mean()})
    cs = pd.DataFrame(cs)
    if mask:
        effect_cols = [c for c in df.columns if c not in ("kind", "contrast", "block", "cond_a", "cond_b", "metric", "estimand",
                                                            "variant", "weighting", "n_scenes", "n_units", "n_clusters",
                                                            "sf_method", "applied", "family_alpha", "mme", "predicted_sign",
                                                            "equiv_margin")]
        df[effect_cols] = df[effect_cols].astype(object).where(df[effect_cols].isna(), MASK)
        for c in ("amplification_unit_mean", "recovery_unit_mean", "fde5_unit_mean"):
            if c in cs:
                cs[c] = MASK
        for d_ in decisions:
            for k in list(d_):
                if k not in ("contrast", "estimand", "family_alpha", "predicted_sign", "mme", "n_clusters", "n_scenes", "n_units"):
                    d_[k] = MASK
    df.to_csv(os.path.join(out, "paired_effects.csv"), index=False)
    cs.to_csv(os.path.join(out, "condition_summary.csv"), index=False)
    json.dump({"decisions": decisions, "flagged_logs_S4": flagged, "alpha": ALPHA, "masked": mask},
              open(os.path.join(out, "decisions.json"), "w"), indent=1, default=str)
    if not mask:
        forest(df, out)
    return df, decisions, flagged


def forest(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for cid, *_ in PRIMARY:
        d = df[(df.kind == "per_drive") & (df.contrast == cid)]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 0.25 * len(d) + 1.5))
        d = d.sort_values(["estimand", "theta"])
        y = np.arange(len(d))
        ax.scatter(d.theta, y, s=8 + 4 * d.n_scenes, c=np.where(d.estimand == "F", "#c0392b", "#2c7fb8"))
        ax.set_yticks(y); ax.set_yticklabels([f"{v} [{e}] n={n}" for v, e, n in zip(d.variant, d.estimand, d.n_scenes)], fontsize=6)
        ax.axvline(0, color="grey", lw=0.8)
        p = df[(df.kind == "primary") & (df.contrast == cid)]
        for r in p.itertuples():
            ax.axvline(r.theta, ls="--", lw=0.8, color="#c0392b" if r.estimand == "F" else "#2c7fb8")
        ax.set_xlabel(f"{cid} amplification difference (scene-weighted, per drive)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "figures", f"forest_{cid}.png"), dpi=150); plt.close(fig)


def denominators(den, ut, units, out, mask=False):
    rows = []
    for log, d in sorted(den.items()):
        r = {"log": log, **{k: v for k, v in d.items() if k != "not_eligible_by_reason"}}
        for k, v in d["not_eligible_by_reason"].items():
            r[f"not_eligible_{k}"] = v
        toks = [t for t, u in units.items() if u["log"] == log]
        r["units_built_scenes"] = len(toks)
        r["units"] = sum(1 + len(units[t]["alternatives"]) for t in toks)
        for blk in BLOCK_DIR:
            b = ut[(ut.block == blk) & (ut.log_id == log)] if len(ut) else ut
            r[f"{blk}_scenes_analysed"] = b[b.exclusion_reason.isna()].scene_id.nunique() if len(b) else 0
            r[f"{blk}_scenes_excluded_harness"] = b[b.exclusion_reason == "harness_exception"].scene_id.nunique() if len(b) else 0
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out, "denominators.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default=C.SPLIT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-effects", action="store_true")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "paired_effects.csv")):
        raise SystemExit(f"{a.out}/paired_effects.csv exists; refusing to overwrite")
    os.makedirs(a.out, exist_ok=True)
    split, sel, den, units, raw, errors, city = load_run(a.run, a.split)
    ut = build_unit_table(split, sel, units, raw, errors, city)
    keep = ["log_id", "drive_id", "scene_id", "stratum", "pi", "weight", "t_star", "perturbation", "condition", "block", "seed",
            "decode_j", "amplification", "fde5", "recovery", "parser_status", "exclusion_reason", "unit_id", "day_vehicle", "city",
            "model", "task_id", "episode_id", "control_t", "reference_branch", "perturbation_requested",
            "perturbation_achieved", "primary_outcome", "entropy1"]
    ut_out = ut[keep].copy() if len(ut) else ut
    if a.mask_effects and len(ut_out):
        for c in ("amplification", "fde5", "recovery", "primary_outcome", "entropy1"):
            ut_out[c] = MASK
    ut_out.to_parquet(os.path.join(a.out, "units.parquet"), index=False)
    denominators(den, ut, units, a.out, a.mask_effects)
    df, dec, flagged = analyse(ut, split, den, a.out, a.mask_effects)
    me = os.path.abspath(__file__)
    json.dump({"script": me, "script_sha256": C.sha256_file(me), "run": os.path.abspath(a.run), "masked": a.mask_effects,
               "n_unit_rows": int(len(ut)), "blocks_present": sorted(set(ut.block)) if len(ut) else [],
               "n_selected": int(len(sel)), "flagged_logs": flagged, "bootstrap_B": B_BOOT, "seed": SEED},
              open(os.path.join(a.out, "analysis_meta.json"), "w"), indent=1)
    print(f"[done] {len(df)} effect rows, {len(dec)} decisions, masked={a.mask_effects}")


if __name__ == "__main__":
    main()
