#!/usr/bin/env python
"""
Analysis of the equal-distance perturbation run (CPU only).

Per scene x condition
  recovery        A+ (P/R/A accept rule, 5 s); invalid output = A-
  amplification   A- AND FDE(5 s) > 3.0 m                      (fixed before running)
  ADE / FDE       5 s and 3 s
  downstream err  fraction of steps after t* != GT;  realign = fraction == GT
  entropy         mean action-token entropy of the generated steps after t*;
                  entropy1 = entropy of the FIRST generated step after the forced token
  err slope       slope of per-pose L2 error over poses t*..9 (m per step)

Analyses
  Q1 Spearman(initial distance, FDE): pooled and within-scene (scene-demeaned ranks)
  Q2 FDE spread inside distance buckets; variance share between buckets / between scenes /
     within scene at equal distance; share of scenes whose equal-distance alternatives MIX
     recovery and amplification, against the decoding-noise floor (original vs original_reseed)
  Q3 entropy of amplifying vs recovering alternatives in mixed scenes (paired), AUROC
  Q4 cross-validated AUROC (log-grouped folds) for amplification from
       distance | distance+direction | scene (leave-one-out rate) | all
  Q5 is the model's own token special: original vs its own equal-distance alternatives
CIs: log-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, PRA_DIR)
import pra_labels as L                                  # noqa: E402
from run_pra_comparison import _pos_to_delta            # noqa: E402

N_ACT = 10
CFG = L.Config()
AMP_FDE = 3.0
REPS = 2000
SECTORS = ["faster", "faster-left", "left", "slower-left", "slower", "slower-right", "right", "faster-right"]


def a_eval(traj, gt, h):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, h), gt_delta=_pos_to_delta(gt, h))
    return L.label_A(s, CFG)[0]


def boot(rows, key, reps=REPS, seed=0):
    items = [(r["log"], float(r[key])) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    if not items:
        return None
    by = defaultdict(list)
    for lg, v in items:
        by[lg].append(v)
    logs = list(by)
    rng = random.Random(seed)
    bs = [np.mean([v for lg in (rng.choice(logs) for _ in logs) for v in by[lg]]) for _ in range(reps)]
    return {"mean": float(np.mean([v for _, v in items])), "median": float(np.median([v for _, v in items])),
            "n": len(items), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def boot_stat(units, fn, reps=REPS, seed=0):
    """units: list of (log, payload); fn(list_of_payloads) -> float."""
    by = defaultdict(list)
    for lg, x in units:
        by[lg].append(x)
    logs = list(by)
    pt = fn([x for _, x in units])
    rng = random.Random(seed)
    bs = []
    for _ in range(reps):
        v = fn([x for lg in (rng.choice(logs) for _ in logs) for x in by[lg]])
        if v is not None and np.isfinite(v):
            bs.append(v)
    if pt is None or not bs:
        return None
    return {"value": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(units)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    # STRICT=1: drop alternatives that only entered through the whole-codebook fill (outside the
    # widest relative tolerance), so conclusions can be checked on genuinely equal-distance tokens.
    strict = os.environ.get("STRICT") == "1"
    suffix = "_strict" if strict else ""

    rows = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        after = list(range(t + 1, N_ACT))
        alt_meta = {f"alt{j}": a for j, a in enumerate(r["alternatives"])}
        for name, c in r["conditions"].items():
            kind = "alt" if name.startswith("alt") else name
            meta = alt_meta.get(name, {})
            if strict and kind == "alt" and not meta.get("matched_strict", True):
                continue
            d = (meta.get("d_gt") if kind == "alt" else (r["geometry"]["d_pred"] if kind.startswith("original") else 0.0))
            acts = c["action_idx"]
            row = {"token": r["token"], "log": r["log"], "group": r["group"], "cond": name, "kind": kind,
                   "t_star": t, "dist": d, "rel_err": meta.get("rel_err", 0.0 if kind.startswith("original") else None),
                   "sector": meta.get("sector", r["geometry"]["pred_sector"] if kind.startswith("original") else "gt"),
                   "same_sector_as_pred": meta.get("same_sector_as_pred"),
                   "vec_forward": meta.get("vec_forward", r["geometry"]["pred_vec_forward"] if kind.startswith("original") else 0.0),
                   "vec_left": meta.get("vec_left", r["geometry"]["pred_vec_left"] if kind.startswith("original") else 0.0),
                   "shape_d_gt": meta.get("shape_d_gt", r["geometry"]["pred_shape_d_gt"] if kind.startswith("original") else 0.0),
                   "valid": c["valid"],
                   "downstream_err": float(np.mean([k >= len(acts) or acts[k] != gt[k] for k in after])) if after else None,
                   "realign": float(np.mean([k < len(acts) and acts[k] == gt[k] for k in after])) if after else None,
                   "entropy": float(np.mean(c["entropy_steps"])) if c["entropy_steps"] else None,
                   "entropy1": float(c["entropy_steps"][0]) if c["entropy_steps"] else None}
            if c["valid"]:
                p, g_ = np.asarray(c["trajectory_pred"], float), np.asarray(gtraj, float)[:, :2]
                e = np.linalg.norm(p - g_, axis=1)
                row.update({"recovery": bool(a_eval(c["trajectory_pred"], gtraj, 10)),
                            "recovery3": bool(a_eval(c["trajectory_pred"], gtraj, 6)),
                            "ade5": float(e.mean()), "fde5": float(e[-1]), "ade3": float(e[:6].mean()), "fde3": float(e[5]),
                            "err_slope": float(np.polyfit(np.arange(t, N_ACT), e[t:], 1)[0]) if N_ACT - t >= 2 else None})
            else:
                row.update({"recovery": False, "recovery3": False, "ade5": None, "fde5": None, "ade3": None, "fde3": None,
                            "err_slope": None})
            row["amplification"] = bool((not row["recovery"]) and row["fde5"] is not None and row["fde5"] > AMP_FDE) \
                if row["fde5"] is not None else (not row["recovery"])
            rows.append(row)

    groups = ["A-", "A+"]
    S = {"n_scenes": {g: sum(r["group"] == g for r in recs) for g in groups},
         "alts_per_scene": {g: float(np.mean([len(r["alternatives"]) for r in recs if r["group"] == g])) for g in groups},
         "rel_used": dict(Counter(r["geometry"]["rel_used"] for r in recs)),
         "alt_rel_err_median": float(np.median([a["rel_err"] for r in recs for a in r["alternatives"]])),
         "alt_abs_dist_diff_median_m": float(np.median([abs(a["d_gt"] - r["geometry"]["d_pred"]) for r in recs for a in r["alternatives"]]))}

    # ---------------- summary table per group x condition kind (+ alt sectors) ----------------
    keys = ["dist", "recovery", "recovery3", "amplification", "ade5", "fde5", "entropy", "entropy1", "err_slope", "downstream_err", "realign"]
    S["table"] = {}
    for g in groups:
        S["table"][g] = {}
        for label, pred in ([("original", lambda x: x["kind"] == "original"), ("original_reseed", lambda x: x["kind"] == "original_reseed"),
                             ("gt", lambda x: x["kind"] == "gt"), ("alt (all)", lambda x: x["kind"] == "alt"),
                             ("alt same direction as original", lambda x: x["kind"] == "alt" and x["same_sector_as_pred"]),
                             ("alt other directions", lambda x: x["kind"] == "alt" and x["same_sector_as_pred"] is False)]
                            + [(f"alt: {s}", (lambda s_: (lambda x: x["kind"] == "alt" and x["sector"] == s_))(s)) for s in SECTORS]):
            sel = [x for x in rows if x["group"] == g and pred(x)]
            S["table"][g][label] = {k: boot(sel, k) for k in keys}
            S["table"][g][label]["n"] = len(sel)

    by_scene = defaultdict(dict)
    for x in rows:
        by_scene[x["token"]][x["cond"]] = x

    S["Q1"], S["Q2"], S["Q3"], S["Q4"], S["Q5"] = {}, {}, {}, {}, {}
    for g in groups + ["ALL"]:
        pert = [x for x in rows if (g == "ALL" or x["group"] == g) and x["kind"] in ("original", "alt") and x["fde5"] is not None]
        # Q1 pooled and within-scene correlation
        S["Q1"][g] = {
            "spearman_dist_fde_pooled": boot_stat([(x["log"], (x["dist"], x["fde5"])) for x in pert],
                                                  lambda ps: spearmanr([p[0] for p in ps], [p[1] for p in ps]).correlation if len(ps) > 5 else None),
        }
        within = []
        for tok, cs in by_scene.items():
            xs = [c for c in cs.values() if c["kind"] in ("original", "alt") and c["fde5"] is not None and (g == "ALL" or c["group"] == g)]
            if len(xs) >= 3 and np.std([c["dist"] for c in xs]) > 0:
                dr = np.argsort(np.argsort([c["dist"] for c in xs])) - (len(xs) - 1) / 2
                fr = np.argsort(np.argsort([c["fde5"] for c in xs])) - (len(xs) - 1) / 2
                within += [(xs[0]["log"], (a, b)) for a, b in zip(dr, fr)]
        S["Q1"][g]["spearman_dist_fde_within_scene"] = boot_stat(
            within, lambda ps: float(np.corrcoef([p[0] for p in ps], [p[1] for p in ps])[0, 1]) if len(ps) > 5 else None)

        # Q2 buckets and variance decomposition over perturbed conditions
        if pert:
            qs = np.quantile([x["dist"] for x in pert], [0.2, 0.4, 0.6, 0.8])
            for x in pert:
                x["bucket"] = int(np.searchsorted(qs, x["dist"]))
            buckets = {}
            for b in range(5):
                sel = [x for x in pert if x["bucket"] == b]
                if sel:
                    f = np.array([x["fde5"] for x in sel])
                    buckets[b] = {"n": len(sel), "dist_range_m": [float(min(x["dist"] for x in sel)), float(max(x["dist"] for x in sel))],
                                  "fde_mean": float(f.mean()), "fde_sd": float(f.std()), "fde_p10": float(np.percentile(f, 10)),
                                  "fde_p90": float(np.percentile(f, 90)),
                                  "amplification_rate": float(np.mean([x["amplification"] for x in sel])),
                                  "recovery_rate": float(np.mean([x["recovery"] for x in sel]))}
            f_all = np.array([x["fde5"] for x in pert]); tot = f_all.var()
            bmean = {b: np.mean([x["fde5"] for x in pert if x["bucket"] == b]) for b in buckets}
            smean = defaultdict(list)
            for x in pert:
                smean[x["token"]].append(x["fde5"])
            smean = {k: np.mean(v) for k, v in smean.items()}
            between_bucket = np.mean([(bmean[x["bucket"]] - f_all.mean()) ** 2 for x in pert])
            between_scene = np.mean([(smean[x["token"]] - f_all.mean()) ** 2 for x in pert])
            within_scene = np.mean([(x["fde5"] - smean[x["token"]]) ** 2 for x in pert])
            S["Q2"][g] = {"buckets": buckets, "fde_variance_total": float(tot),
                          "share_between_distance_buckets": float(between_bucket / tot) if tot else None,
                          "share_between_scenes": float(between_scene / tot) if tot else None,
                          "share_within_scene": float(within_scene / tot) if tot else None}
        # mixing of outcomes among equal-distance alternatives, vs noise floor
        units, noise_units = [], []
        for tok, cs in by_scene.items():
            alts = [c for c in cs.values() if c["kind"] == "alt"]
            if not alts or (g != "ALL" and alts[0]["group"] != g):
                continue
            rec = sum(c["recovery"] for c in alts); amp = sum(c["amplification"] for c in alts)
            fdes = [c["fde5"] for c in alts if c["fde5"] is not None]
            units.append((alts[0]["log"], {"mixed": rec > 0 and amp > 0, "n_rec": rec, "n_amp": amp, "n": len(alts),
                                           "fde_sd": float(np.std(fdes)) if len(fdes) > 1 else np.nan,
                                           "fde_range": float(np.ptp(fdes)) if len(fdes) > 1 else np.nan,
                                           "all_recover": rec == len(alts), "all_amplify": amp == len(alts)}))
            o, o2 = cs.get("original"), cs.get("original_reseed")
            if o and o2 and o["fde5"] is not None and o2["fde5"] is not None:
                noise_units.append((o["log"], {"fde_absdiff": abs(o["fde5"] - o2["fde5"]),
                                               "outcome_flip": o["recovery"] != o2["recovery"],
                                               "tokens_same": cs["original"]["valid"] and True}))
        S["Q2"].setdefault(g, {})
        S["Q2"][g]["scenes_with_mixed_alt_outcomes"] = boot_stat(units, lambda ps: float(np.mean([p["mixed"] for p in ps])) if ps else None)
        S["Q2"][g]["scenes_all_alts_recover"] = boot_stat(units, lambda ps: float(np.mean([p["all_recover"] for p in ps])) if ps else None)
        S["Q2"][g]["scenes_all_alts_amplify"] = boot_stat(units, lambda ps: float(np.mean([p["all_amplify"] for p in ps])) if ps else None)
        S["Q2"][g]["within_scene_alt_fde_sd"] = boot_stat(units, lambda ps: float(np.nanmean([p["fde_sd"] for p in ps])) if ps else None)
        S["Q2"][g]["within_scene_alt_fde_range"] = boot_stat(units, lambda ps: float(np.nanmedian([p["fde_range"] for p in ps])) if ps else None)
        S["Q2"][g]["noise_floor_original_vs_reseed_fde_absdiff"] = boot_stat(noise_units, lambda ps: float(np.mean([p["fde_absdiff"] for p in ps])) if ps else None)
        S["Q2"][g]["noise_floor_outcome_flip_rate"] = boot_stat(noise_units, lambda ps: float(np.mean([p["outcome_flip"] for p in ps])) if ps else None)

        # Q3 entropy: amplifying vs recovering alternatives inside mixed scenes (paired)
        pairs = []
        for tok, cs in by_scene.items():
            alts = [c for c in cs.values() if c["kind"] == "alt" and c["entropy"] is not None]
            if not alts or (g != "ALL" and alts[0]["group"] != g):
                continue
            a = [c for c in alts if c["amplification"]]; r_ = [c for c in alts if c["recovery"]]
            if a and r_:
                pairs.append((alts[0]["log"], {"d_ent": np.mean([c["entropy"] for c in a]) - np.mean([c["entropy"] for c in r_]),
                                               "d_ent1": np.mean([c["entropy1"] for c in a]) - np.mean([c["entropy1"] for c in r_]),
                                               "d_dist": np.mean([c["dist"] for c in a]) - np.mean([c["dist"] for c in r_])}))
        S["Q3"][g] = {"n_mixed_scenes": len(pairs),
                      "entropy_amplifying_minus_recovering": boot_stat(pairs, lambda ps: float(np.mean([p["d_ent"] for p in ps])) if ps else None),
                      "first_step_entropy_amplifying_minus_recovering": boot_stat(pairs, lambda ps: float(np.mean([p["d_ent1"] for p in ps])) if ps else None),
                      "distance_amplifying_minus_recovering_m": boot_stat(pairs, lambda ps: float(np.mean([p["d_dist"] for p in ps])) if ps else None),
                      "wilcoxon_p_entropy": float(wilcoxon([p["d_ent"] for _, p in pairs]).pvalue) if len(pairs) >= 6 else None}
        altr = [x for x in rows if x["kind"] == "alt" and (g == "ALL" or x["group"] == g) and x["entropy"] is not None]
        if len({x["amplification"] for x in altr}) == 2:
            S["Q3"][g]["auroc_entropy_for_amplification"] = float(roc_auc_score([x["amplification"] for x in altr], [x["entropy"] for x in altr]))
            S["Q3"][g]["auroc_first_step_entropy_for_amplification"] = float(roc_auc_score([x["amplification"] for x in altr], [x["entropy1"] for x in altr]))
            S["Q3"][g]["auroc_distance_for_amplification"] = float(roc_auc_score([x["amplification"] for x in altr], [x["dist"] for x in altr]))

        # Q4 cross-validated prediction of amplification among alternatives
        if len({x["amplification"] for x in altr}) == 2 and len({x["log"] for x in altr}) >= 5:
            y = np.array([int(x["amplification"]) for x in altr]); logs = np.array([x["log"] for x in altr])
            scene_rate = []
            for x in altr:                                          # leave-one-out scene amplification rate
                others = [c for c in by_scene[x["token"]].values() if c["kind"] == "alt" and c["cond"] != x["cond"]]
                scene_rate.append(np.mean([c["amplification"] for c in others]) if others else 0.5)
            sec = np.array([[x["sector"] == s for s in SECTORS] for x in altr], float)
            feats = {
                "distance": np.array([[x["dist"]] for x in altr]),
                "distance+direction": np.column_stack([[x["dist"] for x in altr], [x["vec_forward"] for x in altr],
                                                       [x["vec_left"] for x in altr], [x["shape_d_gt"] for x in altr], sec]),
                "scene (leave-one-out)": np.array([[v] for v in scene_rate]),
                "scene+distance+direction": np.column_stack([scene_rate, [x["dist"] for x in altr], [x["vec_forward"] for x in altr],
                                                             [x["vec_left"] for x in altr], [x["shape_d_gt"] for x in altr], sec]),
                "first-step entropy (mediator)": np.array([[x["entropy1"]] for x in altr]),
            }
            S["Q4"][g] = {}
            for name, X in feats.items():
                oof = np.full(len(y), np.nan)
                for tr, te in GroupKFold(n_splits=min(5, len(set(logs)))).split(X, y, logs):
                    if len(set(y[tr])) < 2:
                        continue
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
                    oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
                ok = np.isfinite(oof)
                units_ = [(lg, (yy, pp)) for lg, yy, pp in zip(logs[ok], y[ok], oof[ok])]
                S["Q4"][g][name] = boot_stat(units_, lambda ps: float(roc_auc_score([p[0] for p in ps], [p[1] for p in ps]))
                                             if len({p[0] for p in ps}) == 2 else None, reps=1000)
            # within-scene: same direction as the original token vs other directions
            dpairs = []
            for tok, cs in by_scene.items():
                alts = [c for c in cs.values() if c["kind"] == "alt"]
                if not alts or (g != "ALL" and alts[0]["group"] != g):
                    continue
                sm = [c for c in alts if c["same_sector_as_pred"]]; ot = [c for c in alts if c["same_sector_as_pred"] is False]
                if sm and ot:
                    dpairs.append((alts[0]["log"], np.mean([c["amplification"] for c in sm]) - np.mean([c["amplification"] for c in ot])))
            S["Q4"][g]["same_direction_as_original_minus_other_directions_amplification"] = boot_stat(
                dpairs, lambda ps: float(np.mean(ps)) if ps else None)

        # Q5 original token vs its equal-distance alternatives
        units5 = []
        for tok, cs in by_scene.items():
            o = cs.get("original"); alts = [c for c in cs.values() if c["kind"] == "alt"]
            if not o or not alts or (g != "ALL" and o["group"] != g):
                continue
            af = [c["fde5"] for c in alts if c["fde5"] is not None]
            units5.append((o["log"], {"d_rec": float(o["recovery"]) - np.mean([c["recovery"] for c in alts]),
                                      "d_amp": float(o["amplification"]) - np.mean([c["amplification"] for c in alts]),
                                      "d_fde": (o["fde5"] - np.median(af)) if (o["fde5"] is not None and af) else np.nan,
                                      "pct": (np.mean([f < o["fde5"] for f in af]) if (o["fde5"] is not None and af) else np.nan),
                                      "d_ent": (o["entropy"] - np.mean([c["entropy"] for c in alts if c["entropy"] is not None]))
                                      if o["entropy"] is not None else np.nan}))
        S["Q5"][g] = {
            "recovery_original_minus_alts": boot_stat(units5, lambda ps: float(np.mean([p["d_rec"] for p in ps])) if ps else None),
            "amplification_original_minus_alts": boot_stat(units5, lambda ps: float(np.mean([p["d_amp"] for p in ps])) if ps else None),
            "fde_original_minus_alt_median_m": boot_stat(units5, lambda ps: float(np.nanmean([p["d_fde"] for p in ps])) if ps else None),
            "original_fde_percentile_among_alts": boot_stat(units5, lambda ps: float(np.nanmean([p["pct"] for p in ps])) if ps else None),
            "entropy_original_minus_alts": boot_stat(units5, lambda ps: float(np.nanmean([p["d_ent"] for p in ps])) if ps else None),
            "wilcoxon_p_fde": float(wilcoxon([p["d_fde"] for _, p in units5 if np.isfinite(p["d_fde"])]).pvalue)
            if sum(np.isfinite(p["d_fde"]) for _, p in units5) >= 6 else None,
        }

    with open(os.path.join(args.run, f"summary{suffix}.json"), "w") as f:
        json.dump(S, f, indent=1)
    with open(os.path.join(args.run, f"rows{suffix}.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")

    # ------------------------------------------------------------------ report
    def pc(d):
        return "–" if d is None else f"{100*d['mean']:.1f}% [{100*d['ci95'][0]:.0f}, {100*d['ci95'][1]:.0f}]"

    def nm(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def bs(d, nd=2, pct=False):
        if d is None:
            return "–"
        m = 100 if pct else 1
        u = "%" if pct else ""
        return f"{m*d['value']:+.{nd}f}{u} [{m*d['ci95'][0]:+.{nd}f}, {m*d['ci95'][1]:+.{nd}f}]"

    W = []
    w = W.append
    w("# AutoVLA Natural/Fast — 같은 거리의 action-token perturbation은 같은 결과를 만드는가\n")
    w(f"GPU1, natural fast 경로만. 장면 A− {S['n_scenes']['A-']}, A+ {S['n_scenes']['A+']} (A− 1개당 첫 불일치 step이 같은 A+ 3개). "
      f"장면당 대안 토큰 A− {S['alts_per_scene']['A-']:.1f}개 / A+ {S['alts_per_scene']['A+']:.1f}개, 원래 오차 거리와의 상대 차이 중앙값 "
      f"{100*S['alt_rel_err_median']:.1f}% (절대 {1000*S['alt_abs_dist_diff_median_m']:.1f} mm). 허용오차 단계 사용: {S['rel_used']}. "
      f"증폭 = A− 이면서 FDE > {AMP_FDE} m (사전 고정). [ ]는 log 단위 cluster bootstrap 95% CI.\n")
    w("## 요약 (그룹 × 조건)\n")
    w("| 그룹 | 조건 | n | perturbation 거리 (m) | recovery (A+) | amplification | ADE (m) | FDE (m) | post-mismatch entropy | 오차 증가 (m/step) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in groups:
        for label, d in S["table"][g].items():
            if d["n"] == 0:
                continue
            dist = "0" if label == "gt" else (f"{d['dist']['median']:.3f}" if d["dist"] else "–")
            w(f"| {g} | {label} | {d['n']} | {dist} | {pc(d['recovery'])} | {pc(d['amplification'])} | {nm(d['ade5'])} | {nm(d['fde5'])} | "
              f"{nm(d['entropy'], 3)} | {nm(d['err_slope'], 3)} |")
    w("")
    w("## Q1. 초기 codebook 거리와 FDE의 상관\n")
    w("| 그룹 | Spearman (전체 조건 풀링) | 장면 내 순위 상관 |")
    w("|---|---|---|")
    for g in groups + ["ALL"]:
        q = S["Q1"][g]
        w(f"| {g} | {bs(q['spearman_dist_fde_pooled'])} | {bs(q['spearman_dist_fde_within_scene'])} |")
    w("")
    w("## Q2. 같은 거리 구간 안의 FDE 분산, 장면 안에서 결과가 갈리는가\n")
    w("| 그룹 | 대안들이 recovery와 amplification으로 갈린 장면 | 대안 전부 recovery | 대안 전부 amplification | 장면 내 대안 FDE SD (m) | 장면 내 대안 FDE 범위 중앙값 (m) | noise floor: 같은 토큰 reseed FDE 차이 (m) | noise floor: 결과 뒤집힘 |")
    w("|---|---|---|---|---|---|---|---|")
    for g in groups + ["ALL"]:
        q = S["Q2"][g]
        w(f"| {g} | {bs(q['scenes_with_mixed_alt_outcomes'], 1, True)} | {bs(q['scenes_all_alts_recover'], 1, True)} | "
          f"{bs(q['scenes_all_alts_amplify'], 1, True)} | {bs(q['within_scene_alt_fde_sd'])} | {bs(q['within_scene_alt_fde_range'])} | "
          f"{bs(q['noise_floor_original_vs_reseed_fde_absdiff'])} | {bs(q['noise_floor_outcome_flip_rate'], 1, True)} |")
    w("")
    for g in groups + ["ALL"]:
        q = S["Q2"][g]
        if "buckets" not in q:
            continue
        w(f"**{g}** 거리 5분위 구간 (perturbed 조건: original + 대안). 분산 분해: 거리 구간 간 {100*q['share_between_distance_buckets']:.1f}% · "
          f"장면 간 {100*q['share_between_scenes']:.1f}% · 장면 내 {100*q['share_within_scene']:.1f}%\n")
        w("| 구간 | n | 거리 범위 (m) | FDE 평균 | FDE SD | FDE p10–p90 | amplification | recovery |")
        w("|---|---:|---|---:|---:|---|---:|---:|")
        for b, d in q["buckets"].items():
            w(f"| {b} | {d['n']} | {d['dist_range_m'][0]:.3f}–{d['dist_range_m'][1]:.3f} | {d['fde_mean']:.2f} | {d['fde_sd']:.2f} | "
              f"{d['fde_p10']:.2f}–{d['fde_p90']:.2f} | {100*d['amplification_rate']:.1f}% | {100*d['recovery_rate']:.1f}% |")
        w("")
    w("## Q3. stable vs unstable rollout의 entropy (결과가 갈린 장면 안에서 쌍대 비교)\n")
    w("| 그룹 | 갈린 장면 수 | entropy 차이 (증폭 − 회복) | 첫 생성 step entropy 차이 | 거리 차이 (증폭 − 회복, m) | AUROC entropy / 첫 step entropy / 거리 |")
    w("|---|---:|---|---|---|---|")
    for g in groups + ["ALL"]:
        q = S["Q3"][g]
        au = (f"{q.get('auroc_entropy_for_amplification', float('nan')):.2f} / {q.get('auroc_first_step_entropy_for_amplification', float('nan')):.2f} / "
              f"{q.get('auroc_distance_for_amplification', float('nan')):.2f}") if "auroc_entropy_for_amplification" in q else "–"
        w(f"| {g} | {q['n_mixed_scenes']} | {bs(q['entropy_amplifying_minus_recovering'], 3)} | {bs(q['first_step_entropy_amplifying_minus_recovering'], 3)} | "
          f"{bs(q['distance_amplifying_minus_recovering_m'], 4)} | {au} |")
    w("")
    w("## Q4. 무엇이 증폭을 설명하는가 (대안 조건, log 단위 교차검증 AUROC)\n")
    w("| 그룹 | 거리만 | 거리+방향 | 장면 (leave-one-out) | 장면+거리+방향 | 첫 step entropy (매개변수) | 원래 토큰과 같은 방향 − 다른 방향 amplification |")
    w("|---|---|---|---|---|---|---|")
    for g in groups + ["ALL"]:
        q = S["Q4"].get(g, {})
        if not q:
            continue
        w(f"| {g} | {bs(q.get('distance'))} | {bs(q.get('distance+direction'))} | {bs(q.get('scene (leave-one-out)'))} | "
          f"{bs(q.get('scene+distance+direction'))} | {bs(q.get('first-step entropy (mediator)'))} | "
          f"{bs(q.get('same_direction_as_original_minus_other_directions_amplification'), 1, True)} |")
    w("")
    w("## Q5. 모델이 원래 고른 토큰이 같은 거리의 대안보다 특별히 불안정한가\n")
    w("| 그룹 | recovery (원래 − 대안 평균) | amplification (원래 − 대안 평균) | FDE (원래 − 대안 중앙값, m) | 원래 FDE의 대안 내 백분위 | entropy (원래 − 대안) | Wilcoxon p (FDE) |")
    w("|---|---|---|---|---|---|---|")
    for g in groups + ["ALL"]:
        q = S["Q5"][g]
        wp = "–" if q["wilcoxon_p_fde"] is None else f"{q['wilcoxon_p_fde']:.2g}"
        w(f"| {g} | {bs(q['recovery_original_minus_alts'], 1, True)} | {bs(q['amplification_original_minus_alts'], 1, True)} | "
          f"{bs(q['fde_original_minus_alt_median_m'])} | {bs(q['original_fde_percentile_among_alts'], 1, True)} | "
          f"{bs(q['entropy_original_minus_alts'], 3)} | {wp} |")
    with open(os.path.join(args.run, f"EQUAL_DISTANCE{suffix}.md"), "w") as f:
        f.write("\n".join(W))
    print("\n".join(W))


if __name__ == "__main__":
    main()
