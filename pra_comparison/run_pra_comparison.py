#!/usr/bin/env python
"""
ORION vs AutoVLA P/R/A comparison from STORED inference results only (no GPU).

Inputs (read-only)
    ORION    /root/VLA/vla_misalignment_poc/outputs/full_extract/records.jsonl
    AutoVLA  /root/VLA/autovla_misalignment_poc/outputs/full_extract/records.jsonl
             /root/VLA/autovla_misalignment_poc/outputs/annotations/<token>.json

AutoVLA is labeled on arm C (forced ` complex` CoT) because that is the only arm with an
observable perception and reasoning text; its natural arm N has neither. The P/R/A triple is
always taken from ONE generation -- arm C's CoT with arm C's own trajectory. Arm C samples
that hit max_length or emitted more than 10 action tokens are excluded (degenerate output).
Arm N is labeled for A alone, as the natural-behaviour reference.

Outputs -> /root/VLA/pra_comparison/outputs/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pra_labels as L  # noqa: E402

ORION_REC = "/root/VLA/vla_misalignment_poc/outputs/full_extract/records.jsonl"
ORION_DIR = "/root/VLA/vla_misalignment_poc/outputs/full_extract"
AV_REC = "/root/VLA/autovla_misalignment_poc/outputs/full_extract/records.jsonl"
AV_DIR = "/root/VLA/autovla_misalignment_poc/outputs/full_extract"
AV_ANNO = "/root/VLA/autovla_misalignment_poc/outputs/annotations"

HEADLINE = ["P-R-A-", "P+R-A-", "P+R+A-", "P+R+A+"]
ALL8 = [f"P{p}R{r}A{a}" for p in "+-" for r in "+-" for a in "+-"]


# ======================================================================================
# Adapters
# ======================================================================================

def orion_samples() -> list[L.Sample]:
    out = []
    for line in open(ORION_REC):
        r = json.loads(line)
        gr, mr = r["gt_rounds"], r["model_rounds"]
        gt_tl = (r.get("gt_perception") or {}).get("traffic_light")
        known = gt_tl in ("none", "red", "green", "yellow")
        scene = L.scene_from_objects(
            L.chatb2d_objects(gr.get("critical_objects", "")),
            red_light=(gt_tl == "red") if known else None,
            extra_text=gr.get("scene_description", "") or "")
        m_tl = (r.get("model_perception") or {}).get("traffic_light_obj")
        lon, lat = L.orion_decision(mr.get("driving_reason", ""))
        out.append(L.Sample(
            model="ORION", sample_id=r["sample_id"], cluster=r["clip_id"], scene=scene,
            desc_text=(mr.get("scene_description", "") or "") + "\n" + (mr.get("critical_objects", "") or ""),
            desc_objs=L.chatb2d_objects(mr.get("critical_objects", "")),
            rsn_text=mr.get("driving_reason", "") or "",
            decl_lon=lon, decl_lat=lat,
            pred_delta=r["trajectory_pred"][:6], gt_delta=r["trajectory_gt"][:6],
            a_valid=bool(r.get("fut_valid_flag", True)),
            model_red_light=(m_tl == "red") if m_tl in ("none", "red", "green", "yellow") else None,
            gt_light_known=known,
            meta={"scenario": r["scenario"], "tensor_file": os.path.join(ORION_DIR, r["tensor_file"]),
                  "gt_traffic_light": gt_tl, "model_traffic_light_obj": m_tl}))
    return out


def _pos_to_delta(pos, horizon: int) -> list:
    """AutoVLA poses (x fwd, y left) -> ORION deltas (lat +right, lon +forward)."""
    p = np.asarray(pos, dtype=np.float64)[:horizon, :2]
    q = np.stack([-p[:, 1], p[:, 0]], axis=1)
    d = np.diff(np.vstack([np.zeros((1, 2)), q]), axis=0)
    return d.tolist()


def _av_objects(anno: dict) -> list[dict]:
    boxes = np.asarray(anno.get("boxes") or [], dtype=np.float64).reshape(-1, 7)
    names = anno.get("names") or []
    out = []
    for i, b in enumerate(boxes):
        n = names[i] if i < len(names) else ""
        c = "vehicle" if n == "vehicle" else ("vru" if n in ("pedestrian", "bicycle") else None)
        if c:
            out.append({"cls": c, "lat": -float(b[1]), "lon": float(b[0])})
    return out


def autovla_samples(horizon: int) -> tuple[list[L.Sample], list[L.Sample], Counter]:
    arm_c, arm_n, excl = [], [], Counter()
    for line in open(AV_REC):
        r = json.loads(line)
        tok = r["token"]
        anno = json.load(open(os.path.join(AV_ANNO, f"{tok}.json")))
        scene = L.scene_from_objects(_av_objects(anno), red_light=None)
        gt_d = _pos_to_delta(r["trajectory_gt"], horizon)
        base = {"scenario": r.get("map_name"), "log_name": r.get("log_name")}

        n = r["arms"]["N"]
        arm_n.append(L.Sample(
            model="AutoVLA-N", sample_id=tok, cluster=r["log_name"], scene=scene,
            desc_text="", desc_objs=[], rsn_text="", decl_lon=None, decl_lat=None,
            pred_delta=_pos_to_delta(n["trajectory_pred"], horizon) if len(n["trajectory_pred"]) >= horizon else None,
            gt_delta=gt_d, meta=base))

        c = r["arms"]["C"]
        if c["truncated"] or c["runaway_action_tokens"]:
            excl["truncated_or_runaway"] += 1
            continue
        if len(c["trajectory_pred"]) < horizon:
            excl["short_trajectory"] += 1
            continue
        sec = L.autovla_sections(c["generated_text"])
        lon, lat = L.autovla_decision(sec.get("best driving action", ""))
        arm_c.append(L.Sample(
            model="AutoVLA", sample_id=tok, cluster=r["log_name"], scene=scene,
            desc_text=sec.get("scene description", "") + "\n" + sec.get("critical object description", ""),
            desc_objs=[],
            rsn_text=sec.get("reasoning on intent", "") + "\n" + sec.get("best driving action", ""),
            decl_lon=lon, decl_lat=lat,
            pred_delta=_pos_to_delta(c["trajectory_pred"], horizon), gt_delta=gt_d,
            gt_scene_known=bool(anno.get("boxes")),
            meta={**base, "tensor_file": os.path.join(AV_DIR, c["tensor_file"]) if c["tensor_file"] else None,
                  "tensor_file_armN": os.path.join(AV_DIR, n["tensor_file"])}))
    return arm_c, arm_n, excl


# ======================================================================================
# Statistics
# ======================================================================================

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cluster_boot(rows: list[dict], stat, reps: int = 2000, seed: int = 0) -> list[float]:
    by = defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    keys = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in by[k]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def summarise(rows: list[dict]) -> dict:
    dec = [r for r in rows if "?" not in r["group"]]
    g = Counter(r["group"] for r in rows)
    n = len(dec)
    fails = [r for r in dec if r["A"] is False]

    def stage_share(stage):
        def f(sample):
            fa = [r for r in sample if "?" not in r["group"] and r["A"] is False]
            if not fa:
                return None
            if stage == "perception":
                k = sum(r["P"] is False for r in fa)
            elif stage == "reasoning":
                k = sum(r["P"] and r["R"] is False for r in fa)
            else:
                k = sum(r["P"] and r["R"] for r in fa)
            return k / len(fa)
        return f

    def rate(group):
        return lambda s: (sum(r["group"] == group for r in s if "?" not in r["group"])
                          / max(1, sum("?" not in r["group"] for r in s)))

    attribution = {}
    for st, label in (("perception", "P-"), ("reasoning", "P+R-"), ("interface", "P+R+")):
        k = round(stage_share(st)(dec) * len(fails)) if fails else 0
        attribution[st] = {"definition": f"A- with {label}", "count": k,
                           "share_of_A_minus": (k / len(fails)) if fails else 0.0,
                           "ci95_cluster_boot": cluster_boot(dec, stage_share(st))}
    # Declared and executed classes are judged by one accept function, so whenever they are
    # the same class the R_d and A verdicts must agree. Any violation is a labeler bug.
    viol = 0
    same_class_in_prra = 0
    for r in dec:
        rd, ad = r["R_detail"], r["A_detail"]
        # The at-rest text rule deliberately reads the word through the reasoning, so it is
        # the one place declared == executed class may legitimately disagree; skip it.
        if rd.get("decl_lat") is None or rd.get("Rd_at_rest_text_rule"):
            continue
        if rd["decl_lon"] == ad["pred_lon"] and rd["decl_lat"] == ad["pred_lat"]:
            if (rd["Rd_lon_ok"] and rd["Rd_lat_ok"]) != (ad["lon_ok"] and ad["lat_ok"]):
                viol += 1
            if r["group"] == "P+R+A-":
                same_class_in_prra += 1
    return {
        "consistency_violations_Rd_vs_A": viol,
        "P+R+A-_with_declared_equal_executed": same_class_in_prra,
        "n_total": len(rows), "n_decidable": n,
        "n_undecidable": len(rows) - n,
        "undecidable_breakdown": dict(Counter(r["group"] for r in rows if "?" in r["group"])),
        "groups_all8": {k: {"count": g.get(k, 0), "rate": g.get(k, 0) / n if n else 0.0,
                            "ci95_wilson": wilson(g.get(k, 0), n)} for k in ALL8},
        "headline": {k: {"count": g.get(k, 0), "rate": g.get(k, 0) / n if n else 0.0,
                         "ci95_cluster_boot": cluster_boot(dec, rate(k))} for k in HEADLINE},
        "marginals": {"P+": sum(r["P"] is True for r in dec) / max(n, 1),
                      "R+": sum(r["R"] is True for r in dec) / max(n, 1),
                      "A+": sum(r["A"] is True for r in dec) / max(n, 1)},
        "n_A_minus": len(fails),
        "A_minus_attribution": attribution,
        "top_fail_reasons": {
            "P": Counter(x for r in dec for x in r["P_detail"].get("fails", [])).most_common(8),
            "R": Counter(x.split(":")[0] for r in dec for x in r["R_detail"].get("fails", [])).most_common(8),
            "A": Counter(x.split(":")[0] for r in fails for x in r["A_detail"]["reason"].split(",")).most_common(),
        },
    }


# ======================================================================================

def run_config(cfg: L.Config, orion: list, av_c: list, av_n: list) -> dict:
    lab = {m: [L.label(s, cfg) for s in S] for m, S in
           (("ORION", orion), ("AutoVLA", av_c), ("AutoVLA-N", av_n))}
    return lab


def diagnostics(orion: list, av_c: list) -> None:
    """Parser coverage and the lon_class mirror check, printed before anything is trusted."""
    import analysis.trajectory_metrics as TM
    mism = 0
    for s in (orion + av_c)[::7]:
        for d in (s.pred_delta, s.gt_delta):
            if not d:
                continue
            k = TM.action_semantics(d)
            mism += L.lon_class(k.v_start, k.v_end, k.total_distance) != k.longitudinal
    print(f"[diag] lon_class mirror mismatches: {mism}")
    for name, S in (("ORION", orion), ("AutoVLA", av_c)):
        print(f"[diag] {name}: n={len(S)}  decl_lon parsed {sum(s.decl_lon is not None for s in S)}"
              f"  decl_lat parsed {sum(s.decl_lat is not None for s in S)}"
              f"  empty desc {sum(not s.desc_text.strip() for s in S)}"
              f"  empty rsn {sum(not s.rsn_text.strip() for s in S)}")
        print(f"        decl_lon {Counter(s.decl_lon for s in S).most_common()}")
        print(f"        GT lead {sum(s.scene.lead for s in S)}  vru_hazard {sum(s.scene.vru_hazard for s in S)}"
              f"  red {sum(bool(s.scene.red_light) for s in S)}  veh_near {sum(s.scene.veh_near for s in S)}"
              f"  vru_near {sum(s.scene.vru_near for s in S)}")
        print(f"        desc lead_claim {sum(L.lead_claim(s.desc_text) for s in S)}"
              f"  desc vru {sum(L.mentioned(s.desc_text, L.VRU) for s in S)}"
              f"  rsn veh {sum(L.mentioned(s.rsn_text, L.VEH) for s in S)}"
              f"  rsn vru {sum(L.mentioned(s.rsn_text, L.VRU) for s in S)}")


def write_audit(rows: list[dict], samples: dict, path: str, groups: list[str], k: int, seed: int) -> None:
    rng = random.Random(seed)
    with open(path, "w") as f:
        for g in groups:
            pool = [r for r in rows if r["group"] == g]
            pick = rng.sample(pool, min(k, len(pool)))
            f.write(f"\n\n######## {g}  (showing {len(pick)} of {len(pool)})\n")
            for r in pick:
                s = samples[r["sample_id"]]
                f.write(f"\n=== {r['sample_id']}  cluster={r['cluster']}\n")
                f.write(f"scene: {json.dumps(r['scene'])}\n")
                f.write(f"A: {json.dumps(r['A_detail'])}\n")
                f.write(f"P: {json.dumps(r['P_detail'])}\n")
                f.write(f"R: {json.dumps(r['R_detail'])}\n")
                f.write(f"--- description ---\n{s.desc_text.strip()[:900]}\n")
                f.write(f"--- reasoning ---\n{s.rsn_text.strip()[:900]}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--audit-k", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    orion = orion_samples()
    av_c, av_n, excl = autovla_samples(horizon=10)
    av_c3, av_n3, _ = autovla_samples(horizon=6)
    print(f"[load] ORION {len(orion)}  AutoVLA armC {len(av_c)} (excluded {dict(excl)})  armN {len(av_n)}")
    diagnostics(orion, av_c)
    if args.diagnose_only:
        return

    configs = [
        (L.Config(name="headline"), "native"),
        (L.Config(name="strict_no_tolerance", tol_v_min=0.0, tol_v_rel=0.0, tol_lat=0.0, tol_head=0.0), "native"),
        (L.Config(name="lenient_stop_decel", lenient_stop_decel=True), "native"),
        (L.Config(name="with_traffic_light", use_traffic_light=True), "native"),
        (L.Config(name="p_miss_only", p_hallucination=False), "native"),
        (L.Config(name="at_rest_neither_accept", at_rest_neither_accept=True), "native"),
        (L.Config(name="with_traffic_light_at_rest_neither_accept", use_traffic_light=True,
                  at_rest_neither_accept=True), "native"),
        (L.Config(name="with_traffic_light_p_miss_only", use_traffic_light=True, p_hallucination=False), "native"),
        (L.Config(name="common_3s_horizon"), "3s"),
    ]
    results = {}
    headline_rows = None
    for cfg, hz in configs:
        c, n = (av_c, av_n) if hz == "native" else (av_c3, av_n3)
        lab = run_config(cfg, orion, c, n)
        results[cfg.name] = {
            "config": vars(cfg), "autovla_horizon": "5s (10 poses)" if hz == "native" else "3s (6 poses)",
            "ORION": summarise(lab["ORION"]),
            "AutoVLA": summarise(lab["AutoVLA"]),
            "AutoVLA_armN_A_only": {
                "n": sum(r["A"] is not None for r in lab["AutoVLA-N"]),
                "A_minus_rate": (sum(r["A"] is False for r in lab["AutoVLA-N"])
                                 / max(1, sum(r["A"] is not None for r in lab["AutoVLA-N"])))},
        }
        if cfg.name == "headline":
            headline_rows = lab
        print(f"[{cfg.name}] ORION " + " ".join(
            f"{k}={results[cfg.name]['ORION']['groups_all8'][k]['count']}" for k in HEADLINE)
            + " | AutoVLA " + " ".join(
            f"{k}={results[cfg.name]['AutoVLA']['groups_all8'][k]['count']}" for k in HEADLINE))

    # --- per-sample tables and subsets from the headline configuration ------------------
    for model in ("ORION", "AutoVLA"):
        rows = headline_rows[model]
        with open(os.path.join(args.out, f"labels_{model}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        sub = os.path.join(args.out, "subsets", model)
        os.makedirs(sub, exist_ok=True)
        by = defaultdict(list)
        for r in rows:
            by[r["group"]].append({"sample_id": r["sample_id"], "cluster": r["cluster"],
                                   "tensor_file": r.get("tensor_file"),
                                   "A": r["A_detail"], "P_fails": r["P_detail"].get("fails"),
                                   "R_fails": r["R_detail"].get("fails")})
        for g, items in by.items():
            slug = g.replace("+", "p").replace("-", "m").replace("?", "u")
            with open(os.path.join(sub, f"{slug}.json"), "w") as f:
                json.dump({"model": model, "group": g, "n": len(items),
                           "labeling": "pra_comparison headline config", "samples": items}, f, indent=1)
        smap = {s.sample_id: s for s in (orion if model == "ORION" else av_c)}
        write_audit(rows, smap, os.path.join(args.out, f"audit_{model}.txt"),
                    ["P+R+A-", "P+R-A-", "P-R-A-", "P+R+A+"], args.audit_k, seed=1)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"inputs": {"orion": ORION_REC, "autovla": AV_REC},
                   "autovla_excluded": dict(excl), "results": results}, f, indent=1)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
