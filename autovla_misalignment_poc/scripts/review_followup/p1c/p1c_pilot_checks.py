#!/usr/bin/env python
"""
P1-C pilot checks (protocol §6) - CPU helpers around the GPU runs made by p1c_pilot.sh. Never reads an effect.
  p1-tokens  --out F [--n 30]            30 PoC scenes with a valid full_extract arm-N record, sha256("P1C-P1|tok") order
  p3-tree    --tokens F --n 5 --tree D   symlink tree with ONLY CAM_F0/L1/R1 of the logs of the first n tokens
  report     --pilot DIR                 evaluate P1 (Planner natural pass vs full_extract arm N), P2 (H8 debug identities
                                         + record completeness / per-condition denominators), P3 (3-camera tree run ==
                                         normal run; model reads 3 cameras), ED-wrapper parity with stored dev ED,
                                         runtimes, disk; writes DIR/checks.json and DIR/PILOT_OK if all pass.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

FULL = f"{C.POC}/outputs/full_extract/records.jsonl"
DEV_ED = f"{C.POC}/outputs/equal_distance_perturbation/records.jsonl"
POC_SCENES = f"{C.NUP}/navtest_poc"


def iter_full(want=None):
    for l in open(FULL):
        r = json.loads(l)
        if want is None or r["token"] in want:
            yield r


def cmd_p1_tokens(a):
    if os.path.exists(a.out):
        print("exists"); return
    ok = []
    for r in iter_full():
        n = r["arms"]["N"]
        if not n["cot_present"] and not n["runaway_action_tokens"] and len(n["trajectory_pred"]) == 10:
            ok.append(r["token"])
    ok.sort(key=lambda t: hashlib.sha256(f"P1C-P1|{t}".encode()).hexdigest())
    json.dump(ok[:a.n], open(a.out, "w"))
    print(f"{min(a.n, len(ok))} of {len(ok)} valid PoC scenes")


def cmd_p3_tree(a):
    toks = json.load(open(a.tokens))[:a.n]
    tree = os.path.abspath(a.tree)
    assert tree.startswith(C.P1C + "/pilot_20261008/"), tree
    os.makedirs(tree, exist_ok=True)
    logs = sorted({C.log_of_scene(json.load(open(f"{POC_SCENES}/{t}.json"))) for t in toks})
    for L in logs:
        os.makedirs(os.path.join(tree, L), exist_ok=True)
        for cam in C.CAMS:
            dst = os.path.join(tree, L, cam)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(C.SENSOR, L, cam), dst)
    json.dump(toks, open(os.path.join(os.path.dirname(tree), "p3_tokens.json"), "w"))
    print(f"tree with {len(logs)} logs x {C.CAMS}; available cameras in the real dirs: "
          f"{sorted({c for L in logs for c in os.listdir(os.path.join(C.SENSOR, L))})}")


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def cmd_report(a):
    P = os.path.abspath(a.pilot)
    out = {}
    A0 = 151665  # recomputed below from records if possible
    # ---- P1: Planner natural pass vs full_extract arm N on 30 PoC scenes -------------------------------------
    s1 = {r["token"]: r for r in read_jsonl(f"{P}/checks/p1_stage1/records.jsonl")}
    ref = {r["token"]: r for r in iter_full(set(s1))}
    rows = []
    for t, r in s1.items():
        f = ref[t]; n = f["arms"]["N"]
        A0 = r["arms"]["N"]["action_token_ids"][0] - r["arms"]["N"]["pred_action_idx"][0]
        fa = next(j for j, x in enumerate(n["token_ids"]) if x >= A0)
        stub_ref = n["token_ids"][:fa]
        stub_new = r["arms"]["N"]["token_ids"][:r["p1c"]["stub_len"]]
        pr_ref, pr_new = n["pred_action_idx"][:10], r["arms"]["N"]["pred_action_idx"]
        gt_ref = f["gt_action_idx"]
        ts = lambda p, g: next((k for k in range(10) if p[k] != g[k]), None)
        rows.append({"token": t, "actions_equal": pr_ref == pr_new, "stub_equal": stub_ref == stub_new,
                     "gt_equal": gt_ref == r["gt_action_idx"], "t_star_equal": ts(pr_ref, gt_ref) == r["p1c"]["t_star"],
                     "first_token_equal": pr_ref[0] == pr_new[0],
                     "n_equal_positions": sum(x == y for x, y in zip(pr_ref, pr_new))})
    n = len(rows)
    p1 = {"n_scenes": n, "n_errors": len(read_jsonl(f"{P}/checks/p1_stage1/errors.jsonl")),
          "full_sequence_agreement": sum(r["actions_equal"] for r in rows) / max(1, n),
          "stub_ids_differ_rate": 1 - sum(r["stub_equal"] for r in rows) / max(1, n),
          "gt_tokens_agreement": sum(r["gt_equal"] for r in rows) / max(1, n),
          "t_star_agreement": sum(r["t_star_equal"] for r in rows) / max(1, n),
          "first_token_agreement": sum(r["first_token_equal"] for r in rows) / max(1, n),
          "mean_equal_positions": sum(r["n_equal_positions"] for r in rows) / max(1, n),
          "per_scene": rows}
    p1["pass"] = n >= 30 and p1["full_sequence_agreement"] >= 0.85 and p1["stub_ids_differ_rate"] <= 0.05
    out["P1"] = p1
    # ---- P3a: 3-camera symlink tree run == normal run ------------------------------------------------------------
    p3 = {r["token"]: r for r in read_jsonl(f"{P}/checks/p3_stage1_3cam/records.jsonl")}
    same = [p3[t]["arms"]["N"]["token_ids"] == s1[t]["arms"]["N"]["token_ids"] for t in p3 if t in s1]
    ent = [max(abs(x - y) for x, y in zip(p3[t]["p1c"]["entropy_steps"], s1[t]["p1c"]["entropy_steps"])) for t in p3 if t in s1]
    src = open("/root/VLA/autovla/models/autovla.py").read()
    out["P3_cameras"] = {"n_scenes": len(same), "token_ids_identical": sum(same), "max_abs_entropy_diff": max(ent) if ent else None,
                         "model_camera_types_line": next((l.strip() for l in src.splitlines() if "camera_types = [" in l), None),
                         "vs_full_extract_note": "P1 compares runs on today's PoC dirs (F0/L1/R1 + partial L0/R0) with full_extract "
                                                 "records made when the PoC logs had all 8 cameras",
                         "pass": len(same) >= 3 and all(same)}
    # ---- ED wrapper parity with stored dev ED -------------------------------------------------------------------
    dev = {r["token"]: r for r in read_jsonl(DEV_ED)}
    par = []
    for r in read_jsonl(f"{P}/checks/ed_parity/records.jsonl"):
        d = dev[r["token"]]
        par.append({"token": r["token"], "alternatives_equal": [x["token"] for x in r["alternatives"]] == [x["token"] for x in d["alternatives"]],
                    "conditions_equal": {c: r["conditions"][c]["action_idx"] == d["conditions"][c]["action_idx"] for c in d["conditions"]}})
    out["ED_parity"] = {"n_scenes": len(par), "alternatives_identical": sum(p["alternatives_equal"] for p in par),
                        "condition_rows_identical": sum(sum(p["conditions_equal"].values()) for p in par),
                        "condition_rows": sum(len(p["conditions_equal"]) for p in par), "per_scene": par,
                        "pass": len(par) >= 1 and all(p["alternatives_equal"] for p in par)}
    # ---- P2 (protocol): H8 debug identities ----------------------------------------------------------------------
    dbg = json.load(open(f"{P}/checks/h8_debug/debug_checks.json")) if os.path.exists(f"{P}/checks/h8_debug/debug_checks.json") else []
    out["P2_h8_debug"] = {"n_debug_steps": len(dbg),
                          "emb_patch_vs_recent_gt_logits_L1": [d["emb_patch_vs_recent_gt_logits_L1"] for d in dbg],
                          "L35_patch_vs_gt_history_logits_L1": [d["L35_patch_vs_gt_history_logits_L1"] for d in dbg],
                          "selfpatch_js_vs_normal": [d["selfpatch_js_vs_normal"] for d in dbg],
                          "prefix_kv_checksum": "asserted unchanged after every unit inside prev_action_state_patching.py "
                                                "(run aborts the unit otherwise); see h8_debug errors.jsonl",
                          "h8_debug_unit_errors": len(read_jsonl(f"{P}/checks/h8_debug/errors.jsonl"))}
    out["P2_h8_debug"]["pass"] = (len(dbg) >= 2 and all(x == 0 for x in out["P2_h8_debug"]["emb_patch_vs_recent_gt_logits_L1"])
                                  and all(x == 0 for x in out["P2_h8_debug"]["L35_patch_vs_gt_history_logits_L1"])
                                  and out["P2_h8_debug"]["h8_debug_unit_errors"] == 0)
    # ---- P2 (records complete, per-condition denominators) --------------------------------------------------------
    units = read_jsonl(f"{P}/units/records.jsonl")
    sel = []
    for f in sorted(glob.glob(f"{P}/selection/shard_*.json")):
        if not re.fullmatch(r"shard_\d+\.json", os.path.basename(f)):
            continue
        sel += json.load(open(f))["selected"]
    exp_units = {u["token"]: 1 + len(u["alternatives"]) for u in units}
    comp = {"selected_scenes": len(sel), "unit_scenes": len(units), "units": sum(exp_units.values()),
            "units_errors": len(read_jsonl(f"{P}/units/errors.jsonl")), "blocks": {}}
    for name in ("action_history_causal", "prev_action_state_patching", "temporal_feedback_window", "motion_semantics_ablation"):
        recs = read_jsonl(f"{P}/mech/{name}/records.jsonl")
        if name == "action_history_causal":
            got = {(r["token"], p) for r in recs for p in r["perturbations"]}
            conds = {c for r in recs for p in r["perturbations"].values() for c in p["conditions"]}
        else:
            got = {(r["token"], r["perturbation"]) for r in recs}
            key = "conditions" if name == "motion_semantics_ablation" else "rows"
            conds = {c for r in recs for c in r[key]}
        exp = {(t, p) for t, n_ in exp_units.items() for p in (["original"] + [f"alt{j}" for j in range(n_ - 1)])}
        meta = json.load(open(f"{P}/mech/{name}/run_meta.json")) if os.path.exists(f"{P}/mech/{name}/run_meta.json") else {}
        comp["blocks"][name] = {"units_expected": len(exp), "units_present": len(got & exp), "missing": len(exp - got),
                                "n_conditions": len(conds), "errors": len(read_jsonl(f"{P}/mech/{name}/errors.jsonl")),
                                "elapsed_min": meta.get("elapsed_min"), "gpu": meta.get("gpu"),
                                "records_bytes": os.path.getsize(f"{P}/mech/{name}/records.jsonl") if recs else 0}
    comp["pass"] = (len(sel) == len(units) and all(b["missing"] == 0 and b["errors"] == 0 for b in comp["blocks"].values()))
    out["P2_records_complete"] = comp
    ana = f"{P}/analysis_masked"
    if os.path.exists(f"{ana}/denominators.csv"):
        import pandas as pd
        out["P2_records_complete"]["denominators_csv"] = pd.read_csv(f"{ana}/denominators.csv").to_dict("records")
        cs = pd.read_csv(f"{ana}/condition_summary.csv")
        out["P2_records_complete"]["per_condition_units"] = cs[["block", "condition", "stratum", "n_units", "n_scenes", "n_invalid"]].to_dict("records")
        out["analysis_executes_masked"] = True
    # ---- runtimes -------------------------------------------------------------------------------------------------
    gt = read_jsonl(f"{P}/gpu_time.jsonl")
    out["gpu_steps"] = gt
    s1t = read_jsonl(f"{P}/stage1/timing.jsonl")
    out["stage1_s_per_scene"] = sum(x["s"] for x in s1t) / max(1, len(s1t))
    out["p1_stage1_s_per_scene"] = (lambda x: sum(y["s"] for y in x) / max(1, len(x)))(read_jsonl(f"{P}/checks/p1_stage1/timing.jsonl"))
    ut = read_jsonl(f"{P}/units/timing.jsonl")
    out["units_s_per_scene"] = sum(x["s"] for x in ut) / max(1, len(ut))
    out["download"] = read_jsonl(f"{P}/fetch/download_times.jsonl")
    out["all_pass"] = all(out[k]["pass"] for k in ("P1", "P3_cameras", "ED_parity", "P2_h8_debug", "P2_records_complete"))
    json.dump(out, open(f"{P}/checks.json", "w"), indent=1)
    if out["all_pass"]:
        open(f"{P}/PILOT_OK", "w").write("pilot checks P1-P3 passed; see checks.json\n")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else None) for k, v in out.items() if isinstance(v, dict)}))
    print(f"P1 agreement {p1['full_sequence_agreement']:.3f}, stub differ {p1['stub_ids_differ_rate']:.3f}; all_pass={out['all_pass']}")


def cmd_p1_detail(a):
    """agreement of each stage-1 procedure with stored full_extract arm N and with each other (dev PoC scenes only)."""
    import math
    sys.path.insert(0, C.SCRIPTS)
    from analyze_action_history import a_eval
    P = os.path.abspath(a.pilot)
    runs = {"planner": f"{P}/checks/p1_stage1/records.jsonl", "armN_generate": f"{P}/checks/p1b_armN_generate/records.jsonl"}
    R = {k: {r["token"]: r for r in read_jsonl(v)} for k, v in runs.items()}
    toks = sorted(set.intersection(*[set(x) for x in R.values()]))
    ref = {r["token"]: r for r in iter_full(set(toks))}

    def lab(pred, traj, gt, gtraj):
        t = next((k for k in range(10) if pred[k] != gt[k]), None)
        return t, ("A+" if a_eval(traj, gtraj) else "A-"), (t is not None and t <= C.T_STAR_MAX)

    def wilson(k, n):
        if n == 0:
            return [None, None]
        p, z = k / n, 1.96
        c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(c - h, 3), round(c + h, 3)]

    def get(src, t):
        if src == "full_extract":
            r = ref[t]; n = r["arms"]["N"]
            return n["pred_action_idx"][:10], n["trajectory_pred"], r["gt_action_idx"], r["trajectory_gt"]
        r = R[src][t]; n = r["arms"]["N"]
        return n["pred_action_idx"][:10], n["trajectory_pred"], r["gt_action_idx"], r["trajectory_gt"]

    out = {"n_scenes": len(toks)}
    for x, y in (("planner", "full_extract"), ("armN_generate", "full_extract"), ("planner", "armN_generate")):
        k = {"full_seq": 0, "first_token": 0, "t_star": 0, "a_label": 0, "eligible_and_stratum": 0}
        for t in toks:
            px, tx, gx, gtx = get(x, t); py, ty, gy, gty = get(y, t)
            lx, ly = lab(px, tx, gx, gtx), lab(py, ty, gy, gty)
            k["full_seq"] += px == py; k["first_token"] += px[0] == py[0]; k["t_star"] += lx[0] == ly[0]
            k["a_label"] += lx[1] == ly[1]; k["eligible_and_stratum"] += (lx[2], lx[1] if lx[2] else None) == (ly[2], ly[1] if ly[2] else None)
        out[f"{x}_vs_{y}"] = {m: {"agree": v, "rate": round(v / len(toks), 4), "wilson95": wilson(v, len(toks))} for m, v in k.items()}
    json.dump(out, open(f"{P}/checks/p1_detail.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("p1-tokens"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=30)
    p = sp.add_parser("p3-tree"); p.add_argument("--tokens", required=True); p.add_argument("--n", type=int, default=5)
    p.add_argument("--tree", required=True)
    p = sp.add_parser("report"); p.add_argument("--pilot", required=True)
    p = sp.add_parser("p1-detail"); p.add_argument("--pilot", required=True)
    a = ap.parse_args()
    {"p1-tokens": cmd_p1_tokens, "p3-tree": cmd_p3_tree, "report": cmd_report, "p1-detail": cmd_p1_detail}[a.cmd](a)


if __name__ == "__main__":
    main()
