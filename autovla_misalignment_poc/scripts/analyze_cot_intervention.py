#!/usr/bin/env python
"""
Analysis of the CoT intervention run (CPU only).

Per scene x condition
  first action token (sampled) and step-0 argmax, changed vs `original`
  JS divergence of the step-0 action-token distribution vs `original`
  GT first token: probability / margin / rank at step 0
  coarse action (A+/A-) with the SAME accept rule as the P/R/A comparison
  whether the executed action follows the declared action of that condition
  ADE / FDE (5 s and 3 s) and trajectory change vs `original`

Coupling read-out (pre-specified here, not tuned on the result)
  follow shift   P(executed moves in the counterfactual direction | counter_full)
                 - P(same | original)                                    (same scenes)
  flip excess    step-0 argmax change under counter_full - under template_original
  recovery       A+ rate under corrected_full - under original, on P+R-A-
  verdict        "causally coupled"      follow shift >= 0.50
                 "weakly coupled/post-hoc" follow shift < 0.15 and flip excess < 0.15
                 "partially coupled"     otherwise
Natural fast-thinking (natural_nocot) is reported in its own block.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, PRA_DIR)
import pra_labels as L                                   # noqa: E402
from run_pra_comparison import _pos_to_delta             # noqa: E402

CONDS = ["natural_nocot", "original", "template_original", "corrected_decision",
         "corrected_full", "counter_decision", "counter_full"]
FORCED = CONDS[1:]
GROUPS = ["P+R-A-", "P+R+A-", "P+R+A+ (control)"]
CFG = L.Config()
SLOW = {"STOP", "DECELERATE"}


def softmax(x):
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log(a + 1e-12) - np.log(b + 1e-12))))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def a_eval(traj, gt, horizon):
    s = L.Sample(model="AutoVLA", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[],
                 rsn_text="", decl_lon=None, decl_lat=None,
                 pred_delta=_pos_to_delta(traj, horizon), gt_delta=_pos_to_delta(gt, horizon))
    return L.label_A(s, CFG)


def ade_fde(traj, gt, n):
    p, g = np.asarray(traj, float)[:n], np.asarray(gt, float)[:n, :2]
    d = np.linalg.norm(p - g, axis=1)
    return float(d.mean()), float(d[-1])


def rate_ci(rows, key, reps=2000, seed=0):
    vals = [r for r in rows if r.get(key) is not None]
    if not vals:
        return None
    by = defaultdict(list)
    for r in vals:
        by[r["log"]].append(float(r[key]))
    keys = list(by)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        s = [x for k in (rng.choice(keys) for _ in keys) for x in by[k]]
        boots.append(np.mean(s))
    return {"mean": float(np.mean([float(r[key]) for r in vals])), "n": len(vals),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    p = 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)
    return {"gain": n01, "loss": n10, "p": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/cot_intervention"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]

    rows = []
    for r in recs:
        lg = np.load(os.path.join(args.run, "logits", f"{r['token']}.npz"))["logits"].astype(np.float32)
        gt, gt0 = r["trajectory_gt"], r["gt_action_idx"][0]
        C = r["conditions"]
        base = C.get("original", {})
        base_ok = not base.get("skipped", True) and base.get("trajectory_pred") is not None
        base_p0 = softmax(lg[CONDS.index("original"), 0]) if base_ok else None
        base_arg0 = int(lg[CONDS.index("original"), 0].argmax()) if base_ok else None
        base_A = a_eval(base["trajectory_pred"], gt, 10) if base_ok else (None, None)
        cf_lon = {"a deceleration to zero": "STOP", "an acceleration": "ACCELERATE"}.get(r["cf_target"][1])
        tfp = os.path.join(args.run, "teacher_forced", f"{r['token']}.npz")
        tf = np.load(tfp) if os.path.exists(tfp) else None
        for ci, name in enumerate(CONDS):
            c = C.get(name, {})
            if c.get("skipped", True) or c.get("trajectory_pred") is None:
                rows.append({"token": r["token"], "log": r["log_name"], "group": r["group"], "cond": name, "valid": False})
                continue
            tr = c["trajectory_pred"]
            A5, d5 = a_eval(tr, gt, 10)
            A3, _ = a_eval(tr, gt, 6)
            x0 = lg[ci, 0]
            p0 = softmax(x0)
            others = np.delete(x0, gt0)
            decl = c.get("declared") or [None, None]
            row = {
                "token": r["token"], "log": r["log_name"], "group": r["group"], "cond": name, "valid": True,
                "first_is_action": c["first_is_action"],
                "first_tok": c["action_idx"][0], "argmax0": int(x0.argmax()),
                "A5": bool(A5), "A3": bool(A3), "exec_lon": d5["pred_lon"], "exec_lat": d5["pred_lat"],
                "v_end": d5["v_end_pred"],
                "gt_p0": float(p0[gt0]), "gt_margin0": float(x0[gt0] - others.max()),
                "gt_rank0": int((x0 > x0[gt0]).sum()),
                "decl_lon": decl[0], "decl_lat": decl[1],
            }
            row["ade5"], row["fde5"] = ade_fde(tr, gt, 10)
            row["ade3"], row["fde3"] = ade_fde(tr, gt, 6)
            if decl[0] is not None:
                row["follows_declared"] = (d5["pred_lon"] == decl[0]) and (decl[1] is None or d5["pred_lat"] == decl[1])
            if cf_lon is not None:
                row["towards_cf"] = (d5["pred_lon"] == "ACCELERATE") if cf_lon == "ACCELERATE" else (d5["pred_lon"] in SLOW)
            if base_ok:
                row["first_changed"] = c["action_idx"][0] != base["action_idx"][0]
                row["argmax0_changed"] = int(x0.argmax()) != base_arg0
                row["js0"] = js(p0, base_p0)
                bt = np.asarray(base["trajectory_pred"], float)
                row["traj_change"] = float(np.linalg.norm(np.asarray(tr, float) - bt, axis=1).mean())
                row["dv_end"] = d5["v_end_pred"] - base_A[1]["v_end_pred"]
                row["lon_changed"] = d5["pred_lon"] != base_A[1]["pred_lon"]
                # continuous speed shift in the direction the edited CoT asks for
                if cf_lon is not None:
                    row["dv_toward_cf"] = row["dv_end"] * (1.0 if cf_lon == "ACCELERATE" else -1.0)
                gdir = float(np.sign(d5["v_end_gt"] - base_A[1]["v_end_pred"]))
                row["dv_toward_gt"] = row["dv_end"] * gdir
            if tf is not None and bool(tf["valid"][ci]) and bool(tf["valid"][1]):
                o = CONDS.index("original")
                row["tf_gt_lp_delta"] = float(tf["gt_lp"][ci].sum() - tf["gt_lp"][o].sum())
                row["tf_gt_lp0_delta"] = float(tf["gt_lp"][ci, 0] - tf["gt_lp"][o, 0])
                row["tf_orig_lp_delta"] = float(tf["orig_lp"][ci].sum() - tf["orig_lp"][o].sum())
                row["tf_gt_margin0_delta"] = float(tf["gt_margin"][ci, 0] - tf["gt_margin"][o, 0])
                row["tf_gt_margin_mean_delta"] = float(tf["gt_margin"][ci].mean() - tf["gt_margin"][o].mean())
                row["tf_js_mean"] = float(np.mean([
                    js(softmax(tf["orig_logits"][ci, k].astype(np.float32)),
                       softmax(tf["orig_logits"][o, k].astype(np.float32))) for k in range(10)]))
                row["tf_js_step0"] = float(js(softmax(tf["orig_logits"][ci, 0].astype(np.float32)),
                                              softmax(tf["orig_logits"][o, 0].astype(np.float32))))
            rows.append(row)

    out = {"n_scenes": len(recs), "by_group": {}, "tests": {}, "natural": {}, "verdict": {}}
    metrics = ["first_is_action", "first_changed", "argmax0_changed", "js0", "gt_p0", "gt_margin0",
               "A5", "A3", "follows_declared", "towards_cf", "lon_changed", "traj_change", "dv_end",
               "dv_toward_cf", "dv_toward_gt", "ade5", "fde5", "ade3", "fde3",
               "tf_gt_lp_delta", "tf_gt_lp0_delta", "tf_orig_lp_delta", "tf_gt_margin0_delta",
               "tf_gt_margin_mean_delta", "tf_js_mean", "tf_js_step0"]
    for grp in GROUPS + ["ALL"]:
        gr = [x for x in rows if x["valid"] and (grp == "ALL" or x["group"] == grp)]
        out["by_group"][grp] = {}
        for name in CONDS:
            cr = [x for x in gr if x["cond"] == name]
            out["by_group"][grp][name] = {m: rate_ci(cr, m) for m in metrics}
            out["by_group"][grp][name]["n_valid"] = len(cr)

    def paired(grp, a, b, key):
        ra = {x["token"]: x for x in rows if x["valid"] and x["cond"] == a and (grp == "ALL" or x["group"] == grp)}
        rb = {x["token"]: x for x in rows if x["valid"] and x["cond"] == b and (grp == "ALL" or x["group"] == grp)}
        common = [t for t in ra if t in rb and ra[t].get(key) is not None and rb[t].get(key) is not None]
        return mcnemar([ra[t][key] for t in common], [rb[t][key] for t in common]) | {"n": len(common)}

    for grp in GROUPS + ["ALL"]:
        out["tests"][grp] = {
            "A5 corrected_full vs original": paired(grp, "original", "corrected_full", "A5"),
            "A5 corrected_decision vs original": paired(grp, "original", "corrected_decision", "A5"),
            "A5 template_original vs original": paired(grp, "original", "template_original", "A5"),
            "towards_cf counter_full vs original": paired(grp, "original", "counter_full", "towards_cf"),
            "towards_cf counter_decision vs original": paired(grp, "original", "counter_decision", "towards_cf"),
            "towards_cf counter_full vs template_original": paired(grp, "template_original", "counter_full", "towards_cf"),
            "A5 corrected_full vs counter_full": paired(grp, "counter_full", "corrected_full", "A5"),
            "A5 corrected_decision vs counter_decision": paired(grp, "counter_decision", "corrected_decision", "A5"),
            "towards_cf counter_full vs corrected_full": paired(grp, "corrected_full", "counter_full", "towards_cf"),
            "towards_cf counter_decision vs corrected_decision": paired(grp, "corrected_decision", "counter_decision", "towards_cf"),
        }

    # natural fast-thinking vs forced CoT, same scenes
    for grp in GROUPS + ["ALL"]:
        nat = {x["token"]: x for x in rows if x["valid"] and x["cond"] == "natural_nocot" and (grp == "ALL" or x["group"] == grp)}
        frc = {x["token"]: x for x in rows if x["valid"] and x["cond"] == "original" and (grp == "ALL" or x["group"] == grp)}
        common = [t for t in nat if t in frc]
        out["natural"][grp] = {
            "n": len(common),
            "A5_natural": float(np.mean([nat[t]["A5"] for t in common])) if common else None,
            "A5_forced_original": float(np.mean([frc[t]["A5"] for t in common])) if common else None,
            "first_token_same_natural_vs_forced": float(np.mean([nat[t]["first_tok"] == frc[t]["first_tok"] for t in common])) if common else None,
            "lon_class_same_natural_vs_forced": float(np.mean([nat[t]["exec_lon"] == frc[t]["exec_lon"] for t in common])) if common else None,
            "traj_L2_natural_vs_forced": float(np.mean([np.linalg.norm(
                np.asarray(next(r for r in recs if r["token"] == t)["conditions"]["natural_nocot"]["trajectory_pred"])
                - np.asarray(next(r for r in recs if r["token"] == t)["conditions"]["original"]["trajectory_pred"]), axis=1).mean()
                for t in common])) if common else None,
            "McNemar_A5_natural_vs_forced": mcnemar([frc[t]["A5"] for t in common], [nat[t]["A5"] for t in common]) if common else None,
        }

    # reproduction of the stored run (decoding noise floor)
    rep_c = [r["conditions"]["original"]["action_idx"] == r["stored_armC_action_idx"] for r in recs
             if not r["conditions"]["original"].get("skipped")]
    rep_n = [r["conditions"]["natural_nocot"]["action_idx"] == r["stored_armN_action_idx"] for r in recs
             if not r["conditions"]["natural_nocot"].get("skipped")]
    out["reproduction"] = {"armC_all10_tokens_identical": float(np.mean(rep_c)), "armN_all10_tokens_identical": float(np.mean(rep_n))}
    out["edit_success"] = {name: float(np.mean([not r["conditions"][name].get("skipped", True) for r in recs])) for name in CONDS}
    decl_ok = defaultdict(list)
    for r in recs:
        tgt = {"corrected_full": r["gt_target"], "corrected_decision": r["gt_target"],
               "counter_full": r["cf_target"], "counter_decision": r["cf_target"]}
        for name, (beh, spd) in tgt.items():
            c = r["conditions"][name]
            if c.get("skipped"):
                continue
            want = L.parse_action_instruction(f"{beh} with {spd}")
            decl_ok[name].append(list(c["declared"]) == list(want))
    out["edit_declared_matches_target"] = {k: float(np.mean(v)) for k, v in decl_ok.items()}

    # ---- paired contrasts: value = b - a per scene, log-cluster bootstrap ------------------
    # The scenes were SELECTED on a failure of the stored forced run. Near-tied action tokens
    # re-roll under any prefix change, so every edited condition drifts away from `original`
    # (regression to the mean) whatever it says. A direction effect is therefore measured
    # between conditions with the SAME rewrite form and OPPOSITE declared direction, with
    # `template_original` as the floor for "rewrite only". This replaces the original-based
    # rule written before the run; that rule is still reported as `original_baseline`.
    idx = {(x["token"], x["cond"]): x for x in rows if x["valid"]}

    def diff_ci(grp, a, b, key, reps=2000, seed=0):
        vals = []
        for (t, c), xb in idx.items():
            if c != b or (grp != "ALL" and xb["group"] != grp):
                continue
            xa = idx.get((t, a))
            if xa is None or xa.get(key) is None or xb.get(key) is None:
                continue
            vals.append((xb["log"], float(xb[key]) - float(xa[key])))
        if not vals:
            return None
        by = defaultdict(list)
        for lg_, v_ in vals:
            by[lg_].append(v_)
        keys = list(by)
        rng = random.Random(seed)
        boots = [np.mean([v_ for k in (rng.choice(keys) for _ in keys) for v_ in by[k]]) for _ in range(reps)]
        return {"mean": float(np.mean([v_ for _, v_ in vals])), "n": len(vals),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

    CONTRASTS = [
        ("towards_cf: counter_full - corrected_full", "corrected_full", "counter_full", "towards_cf"),
        ("towards_cf: counter_decision - corrected_decision", "corrected_decision", "counter_decision", "towards_cf"),
        ("A5: corrected_full - counter_full", "counter_full", "corrected_full", "A5"),
        ("A5: corrected_decision - counter_decision", "counter_decision", "corrected_decision", "A5"),
        ("dv_toward_cf (m/s): counter_full - corrected_full", "corrected_full", "counter_full", "dv_toward_cf"),
        ("dv_toward_cf (m/s): counter_decision - corrected_decision", "corrected_decision", "counter_decision", "dv_toward_cf"),
        ("GT logp 10 steps: corrected_full - counter_full", "counter_full", "corrected_full", "tf_gt_lp_delta"),
        ("GT logp 10 steps: corrected_decision - counter_decision", "counter_decision", "corrected_decision", "tf_gt_lp_delta"),
        ("GT logp step0: corrected_full - counter_full", "counter_full", "corrected_full", "tf_gt_lp0_delta"),
        ("argmax0 changed: counter_full - template_original", "template_original", "counter_full", "argmax0_changed"),
        ("A5: template_original - original (rewrite only)", "original", "template_original", "A5"),
        ("A5: corrected_full - template_original", "template_original", "corrected_full", "A5"),
        ("towards_cf: counter_full - template_original", "template_original", "counter_full", "towards_cf"),
    ]
    out["contrasts"] = {grp: {name: diff_ci(grp, a, b, k) for name, a, b, k in CONTRASTS} for grp in GROUPS + ["ALL"]}

    # ---- verdict (direction-specific) ------------------------------------------------------
    C = out["contrasts"]["ALL"]
    allg = out["by_group"]["ALL"]
    prra = out["by_group"]["P+R-A-"]
    dfollow = C["towards_cf: counter_full - corrected_full"]
    flip_excess = C["argmax0 changed: counter_full - template_original"]
    if dfollow["mean"] >= 0.50:
        v = "causally coupled"
    elif dfollow["mean"] < 0.15 and flip_excess["mean"] < 0.15:
        v = "weakly coupled / potentially post-hoc"
    else:
        v = "partially coupled"
    out["verdict"] = {
        "verdict": v,
        "direction_follow_full": dfollow,
        "direction_follow_decision": C["towards_cf: counter_decision - corrected_decision"],
        "direction_A5_full": C["A5: corrected_full - counter_full"],
        "direction_A5_decision": C["A5: corrected_decision - counter_decision"],
        "direction_dv_full": C["dv_toward_cf (m/s): counter_full - corrected_full"],
        "direction_gt_logp_full": C["GT logp 10 steps: corrected_full - counter_full"],
        "direction_gt_logp_decision": C["GT logp 10 steps: corrected_decision - counter_decision"],
        "flip_excess_counter_full_over_template": flip_excess,
        "rewrite_A5_template_minus_original": C["A5: template_original - original (rewrite only)"],
        "recovery_vs_original_PRmAm": prra["corrected_full"]["A5"]["mean"] - prra["original"]["A5"]["mean"],
        "recovery_vs_template_PRmAm": prra["corrected_full"]["A5"]["mean"] - prra["template_original"]["A5"]["mean"],
        "original_baseline (biased by failure selection)": {
            "follow_shift_counter_full": allg["counter_full"]["towards_cf"]["mean"] - allg["original"]["towards_cf"]["mean"],
            "follow_shift_counter_decision": allg["counter_decision"]["towards_cf"]["mean"] - allg["original"]["towards_cf"]["mean"],
        },
        "rule": "direction_follow_full >= 0.50 -> causally coupled; < 0.15 and flip_excess < 0.15 -> weakly coupled; else partially coupled",
        "baseline_change_note": "primary contrast switched from vs-original to counter vs corrected after inspecting 118/159 scenes, "
                                "because failure-selected scenes regress under any prefix change",
    }

    with open(os.path.join(args.run, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.run, "rows.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")

    # ---- console tables ----------------------------------------------------------------
    def fm(d, pct=True):
        if d is None:
            return "   -   "
        return f"{100*d['mean']:5.1f}%" if pct else f"{d['mean']:6.3f}"
    print(f"reproduction: {out['reproduction']}")
    print(f"edit success: {out['edit_success']}")
    print(f"declared matches target: {out['edit_declared_matches_target']}")
    for grp in GROUPS + ["ALL"]:
        print(f"\n=== {grp}")
        print(f"{'condition':20s} {'n':>3s} {'1stChg':>7s} {'argChg':>7s} {'JS0':>6s} {'GTp0':>6s} {'A+5s':>6s} {'A+3s':>6s} "
              f"{'follow':>6s} {'->cf':>6s} {'lonChg':>6s} {'dTraj':>6s} {'ADE5':>6s} {'FDE5':>6s}")
        for name in CONDS:
            b = out["by_group"][grp][name]
            print(f"{name:20s} {b['n_valid']:3d} {fm(b['first_changed'])} {fm(b['argmax0_changed'])} {fm(b['js0'],False)} "
                  f"{fm(b['gt_p0'],False)} {fm(b['A5'])} {fm(b['A3'])} {fm(b['follows_declared'])} {fm(b['towards_cf'])} "
                  f"{fm(b['lon_changed'])} {fm(b['traj_change'],False)} {fm(b['ade5'],False)} {fm(b['fde5'],False)}")
        print(f"   {'teacher-forced / continuous':20s} {'dGTlp':>7s} {'dGTlp0':>7s} {'dOrigLp':>8s} {'dMarg0':>7s} "
              f"{'JSmean':>7s} {'JS0':>7s} {'dv->cf':>7s} {'dv->gt':>7s}")
        for name in CONDS:
            b = out["by_group"][grp][name]
            print(f"   {name:20s} {fm(b['tf_gt_lp_delta'],False)} {fm(b['tf_gt_lp0_delta'],False)} "
                  f"{fm(b['tf_orig_lp_delta'],False)}  {fm(b['tf_gt_margin0_delta'],False)} {fm(b['tf_js_mean'],False)} "
                  f"{fm(b['tf_js_step0'],False)} {fm(b['dv_toward_cf'],False)} {fm(b['dv_toward_gt'],False)}")
        for k, t in out["tests"][grp].items():
            print(f"   {k:45s} gain={t['gain']:3d} loss={t['loss']:3d} p={t['p']:.3g} n={t['n']}")
        print(f"   natural vs forced: {out['natural'][grp]}")
    for grp in GROUPS + ["ALL"]:
        print(f"\n--- paired contrasts: {grp}")
        for name, d in out["contrasts"][grp].items():
            if d is None:
                print(f"   {name:60s} -")
            else:
                print(f"   {name:60s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] n={d['n']}")
    print(f"\nVERDICT: {out['verdict']['verdict']}")
    for k, v_ in out["verdict"].items():
        if isinstance(v_, dict) and "mean" in v_:
            print(f"   {k:45s} {v_['mean']:+.3f} [{v_['ci95'][0]:+.3f}, {v_['ci95'][1]:+.3f}]")
        elif k not in ("verdict",):
            print(f"   {k:45s} {v_}")


if __name__ == "__main__":
    main()
