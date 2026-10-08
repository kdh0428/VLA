#!/usr/bin/env python
"""
R8R9 raw recovery: per-unit table + P0-B cluster statistics for re-runs of exp 8 (prev_action_state_patching.py)
and exp 9 (temporal_feedback_window.py), both run UNMODIFIED into new directories (r8r9_run.sh). CPU only.

Metrics: analyze_action_history.unit_metrics (the original metric code; amplification = A- and FDE5 > 3 m).
Statistics: P0-B estimators (p1_stats = copy of p0b_analysis helpers): scene-weighted, log block bootstrap
(B = 10,000), log-level sign-flip (exact <= 20 logs), cluster-t sensitivity, unit/log weights as sensitivities.
Families (predeclared in protocol.md):
  R8 primary   A-: reverse@emb - gt_history (Reverse-Full; P0-B AutoVLA primary family member), amplification
  R8 secondary recent_gt - normal, gt_history - normal (this harness's own Normal), reverse@L - gt_history for every
               representative layer, FDE5 versions; A+ and pooled
  R9           A-: win_w - normal (w = 1..4), gt_history - normal, win_w - gt_history; delay rows; FDE5; A+
Also writes comparison_with_original_summary.csv: unit-weighted rates from the recovered raw vs the original
summary.json (A- / A+ level amplification per row), and a reproduction table vs exp 7 raw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402
import p1_stats as ST  # noqa: E402
from analyze_action_history import unit_metrics  # noqa: E402

REP_LAYERS = ["emb", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L35"]


def load(path, exp):
    rows, n = [], 0
    for line in open(os.path.join(path, "records.jsonl")):
        r = json.loads(line); n += 1
        t = r["t_star"]
        for cond, x in r["rows"].items():
            ents = x.get("ent") or []
            m = unit_metrics(x["action_idx"], x["trajectory_pred"], ents, r["gt"], r["trajectory_gt"], t)
            e = np.linalg.norm(np.asarray(x["trajectory_pred"], float) - np.asarray(r["trajectory_gt"], float)[:, :2], axis=1)
            rows.append({"model": "AutoVLA", "experiment": exp, "unit_id": f"{r['token']}|{r['perturbation']}", "log_id": r["log"],
                         "scene_id": r["token"], "task_id": None, "episode_id": None, "seed": 0, "group": r["group"], "t_star": t,
                         "perturbation": r["perturbation"], "condition": cond, "decode_j": t, "control_t": None,
                         "reference_branch": ("gt" if cond != "normal" else None), "perturbed_span": f"[{t},{t}]",
                         "perturbation_requested": None, "perturbation_achieved": None,
                         "intervention_count": x.get("n_gt_context_positions"), "parser_status": "ok", "exclusion_reason": None,
                         "primary_outcome": float(m["amplification"]), "amplification": float(m["amplification"]),
                         "recovery": float(m["recovery"]), "fde5": m["fde5"], "ade5": m["ade5"],
                         "amp2": float((not m["recovery"]) and e[-1] > 2), "amp4": float((not m["recovery"]) and e[-1] > 4),
                         "entropy1": m["entropy1"], "action_idx": json.dumps(x["action_idx"]),
                         "prior_exp7_match": (None if not r.get("prior_action_history") or cond not in r["prior_action_history"]
                                              else bool(r["prior_action_history"][cond] == x["action_idx"]))})
    return pd.DataFrame(rows), n


def orig_level(summary, exp, group, cond):
    try:
        if exp == "R8":
            return summary["groups"][group]["all"]["rows"][cond]["level"]["amplification"]["mean"]
        return summary["subsets"]["all perturbations"][group]["rows"][cond]["level"]["amplification"]["mean"]
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing exp8/ and exp9/ raw outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames, n_rec = [], {}
    for exp, sub in (("R8", "exp8"), ("R9", "exp9")):
        p = os.path.join(args.raw, sub)
        if os.path.exists(os.path.join(p, "records.jsonl")):
            df, n = load(p, exp); frames.append(df); n_rec[exp] = n
    U = pd.concat(frames, ignore_index=True)
    U.to_parquet(os.path.join(args.out, "units.parquet"), index=False)

    eff, seed, loo = [], ST.SEED, []
    fam8 = [("R8_primary", "A-", "reverse@emb", "gt_history", "amplification")]
    sec8 = [("recent_gt", "normal"), ("gt_history", "normal")] + [(f"reverse@{l}", "gt_history") for l in REP_LAYERS]
    for grp in ("A-", "A+", "all"):
        gm = (U.group == grp) if grp != "all" else pd.Series(True, index=U.index)
        m8 = gm & (U.experiment == "R8")
        if m8.any():
            for fam, g, a, b, met in fam8:
                if grp == g:
                    df = ST.paired_frame(U, a, b, met, m8)
                    eff.append(ST.effect_row(fam, f"{a} - {b}", grp, met, df, seed))
                    loo += ST.loo_rows(fam, f"{a} - {b}", grp, met, df, seed); seed += 7
            for a, b in sec8:
                for met in ("amplification", "fde5"):
                    if grp == "A-" and (a, b, met) == ("reverse@emb", "gt_history", "amplification"):
                        continue
                    df = ST.paired_frame(U, a, b, met, m8)
                    if len(df):
                        eff.append(ST.effect_row(f"R8_secondary_{grp}", f"{a} - {b}", grp, met, df, seed))
                    seed += 7
        m9 = gm & (U.experiment == "R9")
        if m9.any():
            pairs = [(f"win{w}", "normal") for w in (1, 2, 3, 4)] + [("gt_history", "normal")] + \
                    [(f"win{w}", "gt_history") for w in (1, 2, 3, 4)] + \
                    [(f"delay{s}_len1", "normal") for s in (2, 3, 4)] + [(f"delay{s}_len2", "normal") for s in (2, 3)]
            for a, b in pairs:
                for met in ("amplification", "fde5"):
                    df = ST.paired_frame(U, a, b, met, m9)
                    if len(df):
                        eff.append(ST.effect_row(f"R9_{grp}", f"{a} - {b}", grp, met, df, seed))
                    seed += 7
    ST.add_holm(eff, "R8_primary")
    pd.DataFrame(eff).to_csv(os.path.join(args.out, "paired_effects.csv"), index=False)
    pd.DataFrame(loo).to_csv(os.path.join(args.out, "loo_logs.csv"), index=False)

    # condition summary (scene-weighted level, log CI) + comparison with original summaries
    cs, cmp_rows = [], []
    origs = {}
    for exp, path in (("R8", "outputs/prev_action_state_patching/summary.json"), ("R9", "outputs/temporal_feedback_window/summary.json")):
        try:
            origs[exp] = json.load(open(os.path.join(C.POC_DIR, path)))
        except Exception:
            origs[exp] = None
    for (exp, grp, cond), g in U.groupby(["experiment", "group", "condition"]):
        df = g[["log_id", "scene_id", "amplification"]].rename(columns={"log_id": "log", "scene_id": "scene", "amplification": "d"})
        res = ST.log_cluster_effects(df, with_tests=False, b=2000)
        cs.append({"experiment": exp, "group": grp, "condition": cond, "metric": "amplification",
                   "scene_mean": res["scene"]["estimate"], "ci95_lo": res["scene"]["ci95"][0], "ci95_hi": res["scene"]["ci95"][1],
                   "unit_mean": res["unit"]["estimate"], "n_units": res["n_units"], "n_scenes": res["n_scenes"], "n_logs": res["n_clusters"],
                   "fde5_unit_mean": float(g.fde5.mean())})
        o = orig_level(origs.get(exp), exp, grp, cond) if origs.get(exp) else None
        if o is not None:
            cmp_rows.append({"experiment": exp, "group": grp, "condition": cond, "original_unit_rate": o,
                             "recovered_unit_rate": float(g.amplification.mean()), "diff_pp": 100 * (float(g.amplification.mean()) - o),
                             "n_units": len(g), "share_tokens_equal_exp7_raw": float(g.prior_exp7_match.dropna().mean())
                             if g.prior_exp7_match.notna().any() else None})
    pd.DataFrame(cs).to_csv(os.path.join(args.out, "condition_summary.csv"), index=False)
    pd.DataFrame(cmp_rows).to_csv(os.path.join(args.out, "comparison_with_original_summary.csv"), index=False)
    print(f"[done] records {n_rec}; unit rows {len(U)}; effects {len(eff)}; comparison rows {len(cmp_rows)}")


if __name__ == "__main__":
    main()
