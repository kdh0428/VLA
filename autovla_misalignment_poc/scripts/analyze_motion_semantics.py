#!/usr/bin/env python
"""
Analysis of motion_semantics_ablation (P5). CPU only. Same unit metrics and statistics as analyze_prev_action_identity
(amplification = A- and FDE5 > 3 m, recovery = P/R/A accept at 5 s, ADE / FDE, downstream entropy) plus the mean
log-prob of the generated steps. Per group (A-, A+): condition means (log-cluster bootstrap), paired vs normal
(McNemar / Wilcoxon), fraction of the Recent-GT effect, achieved substitution geometry.

  python scripts/analyze_motion_semantics.py [--run outputs/motion_semantics_ablation]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import cboot, unit_metrics   # noqa: E402
from analyze_prev_action_identity import frac_effect, paired   # noqa: E402

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ["normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"]
SUBS = ROWS[2:]
NAMES = {"normal": "원래 모델 생성 token", "recent_gt": "GT previous action", "dir_ok_mag_wrong": "방향 정답 + 크기 오답",
         "dir_wrong_mag_ok": "크기 정답 + 방향 오답", "mirror_same_dist": "같은 거리, 반대 방향", "random_mag_matched": "크기만 맞춘 무작위"}
M = ["amplification", "recovery", "ade5", "fde5", "entropy", "loglik"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/motion_semantics_ablation"))
    a = ap.parse_args()
    units = []
    for l in open(os.path.join(a.run, "records.jsonl")):
        r = json.loads(l); t = r["t_star"]
        u = {"token": r["token"], "log": r["log"], "group": r["group"]}
        for c in ROWS:
            x = r["conditions"][c]
            u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], t)
            u[c]["loglik"] = float(np.mean(x["logprob_steps"])) if x.get("logprob_steps") else None
            if c in SUBS and x.get("substitutions"):
                u[c + "_sub"] = {k: float(np.mean([s[k] for s in x["substitutions"]])) for k in
                                 ("e_own", "d_sub_gt", "dmag_sub_gt", "dphi_sub_gt_deg", "mag_gt")}
                u[c + "_sub"]["abs_dmag"] = float(np.mean([abs(s["dmag_sub_gt"]) for s in x["substitutions"]]))
        units.append(u)
    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in ("A-", "A+")}, "groups": {}}
    out = []
    for g in ("A-", "A+"):
        us = [u for u in units if u["group"] == g]
        blk = {"conditions": {c: {m: cboot([(u["log"], u[c][m]) for u in us if u[c][m] is not None]) for m in M} for c in ROWS},
               "vs_normal": {c: {m: paired(us, c, "normal", m) for m in M} for c in ROWS[1:]},
               "vs_recent_gt": {c: {m: paired(us, c, "recent_gt", m) for m in ("amplification", "fde5")} for c in SUBS},
               "frac_of_recent_gt_effect": {c: {m: frac_effect(us, c, m) for m in ("amplification", "fde5")} for c in SUBS},
               "substitution_geometry": {c: {k: float(np.mean([u[c + "_sub"][k] for u in us if c + "_sub" in u]))
                                             for k in ("e_own", "d_sub_gt", "abs_dmag", "dphi_sub_gt_deg", "mag_gt")} for c in SUBS}}
        S["groups"][g] = blk
        out.append(f"\n== {g} ({len(us)} units) ==")
        out.append("| 직전 token | 증폭 % [95% CI] | 회복 % | ADE | FDE | 이후 엔트로피 | 이후 log-prob | Recent-GT 효과 대비(증폭) | p vs normal (증폭) |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in ROWS:
            cd = blk["conditions"][c]; fr = blk["frac_of_recent_gt_effect"].get(c, {}).get("amplification")
            vn = blk["vs_normal"].get(c, {}).get("amplification")
            out.append(f"| {NAMES[c]} | {cd['amplification']['mean']*100:.1f} [{cd['amplification']['ci95'][0]*100:.1f}, {cd['amplification']['ci95'][1]*100:.1f}] | "
                       f"{cd['recovery']['mean']*100:.1f} | {cd['ade5']['mean']:.2f} | {cd['fde5']['mean']:.2f} | {cd['entropy']['mean']:.3f} | "
                       f"{cd['loglik']['mean']:.3f} | {('%.2f' % fr['frac']) if fr else '–'} | {('%.2g' % vn['mcnemar']['p']) if vn else '–'} |")
        out.append("\n달성된 대체 기하 (평균): 자기 오차 e, 대체–GT 거리, |Δ크기|, Δ방향(도)")
        for c, v in blk["substitution_geometry"].items():
            out.append(f"- {NAMES[c]}: e = {v['e_own']:.3f} m, d(sub, GT) = {v['d_sub_gt']:.3f} m, |dmag| = {v['abs_dmag']:.3f} m, "
                       f"dphi = {v['dphi_sub_gt_deg']:.2f} deg (GT 속도 {v['mag_gt']:.2f} m/0.5 s)")
    json.dump(S, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    txt = "\n".join(out); open(os.path.join(a.run, "analysis.md"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
