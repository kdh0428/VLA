#!/usr/bin/env python
"""
Analysis of the action-history intervention (CPU only).

Unit = (scene, perturbation). Metrics per condition, from t* on:
  amplification  A- AND FDE(5 s) > 3.0 m (same definition as the equal-distance experiment)
  recovery       A+ (P/R/A accept rule, 5 s)
  ADE / FDE      5 s
  downstream err steps after t* whose token != GT;  realign = fraction == GT
  entropy        mean action-token entropy over the generated steps; entropy1 = first step
  err growth     slope of per-pose L2 error over poses t*..9 (m per step)

Contrasts: condition - normal, per unit, paired; log-cluster bootstrap 95% CI; McNemar for the
binary metrics, Wilcoxon for continuous ones. Reported for A- and A+ separately, for all
perturbations and for the perturbations that amplified in the stored equal-distance run
("previously unstable"). The A- vs A+ interaction of each contrast is bootstrapped too.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, PRA_DIR)
import pra_labels as L                                  # noqa: E402
from run_pra_comparison import _pos_to_delta            # noqa: E402

CONDS = ["normal", "hist_attn_mask", "recent_attn_mask", "hist_emb_neutral", "recent_emb_neutral", "gt_history", "recent_gt"]
NAMES = {"normal": "Normal AR", "hist_attn_mask": "Action-history attention mask",
         "recent_attn_mask": "Recent-action attention mask",
         "hist_emb_neutral": "[OOD] Action-history embedding neutralised",
         "recent_emb_neutral": "[OOD] Recent-action embedding neutralised",
         "gt_history": "GT-history", "recent_gt": "Recent-GT (직전 1개만 GT)"}
OOD = {"hist_emb_neutral", "recent_emb_neutral"}
N_ACT = 10
CFG = L.Config()
AMP_FDE = 3.0
REPS = 2000
METRICS = ["amplification", "recovery", "ade5", "fde5", "downstream_err", "realign", "entropy", "entropy1", "err_slope"]
BINARY = {"amplification", "recovery"}


def a_eval(traj, gt):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, 10), gt_delta=_pos_to_delta(gt, 10))
    return bool(L.label_A(s, CFG)[0])


def unit_metrics(acts, traj, ents, gt, gtraj, t):
    after = list(range(t + 1, N_ACT))
    p, g = np.asarray(traj, float), np.asarray(gtraj, float)[:, :2]
    e = np.linalg.norm(p - g, axis=1)
    rec = a_eval(traj, gtraj)
    return {"recovery": rec, "amplification": (not rec) and float(e[-1]) > AMP_FDE,
            "ade5": float(e.mean()), "fde5": float(e[-1]),
            "downstream_err": float(np.mean([acts[k] != gt[k] for k in after])) if after else None,
            "realign": float(np.mean([acts[k] == gt[k] for k in after])) if after else None,
            "entropy": float(np.mean(ents)) if ents else None, "entropy1": float(ents[0]) if ents else None,
            "err_slope": float(np.polyfit(np.arange(t, N_ACT), e[t:], 1)[0]) if N_ACT - t >= 2 else None}


def cboot(items, reps=REPS, seed=0):
    """items: (log, value). mean and log-cluster CI."""
    items = [(lg, float(v)) for lg, v in items if v is not None and np.isfinite(float(v))]
    if not items:
        return None
    by = defaultdict(list)
    for lg, v in items:
        by[lg].append(v)
    logs = list(by)
    rng = random.Random(seed)
    bs = [np.mean([v for lg in (rng.choice(logs) for _ in logs) for v in by[lg]]) for _ in range(reps)]
    return {"mean": float(np.mean([v for _, v in items])), "n": len(items),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/action_history_causal"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]

    units = []
    repro = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        for pname, pr in r["perturbations"].items():
            u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": pname, "t_star": t}
            # outcome of this perturbation in the stored equal-distance run (for the subset only)
            stored = pr["stored_normal_action_idx"]
            u["stored_normal_reproduced"] = pr["conditions"]["normal"]["action_idx"] == stored
            repro.append(u["stored_normal_reproduced"])
            for c in CONDS:
                x = pr["conditions"][c]
                u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], gt, gtraj, t)
            units.append(u)
    # stored amplification per perturbation: recompute from the equal-distance records
    ed_path = os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl")
    stored_amp = {}
    for line in open(ed_path):
        r = json.loads(line)
        for name, c in r["conditions"].items():
            if name.startswith("alt") or name == "original":
                if c.get("trajectory_pred"):
                    e = np.linalg.norm(np.asarray(c["trajectory_pred"], float) - np.asarray(r["trajectory_gt"], float)[:, :2], axis=1)
                    stored_amp[(r["token"], name)] = bool((not a_eval(c["trajectory_pred"], r["trajectory_gt"])) and float(e[-1]) > AMP_FDE)
    for u in units:
        u["stored_amplified"] = stored_amp.get((u["token"], u["pert"]))

    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in ("A-", "A+")},
         "n_scenes": {g: sum(r["group"] == g for r in recs) for g in ("A-", "A+")},
         "normal_reproduces_stored_equal_distance_tokens": float(np.mean(repro)),
         "subsets": {}}

    def subset(name, pred):
        out = {}
        for g in ("A-", "A+"):
            us = [u for u in units if u["group"] == g and pred(u)]
            block = {"n": len(us), "n_scenes": len({u["token"] for u in us}), "conditions": {}, "vs_normal": {}}
            for c in CONDS:
                block["conditions"][c] = {m: cboot([(u["log"], u[c][m]) for u in us]) for m in METRICS}
            for c in CONDS[1:]:
                block["vs_normal"][c] = {}
                for m in METRICS:
                    pairs = [(u["log"], u[c][m], u["normal"][m]) for u in us if u[c][m] is not None and u["normal"][m] is not None]
                    d = cboot([(lg, float(a) - float(b)) for lg, a, b in pairs])
                    if d and m in BINARY:
                        gain = sum(bool(a) and not bool(b) for _, a, b in pairs)   # condition true, normal false
                        loss = sum(bool(b) and not bool(a) for _, a, b in pairs)
                        d["mcnemar"] = {"cond_only": gain, "normal_only": loss,
                                        "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
                    elif d:
                        diffs = np.array([float(a) - float(b) for _, a, b in pairs])
                        d["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
                    block["vs_normal"][c][m] = d
            out[g] = block
        # interaction: [cond - normal](A-) - [cond - normal](A+), logs resampled jointly
        out["interaction"] = {}
        for c in CONDS[1:]:
            out["interaction"][c] = {}
            for m in ("amplification", "recovery", "fde5", "downstream_err"):
                bn, bp = defaultdict(list), defaultdict(list)
                for u in units:
                    if not pred(u) or u[c][m] is None or u["normal"][m] is None:
                        continue
                    (bn if u["group"] == "A-" else bp)[u["log"]].append(float(u[c][m]) - float(u["normal"][m]))
                logs = sorted(set(bn) | set(bp))
                if not bn or not bp:
                    out["interaction"][c][m] = None
                    continue

                def gap(ls):
                    a = [v for lg in ls for v in bn.get(lg, [])]; b = [v for lg in ls for v in bp.get(lg, [])]
                    return (np.mean(a) - np.mean(b)) if a and b else np.nan
                rng = random.Random(3)
                bs = [x for x in (gap([rng.choice(logs) for _ in logs]) for _ in range(REPS)) if np.isfinite(x)]
                out["interaction"][c][m] = {"gap": float(gap(logs)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        S["subsets"][name] = out

    subset("all perturbations", lambda u: True)
    subset("previously amplified perturbations", lambda u: u["stored_amplified"] is True)
    subset("original token only", lambda u: u["pert"] == "original")

    with open(os.path.join(args.run, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

    # ------------------------------------------------------------------ report
    def pc(d):
        return "–" if d is None else f"{100*d['mean']:.1f}% [{100*d['ci95'][0]:.0f}, {100*d['ci95'][1]:.0f}]"

    def nm(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def dl(d, m):
        if d is None:
            return "–"
        if m in BINARY:
            s = f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]"
            return s + f", p={d['mcnemar']['p']:.2g}"
        if m in ("downstream_err", "realign"):
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}], p={d['wilcoxon_p']:.2g}"
        nd = 3 if m in ("entropy", "entropy1", "err_slope") else 2
        return f"{d['mean']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}], p={d['wilcoxon_p']:.2g}"

    W = []
    w = W.append
    w("# AutoVLA Natural/Fast — 이전 action token에 대한 self-conditioning이 오류 증폭을 유지하는가\n")
    w(f"GPU1. equal-distance 실험의 장면과 perturbation을 그대로 사용: 장면 A− {S['n_scenes']['A-']} / A+ {S['n_scenes']['A+']}, "
      f"(장면, perturbation) 단위 A− {S['n_units']['A-']} / A+ {S['n_units']['A+']}. prefix와 첫 mismatch step의 강제 토큰은 조건 간 동일, "
      "이후 action 생성 방식만 다릅니다. 증폭 = A− 이면서 FDE > 3 m. [ ]는 log 단위 cluster bootstrap 95% CI, p는 McNemar(이진)/Wilcoxon(연속).\n")
    w(f"점검: 이 harness의 Normal AR이 equal-distance 실험에서 저장된 토큰 10개를 완전히 재현한 비율 "
      f"{100*S['normal_reproduces_stored_equal_distance_tokens']:.1f}% (action 토큰만 샘플링하는 수동 decoding 루프이므로 완전 일치는 기대하지 않음; 모든 비교는 이 harness의 Normal AR 대비).\n")
    for sname, block in S["subsets"].items():
        w(f"## {sname}\n")
        for g in ("A-", "A+"):
            B = block[g]
            w(f"### {g} (단위 {B['n']}, 장면 {B['n_scenes']})\n")
            w("| 조건 | amplification | A+ recovery | ADE (m) | FDE (m) | downstream token error | GT re-alignment | post-mismatch entropy | 첫 step entropy | error growth (m/step) |")
            w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for c in CONDS:
                d = B["conditions"][c]
                w(f"| {NAMES[c]} | {pc(d['amplification'])} | {pc(d['recovery'])} | {nm(d['ade5'])} | {nm(d['fde5'])} | "
                  f"{pc(d['downstream_err'])} | {pc(d['realign'])} | {nm(d['entropy'], 3)} | {nm(d['entropy1'], 3)} | {nm(d['err_slope'], 3)} |")
            w("")
            w("조건 − Normal AR (쌍대):\n")
            w("| 조건 | Δamplification | Δrecovery | ΔFDE (m) | Δdownstream error | Δre-alignment | Δentropy | Δerror growth |")
            w("|---|---|---|---|---|---|---|---|")
            for c in CONDS[1:]:
                v = B["vs_normal"][c]
                w(f"| {NAMES[c]} | {dl(v['amplification'], 'amplification')} | {dl(v['recovery'], 'recovery')} | {dl(v['fde5'], 'fde5')} | "
                  f"{dl(v['downstream_err'], 'downstream_err')} | {dl(v['realign'], 'realign')} | {dl(v['entropy'], 'entropy')} | {dl(v['err_slope'], 'err_slope')} |")
            w("")
        w("A− vs A+ 차이 ([조건 − Normal](A−) − [조건 − Normal](A+)):\n")
        w("| 조건 | Δamplification 차이 | Δrecovery 차이 | ΔFDE 차이 (m) | Δdownstream error 차이 |")
        w("|---|---|---|---|---|")
        for c in CONDS[1:]:
            I = block["interaction"][c]
            f = lambda d, pct: "–" if d is None else (f"{100*d['gap']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]" if pct  # noqa: E731
                                                     else f"{d['gap']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")
            w(f"| {NAMES[c]} | {f(I['amplification'], True)} | {f(I['recovery'], True)} | {f(I['fde5'], False)} | {f(I['downstream_err'], True)} |")
        w("")
    # ---- validity of each intervention: is the first free step still in distribution? ----
    w("## 개입 타당성: 첫 생성 step entropy (Normal 대비)\n")
    w("첫 step entropy가 Normal보다 크게(>1 nat) 오르면 모델이 본 적 없는 입력으로 판단해, 그 조건의 결과는 self-conditioning 제거가 아니라 "
      "OOD 손상으로 해석합니다. `[OOD]` 조건은 sanity 단계에서 이 문제가 확인되어 판정에서 제외합니다.\n")
    w("| 조건 | A− Δ첫 step entropy | A+ Δ첫 step entropy | 판정 사용 |")
    w("|---|---|---|---|")
    allb = S["subsets"]["all perturbations"]
    for c in CONDS[1:]:
        dn = allb["A-"]["vs_normal"][c]["entropy1"]; dp = allb["A+"]["vs_normal"][c]["entropy1"]
        big = any(d is not None and d["mean"] > 1.0 for d in (dn, dp))
        use = "아니오 (OOD)" if (c in OOD or big) else "예"
        w(f"| {NAMES[c]} | {dl(dn, 'entropy1')} | {dl(dp, 'entropy1')} | {use} |")
    w("")
    with open(os.path.join(args.run, "ACTION_HISTORY_CAUSAL.md"), "w") as fh:
        fh.write("\n".join(W))
    print("\n".join(W))


if __name__ == "__main__":
    main()
