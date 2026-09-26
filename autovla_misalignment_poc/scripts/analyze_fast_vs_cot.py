#!/usr/bin/env python
"""
Paired analysis of the unbiased Natural / Fast / CoT run (CPU only).

Scoring
  A+       the same accept rule as the P/R/A comparison (semantic coarse action, GT-relative
           speed class, speed-scaled tolerance), 5 s horizon; 3 s reported too.
           Intention-to-treat: an output without 10 action tokens counts as A-.
  ADE/FDE  metres, AutoVLA frame; for a pairwise contrast only scenes valid in BOTH conditions.
  entropy  of the action-token distribution at the step that emits the first action token.

Uncertainty
  log-cluster bootstrap (2000) for every difference; McNemar for A+, Wilcoxon signed-rank
  for continuous metrics.

Strata (neither uses the outcome of this run)
  difficulty   hard = GT trajectory stops / slows / turns, or a hazard is present
               (lead vehicle <= 15 m or pedestrian hazard in the annotation); easy = otherwise
  uncertainty  tertiles of the first-action entropy from the EARLIER, separate natural run
               (outputs/gpu1 logit lens, final layer, step 0)
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
import pra_labels as L                                        # noqa: E402
from run_pra_comparison import _av_objects, _pos_to_delta     # noqa: E402

CONDS = ["natural", "fast", "cot"]
NAMES = {"natural": "Natural", "fast": "Forced Fast", "cot": "Forced CoT"}
PAIRS = [("natural", "fast"), ("natural", "cot"), ("fast", "cot")]
CFG = L.Config()
REPS = 2000


def a_eval(traj, gt, horizon):
    s = L.Sample(model="AutoVLA", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[],
                 rsn_text="", decl_lon=None, decl_lat=None,
                 pred_delta=_pos_to_delta(traj, horizon), gt_delta=_pos_to_delta(gt, horizon))
    return L.label_A(s, CFG)


def boot_mean(rows, key, reps=REPS, seed=0):
    vals = [(r["log"], float(r[key])) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    if not vals:
        return None
    by = defaultdict(list)
    for lg, v in vals:
        by[lg].append(v)
    keys = list(by)
    rng = random.Random(seed)
    boots = [np.mean([v for k in (rng.choice(keys) for _ in keys) for v in by[k]]) for _ in range(reps)]
    return {"mean": float(np.mean([v for _, v in vals])), "n": len(vals),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def paired(scn, a, b, key, binary=False):
    """b - a per scene, cluster bootstrap by log, plus McNemar / Wilcoxon."""
    d = []
    xa, xb = [], []
    for s in scn:
        va, vb = s[a].get(key), s[b].get(key)
        if va is None or vb is None or not np.isfinite(float(va)) or not np.isfinite(float(vb)):
            continue
        d.append({"log": s["log"], "v": float(vb) - float(va)})
        xa.append(va); xb.append(vb)
    if not d:
        return None
    out = boot_mean(d, "v")
    if binary:
        gain = sum((not p) and q for p, q in zip(xa, xb))
        loss = sum(p and (not q) for p, q in zip(xa, xb))
        out["mcnemar"] = {"b_better": gain, "a_better": loss,
                          "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
    else:
        diffs = np.asarray(xb, float) - np.asarray(xa, float)
        out["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/fast_vs_cot_unbiased"))
    ap.add_argument("--stored-natural", default=os.path.join(POC_DIR, "outputs/gpu1/records.jsonl"))
    ap.add_argument("--annotations", default=os.path.join(POC_DIR, "outputs/annotations"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    want = {r["token"] for r in recs}
    stored_H = {}
    for line in open(args.stored_natural):
        x = json.loads(line)
        if x["token"] in want and x.get("logit_lens"):
            stored_H[x["token"]] = x["logit_lens"][0]["layers"][-1]["entropy"]

    # CoT length is recomputed from the saved text (the runner's own count missed the tag in
    # context). Outputs without saved text are the fixed no-CoT stub.
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("/root/VLA/autovla/Qwen2.5-VL-3B-Instruct")
    stub_len = len(tk.encode("<think>\nThis is a straightforward scenario, and a direct decision can be made.\n",
                             add_special_tokens=False))

    def think_len(x):
        txt = x.get("text")
        if not txt:
            return stub_len
        body = txt.split("</think>")[0] if "</think>" in txt else txt.split("<answer>")[0]
        return len(tk.encode(body, add_special_tokens=False))

    scn = []
    for r in recs:
        gt = r["trajectory_gt"]
        kg = L.kinematics(_pos_to_delta(gt, 10))
        anno = json.load(open(os.path.join(args.annotations, f"{r['token']}.json")))
        sc = L.scene_from_objects(_av_objects(anno))
        hard = (kg.longitudinal in ("STOP", "DECELERATE")) or (kg.lateral != "STRAIGHT") or bool(sc.hazards(CFG))
        s = {"token": r["token"], "log": r["log_name"], "map": r["map_name"],
             "difficulty": "hard" if hard else "easy", "gt_class": kg.label,
             "stored_entropy": stored_H.get(r["token"])}
        for c in CONDS:
            x = r["conditions"][c]
            row = {"valid": x["valid"], "cot_present": x["cot_present"], "think_tokens": think_len(x),
                   "truncated": x["truncated"], "runaway": x["runaway"], "first_action": x.get("first_action"),
                   "entropy": x.get("first_entropy_action"), "gt_margin0": x.get("gt_margin0"),
                   "gt_prob0": x.get("gt_prob0"), "action_idx": x["action_idx"], "n_generated": x["n_generated"]}
            if x["valid"]:
                A5, d5 = a_eval(x["trajectory_pred"], gt, 10)
                A3, _ = a_eval(x["trajectory_pred"], gt, 6)
                p, g_ = np.asarray(x["trajectory_pred"], float), np.asarray(gt, float)[:, :2]
                e = np.linalg.norm(p - g_, axis=1)
                row.update({"A5": bool(A5), "A3": bool(A3), "exec": d5["pred"], "ade5": float(e.mean()),
                            "fde5": float(e[-1]), "ade3": float(e[:6].mean()), "fde3": float(e[5])})
            else:
                row.update({"A5": False, "A3": False, "exec": None,
                            "ade5": None, "fde5": None, "ade3": None, "fde3": None})
            s[c] = row
        scn.append(s)

    Hs = np.array([s["stored_entropy"] for s in scn if s["stored_entropy"] is not None])
    q1, q2 = np.quantile(Hs, [1 / 3, 2 / 3])
    for s in scn:
        h = s["stored_entropy"]
        s["uncertainty"] = None if h is None else ("low" if h <= q1 else ("mid" if h <= q2 else "high"))

    def cond_table(sub):
        t = {}
        for c in CONDS:
            rows = [{"log": s["log"], **s[c]} for s in sub]
            t[c] = {k: boot_mean(rows, k) for k in ("A5", "A3", "ade5", "fde5", "ade3", "fde3", "entropy",
                                                      "gt_margin0", "gt_prob0", "valid", "cot_present",
                                                      "think_tokens", "truncated", "runaway")}
            # ADE/FDE on scenes valid under ALL three conditions, so the rows are comparable
            allv = [{"log": s["log"], **s[c]} for s in sub if all(s[k]["valid"] for k in CONDS)]
            t[c]["ade5_allvalid"] = boot_mean(allv, "ade5")
            t[c]["fde5_allvalid"] = boot_mean(allv, "fde5")
        return t

    def pair_table(sub):
        out = {}
        for a, b in PAIRS:
            key = f"{b} - {a}"
            out[key] = {
                "A5": paired(sub, a, b, "A5", binary=True),
                "A3": paired(sub, a, b, "A3", binary=True),
                "ade5": paired(sub, a, b, "ade5"), "fde5": paired(sub, a, b, "fde5"),
                "ade3": paired(sub, a, b, "ade3"), "fde3": paired(sub, a, b, "fde3"),
                "entropy": paired(sub, a, b, "entropy"), "gt_margin0": paired(sub, a, b, "gt_margin0"),
                "first_token_same": float(np.mean([s[a]["first_action"] == s[b]["first_action"] for s in sub
                                                   if s[a]["first_action"] is not None and s[b]["first_action"] is not None])),
                "all10_tokens_same": float(np.mean([s[a]["action_idx"] == s[b]["action_idx"] for s in sub])),
                "coarse_action_same": float(np.mean([s[a]["exec"] == s[b]["exec"] for s in sub
                                                     if s[a]["exec"] and s[b]["exec"]])),
            }
        return out

    summary = {"n_scenes": len(scn), "maps": dict(sorted({m: sum(s["map"] == m for s in scn) for m in {s["map"] for s in scn}}.items())),
               "n_logs": len({s["log"] for s in scn}),
               "difficulty_counts": {k: sum(s["difficulty"] == k for s in scn) for k in ("easy", "hard")},
               "uncertainty_cut": [float(q1), float(q2)],
               "overall": {"conditions": cond_table(scn), "pairs": pair_table(scn)}, "strata": {}}
    for name, key, vals in (("difficulty", "difficulty", ("easy", "hard")),
                            ("uncertainty", "uncertainty", ("low", "mid", "high"))):
        summary["strata"][name] = {}
        for v in vals:
            sub = [s for s in scn if s[key] == v]
            summary["strata"][name][v] = {"n": len(sub), "conditions": cond_table(sub), "pairs": pair_table(sub)}

    # does the CoT effect (cot - fast) differ between strata?  bootstrap by log of the gap
    def interaction(key, lo, hi, metric):
        by = defaultdict(lambda: {lo: [], hi: []})
        for s in scn:
            if s[key] in (lo, hi) and s["fast"].get(metric) is not None and s["cot"].get(metric) is not None:
                by[s["log"]][s[key]].append(float(s["cot"][metric]) - float(s["fast"][metric]))
        logs = list(by)
        rng = random.Random(1)

        def gap(ls):
            h = [v for lg in ls for v in by[lg][hi]]
            l_ = [v for lg in ls for v in by[lg][lo]]
            return (np.mean(h) - np.mean(l_)) if h and l_ else np.nan
        pt = gap(logs) if logs else np.nan
        boots = [gap([rng.choice(logs) for _ in logs]) for _ in range(REPS)] if logs else []
        boots = [b for b in boots if np.isfinite(b)]
        if not np.isfinite(pt) or not boots:
            return None                              # a stratum is empty: no interaction to estimate
        return {"gap": float(pt), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    summary["interaction_cot_minus_fast"] = {
        "A5 hard - easy": interaction("difficulty", "easy", "hard", "A5"),
        "A5 high - low uncertainty": interaction("uncertainty", "low", "high", "A5"),
        "ade5 hard - easy": interaction("difficulty", "easy", "hard", "ade5"),
        "ade5 high - low uncertainty": interaction("uncertainty", "low", "high", "ade5"),
    }
    summary["reproduction"] = {
        "natural_all10_same_as_stored_natural": float(np.mean([r["conditions"]["natural"]["action_idx"] == r["stored_armN_action_idx"] for r in recs])),
        "fast_all10_same_as_stored_natural": float(np.mean([r["conditions"]["fast"]["action_idx"] == r["stored_armN_action_idx"] for r in recs])),
        "cot_first_token_same_as_stored_cot": float(np.mean([r["conditions"]["cot"]["action_idx"][:1] == r["stored_armC_action_idx"][:1] for r in recs])),
        "natural_cot_rate": float(np.mean([r["conditions"]["natural"]["cot_present"] for r in recs])),
    }

    with open(os.path.join(args.run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.run, "scenes.jsonl"), "w") as f:
        for s in scn:
            f.write(json.dumps(s) + "\n")

    # ---------------------------------------------------------------- console + markdown
    def pc(d):
        return "–" if d is None else f"{100 * d['mean']:.1f}% [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]"

    def nm(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def dpc(d):
        if d is None:
            return "–"
        s = f"{100 * d['mean']:+.1f}%p [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
        if "mcnemar" in d:
            s += f", McNemar p={d['mcnemar']['p']:.3g} ({d['mcnemar']['b_better']}↑/{d['mcnemar']['a_better']}↓)"
        return s

    def dnm(d, nd=2):
        if d is None:
            return "–"
        return f"{d['mean']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}], Wilcoxon p={d['wilcoxon_p']:.3g} (n={d['n']})"

    O = summary["overall"]
    W = []
    w = W.append
    w("# AutoVLA — Natural vs Forced Fast vs Forced CoT (unbiased, paired)\n")
    rp = summary["reproduction"]
    w(f"장면 {summary['n_scenes']}개 (전체 2,748개에서 도시 층화 무작위 추출, 결과·실패 여부를 보지 않음; 로그 {summary['n_logs']}개). "
      f"세 조건은 같은 장면·prompt·vision 입력·seed로 한 harness에서 생성했습니다. [ ]는 log 단위 cluster bootstrap 95% CI.\n")
    w("## 요약\n")
    w("| Condition | A+ (5 s) | ADE (m) | FDE (m) | First-token entropy |")
    w("|---|---:|---:|---:|---:|")
    for c in CONDS:
        t = O["conditions"][c]
        w(f"| {NAMES[c]} | {pc(t['A5'])} | {nm(t['ade5_allvalid'])} | {nm(t['fde5_allvalid'])} | {nm(t['entropy'], 3)} |")
    w("")
    w("A+는 모든 장면 기준(유효한 action 10개를 못 낸 출력은 A−). ADE/FDE는 세 조건 모두 유효한 장면 기준 5 s horizon, "
      "entropy는 첫 action token을 내는 step의 action-token 분포.\n")

    w("## 쌍대 비교 (뒤 − 앞)\n")
    w("| 비교 | ΔA+ (5 s) | ΔADE 5 s (m) | ΔFDE 5 s (m) | Δ entropy | 첫 토큰 동일 | 10 토큰 동일 | coarse action 동일 |")
    w("|---|---|---|---|---|---:|---:|---:|")
    for k, v in O["pairs"].items():
        w(f"| {k} | {dpc(v['A5'])} | {dnm(v['ade5'])} | {dnm(v['fde5'])} | {dnm(v['entropy'], 3)} | "
          f"{100 * v['first_token_same']:.1f}% | {100 * v['all10_tokens_same']:.1f}% | {100 * v['coarse_action_same']:.1f}% |")
    w("")
    w("### 3 s horizon\n")
    w("| 비교 | ΔA+ (3 s) | ΔADE 3 s | ΔFDE 3 s |")
    w("|---|---|---|---|")
    for k, v in O["pairs"].items():
        w(f"| {k} | {dpc(v['A3'])} | {dnm(v['ade3'])} | {dnm(v['fde3'])} |")
    w("")

    w("## 조건별 상세\n")
    w("| Condition | A+ 3 s | 유효 출력 | CoT 발생 | think 토큰 수 | max_length 도달 | action 토큰 폭주 | GT margin (step 0) | GT prob (step 0) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in CONDS:
        t = O["conditions"][c]
        w(f"| {NAMES[c]} | {pc(t['A3'])} | {pc(t['valid'])} | {pc(t['cot_present'])} | {nm(t['think_tokens'], 0)} | "
          f"{pc(t['truncated'])} | {pc(t['runaway'])} | {nm(t['gt_margin0'])} | {nm(t['gt_prob0'], 3)} |")
    w("")

    w("## 장면 난이도·불확실성별 CoT 효과\n")
    w(f"난이도: hard = GT가 정지/감속/회전하거나 위험(선행차 ≤15 m, 보행자 위험)이 있음 "
      f"(easy {summary['difficulty_counts']['easy']}, hard {summary['difficulty_counts']['hard']}). "
      f"불확실성: **이전의 별도 natural run**의 첫 action entropy 3분위 (경계 {q1:.3f}, {q2:.3f}) — 이번 결과를 쓰지 않음.\n")
    w("| 층 | n | A+ Fast | A+ CoT | ΔA+ CoT−Fast | ΔADE CoT−Fast (m) | ΔA+ CoT−Natural |")
    w("|---|---:|---:|---:|---|---|---|")
    for name in ("difficulty", "uncertainty"):
        for v, st in summary["strata"][name].items():
            pt = st["pairs"]
            w(f"| {name}: {v} | {st['n']} | {pc(st['conditions']['fast']['A5'])} | {pc(st['conditions']['cot']['A5'])} | "
              f"{dpc(pt['cot - fast']['A5'])} | {dnm(pt['cot - fast']['ade5'])} | {dpc(pt['cot - natural']['A5'])} |")
    w("")
    w("층간 차이 (CoT−Fast 효과가 층에 따라 다른가):\n")
    for k, v in summary["interaction_cot_minus_fast"].items():
        unit = "%p" if k.startswith("A5") else " m"
        mult = 100 if k.startswith("A5") else 1
        if v is None:
            w(f"- {k}: – (한쪽 층이 비어 있음)")
        else:
            w(f"- {k}: {mult * v['gap']:+.2f}{unit} [{mult * v['ci95'][0]:+.2f}, {mult * v['ci95'][1]:+.2f}]")
    w("")
    w("## 점검\n")
    w(f"- Natural 조건에서 CoT를 스스로 연 비율: {100 * rp['natural_cot_rate']:.1f}%")
    w(f"- 저장된 natural run과 10개 action token 완전 일치: natural {100 * rp['natural_all10_same_as_stored_natural']:.1f}%, "
      f"fast {100 * rp['fast_all10_same_as_stored_natural']:.1f}%; CoT 첫 토큰이 저장된 forced-CoT run과 일치: "
      f"{100 * rp['cot_first_token_same_as_stored_cot']:.1f}%")
    w(f"- 장면 수: 도시별 {summary['maps']}")
    with open(os.path.join(args.run, "FAST_VS_COT.md"), "w") as f:
        f.write("\n".join(W))

    print("\n".join(W))


if __name__ == "__main__":
    main()
