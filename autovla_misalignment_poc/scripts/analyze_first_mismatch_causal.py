#!/usr/bin/env python
"""
Analysis of the first-mismatch intervention (CPU only).

Per scene x condition
  downstream token error   fraction of steps after t* whose token != GT
  A+ (coarse recovery)     P/R/A accept rule, 5 s (3 s also); an invalid output counts as A-
  ADE / FDE                metres vs GT (5 s and 3 s)
  divergence               mean L2 between this condition's trajectory and `original`'s
  error growth             slope of per-pose L2 error over poses t*..9 (m per 0.5 s)
  realignment              fraction of steps after t* that equal GT again

Contrasts (paired, log-cluster bootstrap 95% CI; McNemar for A+, Wilcoxon for continuous)
  gt - original, nn - original, gt - nn            within each group
  [gt - original](A-) - [gt - original](A+)        the interaction the claim rests on
A+ is reported both in full and STEP-MATCHED to A- (A+ scenes reweighted so the distribution
of the first-mismatch step equals A-'s), because the two groups differ in where the first
mismatch happens.

"Why" analysis (stored natural run, no intervention): what distinguishes A- from A+ among scenes
that all had a small first deviation -- univariate AUROC with log-cluster CI.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, PRA_DIR)
import pra_labels as L                                        # noqa: E402
from run_pra_comparison import _av_objects, _pos_to_delta     # noqa: E402

CONDS = ["original", "gt", "nn"]
N_ACT = 10
CFG = L.Config()
REPS = 2000


def a_eval(traj, gt, h):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, h), gt_delta=_pos_to_delta(gt, h))
    return L.label_A(s, CFG)


def softmax(x):
    x = x.astype(np.float64); x = x - x.max(); e = np.exp(x); return e / e.sum()


def wmean_boot(rows, key, weight="w", reps=REPS, seed=0):
    items = [(r["log"], float(r[key]), float(r.get(weight, 1.0))) for r in rows
             if r.get(key) is not None and np.isfinite(float(r[key]))]
    if not items:
        return None
    by = defaultdict(list)
    for lg, v, w in items:
        by[lg].append((v, w))
    logs = list(by)
    rng = random.Random(seed)

    def wm(pts):
        v = np.array([p[0] for p in pts]); w = np.array([p[1] for p in pts]); return float((v * w).sum() / w.sum())
    boots = [wm([p for lg in (rng.choice(logs) for _ in logs) for p in by[lg]]) for _ in range(reps)]
    return {"mean": wm([(v, w) for _, v, w in items]), "n": len(items),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def paired(rows_by_token, a, b, key, binary, weight=False):
    """b - a per scene."""
    d, xa, xb = [], [], []
    for t, r in rows_by_token.items():
        va, vb = r[a].get(key), r[b].get(key)
        if va is None or vb is None:
            continue
        d.append({"log": r["log"], "v": float(vb) - float(va), "w": r["w"] if weight else 1.0})
        xa.append(va); xb.append(vb)
    if not d:
        return None
    out = wmean_boot(d, "v")
    if binary:
        gain = sum((not p) and q for p, q in zip(xa, xb)); loss = sum(p and (not q) for p, q in zip(xa, xb))
        out["mcnemar"] = {"gain": gain, "loss": loss, "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
    else:
        diffs = np.asarray(xb, float) - np.asarray(xa, float)
        out["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
    return out


def interaction(neg, pos, a, b, key, weight_pos, reps=REPS, seed=1):
    """[b-a](A-) - [b-a](A+), logs resampled jointly."""
    def items(g, w):
        out = defaultdict(list)
        for r in g.values():
            if r[a].get(key) is None or r[b].get(key) is None:
                continue
            out[r["log"]].append((float(r[b][key]) - float(r[a][key]), r["w"] if w else 1.0))
        return out
    bn, bp = items(neg, False), items(pos, weight_pos)
    logs = sorted(set(bn) | set(bp))

    def gap(ls):
        n = [p for lg in ls for p in bn.get(lg, [])]; q = [p for lg in ls for p in bp.get(lg, [])]
        if not n or not q:
            return np.nan
        wn = np.average([x[0] for x in n], weights=[x[1] for x in n]); wp = np.average([x[0] for x in q], weights=[x[1] for x in q])
        return wn - wp
    if not logs or not np.isfinite(gap(logs)):
        return None                                   # one group has no usable scenes
    rng = random.Random(seed)
    boots = [gap([rng.choice(logs) for _ in logs]) for _ in range(reps)]
    boots = [x for x in boots if np.isfinite(x)]
    if not boots:
        return None
    return {"gap": float(gap(logs)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_boot": float(min(1.0, 2 * min((np.array(boots) <= 0).mean(), (np.array(boots) >= 0).mean())))}


def auroc_ci(vals, labels, logs, reps=1000, seed=2):
    vals, labels, logs = np.asarray(vals, float), np.asarray(labels, int), np.asarray(logs)
    ok = np.isfinite(vals)
    vals, labels, logs = vals[ok], labels[ok], logs[ok]
    if len(np.unique(labels)) < 2:
        return None
    pt = float(roc_auc_score(labels, vals))
    ul = np.unique(logs)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        sel = np.concatenate([np.where(logs == lg)[0] for lg in rng.choice(ul, len(ul))])
        if len(np.unique(labels[sel])) == 2:
            boots.append(roc_auc_score(labels[sel], vals[sel]))
    return {"auroc": pt, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(vals))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/first_mismatch_causal"))
    ap.add_argument("--tensors", default=os.path.join(POC_DIR, "outputs/full_extract/tensors"))
    ap.add_argument("--annotations", default=os.path.join(POC_DIR, "outputs/annotations"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]

    scenes = {}
    for r in recs:
        gt_traj, gt, t = r["trajectory_gt"], r["gt"], r["t_star"]
        lg = np.load(os.path.join(args.run, "logits", f"{r['token']}.npz"))["step_action_logits"].astype(np.float32)
        s = {"token": r["token"], "group": r["group"], "log": r["log"], "t_star": t}
        base_traj = r["conditions"]["original"].get("trajectory_pred")
        for ci, c in enumerate(CONDS):
            x = r["conditions"][c]
            row = {"valid": x["valid"]}
            acts = x["action_idx"]
            after = list(range(t + 1, N_ACT))
            row["downstream_err"] = (float(np.mean([k >= len(acts) or acts[k] != gt[k] for k in after])) if after else None)
            row["realign"] = (float(np.mean([k < len(acts) and acts[k] == gt[k] for k in after])) if after else None)
            row["shift_by_one"] = (float(np.mean([k < len(acts) and acts[k] == gt[k - 1] for k in after])) if after else None)
            if x["valid"]:
                A5, _ = a_eval(x["trajectory_pred"], gt_traj, 10)
                A3, _ = a_eval(x["trajectory_pred"], gt_traj, 6)
                p, g_ = np.asarray(x["trajectory_pred"], float), np.asarray(gt_traj, float)[:, :2]
                e = np.linalg.norm(p - g_, axis=1)
                row.update({"A5": bool(A5), "A3": bool(A3), "ade5": float(e.mean()), "fde5": float(e[-1]),
                            "ade3": float(e[:6].mean()), "fde3": float(e[5]), "pose_err": e.tolist()})
                ks = np.arange(t, N_ACT)
                row["err_slope"] = float(np.polyfit(ks, e[t:], 1)[0]) if len(ks) >= 2 else None
                row["div_vs_original"] = (float(np.linalg.norm(p - np.asarray(base_traj, float), axis=1).mean())
                                          if base_traj is not None else None)
                ent = [float(-(softmax(lg[ci, k]) * np.log(softmax(lg[ci, k]) + 1e-12)).sum()) for k in after
                       if np.isfinite(lg[ci, k]).all()]
                row["entropy_after"] = float(np.mean(ent)) if ent else None
            else:
                row.update({"A5": False, "A3": False, "ade5": None, "fde5": None, "ade3": None, "fde3": None,
                            "err_slope": None, "div_vs_original": None, "entropy_after": None})
            s[c] = row
        # scene factors for the "why" analysis (stored natural run, no intervention)
        v = float(np.linalg.norm(r["velocity"][:2]))
        kg = L.kinematics(_pos_to_delta(gt_traj, 10))
        anno = json.load(open(os.path.join(args.annotations, f"{r['token']}.json")))
        sc = L.scene_from_objects(_av_objects(anno))
        z = np.load(os.path.join(args.tensors, f"{r['token']}_N.npz"))
        al = z["action_logits"][-1].astype(np.float32)             # (11, 2048): row k generates action k
        p_t = softmax(al[t])
        pred_t = r["pred"][t]
        s.update({
            "speed": v, "gt_turn": int(kg.lateral != "STRAIGHT"), "gt_slow": int(kg.longitudinal in ("STOP", "DECELERATE")),
            "hazard": int(bool(sc.hazards(CFG))), "d_pred_gt": r["d_pred_gt"], "nn_rank_pred": r["nn_rank_pred"],
            "entropy_at_t": float(-(p_t * np.log(p_t + 1e-12)).sum()),
            "margin_pred_over_gt_at_t": float(al[t, pred_t] - al[t, r["gt"][t]]),
            "gt_prob_at_t": float(p_t[r["gt"][t]]),
            "stored_downstream_err": float(np.mean([r["pred"][k] != gt[k] for k in range(t + 1, N_ACT)])) if t < N_ACT - 1 else None,
            "stored_shift_by_one": float(np.mean([r["pred"][k] == gt[k - 1] for k in range(t + 1, N_ACT)])) if t < N_ACT - 1 else None,
            "stored_entropy_after": float(np.mean([-(softmax(al[k]) * np.log(softmax(al[k]) + 1e-12)).sum() for k in range(t + 1, N_ACT)])) if t < N_ACT - 1 else None,
            "remaining_steps": N_ACT - t - 1,
        })
        scenes[r["token"]] = s

    neg = {k: v for k, v in scenes.items() if v["group"] == "A-"}
    pos_all = {k: v for k, v in scenes.items() if v["group"] == "A+"}
    tn = Counter(v["t_star"] for v in neg.values())
    tp = Counter(v["t_star"] for v in pos_all.values())
    for v in pos_all.values():                     # step-matched importance weights
        v["w"] = (tn.get(v["t_star"], 0) / max(len(neg), 1)) / (tp[v["t_star"]] / len(pos_all))
    for v in neg.values():
        v["w"] = 1.0
    pos_matched = {k: v for k, v in pos_all.items() if v["w"] > 0}

    def group_summary(g, weighted):
        out = {}
        for c in CONDS:
            rows = [{"log": v["log"], "w": v["w"] if weighted else 1.0, **v[c]} for v in g.values()]
            out[c] = {k: wmean_boot(rows, k) for k in ("A5", "A3", "downstream_err", "realign", "shift_by_one",
                                                        "ade5", "fde5", "ade3", "fde3", "div_vs_original",
                                                        "err_slope", "entropy_after", "valid")}
        out["contrasts"] = {}
        for a, b in (("original", "gt"), ("original", "nn"), ("nn", "gt")):
            out["contrasts"][f"{b} - {a}"] = {k: paired(g, a, b, k, k in ("A5", "A3"), weight=weighted)
                                              for k in ("A5", "A3", "downstream_err", "ade5", "fde5", "ade3", "fde3", "err_slope")}
        return out

    summary = {
        "n": {"A-": len(neg), "A+": len(pos_all), "A+ step-matched (weighted)": len(pos_matched)},
        "t_star_hist": {"A-": dict(sorted(tn.items())), "A+": dict(sorted(tp.items()))},
        "A-": group_summary(neg, False),
        "A+ (all)": group_summary(pos_all, False),
        "A+ (step-matched)": group_summary(pos_matched, True),
        "interaction": {},
        "why": {},
    }
    for lab, pos, w in (("vs A+ step-matched", pos_matched, True), ("vs A+ all", pos_all, False)):
        summary["interaction"][lab] = {f"{k}: [gt-original](A-) - [gt-original](A+)": interaction(neg, pos, "original", "gt", k, w)
                                       for k in ("A5", "downstream_err", "ade5", "fde5")}
        summary["interaction"][lab].update({f"{k}: [nn-original](A-) - [nn-original](A+)": interaction(neg, pos, "original", "nn", k, w)
                                            for k in ("A5", "ade5")})

    # reproduction of the stored group label under the in-harness original rerun
    summary["reproduction"] = {
        "A- still A- under rerun original": float(np.mean([not v["original"]["A5"] for v in neg.values()])),
        "A+ still A+ under rerun original": float(np.mean([v["original"]["A5"] for v in pos_all.values()])),
    }
    # does the NN deviation also fail? (scene fragility vs token-specific)
    summary["nn_fragility"] = {
        "A-: nn also A-": float(np.mean([not v["nn"]["A5"] for v in neg.values()])),
        "A+: nn also A+": float(np.mean([v["nn"]["A5"] for v in pos_all.values()])),
        "d_nn_gt median (m)": float(np.median([r["d_nn_gt"] for r in recs])),
        "d_pred_gt median (m)": float(np.median([r["d_pred_gt"] for r in recs])),
    }

    feats = ["t_star", "remaining_steps", "speed", "gt_turn", "gt_slow", "hazard", "d_pred_gt", "nn_rank_pred",
             "entropy_at_t", "margin_pred_over_gt_at_t", "gt_prob_at_t", "stored_entropy_after",
             "stored_shift_by_one", "stored_downstream_err"]
    allv = list(scenes.values())
    labels = [int(v["group"] == "A-") for v in allv]
    logs = [v["log"] for v in allv]
    for f_ in feats:
        vals = [np.nan if v.get(f_) is None else float(v[f_]) for v in allv]
        summary["why"][f_] = {
            "A- median": float(np.nanmedian([x for x, l in zip(vals, labels) if l == 1])),
            "A+ median": float(np.nanmedian([x for x, l in zip(vals, labels) if l == 0])),
            "auroc_higher_means_A-": auroc_ci(vals, labels, logs),
        }
    # step-matched version of the key "why" features
    summary["why_step_matched"] = {}
    for f_ in ("speed", "gt_turn", "gt_slow", "d_pred_gt", "entropy_at_t", "stored_entropy_after", "stored_shift_by_one"):
        a_ = [v[f_] for v in neg.values() if v.get(f_) is not None]
        ws = [(v[f_], v["w"]) for v in pos_all.values() if v.get(f_) is not None and v["w"] > 0]
        summary["why_step_matched"][f_] = {
            "A- mean": float(np.mean(a_)) if a_ else float("nan"),
            "A+ step-matched mean": float(np.average([x for x, _ in ws], weights=[w for _, w in ws])) if ws else float("nan")}

    with open(os.path.join(args.run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.run, "scenes.jsonl"), "w") as f:
        for v in scenes.values():
            f.write(json.dumps(v) + "\n")

    # ------------------------------------------------------------------ report
    def pc(d):
        return "–" if d is None else f"{100*d['mean']:.1f}% [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}]"

    def nm(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def dpc(d):
        if d is None:
            return "–"
        s = f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]"
        return s + (f", McNemar p={d['mcnemar']['p']:.2g} ({d['mcnemar']['gain']}↑/{d['mcnemar']['loss']}↓)" if "mcnemar" in d else "")

    def dnm(d, nd=2, pct=False):
        if d is None:
            return "–"
        m = 100 if pct else 1
        u = "%p" if pct else ""
        return f"{m*d['mean']:+.{nd if not pct else 1}f}{u} [{m*d['ci95'][0]:+.{nd if not pct else 1}f}, {m*d['ci95'][1]:+.{nd if not pct else 1}f}], p={d.get('wilcoxon_p', float('nan')):.2g}"

    def ix(d, pct=False):
        if d is None:
            return "–"
        m = 100 if pct else 1
        u = "%p" if pct else " m"
        return f"{m*d['gap']:+.2f}{u} [{m*d['ci95'][0]:+.2f}, {m*d['ci95'][1]:+.2f}], p={d['p_boot']:.2g}"

    W = []
    w = W.append
    w("# AutoVLA Natural/Fast — 첫 action-token mismatch의 인과 효과\n")
    w(f"GPU1, natural fast 경로(full extraction arm N)만 사용. 첫 불일치가 있는 장면 전부: A− {len(neg)}, A+ {len(pos_all)}. "
      "그룹은 저장된 natural run으로 개입 전에 고정했습니다. [ ]는 log 단위 cluster bootstrap 95% CI.\n")
    w("A+는 두 가지로 보고합니다: 전체, 그리고 **첫 불일치 step 분포를 A−에 맞춘 가중 비교(step-matched)**. 두 그룹은 첫 불일치 위치가 달라 "
      "남은 step 수가 다르기 때문입니다.\n")
    w("## 요약: A− vs A+\n")
    w("| 지표 | A− original | A− GT 1-token | A− NN token | A+ (step-matched) original | A+ GT 1-token | A+ NN token |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    N_, P_ = summary["A-"], summary["A+ (step-matched)"]
    for key, name, fmt in (("A5", "A+ (coarse action, 5 s)", pc), ("A3", "A+ (3 s)", pc),
                           ("downstream_err", "downstream token error", pc), ("realign", "이후 step GT 재정렬", pc),
                           ("ade5", "ADE 5 s (m)", nm), ("fde5", "FDE 5 s (m)", nm),
                           ("div_vs_original", "original 대비 궤적 발산 (m)", nm), ("err_slope", "오차 증가 속도 (m/step)", lambda d: nm(d, 3)),
                           ("entropy_after", "이후 step action entropy", lambda d: nm(d, 3))):
        w(f"| {name} | {fmt(N_['original'][key])} | {fmt(N_['gt'][key])} | {fmt(N_['nn'][key])} | "
          f"{fmt(P_['original'][key])} | {fmt(P_['gt'][key])} | {fmt(P_['nn'][key])} |")
    w("")
    w("## GT correction 효과 (조건 − original, 장면별 쌍대)\n")
    w("| 그룹 | 대조 | ΔA+ 5 s | Δdownstream error | ΔADE 5 s (m) | ΔFDE 5 s (m) | Δ오차 증가 속도 |")
    w("|---|---|---|---|---|---|---|")
    for gname in ("A-", "A+ (step-matched)", "A+ (all)"):
        for cname in ("gt - original", "nn - original", "gt - nn"):
            c = summary[gname]["contrasts"][cname]
            w(f"| {gname} | {cname} | {dpc(c['A5'])} | {dnm(c['downstream_err'], pct=True)} | {dnm(c['ade5'])} | {dnm(c['fde5'])} | {dnm(c['err_slope'], 3)} |")
    w("")
    w("## 핵심 비교: GT correction 효과가 A−에서 더 큰가 (interaction)\n")
    w("| 비교 대상 | ΔA+ 차이 | Δdownstream error 차이 | ΔADE 차이 | ΔFDE 차이 | NN: ΔA+ 차이 | NN: ΔADE 차이 |")
    w("|---|---|---|---|---|---|---|")
    for lab, I in summary["interaction"].items():
        g = lambda k: I[k]  # noqa: E731
        w(f"| {lab} | {ix(g('A5: [gt-original](A-) - [gt-original](A+)'), True)} | {ix(g('downstream_err: [gt-original](A-) - [gt-original](A+)'), True)} | "
          f"{ix(g('ade5: [gt-original](A-) - [gt-original](A+)'))} | {ix(g('fde5: [gt-original](A-) - [gt-original](A+)'))} | "
          f"{ix(g('A5: [nn-original](A-) - [nn-original](A+)'), True)} | {ix(g('ade5: [nn-original](A-) - [nn-original](A+)'))} |")
    w("")
    w("## 왜 비슷하게 작은 편차가 어떤 장면에서만 증폭되는가 (저장된 natural run, 개입 없음)\n")
    w("| 요인 | A− 중앙값 | A+ 중앙값 | AUROC (높을수록 A−) [CI] |")
    w("|---|---:|---:|---|")
    for f_, d in summary["why"].items():
        a = d["auroc_higher_means_A-"]
        w(f"| {f_} | {d['A- median']:.3f} | {d['A+ median']:.3f} | " + ("–" if a is None else f"{a['auroc']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]") + " |")
    w("")
    w("step-matched 평균 (첫 불일치 위치를 맞춘 뒤에도 남는 차이):\n")
    w("| 요인 | A− | A+ step-matched |")
    w("|---|---:|---:|")
    for f_, d in summary["why_step_matched"].items():
        w(f"| {f_} | {d['A- mean']:.3f} | {d['A+ step-matched mean']:.3f} |")
    w("")
    w("## 점검\n")
    w(f"- 첫 불일치 step 분포: A− {summary['t_star_hist']['A-']}, A+ {summary['t_star_hist']['A+']}")
    w(f"- 저장된 그룹이 이 harness의 original 재생성에서도 유지: A− {100*summary['reproduction']['A- still A- under rerun original']:.1f}%, "
      f"A+ {100*summary['reproduction']['A+ still A+ under rerun original']:.1f}%")
    w(f"- 편차 크기: 원래 예측 토큰–GT 변위 중앙값 {summary['nn_fragility']['d_pred_gt median (m)']:.3f} m, NN 토큰–GT {summary['nn_fragility']['d_nn_gt median (m)']:.3f} m")
    w(f"- NN 토큰을 넣어도 결과 유지: A−에서 여전히 실패 {100*summary['nn_fragility']['A-: nn also A-']:.1f}%, A+에서 여전히 정상 {100*summary['nn_fragility']['A+: nn also A+']:.1f}%")
    with open(os.path.join(args.run, "FIRST_MISMATCH_CAUSAL.md"), "w") as f:
        f.write("\n".join(W))
    print("\n".join(W))


if __name__ == "__main__":
    main()
