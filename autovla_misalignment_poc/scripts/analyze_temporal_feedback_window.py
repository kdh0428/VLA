#!/usr/bin/env python
"""
Analysis of the temporal feedback window experiment (CPU only).

Metric definitions are imported unchanged from analyze_action_history:
  amplification = A- AND FDE(5 s) > 3 m; recovery = A+ (P/R/A accept rule, 5 s); ADE/FDE 5 s;
  downstream token error / GT re-alignment over steps after t*; entropy = mean post-mismatch
  action entropy; error growth = slope of per-pose L2 error over poses t*..9 (m/step).
Additional (this experiment):
  post_release_realign  fraction of executed tokens == GT over the steps whose context contains a
                        self-generated token AFTER the window (k >= t* + start + length + 1);
                        paired with GT-history at exactly the same steps -> self-sustain gap.
  post_release_entropy  mean entropy over the same steps.
Window length is clamped by the horizon: positions t*+1..8 are the only context positions, so a
w-step window at t* is effectively min(w, 8 - t*) steps and equals GT-history once w >= 8 - t*.

Critical window: smallest w whose fraction of the Normal -> GT-history effect
f(w) = (m_normal - m_w) / (m_normal - m_full) reaches 0.9 (amplification for A-, FDE both).
Its uncertainty = the distribution of that smallest w over log-cluster bootstrap replicates.
Stats: paired McNemar (binary) / Wilcoxon (continuous) vs Normal, vs the previous window (w-1)
and vs Full; 95% CI = log-cluster bootstrap (2000). A- and A+ reported separately; A- vs A+
interaction bootstrapped with logs resampled jointly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(POC_DIR, "scripts"))
from analyze_action_history import unit_metrics, AMP_FDE  # noqa: E402
from analyze_prev_action_state_patching import Boot       # noqa: E402

N_ACT = 10
WINS = [f"win{w}" for w in range(1, 9)]
METRICS = ["amplification", "recovery", "ade5", "fde5", "downstream_err", "realign", "entropy", "err_slope",
           "post_release_realign", "post_release_entropy", "self_sustain_gap"]
BINARY = {"amplification", "recovery"}
PCT = {"downstream_err", "realign", "post_release_realign", "self_sustain_gap"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/temporal_feedback_window"))
    ap.add_argument("--narrative", default=None)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "run_meta.json")))
    ROWS = meta["rows"]
    names = [r["name"] for r in ROWS]
    spec = {r["name"]: r for r in ROWS}
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    ep = os.path.join(args.run, "errors.jsonl")
    errs = [json.loads(l) for l in open(ep)] if os.path.exists(ep) else []

    units = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": r["perturbation"], "t_star": t, "m": {}}
        acts = {nm: r["rows"][nm]["action_idx"] for nm in names}
        for nm in names:
            x = r["rows"][nm]
            mm = unit_metrics(x["action_idx"], x["trajectory_pred"], x["ent"], gt, gtraj, t)
            s = spec[nm]
            if s["length"] == 0:
                rel = list(range(t + 2, N_ACT))                 # every step that can see a self-generated token
            else:
                rel = list(range(t + s["start"] + s["length"] + 1, N_ACT)) if s["length"] < 99 else []
            if rel:
                mm["post_release_realign"] = float(np.mean([acts[nm][k] == gt[k] for k in rel]))
                mm["post_release_entropy"] = float(np.mean([x["ent"][k - t - 1] for k in rel]))
                mm["self_sustain_gap"] = mm["post_release_realign"] - float(np.mean([acts["gt_history"][k] == gt[k] for k in rel]))
            else:
                mm["post_release_realign"] = mm["post_release_entropy"] = mm["self_sustain_gap"] = None
            mm["eff_len"] = min(s["length"], max(0, 8 - t - s["start"] + 1))
            u["m"][nm] = mm
        u["checks"] = {
            "prefix_identical": all(a[:t + 1] == acts["normal"][:t + 1] for a in acts.values()),
            "first_free_step_identical": all(a[t + 1] == acts["normal"][t + 1] for a in acts.values()) if t + 1 < N_ACT else True,
            "win_saturated_eq_gt_history": all(acts[f"win{w}"] == acts["gt_history"] for w in range(1, 9) if w >= 8 - t),
            "gt_context_count_ok": all(r["rows"][f"win{w}"]["n_gt_context_positions"] == min(w, max(0, 8 - t)) for w in range(1, 9)),
            "normal_no_gt_context": r["rows"]["normal"]["n_gt_context_positions"] == 0,
        }
        pri = r.get("prior_action_history") or {}
        u["repro"] = {c: (acts[c] == pri[c]) if c in pri else None for c in ("normal", "gt_history")}
        units.append(u)

    groups = ("A-", "A+")
    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in groups},
         "n_scenes": {g: len({u["token"] for u in units if u["group"] == g}) for g in groups},
         "t_star_distribution": {g: dict(sorted(defaultdict(int, {t: sum(u["group"] == g and u["t_star"] == t for u in units)
                                                                  for t in range(10)}).items())) for g in groups},
         "n_errors": len(errs), "errors": errs, "rows": ROWS, "amplification_fde_threshold_m": AMP_FDE}
    S["sanity"] = {k: float(np.mean([u["checks"][k] for u in units])) for k in units[0]["checks"]}
    for c in ("normal", "gt_history"):
        v = [u["repro"][c] for u in units if u["repro"][c] is not None]
        S["sanity"][f"reproduces_action_history_causal_{c}"] = float(np.mean(v)) if v else None

    CURVE = ["normal"] + WINS

    def stats_block(us, seed=0):
        if not us:
            return None
        bt = Boot([u["log"] for u in us], seed=seed)
        nc = len(bt.labels)
        out = {"n": len(us), "n_scenes": len({u["token"] for u in us}), "rows": {}}
        for nm in names:
            d = {"level": {m: bt([u["m"][nm][m] for u in us]) for m in METRICS}, "vs_normal": {}, "vs_prev": {}, "vs_full": {}}
            prev = None
            if nm.startswith("win"):
                w = int(nm[3:]); prev = "normal" if w == 1 else f"win{w-1}"
            for key, base in (("vs_normal", "normal"), ("vs_prev", prev), ("vs_full", "gt_history")):
                if base is None or base == nm:
                    continue
                for m in METRICS:
                    pairs = [(u["m"][nm][m], u["m"][base][m]) for u in us if u["m"][nm][m] is not None and u["m"][base][m] is not None]
                    if not pairs:
                        d[key][m] = None; continue
                    c = bt([None if a is None or b is None else float(a) - float(b)
                            for a, b in ((u["m"][nm][m], u["m"][base][m]) for u in us)])
                    if m in BINARY:
                        g_ = sum(bool(a) and not bool(b) for a, b in pairs); l_ = sum(bool(b) and not bool(a) for a, b in pairs)
                        c["mcnemar"] = {"row_only": g_, "base_only": l_, "p": 1.0 if g_ + l_ == 0 else float(binomtest(g_, g_ + l_, 0.5).pvalue)}
                    else:
                        dd = np.array([float(a) - float(b) for a, b in pairs])
                        c["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                    d[key][m] = c
            out["rows"][nm] = d
        # fraction of the Normal -> Full effect and bootstrap distribution of the critical window
        out["fraction"] = {}
        out["critical_window"] = {}
        for m in ("amplification", "fde5", "downstream_err", "recovery"):
            vals = {nm: np.array([float(u["m"][nm][m]) if u["m"][nm][m] is not None else np.nan for u in us]) for nm in names}
            ok = np.all([np.isfinite(v) for v in vals.values()], axis=0)
            cs = {nm: np.bincount(bt.cid[ok], weights=vals[nm][ok], minlength=nc) for nm in names}
            gap_pt = cs["normal"].sum() - cs["gt_history"].sum()
            gap_bs = cs["normal"][bt.R].sum(1) - cs["gt_history"][bt.R].sum(1)
            fr, frbs = {}, {}
            for nm in names:
                if nm in ("normal", "gt_history"):
                    continue
                num_pt = cs["normal"].sum() - cs[nm].sum()
                num_bs = cs["normal"][bt.R].sum(1) - cs[nm][bt.R].sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = num_bs / gap_bs
                frbs[nm] = b
                bf = b[np.isfinite(b)]
                fr[nm] = {"mean": float(num_pt / gap_pt) if gap_pt else None,
                          "ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))] if len(bf) else None}
            out["fraction"][m] = fr
            crit = {}
            for thr in (0.5, 0.8, 0.9):
                pt = next((w for w in range(1, 9) if fr[f"win{w}"]["mean"] is not None and fr[f"win{w}"]["mean"] >= thr), None)
                reps = []
                for i in range(len(bt.R)):
                    if not np.isfinite(gap_bs[i]) or gap_bs[i] <= 0:
                        continue
                    reps.append(next((w for w in range(1, 9) if frbs[f"win{w}"][i] >= thr), 9))
                reps = np.array(reps)
                crit[str(thr)] = {"point": pt, "boot_median": float(np.median(reps)) if len(reps) else None,
                                  "boot_ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))] if len(reps) else None,
                                  "boot_dist": {str(w): float(np.mean(reps == w)) for w in range(1, 10)} if len(reps) else None,
                                  "n_valid_reps": int(len(reps))}
            out["critical_window"][m] = crit
        return out

    S["subsets"] = {}
    subsets = {
        "all perturbations": lambda u: True,
        "original token only": lambda u: u["pert"] == "original",
        "t*=0": lambda u: u["t_star"] == 0,
        "t*=1": lambda u: u["t_star"] == 1,
        "t*>=2": lambda u: u["t_star"] >= 2,
    }
    for sname, pred in subsets.items():
        S["subsets"][sname] = {g: stats_block([u for u in units if u["group"] == g and pred(u)]) for g in groups}
    S["subsets"]["Normal-amplified A- units"] = {"A-": stats_block([u for u in units if u["group"] == "A-" and u["m"]["normal"]["amplification"]]), "A+": None}

    # A- vs A+ interaction of (row - normal)
    inter = {}
    logs = sorted({u["log"] for u in units})
    rng = np.random.default_rng(7)
    R = rng.integers(0, len(logs), size=(2000, len(logs)))
    li = {lg: i for i, lg in enumerate(logs)}
    for nm in names[1:]:
        inter[nm] = {}
        for m in ("amplification", "recovery", "fde5", "downstream_err", "realign"):
            sums = {g: np.zeros(len(logs)) for g in groups}; cnts = {g: np.zeros(len(logs)) for g in groups}
            for u in units:
                a, b = u["m"][nm][m], u["m"]["normal"][m]
                if a is None or b is None:
                    continue
                sums[u["group"]][li[u["log"]]] += float(a) - float(b); cnts[u["group"]][li[u["log"]]] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                pt = sums["A-"].sum() / cnts["A-"].sum() - sums["A+"].sum() / cnts["A+"].sum()
                bs = sums["A-"][R].sum(1) / cnts["A-"][R].sum(1) - sums["A+"][R].sum(1) / cnts["A+"][R].sum(1)
            bs = bs[np.isfinite(bs)]
            inter[nm][m] = {"gap": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    S["interaction_Aminus_minus_Aplus"] = inter

    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(args.run, "figures"); os.makedirs(fd, exist_ok=True)
    XS = list(range(0, 9))

    def curve(block, m, scale=1.0):
        rows = block["rows"]
        mu = [rows[nm]["level"][m]["mean"] * scale for nm in CURVE]
        lo = [rows[nm]["level"][m]["ci95"][0] * scale for nm in CURVE]
        hi = [rows[nm]["level"][m]["ci95"][1] * scale for nm in CURVE]
        return mu, lo, hi

    A = S["subsets"]["all perturbations"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    for sname, col in (("all perturbations", "k"), ("original token only", "tab:purple"), ("Normal-amplified A- units", "tab:red")):
        b = S["subsets"][sname]["A-"]
        mu, lo, hi = curve(b, "amplification", 100)
        ax.plot(XS, mu, "-o", color=col, ms=4, label=f"A− {sname} (n={b['n']})"); ax.fill_between(XS, lo, hi, color=col, alpha=.1)
        ax.axhline(b["rows"]["gt_history"]["level"]["amplification"]["mean"] * 100, color=col, ls=":", lw=1)
    b = A["A-"]["rows"]
    for nm, mk in (("delay2_len1", "s"), ("delay3_len1", "s"), ("delay4_len1", "s"), ("delay2_len2", "D"), ("delay3_len2", "D")):
        s = spec[nm]
        ax.plot([s["length"]], [b[nm]["level"]["amplification"]["mean"] * 100], mk, color="tab:orange", ms=6)
        ax.annotate(f"start t*+{s['start']}", (s["length"], b[nm]["level"]["amplification"]["mean"] * 100), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    ax.axhline(A["A+"]["rows"]["normal"]["level"]["amplification"]["mean"] * 100, color="tab:green", ls="--", lw=1, label="A+ Normal AR level")
    ax.set_xlabel("correction window w (context t*+1 … t*+w = GT)"); ax.set_ylabel("A− amplification (%)")
    ax.set_title("A− amplification vs window (dotted = Full GT-history; orange = delayed windows)", fontsize=9); ax.legend(fontsize=7)
    ax = axs[1]
    for g, col in (("A-", "tab:red"), ("A+", "tab:blue")):
        mu, lo, hi = curve(A[g], "fde5")
        ax.plot(XS, mu, "-o", color=col, ms=4, label=f"{g} FDE"); ax.fill_between(XS, lo, hi, color=col, alpha=.12)
        ax.axhline(A[g]["rows"]["gt_history"]["level"]["fde5"]["mean"], color=col, ls=":", lw=1)
    ax.set_xlabel("correction window w"); ax.set_ylabel("FDE 5 s (m)"); ax.set_title("FDE vs window (dotted = Full)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_amplification_fde_vs_window.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, m, lab in ((axs[0], "amplification", "A− amplification"), (axs[1], "fde5", "A− FDE")):
        for sname, col in (("all perturbations", "k"), ("t*=0", "tab:blue"), ("t*=1", "tab:orange"), ("t*>=2", "tab:green")):
            b = S["subsets"][sname]["A-"]
            if not b:
                continue
            fr = b["fraction"][m]
            mu = [0.0] + [fr[w]["mean"] for w in WINS]
            ax.plot(XS, mu, "-o", color=col, ms=4, label=f"{sname} (n={b['n']})")
            if sname == "all perturbations":
                ax.fill_between(XS[1:], [fr[w]["ci95"][0] for w in WINS], [fr[w]["ci95"][1] for w in WINS], color=col, alpha=.1)
        ax.axhline(0.9, color="gray", ls="--", lw=1); ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("correction window w"); ax.set_ylabel("fraction of Normal→Full effect"); ax.set_title(f"{lab}: fraction of GT-history effect by t*")
        ax.legend(fontsize=7); ax.set_ylim(-0.3, 1.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig2_fraction_by_tstar.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for g, col in (("A-", "tab:red"), ("A+", "tab:blue")):
        rows = A[g]["rows"]
        axs[0].plot(range(1, 9), [rows[w]["level"]["self_sustain_gap"]["mean"] * 100 if rows[w]["level"]["self_sustain_gap"] else np.nan for w in WINS],
                    "-o", color=col, label=g)
        axs[1].plot(XS, curve(A[g], "entropy")[0], "-o", color=col, label=f"{g} post-mismatch entropy")
        axs[1].axhline(rows["gt_history"]["level"]["entropy"]["mean"], color=col, ls=":", lw=1)
    axs[0].axhline(0, color="gray", lw=1)
    axs[0].set_xlabel("correction window w"); axs[0].set_ylabel("post-release GT match − GT-history at same steps (%p)")
    axs[0].set_title("self-sustain gap after the window is released (0 = as good as continued correction)", fontsize=9); axs[0].legend()
    axs[1].set_xlabel("correction window w"); axs[1].set_ylabel("nats"); axs[1].set_title("entropy vs window (dotted = Full)"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_self_sustain_entropy.png"), dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ report
    def pc(d):
        return "–" if d is None else f"{100*d['mean']:.1f}% [{100*d['ci95'][0]:.1f}, {100*d['ci95'][1]:.1f}]"

    def nm_(d, nd=2):
        return "–" if d is None else f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"

    def dl(d, m):
        if d is None:
            return "–"
        if m in BINARY:
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] p={d['mcnemar']['p']:.2g}"
        if m in PCT:
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] p={d['wilcoxon_p']:.2g}"
        nd = 3 if m in ("entropy", "err_slope", "post_release_entropy") else 2
        return f"{d['mean']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}] p={d['wilcoxon_p']:.2g}"

    def frs(d):
        return "–" if not d or d["mean"] is None else f"{100*d['mean']:.0f}% [{100*d['ci95'][0]:.0f}, {100*d['ci95'][1]:.0f}]"

    NAR = json.load(open(args.narrative)) if args.narrative and os.path.exists(args.narrative) else {}
    W = []; w = W.append
    w("# AutoVLA Natural/Fast — Temporal Feedback Window\n")
    w("## 1. 결론\n"); w(NAR.get("conclusion", "_(narrative 미작성)_") + "\n")
    w("## 2. 실험 설정\n"); w(NAR.get("setup", "") + "\n")
    w(f"- 단위: 장면 A− {S['n_scenes']['A-']} / A+ {S['n_scenes']['A+']} (A+는 A− 장면당 3개, 같은 첫 mismatch step), (장면, perturbation) 단위 A− {S['n_units']['A-']} / A+ {S['n_units']['A+']}. 실패 {S['n_errors']}.")
    w(f"- t* 분포: A− {S['t_star_distribution']['A-']}, A+ {S['t_star_distribution']['A+']}")
    w("- CI: log 단위 cluster bootstrap 95% (2000회). p: McNemar(이진) / Wilcoxon(연속), 모두 단위별 쌍대.\n")

    w("## 3. Window 길이별 결과 (전체 perturbation)\n")
    w("![fig1](figures/fig1_amplification_fde_vs_window.png)\n")
    for g in groups:
        B = A[g]
        w(f"### {g} (단위 {B['n']}, 장면 {B['n_scenes']})\n")
        w("| 조건 | amplification | A+ recovery | ADE (m) | FDE (m) | downstream token error | GT re-alignment | entropy | error growth (m/step) | Full 효과 대비 (amp) | Full 효과 대비 (FDE) |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        order = ["normal"] + WINS + ["gt_history"] + [n for n in names if n.startswith("delay") or n.startswith("skip")]
        for nm in order:
            L_ = B["rows"][nm]["level"]
            w(f"| {nm} | {pc(L_['amplification'])} | {pc(L_['recovery'])} | {nm_(L_['ade5'])} | {nm_(L_['fde5'])} | {pc(L_['downstream_err'])} | "
              f"{pc(L_['realign'])} | {nm_(L_['entropy'], 3)} | {nm_(L_['err_slope'], 3)} | {frs(B['fraction']['amplification'].get(nm))} | {frs(B['fraction']['fde5'].get(nm))} |")
        w("")
        w("쌍대 비교: vs Normal, 그리고 직전 window 대비 추가 이득(w − (w−1)):\n")
        w("| 조건 | Δamp vs Normal | ΔFDE vs Normal | Δre-align vs Normal | Δentropy vs Normal | Δamp vs w−1 | ΔFDE vs w−1 | Δamp vs Full | ΔFDE vs Full |")
        w("|---|---|---|---|---|---|---|---|---|")
        for nm in WINS + ["gt_history"] + [n for n in names if n.startswith("delay") or n.startswith("skip")]:
            v, p_, f_ = B["rows"][nm]["vs_normal"], B["rows"][nm]["vs_prev"], B["rows"][nm]["vs_full"]
            w(f"| {nm} | {dl(v.get('amplification'), 'amplification')} | {dl(v.get('fde5'), 'fde5')} | {dl(v.get('realign'), 'realign')} | {dl(v.get('entropy'), 'entropy')} | "
              f"{dl(p_.get('amplification'), 'amplification')} | {dl(p_.get('fde5'), 'fde5')} | {dl(f_.get('amplification'), 'amplification')} | {dl(f_.get('fde5'), 'fde5')} |")
        w("")

    w("## 4. Critical window: Full GT-history 효과의 50 / 80 / 90%에 도달하는 최소 w\n")
    w("점 추정 = 전체 표본에서 f(w) ≥ 기준을 만족하는 최소 w. bootstrap = log 재표본마다 최소 w를 구한 분포(9 = w ≤ 8에서 도달 못함).\n")
    w("| subset | group | 지표 | 50% | 80% | 90% (점 추정, bootstrap 중앙값 [95%]) | 90% bootstrap 분포 |")
    w("|---|---|---|---|---|---|---|")
    def cell(x):
        pt = "없음" if x["point"] is None else str(x["point"])
        if x["boot_median"] is None:
            return f"{pt} (bootstrap 불가: Normal−Full 간격 ≤ 0)"
        return f"{pt} (med {x['boot_median']:.0f} [{x['boot_ci95'][0]:.0f}, {x['boot_ci95'][1]:.0f}])"

    for sname in list(subsets.keys()) + ["Normal-amplified A- units"]:
        for g in groups:
            B = S["subsets"][sname].get(g)
            if not B:
                continue
            for m in (("amplification", "fde5", "downstream_err") if g == "A-" else ("fde5", "downstream_err", "recovery", "amplification")):
                c = B["critical_window"][m]
                dist = c["0.9"]["boot_dist"]
                ds = "–" if not dist else ", ".join(f"{k}:{100*v:.0f}%" for k, v in dist.items() if v >= 0.02)
                w(f"| {sname} (n={B['n']}) | {g} | {m} | {cell(c['0.5'])} | {cell(c['0.8'])} | {cell(c['0.9'])} | {ds} |")
    w("")

    w("## 5. 첫 mismatch 위치(t*)별\n")
    w("![fig2](figures/fig2_fraction_by_tstar.png)\n")
    w("window 길이는 horizon에 의해 잘립니다: t*에서 교정 가능한 context 위치는 t*+1..8 (최대 8−t*개).\n")
    for g in groups:
        w(f"### {g}\n")
        w("| t* | n | Normal amp | win1 | win2 | win3 | win4 | Full | Normal FDE | win1 | win2 | win3 | win4 | Full |")
        w("|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|")
        for sname in ("t*=0", "t*=1", "t*>=2"):
            B = S["subsets"][sname][g]
            if not B:
                continue
            R_ = B["rows"]
            cols = ["normal", "win1", "win2", "win3", "win4", "gt_history"]
            w(f"| {sname} | {B['n']} | " + " | ".join(pc(R_[c]["level"]["amplification"]) for c in cols) + " | "
              + " | ".join(nm_(R_[c]["level"]["fde5"]) for c in cols) + " |")
        w("")
        w("Full 효과 대비 비율 (amplification / FDE):\n")
        w("| t* | win1 | win2 | win3 | win4 |")
        w("|---|---|---|---|---|")
        for sname in ("all perturbations", "t*=0", "t*=1", "t*>=2"):
            B = S["subsets"][sname][g]
            if not B:
                continue
            w(f"| {sname} | " + " | ".join(f"{frs(B['fraction']['amplification'][c])} / {frs(B['fraction']['fde5'][c])}" for c in ("win1", "win2", "win3", "win4")) + " |")
        w("")

    w("## 6. Window 해제 후 자기 유지 여부\n")
    w("![fig3](figures/fig3_self_sustain_entropy.png)\n")
    w("post-release GT match = window가 끝난 뒤(context에 window 이후 자기 생성 토큰이 들어간 step) 실행 토큰의 GT 일치율. "
      "self-sustain gap = 같은 step에서 GT-history(교정을 계속한 경우)와의 차이 (0이면 교정을 멈춰도 계속 교정한 것과 같음).\n")
    for g in groups:
        B = A[g]["rows"]
        w(f"### {g}\n")
        w("| window | post-release GT match | GT-history 대비 gap | post-release entropy |")
        w("|---|---|---|---|")
        for nm in ["normal"] + WINS[:6] + [n for n in names if n.startswith("delay")]:
            L_ = B[nm]["level"]
            w(f"| {nm} | {pc(L_['post_release_realign'])} | {pc(L_['self_sustain_gap'])} | {nm_(L_['post_release_entropy'], 3)} |")
        w("")

    w("## 7. 교정 시점: 지연 window와 첫 교정 생략\n")
    w("같은 길이의 교정을 늦게 시작하면(t*+2, t*+3, t*+4) 효과가 유지되는지, 첫 교정만 빼고 나머지를 모두 교정하면(skipfirst) 어떤지.\n")
    for g in groups:
        B = A[g]["rows"]
        w(f"### {g}\n")
        w("| 조건 | 교정 context 위치 | amplification | FDE | Δamp vs 같은 길이 win (시작 t*+1) | ΔFDE vs 같은 길이 win |")
        w("|---|---|---|---|---|---|")
        us = [u for u in units if u["group"] == g]
        bt = Boot([u["log"] for u in us], seed=11)
        for nm in [n for n in names if n.startswith("delay") or n.startswith("skip")] + ["win1", "win2", "gt_history"]:
            s = spec[nm]
            ref = f"win{s['length']}" if s["length"] < 99 else "gt_history"
            pos = f"t*+{s['start']}…" + ("end" if s["length"] >= 99 else f"t*+{s['start']+s['length']-1}")
            cells = []
            for m in ("amplification", "fde5"):
                if ref == nm:
                    cells.append("(기준)"); continue
                pairs = [(u["m"][nm][m], u["m"][ref][m]) for u in us]
                c = bt([float(a) - float(b) for a, b in pairs])
                if m in BINARY:
                    g_ = sum(bool(a) and not bool(b) for a, b in pairs); l_ = sum(bool(b) and not bool(a) for a, b in pairs)
                    c["mcnemar"] = {"p": 1.0 if g_ + l_ == 0 else float(binomtest(g_, g_ + l_, 0.5).pvalue)}
                else:
                    dd = np.array([float(a) - float(b) for a, b in pairs]); c["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                cells.append(dl(c, m))
            L_ = B[nm]["level"]
            w(f"| {nm} | {pos} | {pc(L_['amplification'])} | {nm_(L_['fde5'])} | {cells[0]} | {cells[1]} |")
        w("")

    w("## 8. A− vs A+ 차이\n")
    w("[조건 − Normal](A−) − [조건 − Normal](A+), log 공동 재표본 bootstrap.\n")
    w("| 조건 | Δamplification 차이 | Δrecovery 차이 | ΔFDE 차이 (m) | Δdownstream error 차이 | Δre-alignment 차이 |")
    w("|---|---|---|---|---|---|")
    for nm in WINS[:6] + ["gt_history"]:
        I = inter[nm]
        f1 = lambda d: f"{100*d['gap']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]"  # noqa: E731
        w(f"| {nm} | {f1(I['amplification'])} | {f1(I['recovery'])} | {I['fde5']['gap']:+.2f} [{I['fde5']['ci95'][0]:+.2f}, {I['fde5']['ci95'][1]:+.2f}] | "
          f"{f1(I['downstream_err'])} | {f1(I['realign'])} |")
    w("")

    w("## 9. 부분집합 (original token only / Normal에서 증폭된 A− 단위)\n")
    for sname in ("original token only", "Normal-amplified A- units"):
        for g in groups:
            B = S["subsets"][sname].get(g)
            if not B:
                continue
            w(f"### {sname} — {g} (n={B['n']}, 장면 {B['n_scenes']})\n")
            w("| 조건 | amplification | recovery | FDE (m) | re-align | Full 효과 대비 (amp) | Full 효과 대비 (FDE) |")
            w("|---|---:|---:|---:|---:|---:|---:|")
            for nm in ["normal"] + WINS[:6] + ["gt_history"]:
                L_ = B["rows"][nm]["level"]
                w(f"| {nm} | {pc(L_['amplification'])} | {pc(L_['recovery'])} | {nm_(L_['fde5'])} | {pc(L_['realign'])} | "
                  f"{frs(B['fraction']['amplification'].get(nm))} | {frs(B['fraction']['fde5'].get(nm))} |")
            w("")

    w("## 10. Sanity checks\n")
    w("| 검사 | 결과 |")
    w("|---|---|")
    lab = {"prefix_identical": "첫 mismatch(t* 포함)까지 모든 조건 토큰 동일",
           "first_free_step_identical": "a_{t*+1}은 모든 조건에서 동일 (context가 perturbation뿐이므로 교정 불가)",
           "win_saturated_eq_gt_history": "w ≥ 8−t*인 window = GT-history (토큰 완전 일치)",
           "gt_context_count_ok": "마지막 step에서 GT로 교정된 context 위치 수 = min(w, 8−t*)",
           "normal_no_gt_context": "Normal AR에는 GT context 없음",
           "reproduces_action_history_causal_normal": "이전 action_history Normal AR 토큰 재현율 (batch 수치차)",
           "reproduces_action_history_causal_gt_history": "이전 action_history GT-history 토큰 재현율"}
    for k, t_ in lab.items():
        v = S["sanity"].get(k)
        w(f"| {t_} | {'–' if v is None else f'{100*v:.1f}%'} |")
    w("")
    w("## 11. 해석과 한계\n"); w(NAR.get("interpretation", "") + "\n")
    open(os.path.join(args.run, "TEMPORAL_FEEDBACK_WINDOW.md"), "w").write("\n".join(W))
    print("\n".join(W))


if __name__ == "__main__":
    main()
