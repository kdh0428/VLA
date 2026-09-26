#!/usr/bin/env python
"""
Analysis of layer-wise previous-action state patching (CPU only).

Unit = (scene, perturbation). Metric definitions are imported unchanged from
analyze_action_history (amplification = A- AND FDE(5 s) > 3 m, recovery = A+ accept rule,
ADE/FDE 5 s, downstream token error, GT re-alignment, post-mismatch entropy, error growth).

Next-action distribution metrics (secondary) are taken at the FIRST EFFECTIVE PATCH STEP,
k = t*+2: it is the first step whose preceding token is self-generated, and the last step at
which every row still shares the same earlier context (so the distributions are comparable):
  js1 = JS(row || normal),  kl1 = KL(normal || row),  jsgt1 = JS(row || gt_history),
  pgt1 = P(GT action),  pnorm1 = P(action that Normal AR sampled).
Units with t* = 8 have no effective patch step (one post-mismatch step only); they are kept
in every rollout metric (all patch rows equal Normal there) and counted separately.

Contrasts: row - baseline, paired per unit (baseline = normal; for reverse@ rows baseline =
gt_history). McNemar (binary), Wilcoxon (continuous). 95% CI = cluster bootstrap over logs
(primary; logs contain several scenes, so this is the conservative cluster); scene-cluster CIs
are also reported for the headline amplification contrast.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(POC_DIR, "scripts"))
from analyze_action_history import unit_metrics, AMP_FDE  # noqa: E402  (unchanged metric definitions)

REPS = 2000
ROLL = ["amplification", "recovery", "ade5", "fde5", "downstream_err", "realign", "entropy", "err_slope"]
DIST = ["js1", "kl1", "jsgt1", "pgt1", "pnorm1", "ent1", "pgt_late", "ent_late", "jsgt_late"]
# *_late: mean over steps k >= t*+3 (the only steps at which layer choice can matter: at k = t*+2 the
# preceding token is the only position that differs between rows, so every full patch equals
# GT-history exactly there). Units with t* = 7 have no such step.
METRICS = ROLL + DIST
BINARY = {"amplification", "recovery"}
REF_PRIOR = {"normal": 0.474, "recent_gt": 0.071, "gt_history": 0.041}   # action_history_causal, A- all perturbations


class Boot:
    """Vectorised cluster bootstrap for one group of units."""

    def __init__(self, clusters, reps=REPS, seed=0):
        self.labels = sorted(set(clusters))
        self.cid = np.array([self.labels.index(c) for c in clusters])
        rng = np.random.default_rng(seed)
        self.R = rng.integers(0, len(self.labels), size=(reps, len(self.labels)))

    def __call__(self, vals):
        v = np.array([np.nan if x is None else float(x) for x in vals], float)
        ok = np.isfinite(v)
        if not ok.any():
            return None
        nc = len(self.labels)
        s = np.bincount(self.cid[ok], weights=v[ok], minlength=nc)
        n = np.bincount(self.cid[ok], minlength=nc).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = s[self.R].sum(1) / n[self.R].sum(1)
        bs = bs[np.isfinite(bs)]
        return {"mean": float(v[ok].mean()), "n": int(ok.sum()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def layer_order(lk):
    return -1 if lk == "emb" else int(lk[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(POC_DIR, "outputs/prev_action_state_patching"))
    ap.add_argument("--narrative", default=None, help="json with conclusion/mechanism/limitations text")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "run_meta.json")))
    ROWS = meta["rows"]
    names = [r["name"] for r in ROWS]
    base_of = {r["name"]: ("gt_history" if r["base"] == "gt" and r["name"] != "gt_history" else "normal") for r in ROWS}
    recs = [json.loads(l) for l in open(os.path.join(args.run, "records.jsonl"))]
    errs = [json.loads(l) for l in open(os.path.join(args.run, "errors.jsonl"))] if os.path.exists(os.path.join(args.run, "errors.jsonl")) else []

    # ------------------------------------------------------------------ per-unit metrics
    units = []
    for r in recs:
        t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": r["perturbation"], "t_star": t,
             "effective": t + 2 <= 9, "m": {}}
        for nm in names:
            x = r["rows"][nm]
            mm = unit_metrics(x["action_idx"], x["trajectory_pred"], x["ent"], gt, gtraj, t)
            j = 1 if u["effective"] else None
            for key, src in (("js1", "js_to_normal"), ("kl1", "kl_normal_to_row"), ("jsgt1", "js_to_gt_history"),
                             ("pgt1", "p_gt"), ("pnorm1", "p_normal_choice"), ("ent1", "ent")):
                mm[key] = x[src][j] if j is not None else None
            for key, src in (("pgt_late", "p_gt"), ("ent_late", "ent"), ("jsgt_late", "js_to_gt_history")):
                mm[key] = float(np.mean(x[src][2:])) if len(x[src]) > 2 else None
            u["m"][nm] = mm
        acts = {nm: r["rows"][nm]["action_idx"] for nm in names}
        u["eq"] = {
            "selfpatch_eq_normal": all(acts[n] == acts["normal"] for n in names if n.startswith("selfpatch@")),
            "patch_emb_eq_recent_gt": acts["patch_full@emb"] == acts["recent_gt"],
            "patch_L35_eq_gt_history": acts["patch_full@L35"] == acts["gt_history"],
            "identitysrc_eq_recent_gt": all(acts[n] == acts["recent_gt"] for n in names if n.startswith("identitysrc@")),
            "reverse_L35_eq_normal": acts.get("reverse@L35") == acts["normal"],
            "prefix_identical": r["check_prefix_identical_all_rows"],
            "patch_noop_at_first_step": all(acts[n][t + 1] == acts["normal"][t + 1] for n in names
                                            if names and ROWS[names.index(n)]["base"] == "normal"),
        }
        pri = r.get("prior_action_history") or {}
        u["repro"] = {c: (acts[c] == pri[c]) if c in pri else None for c in ("normal", "gt_history", "recent_gt")}
        units.append(u)

    groups = ("A-", "A+")
    S = {"n_units": {g: sum(u["group"] == g for u in units) for g in groups},
         "n_scenes": {g: len({u["token"] for u in units if u["group"] == g}) for g in groups},
         "n_units_with_effective_patch_step": {g: sum(u["group"] == g and u["effective"] for u in units) for g in groups},
         "n_units_with_late_steps": {g: sum(u["group"] == g and u["t_star"] <= 6 for u in units) for g in groups},
         "n_errors": len(errs), "errors": errs,
         "rows": ROWS, "amplification_fde_threshold_m": AMP_FDE}

    # ------------------------------------------------------------------ sanity
    san = {}
    for key in units[0]["eq"]:
        v = [u["eq"][key] for u in units]
        san[key] = {"rate": float(np.mean(v)), "n": len(v)}
    for c in ("normal", "gt_history", "recent_gt"):
        v = [u["repro"][c] for u in units if u["repro"][c] is not None]
        san[f"reproduces_action_history_causal_{c}"] = {"rate": float(np.mean(v)) if v else None, "n": len(v)}
    dbg_path = os.path.join(args.run, "sanity", "debug_checks.json")
    if os.path.exists(dbg_path):
        dbg = json.load(open(dbg_path))
        pc = [v for d in dbg for v in d["patch_checks"].values()]
        kv = [v for d in dbg for v in d["kv_logits"].values()]
        san["debug"] = {
            "n_samples": len(dbg),
            "max_abs_patched_minus_source_alpha1": max(v["max_abs_tgt_minus_src_alpha1"] or 0 for v in pc),
            "max_other_positions_changed": max(v["other_positions_changed"] for v in pc),
            "max_untouched_rows_changed": max(v["untouched_rows_changed"] for v in pc),
            "max_K_or_V_diff_at_or_before_patch_layer": max(max(v["max_K_diff_at_or_before_patch"] or 0, v["max_V_diff_at_or_before_patch"] or 0) for v in kv),
            "min_K_diff_after_patch_layer": min(v["max_K_diff_after_patch"] for v in kv if v["max_K_diff_after_patch"] is not None),
            "min_V_diff_after_patch_layer": min(v["max_V_diff_after_patch"] for v in kv if v["max_V_diff_after_patch"] is not None),
            "min_logits_L1_vs_normal": min(v["logits_L1_vs_normal"] for v in kv),
            "max_selfpatch_js": max(max(d["selfpatch_js_vs_normal"].values()) for d in dbg),
            "max_emb_patch_vs_recent_gt_logits_L1": max(d["emb_patch_vs_recent_gt_logits_L1"] for d in dbg),
            "max_L35_patch_vs_gt_history_logits_L1": max(d["L35_patch_vs_gt_history_logits_L1"] for d in dbg),
            "per_sample": dbg}
    S["sanity"] = san

    # ------------------------------------------------------------------ stats
    def block(us, cluster="log", seed=0):
        if not us:
            return None
        bt = Boot([u[cluster] for u in us], seed=seed)
        out = {"n": len(us), "n_scenes": len({u["token"] for u in us}), "rows": {}}
        for nm in names:
            base = base_of[nm]
            d = {"level": {}, "vs_base": {}, "baseline": base}
            for m in METRICS:
                d["level"][m] = bt([u["m"][nm][m] for u in us])
                if nm == base:
                    continue
                pairs = [(u["m"][nm][m], u["m"][base][m]) for u in us]
                diffs = [None if a is None or b is None else float(a) - float(b) for a, b in pairs]
                c = bt(diffs)
                if c is None:
                    d["vs_base"][m] = None
                    continue
                ok = [(a, b) for a, b in pairs if a is not None and b is not None]
                if m in BINARY:
                    gain = sum(bool(a) and not bool(b) for a, b in ok)
                    loss = sum(bool(b) and not bool(a) for a, b in ok)
                    c["mcnemar"] = {"row_only": gain, "baseline_only": loss,
                                    "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)}
                else:
                    dd = np.array([float(a) - float(b) for a, b in ok])
                    c["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                d["vs_base"][m] = c
            out["rows"][nm] = d
        # fraction of the Normal -> GT-history amplification gap closed (paired, bootstrap)
        amp = {nm: np.array([float(u["m"][nm]["amplification"]) for u in us]) for nm in names}
        gap = amp["normal"] - amp["gt_history"]
        frac = {}
        for nm in names:
            if not (nm.startswith("patch_") or nm == "recent_gt"):
                continue
            num = amp["normal"] - amp[nm]
            nc = len(bt.labels)
            sn = np.bincount(bt.cid, weights=num, minlength=nc); sg = np.bincount(bt.cid, weights=gap, minlength=nc)
            with np.errstate(invalid="ignore", divide="ignore"):
                bs = sn[bt.R].sum(1) / sg[bt.R].sum(1)
            bs = bs[np.isfinite(bs)]
            frac[nm] = {"mean": float(num.sum() / gap.sum()) if gap.sum() else None,
                        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None}
        out["fraction_of_gt_history_effect"] = frac
        return out

    S["groups"] = {}
    for g in groups:
        us = [u for u in units if u["group"] == g]
        S["groups"][g] = {"all": block(us), "effective_patch_step_only": block([u for u in us if u["effective"]])}
        sc = block(us, cluster="token", seed=1)
        S["groups"][g]["scene_cluster_amplification"] = {nm: {"level": sc["rows"][nm]["level"]["amplification"],
                                                               "vs_base": sc["rows"][nm]["vs_base"].get("amplification")}
                                                          for nm in names}
    # previously amplified subset (A- units whose Normal AR amplified in this run)
    ua = [u for u in units if u["group"] == "A-" and u["m"]["normal"]["amplification"]]
    S["groups"]["A-"]["normal_amplified_subset"] = block(ua)

    # ------------------------------------------------------------------ layer_results.json
    fams = defaultdict(dict)
    for nm in names:
        if "@" not in nm:
            continue
        fam, lk = nm.split("@")
        fams[fam][lk] = nm
    LR = {"layer_keys_note": "emb = input to block 0; L{i} = output of decoder block i",
          "references": {g: {c: S["groups"][g]["all"]["rows"][c]["level"] for c in ("normal", "recent_gt", "gt_history")} for g in groups},
          "families": {}}
    for fam, d in fams.items():
        LR["families"][fam] = {}
        for g in groups:
            B = S["groups"][g]["all"]
            LR["families"][fam][g] = [{"layer": lk, "row": nm,
                                       "level": {m: B["rows"][nm]["level"][m] for m in METRICS},
                                       "vs_baseline": B["rows"][nm]["vs_base"],
                                       "fraction_of_gt_history_effect": B["fraction_of_gt_history_effect"].get(nm)}
                                      for lk, nm in sorted(d.items(), key=lambda kv: layer_order(kv[0]))]
    json.dump(LR, open(os.path.join(args.run, "layer_results.json"), "w"), indent=1)
    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(args.run, "units.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(args.run, "figures")
    os.makedirs(fd, exist_ok=True)
    full = sorted(fams["patch_full"].items(), key=lambda kv: layer_order(kv[0]))
    xs = [layer_order(lk) for lk, _ in full]
    xt = [-1, 0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    xtl = ["emb" if x == -1 else str(x) for x in xt]

    def series(g, rows_, m, scale=1.0):
        B = S["groups"][g]["all"]["rows"]
        mu = [B[nm]["level"][m]["mean"] * scale for _, nm in rows_]
        lo = [B[nm]["level"][m]["ci95"][0] * scale for _, nm in rows_]
        hi = [B[nm]["level"][m]["ci95"][1] * scale for _, nm in rows_]
        return mu, lo, hi

    def refs(ax, g, m, scale, prior=False):
        B = S["groups"][g]["all"]["rows"]
        for c, col, lab in (("normal", "tab:red", "Normal AR"), ("recent_gt", "tab:green", "Recent-GT"), ("gt_history", "tab:blue", "GT-history")):
            v = B[c]["level"][m]["mean"] * scale
            ax.axhline(v, color=col, ls="--", lw=1, label=f"{lab} (this run) {v:.1f}")
            if prior:
                ax.axhline(REF_PRIOR[c] * 100, color=col, ls=":", lw=1, alpha=.6, label=f"{lab} (action_history) {REF_PRIOR[c]*100:.1f}")

    # Fig 1
    fig, ax = plt.subplots(figsize=(10, 5))
    mu, lo, hi = series("A-", full, "amplification", 100)
    ax.plot(xs, mu, "k-o", ms=3, label="full replacement (GT-history source)")
    ax.fill_between(xs, lo, hi, color="k", alpha=.12)
    for a, col in (("0.25", "tab:purple"), ("0.5", "tab:orange"), ("0.75", "tab:olive")):
        fam = fams.get(f"patch_delta{a}")
        if fam:
            rr = sorted(fam.items(), key=lambda kv: layer_order(kv[0]))
            ax.plot([layer_order(k) for k, _ in rr], series("A-", rr, "amplification", 100)[0], "-s", color=col, ms=3, lw=1, label=f"delta α={a}")
    if "reverse" in fams:
        rr = sorted(fams["reverse"].items(), key=lambda kv: layer_order(kv[0]))
        ax.plot([layer_order(k) for k, _ in rr], series("A-", rr, "amplification", 100)[0], "-^", color="tab:gray", ms=3, lw=1,
                label="reverse: GT-history ← Normal state")
    refs(ax, "A-", "amplification", 100, prior=True)
    ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_xlabel("patched layer (previous-action position)")
    ax.set_ylabel("A− amplification (%)"); ax.set_title(f"A− amplification vs patched layer (n={S['n_units']['A-']} units, 95% log-cluster CI)")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(fd, "fig1_amplification_vs_layer.png"), dpi=150); plt.close(fig)

    # Fig 2
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, g in zip(axs, groups):
        mu, lo, hi = series(g, full, "fde5")
        ax.plot(xs, mu, "k-o", ms=3, label="full replacement"); ax.fill_between(xs, lo, hi, color="k", alpha=.12)
        if "reverse" in fams:
            rr = sorted(fams["reverse"].items(), key=lambda kv: layer_order(kv[0]))
            ax.plot([layer_order(k) for k, _ in rr], series(g, rr, "fde5")[0], "-^", color="tab:gray", ms=3, lw=1, label="reverse")
        refs(ax, g, "fde5", 1.0)
        ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_xlabel("patched layer"); ax.set_ylabel("FDE 5 s (m)")
        ax.set_title(f"{g}: FDE vs layer"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig2_fde_vs_layer.png"), dpi=150); plt.close(fig)

    # Fig 3
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for g, ls in zip(groups, ("-", "--")):
        mu, lo, hi = series(g, full, "entropy")
        axs[0].plot(xs, mu, ls, marker="o", ms=3, label=f"{g} full replacement")
        axs[0].fill_between(xs, lo, hi, alpha=.1)
        for c, col in (("normal", "tab:red"), ("gt_history", "tab:blue"), ("recent_gt", "tab:green")):
            axs[0].axhline(S["groups"][g]["all"]["rows"][c]["level"]["entropy"]["mean"], color=col, ls=ls, lw=.8,
                           label=f"{g} {c}")
        mu, lo, hi = series(g, full, "jsgt_late")
        axs[1].plot(xs, mu, ls, marker="o", ms=3, label=f"{g} JS(row‖GT-history), k≥t*+3")
        axs[1].fill_between(xs, lo, hi, alpha=.1)
        for c, col in (("normal", "tab:red"), ("recent_gt", "tab:green")):
            axs[1].axhline(S["groups"][g]["all"]["rows"][c]["level"]["jsgt_late"]["mean"], color=col, ls=ls, lw=.8, label=f"{g} {c}")
    axs[0].set_title("post-mismatch mean entropy (nats)")
    axs[1].set_title("next-action JS to GT-history, steps k≥t*+3\n(at k=t*+2 every layer patch equals GT-history exactly)", fontsize=9)
    for ax in axs:
        ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_xlabel("patched layer"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(fd, "fig3_entropy_vs_layer.png"), dpi=150); plt.close(fig)

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
        if m in ("downstream_err", "realign", "pgt1", "pnorm1", "pgt_late"):
            return f"{100*d['mean']:+.1f}%p [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] p={d['wilcoxon_p']:.2g}"
        nd = 3 if m in ("entropy", "err_slope", "js1", "kl1", "jsgt1", "ent1", "ent_late", "jsgt_late") else 2
        return f"{d['mean']:+.{nd}f} [{d['ci95'][0]:+.{nd}f}, {d['ci95'][1]:+.{nd}f}] p={d['wilcoxon_p']:.2g}"

    NAR = json.load(open(args.narrative)) if args.narrative and os.path.exists(args.narrative) else {}
    W = []
    w = W.append
    w("# AutoVLA Natural/Fast — Layer-wise Previous-Action State Patching\n")
    w("## 1. 결론\n")
    w(NAR.get("conclusion", "_(narrative 미작성)_") + "\n")
    w("## 2. 실험 설정\n")
    w(NAR.get("setup", "") + "\n")
    w(f"- 단위: 장면 A− {S['n_scenes']['A-']} / A+ {S['n_scenes']['A+']}, (장면, perturbation) 단위 A− {S['n_units']['A-']} / A+ {S['n_units']['A+']} "
      f"(equal-distance/action-history와 동일, 새 perturbation 없음). 실패 unit {S['n_errors']}개.")
    w(f"- 실제 patch가 작동하는 step(k = t*+2)이 존재하는 단위: A− {S['n_units_with_effective_patch_step']['A-']} / A+ {S['n_units_with_effective_patch_step']['A+']} (전부; t* ≤ 7). "
      f"layer 선택이 결과를 바꿀 수 있는 step(k ≥ t*+3)이 존재하는 단위: A− {S['n_units_with_late_steps']['A-']} / A+ {S['n_units_with_late_steps']['A+']} (t* = 7 제외; rollout 지표에는 모두 포함).")
    w("- CI: log 단위 cluster bootstrap 95% (2000회). p: McNemar(이진) / Wilcoxon(연속). 기준선: Normal AR (reverse 행은 GT-history).\n")

    w("## 3. 주요 결과\n")
    key_rows = ["normal", "recent_gt", "gt_history"] + [fams["patch_full"][k] for k in ("emb", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L35") if k in fams["patch_full"]]
    for g in groups:
        B = S["groups"][g]["all"]
        w(f"### {g} (단위 {B['n']}, 장면 {B['n_scenes']})\n")
        w("| 조건 | amplification | recovery | FDE (m) | GT re-alignment | error growth (m/step) | post-mismatch entropy | GT-history 효과 대비 비율 |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|")
        for nm in key_rows:
            L_ = B["rows"][nm]["level"]; fr = B["fraction_of_gt_history_effect"].get(nm)
            frs = "–" if not fr or fr["mean"] is None else f"{100*fr['mean']:.0f}% [{100*fr['ci95'][0]:.0f}, {100*fr['ci95'][1]:.0f}]"
            w(f"| {nm} | {pc(L_['amplification'])} | {pc(L_['recovery'])} | {nm_(L_['fde5'])} | {pc(L_['realign'])} | {nm_(L_['err_slope'], 3)} | {nm_(L_['entropy'], 3)} | {frs} |")
        w("")
        w("조건 − Normal AR (쌍대):\n")
        w("| 조건 | Δamplification | Δrecovery | ΔFDE (m) | Δre-alignment | Δerror growth | Δentropy |")
        w("|---|---|---|---|---|---|---|")
        for nm in key_rows[1:]:
            v = B["rows"][nm]["vs_base"]
            w(f"| {nm} | {dl(v['amplification'], 'amplification')} | {dl(v['recovery'], 'recovery')} | {dl(v['fde5'], 'fde5')} | "
              f"{dl(v['realign'], 'realign')} | {dl(v['err_slope'], 'err_slope')} | {dl(v['entropy'], 'entropy')} |")
        w("")

    w("## 4. Layer별 효과\n")
    w("![fig1](figures/fig1_amplification_vs_layer.png)\n\n![fig2](figures/fig2_fde_vs_layer.png)\n\n![fig3](figures/fig3_entropy_vs_layer.png)\n")
    for g in groups:
        B = S["groups"][g]["all"]["rows"]
        w(f"### {g}: full replacement 전체 layer (+ 첫 유효 step의 next-action 분포)\n")
        w("k = t*+2 열: JS(row‖Normal), JS(row‖GT-hist), P(GT), P(Normal action) — 모든 layer에서 GT-history와 동일(구조적). "
          "k ≥ t*+3 열: layer 선택이 영향을 주는 구간의 평균.\n")
        w("| layer | amplification | recovery | FDE (m) | re-align | entropy | JS‖Normal @t*+2 | JS‖GT-hist @t*+2 | P(GT) @t*+2 | P(Normal act) @t*+2 | P(GT) k≥t*+3 | entropy k≥t*+3 | JS‖GT-hist k≥t*+3 |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for nm in ["normal", "recent_gt", "gt_history"] + [n for _, n in full]:
            L_ = B[nm]["level"]
            w(f"| {nm} | {pc(L_['amplification'])} | {pc(L_['recovery'])} | {nm_(L_['fde5'])} | {pc(L_['realign'])} | {nm_(L_['entropy'], 3)} | "
              f"{nm_(L_['js1'], 3)} | {nm_(L_['jsgt1'], 3)} | {pc(L_['pgt1'])} | {pc(L_['pnorm1'])} | {pc(L_['pgt_late'])} | {nm_(L_['ent_late'], 3)} | {nm_(L_['jsgt_late'], 3)} |")
        w("")
    # transition: fraction of the Recent-GT -> GT-history gap reached by patch@layer
    w("### 전이 구간: Recent-GT(=patch@emb) → GT-history(=patch@L35) 간격 중 patch@layer가 도달한 비율\n")
    w("f(l) = (m(patch@l) − m(Recent-GT)) / (m(GT-history) − m(Recent-GT)). 두 끝점은 구조적으로 0과 1. "
      "간격이 작은 지표(amplification 3%p 등)는 잡음이 커서 연속 지표(re-alignment, entropy, P(GT) k≥t*+3)를 우선 봅니다.\n")
    trans = {}
    for g in groups:
        B = S["groups"][g]["all"]["rows"]
        trans[g] = {}
        w(f"**{g}**\n")
        tm = ["realign", "entropy", "pgt_late", "ent_late", "jsgt_late", "fde5", "amplification" if g == "A-" else "recovery"]
        w("| layer | " + " | ".join(tm) + " |")
        w("|---|" + "---:|" * len(tm))
        for lk, nm in full:
            cells = []
            for m in tm:
                a, b, v = B["recent_gt"]["level"][m], B["gt_history"]["level"][m], B[nm]["level"][m]
                f = None if not (a and b and v) or abs(b["mean"] - a["mean"]) < 1e-9 else (v["mean"] - a["mean"]) / (b["mean"] - a["mean"])
                trans[g].setdefault(m, {})[lk] = f
                cells.append("–" if f is None else f"{f:.2f}")
            w(f"| {lk} | " + " | ".join(cells) + " |")
        w("")
        half = {m: next((lk for lk, _ in full if trans[g][m].get(lk) is not None and trans[g][m][lk] >= 0.5), None) for m in tm}
        w(f"50% 도달 layer ({g}): " + ", ".join(f"{m} → {h}" for m, h in half.items()) + "\n")
    S["transition_fraction"] = trans
    json.dump(S, open(os.path.join(args.run, "summary.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    LR["transition_fraction_recent_gt_to_gt_history"] = trans
    json.dump(LR, open(os.path.join(args.run, "layer_results.json"), "w"), indent=1)
    w("### Delta patch h' = h + α(h_GT − h) (대표 layer)\n")
    for g in groups:
        B = S["groups"][g]["all"]["rows"]
        w(f"**{g}** — amplification (A−) / recovery (A+), FDE\n")
        alph = sorted({re.match(r"patch_delta([\d.]+)@", n).group(1) for n in names if n.startswith("patch_delta")}, key=float)
        w("| layer | " + " | ".join(f"α={a}" for a in alph) + " | α=1.0 |")
        w("|---|" + "---|" * (len(alph) + 1))
        met = "amplification" if g == "A-" else "recovery"
        for lk in sorted(fams.get(f"patch_delta{alph[0]}", {}), key=layer_order):
            cells = [f"{pc(B[f'patch_delta{a}@{lk}']['level'][met])}, FDE {B[f'patch_delta{a}@{lk}']['level']['fde5']['mean']:.2f}" for a in alph]
            cells.append(f"{pc(B[f'patch_full@{lk}']['level'][met])}, FDE {B[f'patch_full@{lk}']['level']['fde5']['mean']:.2f}")
            w(f"| {lk} | " + " | ".join(cells) + " |")
        w("")
    if "reverse" in fams:
        w("### Reverse patch (GT-history rollout의 직전-action 위치에 Normal state 삽입) — 충분성 검사\n")
        w("| layer | A− amplification | A− Δ vs GT-history | A+ recovery | A+ Δ vs GT-history | A− FDE |")
        w("|---|---|---|---|---|---|")
        for lk, nm in sorted(fams["reverse"].items(), key=lambda kv: layer_order(kv[0])):
            a, p = S["groups"]["A-"]["all"]["rows"][nm], S["groups"]["A+"]["all"]["rows"][nm]
            w(f"| {lk} | {pc(a['level']['amplification'])} | {dl(a['vs_base']['amplification'], 'amplification')} | {pc(p['level']['recovery'])} | "
              f"{dl(p['vs_base']['recovery'], 'recovery')} | {nm_(a['level']['fde5'])} |")
        w("")

    w("## 5. 통계 검정\n")
    w("Full replacement 각 layer − Normal AR (쌍대, 전체 단위). A− amplification은 scene-cluster CI도 함께 표기.\n")
    for g in groups:
        B = S["groups"][g]["all"]["rows"]; SC = S["groups"][g]["scene_cluster_amplification"]
        w(f"### {g}\n")
        w("| layer | Δamplification (log CI) | Δamplification scene CI | Δrecovery | ΔFDE | Δre-alignment | Δerror growth | Δentropy | ΔP(GT) k≥t*+3 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for nm in ["recent_gt", "gt_history"] + [n for _, n in full]:
            v = B[nm]["vs_base"]; s = SC[nm]["vs_base"]
            sci = "–" if not s else f"[{100*s['ci95'][0]:+.1f}, {100*s['ci95'][1]:+.1f}]"
            w(f"| {nm} | {dl(v['amplification'], 'amplification')} | {sci} | {dl(v['recovery'], 'recovery')} | {dl(v['fde5'], 'fde5')} | "
              f"{dl(v['realign'], 'realign')} | {dl(v['err_slope'], 'err_slope')} | {dl(v['entropy'], 'entropy')} | {dl(v['pgt_late'], 'pgt_late')} |")
        w("")
    # interior layers vs the Recent-GT endpoint (= patch@emb): does anything beyond token identity help?
    w("### 내부 layer − Recent-GT (쌍대; patch@emb와 동일한 기준선)\n")
    w("잔차 patch가 토큰 identity(Recent-GT) 이상을 전달하는지 검정합니다. 차이는 unit별 `patch@l − recent_gt`.\n")
    for g in groups:
        us = [u for u in units if u["group"] == g]
        bt = Boot([u["log"] for u in us], seed=5)
        w(f"**{g}**\n")
        w("| layer | Δamplification | Δrecovery | ΔFDE (m) | Δre-alignment | Δentropy |")
        w("|---|---|---|---|---|---|")
        for lk in ("L4", "L12", "L16", "L20", "L24", "L28", "L31", "L35"):
            nm = f"patch_full@{lk}"; cells = []
            for m in ("amplification", "recovery", "fde5", "realign", "entropy"):
                pairs = [(u["m"][nm][m], u["m"]["recent_gt"][m]) for u in us if u["m"][nm][m] is not None and u["m"]["recent_gt"][m] is not None]
                c = bt([float(a) - float(b) for a, b in pairs]) if pairs else None
                if c and m in BINARY:
                    gn = sum(bool(a) and not bool(b) for a, b in pairs); ls_ = sum(bool(b) and not bool(a) for a, b in pairs)
                    c["mcnemar"] = {"p": 1.0 if gn + ls_ == 0 else float(binomtest(gn, gn + ls_, 0.5).pvalue)}
                elif c:
                    dd = np.array([float(a) - float(b) for a, b in pairs]); c["wilcoxon_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                cells.append(dl(c, m))
            w(f"| {lk} | " + " | ".join(cells) + " |")
        w("")
    ua_b = S["groups"]["A-"]["normal_amplified_subset"]
    if ua_b:
        w(f"### A− 중 이 run의 Normal AR이 증폭된 단위 (n={ua_b['n']}, 장면 {ua_b['n_scenes']})\n")
        w("| 조건 | amplification | FDE (m) |")
        w("|---|---:|---:|")
        for nm in ["normal", "recent_gt", "gt_history"] + [n for _, n in full]:
            L_ = ua_b["rows"][nm]["level"]
            w(f"| {nm} | {pc(L_['amplification'])} | {nm_(L_['fde5'])} |")
        w("")

    w("## 6. Sanity checks\n")
    w("| 검사 | 결과 |")
    w("|---|---|")
    lab = {"selfpatch_eq_normal": "(1) normal→normal patch(emb/L16/L35) 행동 토큰 = Normal AR",
           "prefix_identical": "(5) 첫 mismatch까지(t* 포함) 모든 조건의 궤적 토큰 동일",
           "patch_noop_at_first_step": "시점 검증: k=t*+1(직전 토큰=perturbation, 모든 조건 동일)에서 patch 행 = Normal",
           "patch_emb_eq_recent_gt": "구조적 동치: patch@emb 행동 = Recent-GT",
           "patch_L35_eq_gt_history": "구조적 동치: patch@L35 행동 = GT-history",
           "identitysrc_eq_recent_gt": "구조적 동치: source=Recent-GT patch(모든 layer) = Recent-GT",
           "reverse_L35_eq_normal": "reverse@L35 = Normal AR (역방향 끝점)",
           "reproduces_action_history_causal_normal": "이전 action_history 실험 Normal AR 토큰 재현율 (batch 93 vs 1 수치차)",
           "reproduces_action_history_causal_gt_history": "이전 GT-history 토큰 재현율",
           "reproduces_action_history_causal_recent_gt": "이전 Recent-GT 토큰 재현율"}
    for k, t_ in lab.items():
        if k in san and san[k]["rate"] is not None:
            w(f"| {t_} | {100*san[k]['rate']:.1f}% (n={san[k]['n']}) |")
    if "debug" in san:
        d = san["debug"]
        w(f"| (2) patched activation = GT activation: max|h_patched − h_source| (α=1, 모든 patch layer) | {d['max_abs_patched_minus_source_alpha1']:.3g} (cosine=1) |")
        w(f"| (3) 다른 위치 덮어쓰기: patch 대상 외 위치 최대 변화 / patch 대상 외 row 최대 변화 | {d['max_other_positions_changed']:.3g} / {d['max_untouched_rows_changed']:.3g} |")
        w(f"| KV: patch layer 이하 K/V 차이 (0이어야 함) | {d['max_K_or_V_diff_at_or_before_patch_layer']:.3g} |")
        w(f"| KV: patch layer 이후 K / V 차이 (0보다 커야 함; 최소값) | {d['min_K_diff_after_patch_layer']:.3g} / {d['min_V_diff_after_patch_layer']:.3g} |")
        w(f"| logits: Normal 대비 L1 차이 (최소값) | {d['min_logits_L1_vs_normal']:.3g} |")
        w(f"| self-patch JS vs Normal (최대) | {d['max_selfpatch_js']:.3g} |")
        w(f"| logits L1: patch@emb vs Recent-GT / patch@L35 vs GT-history | {d['max_emb_patch_vs_recent_gt_logits_L1']:.3g} / {d['max_L35_patch_vs_gt_history_logits_L1']:.3g} |")
        w(f"\n디버그 샘플 {d['n_samples']}개 (k = t*+2, perturbation=original)에서 측정. 상세: `sanity/debug_checks.json`.\n")
    w("## 7. Mechanistic interpretation\n")
    w(NAR.get("mechanism", "") + "\n")
    w("## 8. 대안 설명과 한계\n")
    w(NAR.get("limitations", "") + "\n")
    open(os.path.join(args.run, "PREV_ACTION_STATE_PATCHING.md"), "w").write("\n".join(W))
    print("\n".join(W))


if __name__ == "__main__":
    main()
