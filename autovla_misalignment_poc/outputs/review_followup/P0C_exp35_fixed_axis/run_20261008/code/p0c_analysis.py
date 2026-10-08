#!/usr/bin/env python
"""P0-C: experiment 35 fixed-axis paired reanalysis (post-hoc; see ../protocol.md). CPU only, no inference.

  nice -n 19 python p0c_analysis.py <run_dir>
Reads (read-only) exp-35 rollouts/ and rollouts_ext/ episodes.jsonl (streamed), token_table.npz (export_token_table.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

SRC = "/root/VLA/autovla_misalignment_poc/outputs/spatialvla_feedback_protection_ablation"
FILES = [os.path.join(SRC, "rollouts", "episodes.jsonl"), os.path.join(SRC, "rollouts_ext", "episodes.jsonl")]
CFGS = ["r1e4", "r1e2", "r1e1", "r2e2", "r2e1", "r4e1"]
MODES = ["natural", "feedback", "corrected", "reverse"]
TASKS = ["google_robot_move_near", "google_robot_pick_coke_can"]
WIN = (8, 24); T = 80; TAU, HOLD = 0.02, 4; D = 0.3; TEMP = -0.8
B_REPS, PERM_REPS = 2000, 10000
DID = [("e1", "r1e1", "r2e1"), ("e1", "r1e1", "r4e1"), ("e1", "r2e1", "r4e1"),
       ("r1", "r1e1", "r1e2"), ("r1", "r1e1", "r1e4"), ("r1", "r1e2", "r1e4"),
       ("r2", "r2e1", "r2e2")]
CONTRASTS = {"R-N": ("reverse", "natural"), "F-C": ("feedback", "corrected")}
EXTRA_PAIRS = {"C-N": ("corrected", "natural"), "F-N": ("feedback", "natural"), "gen-N": ("natural_gen", "natural")}
TRAJ = ["mean_dev_cm", "final_dev_cm", "max_dev_cm", "area_dev_cmstep", "post_area_dev_cmstep", "cmd_final_m", "cmd_max_m",
        "adiff_win", "adiff_all", "realigned", "realign_t80", "sticky_onsets_win_diff", "sticky_steps_diff"]
SIGNED = {"sticky_onsets_win_diff", "sticky_steps_diff"}
PRIMARY = {"success", "area_dev_cmstep"}

run = sys.argv[1]
TT = np.load(os.path.join(run, "token_table.npz"))
ID2ROW = {int(t): j for j, t in enumerate(TT["ids"])}
NORM, RAW, PHYS = TT["norm"], TT["raw"], TT["phys"]


def perturb(g):  # exact re-implementation of svla_rollouts.perturb_token(T, g, 0.3, "opposite")
    v = NORM[ID2ROW[g]]; vh = v / max(np.linalg.norm(v), 1e-6)
    tgt = v + D * (-vh)
    j = int(np.argmin(np.linalg.norm(NORM - tgt, axis=1)))
    return int(TT["ids"][j]), tgt


def ens_w(n):
    w = np.exp(-TEMP * np.arange(n)); return w / w.sum()


def failure_type(ep):
    fi, ev = ep["final_info"], ep["ever"]
    if ep["success"]:
        return "success"
    if "lifted_object" in fi:
        if not ev.get("is_grasped"):
            return "never_grasped"
        return "dropped_or_lost_grasp" if not fi.get("is_grasped") else "grasped_not_lifted"
    if fi.get("moved_wrong_obj"):
        return "moved_wrong_obj"
    if not fi.get("all_obj_keep_height", True):
        return "object_lifted_or_fell"
    return "target_not_moved" if not fi.get("moved_correct_obj") else "not_near_target"


def sticky_onsets(a6):
    on = np.abs(a6) > 0.5; ons = []; run_ = 0
    for t in range(len(a6)):
        if on[t]:
            if run_ % 10 == 0:
                ons.append(t)
            run_ += 1
        else:
            run_ = 0
    return ons, on


def compact(ep):
    rec = ep["rec"]; mode = ep["cond"].split(":")[1]
    pos = np.array([r["pose"][:3] for r in rec], dtype=np.float64)
    act = np.array([r["action"] for r in rec], dtype=np.float64)
    pert_gen = set(); chunks = []; ens_err = 0.0
    for r in rec:
        t = r["t"]; ages = r["ages"]; w = ens_w(len(ages)); preds = np.array(r["preds"])
        ens_err = max(ens_err, float(np.abs((w[:, None] * preds).sum(0) - np.array(r["action"][:3])).max()))
        if "exec" in r and WIN[0] <= t < WIN[1]:
            ex = r["exec"]; cx = r["ctx"] if r["ctx"] is not None else ex; g = int(r["nat1"])
            ex1, cx1 = int(ex[0][0]), int(cx[0][0])
            info = {"t": t, "g": g, "ex1": ex1, "cx1": cx1}
            if mode in ("feedback", "corrected", "reverse"):
                pert_gen.add(t)
                p = ex1 if mode in ("feedback", "corrected") else cx1
                p_re, tgt = perturb(g)
                vg, vp = NORM[ID2ROW[g]], NORM[ID2ROW[p]]
                w0 = float(w[-1]) if ages[-1] == 0 else 0.0
                info.update({"p": p, "p_match": p == p_re, "null": p == g,
                             "ctx_ok": (mode == "feedback" and cx1 == p) or (mode == "corrected" and cx1 == g) or (mode == "reverse" and ex1 == g),
                             "dnorm": float(np.linalg.norm(vp - vg)), "q_err": float(np.linalg.norm(vp - tgt)),
                             "flip": float(np.linalg.norm(vg)) < D, "g_norm": float(np.linalg.norm(vg)),
                             "clip_p": bool((np.abs(RAW[ID2ROW[p]]) > 1 + 1e-9).any()), "clip_g": bool((np.abs(RAW[ID2ROW[g]]) > 1 + 1e-9).any()),
                             "bound_p": bool((np.abs(vp) >= 1 - 1e-9).any()),
                             "dphys": float(np.linalg.norm(PHYS[ID2ROW[p]] - PHYS[ID2ROW[g]])), "w0": w0,
                             "dir_exec": (w0 if mode != "reverse" else 0.0) * float(np.linalg.norm(PHYS[ID2ROW[p]] - PHYS[ID2ROW[g]]))})
            chunks.append(info)
    # context exposure (weights of preds coming from perturbed chunks)
    ctx_trans = ctx_same = 0.0
    if mode in ("feedback", "reverse"):
        for r in rec:
            w = ens_w(len(r["ages"]))
            for wi, a in zip(w, r["ages"]):
                if (r["t"] - a) in pert_gen:
                    if a >= 1:
                        ctx_trans += wi
                    else:
                        ctx_same += wi
    ons, on = sticky_onsets(act[:, 6])
    return {"task": ep["task"], "seed": ep["seed"], "cfg": ep["cond"].split(":")[0], "mode": mode, "steps": ep["steps"],
            "success": float(ep["success"]), "ftype": failure_type(ep), "pos": pos, "act": act, "chunks": chunks,
            "ctx_trans": ctx_trans, "ctx_same": ctx_same, "ens_err": ens_err,
            "sticky_onsets": ons, "sticky_steps": int(on.sum()), "sticky_steps_win": int(on[WIN[0]:WIN[1]].sum())}


def load():
    cache = "/tmp/claude-0/-root-VLA/3d5b895f-ebec-4409-a7c0-03065c2dc74b/scratchpad/p0c_cache.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    E = {}; sha = {}
    for f in FILES:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for line in fh:
                h.update(line)
                ep = compact(json.loads(line))
                E[(ep["cfg"], ep["mode"], ep["task"], ep["seed"])] = ep
        sha[f] = h.hexdigest()
    pickle.dump((E, sha), open(cache, "wb"))
    return E, sha


def pair(a, b):
    n = min(len(a["pos"]), len(b["pos"]))
    d = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
    c = np.linalg.norm(np.cumsum(a["act"][:n, :3], 0) - np.cumsum(b["act"][:n, :3], 0), axis=1)
    ad = np.linalg.norm(a["act"][:n, :3] - b["act"][:n, :3], axis=1)
    dd = np.full(T, np.nan); dd[:n] = d
    re_t = None
    if n >= T:
        for t in range(24, T - HOLD + 1):
            if np.all(d[t:t + HOLD] <= TAU):
                re_t = t; break
    return {"mean_dev_cm": 100 * d[8:].mean(), "final_dev_cm": 100 * dd[T - 1] if n >= T else np.nan, "max_dev_cm": 100 * d.max(),
            "area_dev_cmstep": 100 * d[8:].sum(), "post_area_dev_cmstep": 100 * d[24:].sum(), "cmd_final_m": c[-1], "cmd_max_m": c.max(),
            "adiff_win": ad[WIN[0]:WIN[1]].mean(), "adiff_all": ad[8:].mean(), "realigned": float(re_t is not None),
            "realign_t80": float(re_t if re_t is not None else T), "_curve": dd,
            "sticky_onsets_win_diff": float(sum(WIN[0] <= o < WIN[1] for o in a["sticky_onsets"]) - sum(WIN[0] <= o < WIN[1] for o in b["sticky_onsets"])),
            "sticky_steps_diff": float(a["sticky_steps"] - b["sticky_steps"])}


# ---------------- statistics ----------------
def boot_idx(keys, seed=0):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k[0]].append(i)
    cols = []
    for t in sorted(by):
        ix = np.array(by[t]); cols.append(ix[rng.integers(0, len(ix), size=(B_REPS, len(ix)))])
    return np.concatenate(cols, 1)


def ci(v, B):
    m = np.nanmean(v[B], 1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(x, y):
    a = int(np.sum((x == 1) & (y == 0))); b = int(np.sum((x == 0) & (y == 1)))
    return a, b, (float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0)


def wil(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def signflip(d, seed=0):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(PERM_REPS, len(d))); obs = abs(d.mean())
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.zeros(m); run_ = 0.0
    for i, j in enumerate(o):
        run_ = max(run_, min(1.0, (m - i) * ps[j])); out[j] = run_
    return out


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); prev = 1.0
    for r in range(m - 1, -1, -1):
        prev = min(prev, ps[o[r]] * m / (r + 1)); q[o[r]] = prev
    return q


def fmt(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    E, sha = load()
    keys80 = [(t, s) for t in TASKS for s in range(40)]
    keys240 = [(t, s) for t in TASKS for s in range(120)]
    B80 = boot_idx(keys80); B240 = boot_idx(keys240)
    task_idx80 = {t: np.array([i for i, k in enumerate(keys80) if k[0] == t]) for t in TASKS}
    task_idx240 = {t: np.array([i for i, k in enumerate(keys240) if k[0] == t]) for t in TASKS}

    def vals(cfg, keys):
        """per-episode success and pair metrics for each contrast in config cfg over keys."""
        out = {}
        for m in MODES + ["natural_gen"]:
            if all((cfg, m, *k) in E for k in keys):
                out[("S", m)] = np.array([E[(cfg, m, *k)]["success"] for k in keys])
        for cname, (a, b) in {**CONTRASTS, **EXTRA_PAIRS}.items():
            if ("S", a) not in out:
                continue
            out[("dS", cname)] = out[("S", a)] - out[("S", b)]
            pm = [pair(E[(cfg, a, *k)], E[(cfg, b, *k)]) for k in keys]
            for f in TRAJ:
                out[(cname, f)] = np.array([p[f] for p in pm], float)
            out[(cname, "_curve")] = np.stack([p["_curve"] for p in pm])
        return out

    V = {c: vals(c, keys80) for c in CFGS}
    VX = {c: vals(c, keys240) for c in ("r1e4", "r4e1")}

    # ---------- condition summary ----------
    cs = []
    for c in CFGS:
        for m in MODES + ["natural_gen"]:
            if ("S", m) not in V[c]:
                continue
            s = V[c][("S", m)]; lo, hi = ci(s, B80)
            row = {"config": c, "condition": m, "n": len(s), "success": fmt(s.mean()), "ci_lo": fmt(lo), "ci_hi": fmt(hi)}
            for t in TASKS:
                row[f"success_{t.replace('google_robot_', '')}"] = fmt(s[task_idx80[t]].mean())
            ft = defaultdict(int)
            for k in keys80:
                ft[E[(c, m, *k)]["ftype"]] += 1
            row["failure_types"] = json.dumps(dict(sorted(ft.items())))
            cn = {"feedback": "F-N", "corrected": "C-N", "reverse": "R-N", "natural_gen": "gen-N"}.get(m)
            if cn and (cn, "mean_dev_cm") in V[c]:
                for f in ("mean_dev_cm", "max_dev_cm", "final_dev_cm", "area_dev_cmstep", "cmd_final_m", "adiff_win", "realigned"):
                    row[f"vsN_{f}"] = fmt(np.nanmean(V[c][(cn, f)]))
            cs.append(row)
    write_csv(os.path.join(run, "condition_summary.csv"), cs)

    # ---------- within-config paired effects ----------
    pe = []; mc_rows = []
    for c in CFGS:
        for cname in list(CONTRASTS) + ["F-N", "C-N", "gen-N"]:
            if ("dS", cname) not in V[c]:
                continue
            x, y = (CONTRASTS.get(cname) or EXTRA_PAIRS[cname])
            for f in ["success"] + TRAJ:
                v = V[c][("dS", cname)] if f == "success" else V[c][(cname, f)]
                lo, hi = ci(v, B80)
                row = {"config": c, "contrast": cname, "metric": f, "n": int(np.sum(~np.isnan(v))), "estimate": fmt(np.nanmean(v)),
                       "ci_lo": fmt(lo), "ci_hi": fmt(hi)}
                if f == "success":
                    a, b, p = mcnemar(V[c][("S", x)], V[c][("S", y)])
                    row.update({"test": "exact McNemar", "x_only_success": a, "y_only_success": b, "p": fmt(p, 5)})
                    if cname in CONTRASTS:
                        mc_rows.append(row)
                elif f in SIGNED:
                    row.update({"test": "Wilcoxon signed-rank (signed X-Y)", "p": fmt(wil(v), 5)})
                else:
                    row.update({"test": "none (deviation >= 0; compare with gen-N noise)", "p": None})
                for t in TASKS:
                    vt = v[task_idx80[t]]; tl = t.replace("google_robot_", "")
                    sub = B80[:, :0]
                    Bt = boot_idx([k for k in keys80 if k[0] == t])
                    l2, h2 = ci(vt, Bt)
                    row[f"{tl}_est"] = fmt(np.nanmean(vt)); row[f"{tl}_ci_lo"] = fmt(l2); row[f"{tl}_ci_hi"] = fmt(h2)
                    if f == "success":
                        row[f"{tl}_p"] = fmt(mcnemar(V[c][("S", x)][task_idx80[t]], V[c][("S", y)][task_idx80[t]])[2], 5)
                pe.append(row)
    hp = holm([r["p"] for r in mc_rows])
    for r, h in zip(mc_rows, hp):
        r["p_holm_12"] = fmt(h, 5)
    write_csv(os.path.join(run, "paired_effects.csv"), pe)

    # ---------- DiD ----------
    dd = []
    for cname in CONTRASTS:
        for f in ["success"] + TRAJ:
            fam = []
            for ax, c1, c2 in DID:
                v1 = V[c1][("dS", cname)] if f == "success" else V[c1][(cname, f)]
                v2 = V[c2][("dS", cname)] if f == "success" else V[c2][(cname, f)]
                d = v2 - v1; lo, hi = ci(d, B80)
                discrete = f in ("success", "realigned")
                p_main = signflip(d) if discrete else wil(d)
                row = {"axis": ax, "contrast_type": cname, "metric": f, "primary": f in PRIMARY, "did": f"{c2} - {c1}",
                       "c1": c1, "c2": c2, "c1_est": fmt(np.nanmean(v1)), "c2_est": fmt(np.nanmean(v2)), "n": int(np.sum(~np.isnan(d))),
                       "estimate": fmt(np.nanmean(d)), "ci_lo": fmt(lo), "ci_hi": fmt(hi),
                       "test": "sign-flip permutation" if discrete else "Wilcoxon signed-rank", "p": fmt(p_main, 6),
                       "p_signflip": fmt(signflip(d), 6)}
                for t in TASKS:
                    tl = t.replace("google_robot_", ""); dt = d[task_idx80[t]]
                    l2, h2 = ci(dt, boot_idx([k for k in keys80 if k[0] == t]))
                    row[f"{tl}_est"] = fmt(np.nanmean(dt)); row[f"{tl}_ci_lo"] = fmt(l2); row[f"{tl}_ci_hi"] = fmt(h2)
                    row[f"{tl}_p"] = fmt(signflip(dt) if discrete else wil(dt), 6)
                fam.append(row)
            for r, h in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm_family7"] = fmt(h, 6)
            dd += fam
    q = bh([r["p"] for r in dd])
    for r, qq in zip(dd, q):
        r["q_bh_all"] = fmt(qq, 6)
    write_csv(os.path.join(run, "did_effects.csv"), dd)

    # ---------- +1.8 cm check: original analyze_protection definition (r1e1 vs r1e4, F_vs_C mean_err, Wilcoxon, no Holm) ----------
    d18 = V["r1e1"][("F-C", "mean_dev_cm")] - V["r1e4"][("F-C", "mean_dev_cm")]
    rng = __import__("random").Random(0); by = defaultdict(list)
    for i, k in enumerate(keys80):
        by[k[0]].append(d18[i])
    bs = sorted(np.mean(sum(([vs[rng.randrange(len(vs))] for _ in vs] for vs in by.values()), [])) for _ in range(2000))
    check18 = {"definition": "mean_{t=8..79} ||p_feedback(t) - p_corrected(t)|| per episode, config r1e1 minus r1e4, seed 0-39 (n=80)",
               "estimate_cm": fmt(d18.mean()), "ci_orig_random_module": [fmt(bs[int(.025 * 2000)]), fmt(bs[int(.975 * 2000) - 1])],
               "ci_this_run": [fmt(x) for x in ci(d18, B80)], "wilcoxon_p": fmt(wil(d18), 5), "signflip_p": fmt(signflip(d18), 5),
               "r1e1_FC_mean_dev": fmt(V["r1e1"][("F-C", "mean_dev_cm")].mean()), "r1e4_FC_mean_dev": fmt(V["r1e4"][("F-C", "mean_dev_cm")].mean()),
               "by_task": {t: fmt(d18[task_idx80[t]].mean()) for t in TASKS}}

    # ---------- dose table ----------
    dose = []
    for c in CFGS:
        for m in ["feedback", "corrected", "reverse", "natural"]:
            eps = [E[(c, m, *k)] for k in keys80]
            ch = [x for e in eps for x in e["chunks"] if "p" in x]
            nch = [sum("p" in x for x in e["chunks"]) for e in eps]
            plans_win = [len(e["chunks"]) for e in eps]
            row = {"config": c, "condition": m, "n_episodes": len(eps),
                   "plan_steps_in_window_per_ep": fmt(np.mean(plans_win), 2),
                   "perturbed_chunks_per_ep": fmt(np.mean(nch), 2), "perturbed_chunks_total": int(np.sum(nch))}
            if ch:
                dn = np.array([x["dnorm"] for x in ch]); dp = np.array([x["dphys"] for x in ch])
                de = np.array([sum(x["dir_exec"] for x in e["chunks"] if "p" in x) for e in eps])
                row.update({"p_matches_perturb_token": fmt(np.mean([x["p_match"] for x in ch])),
                            "context_token_as_specified": fmt(np.mean([x["ctx_ok"] for x in ch])),
                            "null_perturbation_frac(p==g)": fmt(np.mean([x["null"] for x in ch])),
                            "achieved_norm_dist_mean": fmt(dn.mean()), "achieved_norm_dist_median": fmt(np.median(dn)),
                            "achieved_norm_dist_min": fmt(dn.min()), "achieved_norm_dist_max": fmt(dn.max()),
                            "frac_achieved_within_0.2-0.4": fmt(np.mean((dn >= 0.2) & (dn <= 0.4))),
                            "quantization_err_to_target_mean": fmt(np.mean([x["q_err"] for x in ch])),
                            "frac_|v(g)|<0.3_target_flips": fmt(np.mean([x["flip"] for x in ch])),
                            "frac_p_token_clipped_by_tokenizer": fmt(np.mean([x["clip_p"] for x in ch])),
                            "frac_g_token_clipped_by_tokenizer": fmt(np.mean([x["clip_g"] for x in ch])),
                            "frac_p_on_norm_boundary": fmt(np.mean([x["bound_p"] for x in ch])),
                            "step1_phys_shift_m_mean": fmt(dp.mean(), 5),
                            "age0_weight_mean": fmt(np.mean([x["w0"] for x in ch])),
                            "direct_exec_dose_per_chunk_m": fmt(np.mean([x["dir_exec"] for x in ch]), 5),
                            "direct_exec_dose_per_ep_m": fmt(de.mean(), 5)})
            row["ctx_trans_exposure_per_ep"] = fmt(np.mean([e["ctx_trans"] for e in eps]))
            row["ctx_samestep_rotgrip_exposure_per_ep"] = fmt(np.mean([e["ctx_same"] for e in eps]))
            if m != "natural":
                nat = [E[(c, "natural", *k)] for k in keys80]
                adw = [np.linalg.norm(e["act"][8:24, :3] - n["act"][8:24, :3], axis=1) for e, n in zip(eps, nat)]
                row["exec_action_diff_vsN_win_mean_m"] = fmt(np.mean([a.mean() for a in adw]), 5)
                row["exec_action_diff_vsN_win_sum_m"] = fmt(np.mean([a.sum() for a in adw]), 5)
                row["exec_action_diff_vsN_t8_m"] = fmt(np.mean([a[0] for a in adw]), 5)
                row["sticky_first_onset_differs_vsN"] = fmt(np.mean([
                    (e["sticky_onsets"][:1] != n["sticky_onsets"][:1]) for e, n in zip(eps, nat)]))
            row["sticky_steps_per_ep"] = fmt(np.mean([e["sticky_steps"] for e in eps]), 2)
            row["sticky_steps_in_window_per_ep"] = fmt(np.mean([e["sticky_steps_win"] for e in eps]), 2)
            row["sticky_onsets_in_window_per_ep"] = fmt(np.mean([sum(WIN[0] <= o < WIN[1] for o in e["sticky_onsets"]) for e in eps]), 3)
            row["ensemble_recompute_max_abs_err"] = fmt(max(e["ens_err"] for e in eps), 6)
            dose.append(row)
    write_csv(os.path.join(run, "dose_table.csv"), dose)

    # ---------- extension: 240 paired (r1e4, r4e1), not compared with 80 ----------
    ext = []
    parts = {"seed0-119": np.arange(240), "seed0-39": np.array([i for i, k in enumerate(keys240) if k[1] < 40]),
             "seed40-119": np.array([i for i, k in enumerate(keys240) if k[1] >= 40])}
    for part, ix in parts.items():
        kk = [keys240[i] for i in ix]; Bp = boot_idx(kk)
        for cname in CONTRASTS:
            x, y = CONTRASTS[cname]
            for f in ["success", "area_dev_cmstep", "mean_dev_cm", "max_dev_cm", "final_dev_cm", "cmd_final_m", "realigned", "realign_t80"]:
                get = (lambda c: VX[c][("dS", cname)][ix]) if f == "success" else (lambda c: VX[c][(cname, f)][ix])
                for c in ("r1e4", "r4e1"):
                    v = get(c); lo, hi = ci(v, Bp)
                    r = {"part": part, "n": len(ix), "row": "within", "config": c, "contrast_type": cname, "metric": f,
                         "estimate": fmt(v.mean()), "ci_lo": fmt(lo), "ci_hi": fmt(hi)}
                    if f == "success":
                        a, b, p = mcnemar(VX[c][("S", x)][ix], VX[c][("S", y)][ix]); r.update({"x_only": a, "y_only": b, "p": fmt(p, 5), "test": "McNemar"})
                    for t in TASKS:
                        r[t.replace("google_robot_", "") + "_est"] = fmt(v[[i for i, k in enumerate(kk) if k[0] == t]].mean())
                    ext.append(r)
                d = get("r4e1") - get("r1e4"); lo, hi = ci(d, Bp)
                disc = f in ("success", "realigned")
                r = {"part": part, "n": len(ix), "row": "DiD r4e1 - r1e4 (cross-axis; not replanning-only)", "config": "r4e1-r1e4",
                     "contrast_type": cname, "metric": f, "estimate": fmt(d.mean()), "ci_lo": fmt(lo), "ci_hi": fmt(hi),
                     "p": fmt(signflip(d) if disc else wil(d), 6), "test": "sign-flip" if disc else "Wilcoxon"}
                for t in TASKS:
                    r[t.replace("google_robot_", "") + "_est"] = fmt(d[[i for i, k in enumerate(kk) if k[0] == t]].mean())
                ext.append(r)
    write_csv(os.path.join(run, "extension_240.csv"), ext)

    # ---------- units (long, per episode x config x contrast) ----------
    units = []
    for c in CFGS:
        for i, k in enumerate(keys80):
            for cname in CONTRASTS:
                u = {"task_id": k[0], "seed": k[1], "episode_id": f"{k[0]}_s{k[1]}", "config": c, "contrast": cname,
                     "success_diff": V[c][("dS", cname)][i]}
                for f in TRAJ:
                    u[f] = fmt(V[c][(cname, f)][i], 5)
                units.append(u)
    write_csv(os.path.join(run, "units_seed0_39.csv"), units)

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"r1e1": "#2a78d6", "r2e1": "#eb6834", "r4e1": "#1baf7a", "r1e2": "#eda100", "r1e4": "#e87ba4", "r2e2": "#4a3aa7"}
    axes_def = {"e1_fixed_replanning": ["r1e1", "r2e1", "r4e1"], "r1_fixed_ensemble": ["r1e1", "r1e2", "r1e4"], "r2_fixed_ensemble": ["r2e1", "r2e2"]}
    os.makedirs(os.path.join(run, "figures"), exist_ok=True)

    def curve_band(cv, B):
        m = np.nanmean(cv, 0) * 100; bm = np.stack([np.nanmean(cv[b], 0) for b in B[:500]]) * 100
        return m, np.percentile(bm, 2.5, 0), np.percentile(bm, 97.5, 0)
    for name, cfgs in axes_def.items():
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, cname, title in ((axs[0], "F-C", "feedback vs corrected"), (axs[1], "R-N", "reverse vs natural")):
            for c in cfgs:
                m, lo, hi = curve_band(V[c][(cname, "_curve")], B80)
                ax.plot(m, color=col[c], lw=2, label=c); ax.fill_between(np.arange(T), lo, hi, color=col[c], alpha=0.15, lw=0)
            for c in ("r1e4", "r4e1"):
                if c in cfgs:
                    g_ = np.nanmean(V[c][("gen-N", "_curve")], 0) * 100
                    ax.plot(g_, color=col[c], lw=1, ls=":", label=f"{c} rerun noise")
            ax.axvspan(8, 24, color="#999", alpha=0.12, lw=0); ax.axhline(2, color="#999", lw=0.8, ls="--")
            ax.set_title(f"{title} (seed 0-39, n=80)", fontsize=10); ax.set_xlabel("control step"); ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axs[0].set_ylabel("mean TCP position deviation (cm)"); axs[1].legend(fontsize=8, frameon=False)
        fig.suptitle(name.replace("_", " "), fontsize=11); fig.tight_layout()
        fig.savefig(os.path.join(run, "figures", f"error_curves_{name}.png"), dpi=130); plt.close(fig)
    # forest plot of DiD for primary metrics
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))
    for i, cname in enumerate(CONTRASTS):
        for j, f in enumerate(["success", "area_dev_cmstep"]):
            ax = axs[i, j]; rows = [r for r in dd if r["contrast_type"] == cname and r["metric"] == f]
            for y, r in enumerate(rows):
                sc = 100 if f == "success" else 1
                ax.plot([r["ci_lo"] * sc, r["ci_hi"] * sc], [y, y], color="#52514e", lw=2)
                ax.plot(r["estimate"] * sc, y, "o", color="#2a78d6", ms=7)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["did"] for r in rows], fontsize=8); ax.invert_yaxis()
            ax.axvline(0, color="#999", lw=1); ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"DiD of {cname}: {'success (%p)' if f == 'success' else 'deviation area (cm*step)'}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(run, "figures", "did_forest_primary.png"), dpi=130); plt.close(fig)
    # extension curves (240)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cname in zip(axs, CONTRASTS):
        for c in ("r1e4", "r4e1"):
            m, lo, hi = curve_band(VX[c][(cname, "_curve")], B240)
            ax.plot(m, color=col[c], lw=2, label=c); ax.fill_between(np.arange(T), lo, hi, color=col[c], alpha=0.15, lw=0)
        ax.axvspan(8, 24, color="#999", alpha=0.12, lw=0); ax.set_title(f"{cname} deviation, extension seed 0-119 (n=240)", fontsize=10)
        ax.set_xlabel("control step"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("mean TCP position deviation (cm)"); axs[1].legend(fontsize=8, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(run, "figures", "extension240_r1e4_r4e1.png"), dpi=130); plt.close(fig)

    # ---------- manifest ----------
    steps = defaultdict(int)
    for e in E.values():
        steps[e["steps"]] += 1
    man = {"task": "P0-C experiment 35 fixed-axis paired reanalysis (post-hoc, not preregistered)", "date": "2026-10-08",
           "inputs": {f: {"sha256": h} for f, h in sha.items()}, "token_table": "token_table.npz (processor of /root/VLA/spatialvla/spatialvla-4b-224-sft-fractal, CPU)",
           "code": ["code/export_token_table.py", "code/p0c_analysis.py"],
           "python": sys.executable, "n_episodes_loaded": len(E), "episode_length_counts": dict(steps),
           "common_units": len(keys80), "extension_units": len(keys240), "bootstrap": f"task-stratified episode bootstrap {B_REPS}, numpy default_rng(0), shared indices across configs/conditions",
           "permutation_reps": PERM_REPS, "plus_1_8cm_check": check18,
           "outputs": ["protocol.md", "condition_summary.csv", "paired_effects.csv", "did_effects.csv", "dose_table.csv", "extension_240.csv",
                       "units_seed0_39.csv", "figures/", "analysis.md"]}
    json.dump(man, open(os.path.join(run, "manifest.json"), "w"), indent=1, default=str)
    print(json.dumps(check18, indent=1))


if __name__ == "__main__":
    main()
