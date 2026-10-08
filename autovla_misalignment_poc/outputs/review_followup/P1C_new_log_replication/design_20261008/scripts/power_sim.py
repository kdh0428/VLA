#!/usr/bin/env python
"""P1-C precision/power simulation (CPU only, no model inference).

Inputs (read-only):
  P0-B units.parquet (exp07 recent_gt/normal, exp32 dir_wrong_mag_ok/dir_ok_mag_wrong; A-/A+; 16/26 dev logs)
  equal_distance_perturbation/records.jsonl  (dev t* per scene)
  work/eval_pool.json + work/frame_labels_5090.json  (eval logs: drives, per-log eligible A-/A+ counts, A+ t*)
Model:
  - eval cluster = drive (date.time_veh); a drive keeps its real log segments and real per-segment counts
    (eligible = t* <= 7, labels from the existing 5090 natural rows; planning only).
  - each eval drive draws ONE dev A- log (for all its A- scenes) and ONE dev A+ log (for its sampled A+ scenes):
    maximal within-drive correlation (conservative). Scenes are resampled with replacement inside the dev log;
    A+ scenes are drawn with prob ~ eval/dev t* ratio (bins 0,1,2,3,4-7). k units per scene w/o replacement.
  - failure stratum F: all eligible A- scenes (pi = 1). success-mismatch stratum S: m scenes per log segment
    (pi = m / N_S,seg, capped at 1). Population P = all eligible mismatch scenes (F u S), Hajek ratio estimator.
  - CI: cluster-robust linearised SE, t(G-1). Power = CI excludes 0 (alpha .05) and at Holm-first-step alpha .05/3.
  - Reverse-Full has no raw (exp 8): simulated as RN-shaped noise scaled by the summary CI width ratio
    (A- 16.9/20.9 = 0.81, A+ 3.7/4.5 = 0.82), centred on the hypothesised effect. APPROXIMATION.
  - hypotheses: H0 (all effects 0), MME (stratum means scaled so the estimand equals the MME), dev (dev means),
    half_dev (half the dev means).
"""
import json, os, sys
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(D, "work")
P0B = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P0B_cluster_stats/run_20261008/units.parquet"
ED = "/root/VLA/autovla_misalignment_poc/outputs/equal_distance_perturbation/records.jsonl"
R = int(os.environ.get("REPS", 2000))
rng = np.random.default_rng(20261008)
TBIN = lambda t: min(int(t), 4)

# ---------------- dev scene/unit effects ----------------
u = pd.read_parquet(P0B)
u = u[(u.model == "AutoVLA")]
def pair(exp, a, b):
    x = u[u.experiment == exp].pivot_table(index=["unit_id", "scene_id", "log_id", "group"], columns="condition", values="amplification").reset_index()
    return x.assign(d=x[a] - x[b])[["unit_id", "scene_id", "log_id", "group", "d"]]
rn = pair("exp07_action_history_causal", "recent_gt", "normal").rename(columns={"d": "rn"})
dm = pair("exp32_motion_semantics", "dir_wrong_mag_ok", "dir_ok_mag_wrong").rename(columns={"d": "dm"})
du = rn.merge(dm[["unit_id", "dm"]], on="unit_id", how="inner")
assert len(du) == 1408
tstar = {json.loads(l)["token"]: json.loads(l)["t_star"] for l in open(ED)}
du["tbin"] = du.scene_id.map(lambda s: TBIN(tstar[s]))
dev = {}
for g in ("A-", "A+"):
    logs = {}
    for lg, x in du[du.group == g].groupby("log_id"):
        scenes = [(s.tbin.iloc[0], s.rn.to_numpy(), s.dm.to_numpy()) for _, s in x.groupby("scene_id")]
        logs[lg] = scenes
    dev[g] = logs
def scene_mean(g):
    x = du[du.group == g].groupby("scene_id")[["rn", "dm"]].mean()
    return x.rn.mean(), x.dm.mean()
dev_mean = {g: scene_mean(g) for g in ("A-", "A+")}
RV_DEV = {"A-": 0.290, "A+": 0.0307}          # exp 8 summary (unit-weighted, raw unavailable)
RV_SCALE = {"A-": 16.9 / 20.9, "A+": 3.7 / 4.5}

# ---------------- eval pool ----------------
pool = json.load(open(os.path.join(W, "eval_pool.json")))["eval"]
fl = [r for r in json.load(open(os.path.join(W, "frame_labels_5090.json")))]
evlogs = {r["log"] for r in pool}
seg = {}
for r in fl:
    if r["log"] not in evlogs or r["t_star"] is None or r["t_star"] > 7:
        continue
    s = seg.setdefault(r["log"], {"F": 0, "S": 0})
    s["F" if r["a_minus"] else "S"] += 1
drives = {}
for r in pool:
    drives.setdefault(r["drive"], []).append(seg.get(r["log"], {"F": 0, "S": 0}))
DRV = list(drives.values())
ev_t = np.bincount([TBIN(r["t_star"]) for r in fl if r["log"] in evlogs and r["t_star"] is not None and r["t_star"] <= 7 and not r["a_minus"]], minlength=5)
dv_t = np.bincount(du[du.group == "A+"].drop_duplicates("scene_id").tbin, minlength=5)
TW = (ev_t / ev_t.sum()) / np.maximum(dv_t / dv_t.sum(), 1e-9)


# ---------------- centre dev effects on their expectation under the sim's sampling ----------------
# F: one dev log uniformly, scenes uniformly inside it -> E = mean over logs of mean scene effect.
# S: one dev log uniformly, scenes ~ t* weight inside it -> E = mean over logs of weighted scene mean.
def centre(logs, tw):
    cen = np.zeros(2)
    for sc in logs.values():
        w = np.array([tw[s[0]] if tw is not None else 1.0 for s in sc]); w = w / w.sum()
        cen += np.array([[a.mean(), b.mean()] for _, a, b in sc]).T @ w
    return cen / len(logs)
CEN = {"A-": centre(dev["A-"], None), "A+": centre(dev["A+"], TW)}
for g in dev:
    for lg in dev[g]:
        dev[g][lg] = [(t, a - CEN[g][0], b - CEN[g][1]) for t, a, b in dev[g][lg]]
LOGS = {g: list(dev[g].values()) for g in dev}

def draw(logscenes, n, k, tw=None):
    if tw is None:
        idx = rng.integers(0, len(logscenes), n)
    else:
        p = np.array([tw[s[0]] for s in logscenes]); p = p / p.sum()
        idx = rng.choice(len(logscenes), n, p=p)
    tot = np.zeros(2); nu = 0
    for j in idx:
        _, a, b = logscenes[j]
        if k and k < len(a):
            sel = rng.choice(len(a), k, replace=False); a, b = a[sel], b[sel]
        tot += (a.mean(), b.mean()); nu += len(a)
    return tot, nu

def replicate(G, m, k, cl):
    pick = rng.choice(len(DRV), G, replace=G > len(DRV))
    rows = []
    for di in pick:
        segs = DRV[di]
        lf = LOGS["A-"][rng.integers(len(LOGS["A-"]))]; lp = LOGS["A+"][rng.integers(len(LOGS["A+"]))]
        nF = NS = nu = ns = 0; sF = np.zeros(2); sS = np.zeros(2)
        for s in segs:
            if cl == "segment":      # independent dev log per segment (less conservative)
                lf = LOGS["A-"][rng.integers(len(LOGS["A-"]))]; lp = LOGS["A+"][rng.integers(len(LOGS["A+"]))]
            if s["F"]:
                t_, n_ = draw(lf, s["F"], k); sF += t_; nu += n_; ns += s["F"]; nF += s["F"]
            if s["S"]:
                NS += s["S"]
                if m:
                    mm = min(m, s["S"]); t_, n_ = draw(lp, mm, k, TW); sS += t_ * (s["S"] / mm); nu += n_; ns += mm
        rows.append((nF, sF[0], sF[1], NS, sS[0], sS[1], nu, ns))
    return np.array(rows, float)

def est(num, den, weighting):
    keep = den > 0; num, den = num[keep], den[keep]; g = len(num)
    if g < 2:
        return np.nan, np.nan, g
    if weighting == "scene":
        th = num.sum() / den.sum(); z = (num - th * den) / den.mean()
    else:                              # drive-equal: mean of drive-level ratios
        r = num / den; th = r.mean(); z = r - th
    return th, np.sqrt(g / (g - 1) * (z ** 2).sum()) / g, g

from scipy.stats import t as tdist
MME = {("F", "rn"): -0.10, ("F", "rv"): 0.10, ("F", "dm"): 0.05,
       ("P", "rn"): -0.02, ("P", "rv"): 0.02, ("P", "dm"): 0.02}
DEVF = {"rn": dev_mean["A-"][0], "dm": dev_mean["A-"][1], "rv": RV_DEV["A-"]}
DEVS = {"rn": dev_mean["A+"][0], "dm": dev_mean["A+"][1], "rv": RV_DEV["A+"]}
A_HOLM = 0.05 / 3

def run(G, m, k, cl):
    reps = [replicate(G, m, k, cl) for _ in range(R)]
    out = []
    for estd in ("F", "P"):
        if estd == "P" and m == 0:
            continue
        for c in ("rn", "rv", "dm"):
            for wt in ("scene", "drive_equal"):
                res = []
                for X in reps:
                    nF, fr, fd, NS, sr, sd, nu, ns = X.T
                    if c == "rv":
                        cF, cS = -RV_SCALE["A-"] * fr, -RV_SCALE["A+"] * sr
                    else:
                        cF, cS = (fr, sr) if c == "rn" else (fd, sd)
                    hyp = {}
                    for h in ("H0", "MME", "dev", "half_dev"):
                        if estd == "F":
                            mu = {"H0": 0.0, "MME": MME[("F", c)], "dev": DEVF[c], "half_dev": DEVF[c] / 2}[h]
                            hyp[h] = est(cF + nF * mu, nF, wt)
                        else:
                            pd_ = (nF.sum() * DEVF[c] + NS.sum() * DEVS[c]) / (nF.sum() + NS.sum())
                            sc = {"H0": 0.0, "MME": MME[("P", c)] / pd_, "dev": 1.0, "half_dev": 0.5}[h]
                            hyp[h] = est(cF + cS + sc * (nF * DEVF[c] + NS * DEVS[c]), nF + NS, wt)
                    res.append((hyp, nu.sum(), ns.sum() if estd == "P" else nF.sum()))
                A = {h: np.array([r[0][h] for r in res], float) for h in res[0][0]}
                h0 = A["H0"]; ok0 = np.isfinite(h0[:, 1])
                crit05 = np.nanquantile(np.abs(h0[:, 0]), 0.95); critH = np.nanquantile(np.abs(h0[:, 0]), 1 - A_HOLM)
                row = dict(contrast=c, estimand=estd, weighting=wt, cluster_model=cl, drives=G, aplus_per_log=m,
                           units_per_scene=(k or "all"), reps=R, mme=MME[(estd, c)],
                           exp_scenes=np.mean([r[2] for r in res]), exp_units=np.mean([r[1] for r in res]),
                           median_clusters=np.nanmedian(h0[:, 2]), sd_est=np.nanstd(h0[:, 0]),
                           calibrated_halfwidth95=1.96 * np.nanstd(h0[:, 0]),
                           clustert_halfwidth95=np.nanmean(tdist.ppf(0.975, np.maximum(h0[:, 2] - 1, 1)) * h0[:, 1]),
                           clustert_typeI05=np.mean((np.abs(h0[:, 0] / h0[:, 1]) > tdist.ppf(0.975, np.maximum(h0[:, 2] - 1, 1)))[ok0]))
                for h in ("MME", "dev", "half_dev"):
                    a = A[h]; ok = np.isfinite(a[:, 1]); q = tdist.ppf(1 - A_HOLM / 2, np.maximum(a[:, 2] - 1, 1))
                    row[f"true_{h}"] = np.nanmean(a[:, 0])
                    row[f"power_calib05_{h}"] = np.mean(np.abs(a[:, 0]) > crit05)
                    row[f"power_calibHolm_{h}"] = np.mean(np.abs(a[:, 0]) > critH)
                    row[f"power_clustertHolm_{h}"] = np.mean((np.abs(a[:, 0] / a[:, 1]) > q)[ok])
                out.append(row)
    return out

if __name__ == "__main__":
    allr = []
    for cl in ("drive", "segment"):
        for G in (8, 12, 16, 20, 24, 32, 48):
            for m in (0, 2, 4, 8, 16):
                for k in (0, 3):
                    allr += run(G, m, k, cl)
            print(cl, G, flush=True)
    df = pd.DataFrame(allr)
    df.to_csv(os.path.join(D, "power_simulation.csv"), index=False, float_format="%.4f")
    meta = dict(dev_mean_scene={g: dict(rn=v[0], dm=v[1]) for g, v in dev_mean.items()}, centre_under_sim={g: v.tolist() for g, v in CEN.items()},
                rv_dev=RV_DEV, rv_scale=RV_SCALE, eval_drives=len(DRV), eval_F=int(sum(s["F"] for d in DRV for s in d)),
                eval_S=int(sum(s["S"] for d in DRV for s in d)), drives_with_F=int(sum(any(s["F"] for s in d) for d in DRV)),
                F_per_drive=sorted(int(sum(s["F"] for s in d)) for d in DRV), tstar_weights=TW.tolist(), reps=R, seed=20261008,
                note="drives>24 are hypothetical (eval drives resampled with replacement)")
    json.dump(meta, open(os.path.join(W, "power_sim_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
