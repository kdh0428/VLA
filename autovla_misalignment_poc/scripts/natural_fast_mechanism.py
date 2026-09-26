#!/usr/bin/env python
"""
Re-test the AutoVLA failure mechanisms on NATURAL / FAST inference failures (CPU only).

No GPU and no new inference: everything is read from stored tensors.

Sources (same code for all three)
  N  full_extract arm N  -- natural fast path; per-layer h_in / attn_out / mlp_out and
                            action logits at the action query rows          (PRIMARY)
  G  outputs/gpu1        -- the original natural run; residual stream only  (replication)
  C  full_extract arm C  -- forced CoT, scored identically                  (comparison)

Failure definition
  A+/A- from the planned trajectory with the P/R/A comparison's accept rule (semantic coarse
  action, 5 s). Token IDs never define failure. The OLD token-ID grouping (step-0 token != GT)
  is also run, only to bridge to the earlier reports.

Readout
  final RMSNorm + lm_head action rows, fp32, exactly as the earlier decomposition.
  margin = logit(GT token) - logit(rival); rival = generated token if wrong, else runner-up.
  d_attn[l] = margin(h_in + attn_out) - margin(h_in); d_mlp[l] = margin(h_out) - margin(h_in + attn_out)

Groupings
  traj_step0          A- vs A+ at the step-0 action query
  traj_first_mismatch A- at its first token-mismatch step t*, vs A+ scenes whose first
                      mismatch is at the same t* (step-matched: both made a token error, only
                      one of them turned into a trajectory failure)
  token_step0         step-0 token wrong vs right (the earlier definition)

Verdict rules (fixed before running)
  perception preserved   late-layer (L32-L36) AUROC in A- within 0.05 of A+, or higher
  late action formation  median commitment layer >= 32 in both groups AND >= 50% of the final
                         margin gap is created inside L32-L35
  L35 MLP dominance      L35 MLP has the largest |failure - correct| among L28-L35 components
                         and its CI excludes 0
  attention weak         |sum L32-35 MLP difference| >= 3 x |sum L32-35 attention difference|
  small codebook mismatch  median nearest-neighbour rank of the wrong token around the GT token
                         <= 10 (of 2048)
  error amplification    median FDE / pose error at the first mismatch step > 3 AND
                         P(token wrong | earlier wrong) / P(token wrong | clean prefix) > 3
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np                                                  # noqa: E402
from scipy.stats import spearmanr                                   # noqa: E402

POC = "/root/VLA/autovla_misalignment_poc"
PRA = "/root/VLA/pra_comparison"
sys.path.insert(0, PRA)
import pra_labels as L                                              # noqa: E402
from run_pra_comparison import _pos_to_delta                        # noqa: E402

OUT = os.path.join(POC, "outputs/natural_fast_mechanism")
CFG = L.Config()
PROBE_LAYERS = [0, 8, 16, 24, 28, 32, 34, 35, 36]
DEC = list(range(28, 36))
REPS = 1000
N_ACT = 10

RD = np.load(os.path.join(OUT, "readout.npz"))
W = RD["W_act"].astype(np.float32)
G_NORM = RD["norm_w"].astype(np.float32)
EPS = float(RD["eps"])
# token_all = {"veh", "ped", "cyc"}, each (2048, 6, 4, 2); the ego uses the vehicle codebook.
CB = np.asarray(pickle.load(open("/root/VLA/autovla/codebook_cache/agent_vocab.pkl", "rb"))["token_all"]["veh"], np.float32)
DISP = CB[:, -1].mean(1)                                            # (2048, 2) displacement per token


def rmsnorm(x):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * G_NORM


def a_label(traj, gt, horizon=10):
    s = L.Sample(model="", sample_id="", cluster="", scene=L.Scene(), desc_text="", desc_objs=[], rsn_text="",
                 decl_lon=None, decl_lat=None, pred_delta=_pos_to_delta(traj, horizon), gt_delta=_pos_to_delta(gt, horizon))
    return bool(L.label_A(s, CFG)[0])


# ======================================================================================
# per-row mechanics
# ======================================================================================
def row_mechanics(hin, att, mlp, gen_tok, gt_tok):
    """hin/att/mlp: (36, H) at one action query row (att/mlp may be None for residual-only)."""
    if att is not None:
        final = hin[35] + att[35] + mlp[35]
        states = np.concatenate([hin, final[None]], 0)             # 37 residual states
    else:
        states = hin                                               # already 37 states
    logits = rmsnorm(states) @ W.T                                 # (37, 2048)
    fin = logits[-1]
    if gen_tok != gt_tok:
        rival = gen_tok
    else:
        tmp = fin.copy(); tmp[gt_tok] = -np.inf
        rival = int(tmp.argmax())
    u = W[gt_tok] - W[rival]
    lens = rmsnorm(states) @ u                                     # (37,)
    out = {"lens_margin": lens, "argmax": logits.argmax(1), "final_logits": fin, "rival": rival}
    commit = next((l for l in range(37) if all(logits[k].argmax() == gen_tok for k in range(l, 37))), 36)
    out["commit_layer"] = commit
    if att is not None:
        m_in = rmsnorm(hin) @ u
        m_attn = rmsnorm(hin + att) @ u
        m_out = rmsnorm(hin + att + mlp) @ u
        out["d_attn"] = m_attn - m_in
        out["d_mlp"] = m_out - m_attn
        out["m_in"] = m_in
        out["m_out"] = m_out
    return out


def first_mismatch(pred, gt):
    return next((k for k in range(N_ACT) if pred[k] != gt[k]), None)


# ======================================================================================
# loading
# ======================================================================================
def load_source(src: str, limit: int = 0):
    samples = []
    t0 = time.time()
    if src in ("N", "C"):
        it = (json.loads(l) for l in open(os.path.join(POC, "outputs/full_extract/records.jsonl")))
    else:
        it = (json.loads(l) for l in open(os.path.join(POC, "outputs/gpu1/records.jsonl")))
    val_diffs = []
    for i, r in enumerate(it):
        if limit and i >= limit:
            break
        if src in ("N", "C"):
            arm = r["arms"][src]
            if src == "N" and (arm["cot_present"] or arm["runaway_action_tokens"]):
                continue
            if src == "C" and (arm["truncated"] or arm["runaway_action_tokens"]):
                continue
            traj, pred = arm["trajectory_pred"], arm["pred_action_idx"][:N_ACT]
        else:
            if r.get("cot_present"):
                continue
            traj, pred = r["trajectory_pred"], r["pred_action_idx"][:N_ACT]
        gt_traj, gt = r["trajectory_gt"], r["gt_action_idx"][:N_ACT]
        if len(traj) < N_ACT or len(pred) < N_ACT:
            continue
        s = {"token": r["token"], "log": r["log_name"], "src": src, "pred": pred, "gt": gt,
             "A": a_label(traj, gt_traj), "t_star": first_mismatch(pred, gt),
             "perception": r.get("perception") or {}}
        p, g_ = np.asarray(traj, float)[:, :2], np.asarray(gt_traj, float)[:N_ACT, :2]
        e = np.linalg.norm(p - g_, axis=1)
        s.update({"ade": float(e.mean()), "fde": float(e[-1]), "pose_err": e.tolist()})

        rows = {0}
        if s["t_star"] is not None:
            rows.add(s["t_star"])
        if src in ("N", "C"):
            z = np.load(os.path.join(POC, "outputs/full_extract/tensors", f"{r['token']}_{src}.npz"))
            h_in, a_out, m_out = z["h_in"], z["attn_out"], z["mlp_out"]
            qk = z["q_rows_in_kept"] if "q_rows_in_kept" in z.files else np.arange(h_in.shape[1])
            nq = a_out.shape[1]
            s["rows"] = {}
            for k in rows:
                if k >= nq:
                    continue
                hin = h_in[:, int(qk[k])].astype(np.float32)
                mech = row_mechanics(hin, a_out[:, k].astype(np.float32), m_out[:, k].astype(np.float32),
                                     int(pred[k]), int(gt[k]))
                if len(val_diffs) < 20 and k == 0:
                    val_diffs.append(float(np.abs(mech["final_logits"] - z["action_logits"][-1, 0].astype(np.float32)).max()))
                mech.pop("final_logits")
                s["rows"][k] = mech
            hin0 = h_in[:, int(qk[0])].astype(np.float32)
            fin0 = hin0[35] + a_out[35, 0] + m_out[35, 0]
            full0 = np.concatenate([hin0, fin0[None]], 0)
            s["feat"] = full0[PROBE_LAYERS].astype(np.float16)
        else:
            z = np.load(os.path.join(POC, "outputs/gpu1/hidden", f"{r['token']}.npz"))
            s["rows"] = {}
            for k in rows:
                key = f"action_{k}"
                if key not in z.files:
                    continue
                mech = row_mechanics(z[key].astype(np.float32), None, None, int(pred[k]), int(gt[k]))
                mech.pop("final_logits")
                s["rows"][k] = mech
            s["feat"] = z["action_0"].astype(np.float16)[PROBE_LAYERS]
        samples.append(s)
        if len(samples) % 500 == 0:
            print(f"  [{src}] loaded {len(samples)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{src}] {len(samples)} samples, A- {sum(not s['A'] for s in samples)}, "
          f"readout check maxdiff {max(val_diffs) if val_diffs else float('nan'):.3f} ({time.time()-t0:.0f}s)", flush=True)
    return samples


# ======================================================================================
# statistics
# ======================================================================================
def cluster_mean_ci(items, reps=REPS, seed=0):
    """items: list of (log, value)."""
    items = [(lg, float(v)) for lg, v in items if v is not None and np.isfinite(v)]
    if not items:
        return None
    by = defaultdict(list)
    for lg, v in items:
        by[lg].append(v)
    logs = list(by)
    rng = random.Random(seed)
    boots = [np.mean([v for lg in (rng.choice(logs) for _ in logs) for v in by[lg]]) for _ in range(reps)]
    return {"mean": float(np.mean([v for _, v in items])), "n": len(items),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def cluster_diff_ci(fail, ok, reps=REPS, seed=0):
    """mean(fail) - mean(ok), resampling logs jointly; two-sided bootstrap p."""
    fail = [(lg, float(v)) for lg, v in fail if np.isfinite(v)]
    ok = [(lg, float(v)) for lg, v in ok if np.isfinite(v)]
    if not fail or not ok:
        return None
    bf, bo = defaultdict(list), defaultdict(list)
    for lg, v in fail:
        bf[lg].append(v)
    for lg, v in ok:
        bo[lg].append(v)
    logs = sorted(set(bf) | set(bo))
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        pick = [rng.choice(logs) for _ in logs]
        f = [v for lg in pick for v in bf.get(lg, [])]
        o = [v for lg in pick for v in bo.get(lg, [])]
        if f and o:
            boots.append(np.mean(f) - np.mean(o))
    pt = np.mean([v for _, v in fail]) - np.mean([v for _, v in ok])
    boots = np.asarray(boots)
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {"diff": float(pt), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_boot": p, "n_fail": len(fail), "n_ok": len(ok)}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 3:
        return None
    return float(roc_auc_score(y, s))


# ======================================================================================
# analyses
# ======================================================================================
def perception_probes(samples):
    """GT perception labels decoded from the step-0 action-query state; AUROC reported per group."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    variables = {
        "lead_vehicle": lambda p: None if p.get("lead_vehicle") is None else int(bool(p["lead_vehicle"])),
        "pedestrian": lambda p: None if p.get("pedestrian") is None else int(bool(p["pedestrian"])),
        "critical_center": lambda p: None if not p.get("critical_side") else int(p["critical_side"] == "center"),
        "critical_moving": lambda p: None if not p.get("critical_motion") else int(p["critical_motion"] == "moving"),
    }
    res = {}
    for var, fn in variables.items():
        idx = [i for i, s in enumerate(samples) if fn(s["perception"]) is not None]
        y = np.array([fn(samples[i]["perception"]) for i in idx])
        groups = np.array([samples[i]["log"] for i in idx])
        isfail = np.array([not samples[i]["A"] for i in idx])
        res[var] = {"n": len(idx), "n_pos": int(y.sum()), "n_fail": int(isfail.sum()),
                    "n_fail_pos": int(y[isfail].sum()), "layers": {}}
        for li, layer in enumerate(PROBE_LAYERS):
            X = np.stack([samples[i]["feat"][li] for i in idx]).astype(np.float32)
            oof = np.full(len(idx), np.nan)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
            try:
                splits = list(cv.split(X, y, groups))
            except ValueError:
                break
            for tr, te in splits:
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=0.01, max_iter=400, class_weight="balanced")
                clf.fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            ok = np.isfinite(oof)
            a_all = auroc(y[ok], oof[ok])
            a_pos = auroc(y[ok & ~isfail], oof[ok & ~isfail])
            a_neg = auroc(y[ok & isfail], oof[ok & isfail])
            # bootstrap the A- AUROC gap over logs
            gap = None
            if a_pos is not None and a_neg is not None:
                logs = sorted(set(groups))
                rng = random.Random(layer)
                boots = []
                for _ in range(300):
                    pick = [rng.choice(logs) for _ in logs]
                    sel = np.concatenate([np.where(groups == lg)[0] for lg in pick])
                    sel = sel[ok[sel]]
                    ap_ = auroc(y[sel][~isfail[sel]], oof[sel][~isfail[sel]])
                    an_ = auroc(y[sel][isfail[sel]], oof[sel][isfail[sel]])
                    if ap_ is not None and an_ is not None:
                        boots.append(an_ - ap_)
                if boots:
                    gap = {"diff": a_neg - a_pos, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            res[var]["layers"][layer] = {"auroc_all": a_all, "auroc_Aplus": a_pos, "auroc_Aminus": a_neg, "gap_Aminus_minus_Aplus": gap}
    return res


def group_rows(samples, grouping):
    """Return (fail_rows, ok_rows): lists of (sample, row_index)."""
    fail, ok = [], []
    if grouping == "traj_step0":
        for s in samples:
            if 0 in s["rows"]:
                (ok if s["A"] else fail).append((s, 0))
    elif grouping == "token_step0":
        for s in samples:
            if 0 in s["rows"]:
                (fail if s["pred"][0] != s["gt"][0] else ok).append((s, 0))
    elif grouping == "traj_first_mismatch":
        by_t = defaultdict(list)
        for s in samples:
            if s["A"] and s["t_star"] is not None and s["t_star"] in s["rows"]:
                by_t[s["t_star"]].append(s)
        for s in samples:
            if not s["A"] and s["t_star"] is not None and s["t_star"] in s["rows"]:
                fail.append((s, s["t_star"]))
        wanted = Counter(t for _, t in fail)
        for t in wanted:
            ok += [(s, t) for s in by_t.get(t, [])]
    return fail, ok


def decomposition(samples, grouping, has_components):
    fail, ok = group_rows(samples, grouping)
    out = {"n_fail": len(fail), "n_ok": len(ok), "n_logs_fail": len({s["log"] for s, _ in fail})}
    if not fail or not ok:
        return out
    F = lambda rows, key, l: [(s["log"], s["rows"][k][key][l]) for s, k in rows]  # noqa: E731
    lens = {}
    for l in range(37):
        lens[l] = {"fail": cluster_mean_ci(F(fail, "lens_margin", l)), "ok": cluster_mean_ci(F(ok, "lens_margin", l)),
                   "diff": cluster_diff_ci(F(fail, "lens_margin", l), F(ok, "lens_margin", l))}
    out["lens"] = lens
    gap_final = lens[36]["ok"]["mean"] - lens[36]["fail"]["mean"]
    gap_32 = lens[32]["ok"]["mean"] - lens[32]["fail"]["mean"]
    out["final_gap"] = gap_final
    out["gap_at_L32_input"] = gap_32
    out["share_gap_created_L32_L35"] = (gap_final - gap_32) / gap_final if gap_final else None
    sig = [l for l in range(37) if lens[l]["diff"] and lens[l]["diff"]["ci95"][1] < 0]
    out["first_layer_diff_ci_below0_and_stays"] = next((l for l in range(37) if all(m in sig for m in range(l, 37))), None)
    out["commit_layer_median"] = {"fail": float(np.median([s["rows"][k]["commit_layer"] for s, k in fail])),
                                  "ok": float(np.median([s["rows"][k]["commit_layer"] for s, k in ok]))}
    if has_components:
        cells = {}
        for l in DEC:
            for comp in ("attn", "mlp"):
                key = f"d_{comp}"
                cells[f"L{l}_{comp}"] = {"ok": cluster_mean_ci(F(ok, key, l)), "fail": cluster_mean_ci(F(fail, key, l)),
                                         "diff": cluster_diff_ci(F(fail, key, l), F(ok, key, l))}
        out["cells"] = cells
        sums = {}
        for comp in ("attn", "mlp"):
            f_ = [(s["log"], float(sum(s["rows"][k][f"d_{comp}"][l] for l in range(32, 36)))) for s, k in fail]
            o_ = [(s["log"], float(sum(s["rows"][k][f"d_{comp}"][l] for l in range(32, 36)))) for s, k in ok]
            sums[comp] = {"ok": cluster_mean_ci(o_), "fail": cluster_mean_ci(f_), "diff": cluster_diff_ci(f_, o_)}
        out["sum_L32_35"] = sums
        out["mlp_over_attn_separation"] = abs(sums["mlp"]["diff"]["diff"]) / max(abs(sums["attn"]["diff"]["diff"]), 1e-9)
        absd = {k: abs(v["diff"]["diff"]) for k, v in cells.items() if v["diff"]}
        out["largest_cell"] = max(absd, key=absd.get)
        l35 = cells["L35_mlp"]["diff"]
        out["L35_mlp_share_of_total_component_diff"] = abs(l35["diff"]) / sum(absd.values())
    return out


def codebook_and_amplification(samples):
    rng = np.random.default_rng(0)
    rand_pairs = rng.integers(0, 2048, size=(20000, 2))
    rand_d = np.linalg.norm(DISP[rand_pairs[:, 0]] - DISP[rand_pairs[:, 1]], axis=1)
    order_cache = {}

    def nn_rank(gt_tok, tok):
        if gt_tok not in order_cache:
            d = np.linalg.norm(DISP - DISP[gt_tok], axis=1)
            order_cache[gt_tok] = np.argsort(np.argsort(d, kind="stable"), kind="stable")
        return int(order_cache[gt_tok][tok])

    res = {"random_pair_displacement_m": {"median": float(np.median(rand_d)), "p10": float(np.percentile(rand_d, 10))}}
    for grp, pick in (("A-", [s for s in samples if not s["A"] and s["t_star"] is not None]),
                      ("A+ with a token mismatch", [s for s in samples if s["A"] and s["t_star"] is not None]),
                      # the earlier reports' population: step-0 token wrong (t_star is then 0)
                      ("step-0 token wrong (old definition)", [s for s in samples if s["pred"][0] != s["gt"][0]])):
        if not pick:
            continue
        dist, rank, shape_d, ratio, e_at = [], [], [], [], []
        for s in pick:
            t = s["t_star"]
            g, w_ = s["gt"][t], s["pred"][t]
            dist.append(float(np.linalg.norm(DISP[w_] - DISP[g])))
            rank.append(nn_rank(g, w_))
            shape_d.append(float(np.linalg.norm(CB[w_] - CB[g], axis=-1).mean()))
            e_at.append(s["pose_err"][t])
            ratio.append(s["fde"] / max(s["pose_err"][t], 0.05))
        dist, rank = np.asarray(dist), np.asarray(rank)
        res[grp] = {
            "n": len(pick),
            "first_mismatch_step_hist": dict(Counter(s["t_star"] for s in pick)),
            "wrong_token_displacement_m": {"median": float(np.median(dist)), "p75": float(np.percentile(dist, 75)),
                                           "percentile_vs_random_pairs": float((rand_d[None, :] <= np.median(dist)).mean() * 100)},
            "nn_rank_of_wrong_token": {"median": float(np.median(rank)), "share_rank_le10": float((rank <= 10).mean()),
                                       "share_rank_le50": float((rank <= 50).mean())},
            "full_shape_distance_m": float(np.median(shape_d)),
            "pose_error_at_first_mismatch_m": float(np.median(e_at)),
            "ade_m": float(np.median([s["ade"] for s in pick])), "fde_m": float(np.median([s["fde"] for s in pick])),
            "fde_over_first_step_error_median": float(np.median(ratio)),
            "spearman_first_displacement_vs_fde": _spearman_ci([(s["log"], d, s["fde"]) for s, d in zip(pick, dist)]),
            "spearman_first_displacement_vs_ade": _spearman_ci([(s["log"], d, s["ade"]) for s, d in zip(pick, dist)]),
            "pose_error_curve_mean": np.mean([s["pose_err"] for s in pick], 0).tolist(),
        }
    res["A+ no mismatch pose_error_curve_mean"] = np.mean([s["pose_err"] for s in samples if s["t_star"] is None], 0).tolist()
    # autoregressive propagation over all samples of the source
    clean_w = clean_n = dirty_w = dirty_n = 0
    for s in samples:
        wrong_before = False
        for k in range(1, N_ACT):
            wrong_before = wrong_before or s["pred"][k - 1] != s["gt"][k - 1]
            wk = s["pred"][k] != s["gt"][k]
            if wrong_before:
                dirty_n += 1; dirty_w += wk
            else:
                clean_n += 1; clean_w += wk
    res["propagation"] = {"p_wrong_given_clean_prefix": clean_w / max(clean_n, 1),
                          "p_wrong_given_earlier_wrong": dirty_w / max(dirty_n, 1),
                          "ratio": (dirty_w / max(dirty_n, 1)) / max(clean_w / max(clean_n, 1), 1e-9),
                          "n_clean": clean_n, "n_dirty": dirty_n}
    return res


def _spearman_ci(items, reps=REPS):
    if len(items) < 5:
        return None
    rho = float(spearmanr([d for _, d, _ in items], [f for _, _, f in items]).correlation)
    by = defaultdict(list)
    for lg, d, f in items:
        by[lg].append((d, f))
    logs = list(by)
    rng = random.Random(3)
    boots = []
    for _ in range(reps):
        pts = [p for lg in (rng.choice(logs) for _ in logs) for p in by[lg]]
        if len(pts) >= 5:
            boots.append(spearmanr([p[0] for p in pts], [p[1] for p in pts]).correlation)
    boots = [b for b in boots if np.isfinite(b)]
    return {"rho": rho, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
            "n": len(items)}


def verdicts(p, dec, cba):
    v = {}
    late = [l for l in (32, 34, 35, 36)]
    lv = p.get("lead_vehicle", {}).get("layers", {})
    gaps = [lv[l]["auroc_Aminus"] - lv[l]["auroc_Aplus"] for l in late
            if l in lv and lv[l]["auroc_Aminus"] is not None and lv[l]["auroc_Aplus"] is not None]
    v["perception_preserved"] = None if not gaps else {"mean_late_gap_lead_vehicle": float(np.mean(gaps)),
                                                       "holds": bool(np.mean(gaps) >= -0.05)}
    d = dec.get("traj_step0", {})
    if "commit_layer_median" in d:
        v["late_action_formation"] = {"commit_median": d["commit_layer_median"],
                                      "share_gap_L32_35": d["share_gap_created_L32_L35"],
                                      "holds": bool(min(d["commit_layer_median"].values()) >= 32
                                                    and (d["share_gap_created_L32_L35"] or 0) >= 0.5)}
    for gname in ("traj_step0", "traj_first_mismatch"):
        d = dec.get(gname, {})
        if "cells" in d:
            l35 = d["cells"]["L35_mlp"]["diff"]
            v[f"L35_mlp_dominance[{gname}]"] = {"largest_cell": d["largest_cell"], "L35_mlp_diff": l35,
                                                "holds": bool(d["largest_cell"] == "L35_mlp" and (l35["ci95"][1] < 0 or l35["ci95"][0] > 0))}
            v[f"attention_weak[{gname}]"] = {"mlp_over_attn": d["mlp_over_attn_separation"],
                                             "holds": bool(d["mlp_over_attn_separation"] >= 3)}
    if "A-" in cba:
        v["small_codebook_mismatch"] = {"median_nn_rank": cba["A-"]["nn_rank_of_wrong_token"]["median"],
                                        "holds": bool(cba["A-"]["nn_rank_of_wrong_token"]["median"] <= 10)}
        v["error_amplification"] = {"fde_over_first_error": cba["A-"]["fde_over_first_step_error_median"],
                                    "propagation_ratio": cba["propagation"]["ratio"],
                                    "holds": bool(cba["A-"]["fde_over_first_step_error_median"] > 3 and cba["propagation"]["ratio"] > 3)}
    return v


def main():
    limit = int(os.environ.get("LIMIT", "0"))
    summary = {}
    for src in ("N", "G", "C"):
        samples = load_source(src, limit)
        has_comp = src in ("N", "C")
        t0 = time.time()
        dec = {g: decomposition(samples, g, has_comp) for g in ("traj_step0", "traj_first_mismatch", "token_step0")}
        print(f"[{src}] decomposition {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        pr = perception_probes(samples)
        print(f"[{src}] probes {time.time()-t0:.0f}s", flush=True)
        cba = codebook_and_amplification(samples)
        summary[src] = {"n": len(samples), "n_Aminus": sum(not s["A"] for s in samples),
                        "n_Aminus_step0_token_wrong": sum((not s["A"]) and s["pred"][0] != s["gt"][0] for s in samples),
                        "n_Aplus_step0_token_wrong": sum(s["A"] and s["pred"][0] != s["gt"][0] for s in samples),
                        "decomposition": dec, "perception": pr, "codebook_amplification": cba,
                        "verdicts": verdicts(pr, dec, cba)}
        with open(os.path.join(OUT, f"per_sample_{src}.jsonl"), "w") as f:
            for s in samples:
                f.write(json.dumps({k: s[k] for k in ("token", "log", "A", "t_star", "pred", "gt", "ade", "fde")}
                                   | {"commit_layer_step0": s["rows"].get(0, {}).get("commit_layer")}) + "\n")
        del samples
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"[{src}] verdicts: {json.dumps(summary[src]['verdicts'], default=str)[:900]}", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
