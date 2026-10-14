#!/usr/bin/env python
"""
P2-B evaluation (review_followup3, 2026-10-14; POST-HOC, CPU only).  Spec: P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md
sections 6-8 (locked; checked).  Same units / phi / estimator as P2-A (p2_motion_baselines.phi_pair, c_baseline_window.drive_row).

Outputs (P2/): extrap_tests_output.txt, extrap_units.parquet, extrap_coverage.csv, extrap_val_accuracy.csv,
               LEARNED_EXTRAPOLATOR_RESULTS.csv, extrap_run_meta.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_extrap_common as X  # noqa: E402
import p2_kinematics as K  # noqa: E402
import p2_motion_baselines as MB  # noqa: E402  (phi_pair, METRICS, CB, SC, P1A)
from p2_extrap_data import check_lock, OUT, DD  # noqa: E402
from p2_extrap_train import load, build_rows  # noqa: E402

CB, SC, C = MB.CB, MB.SC, MB.C
SEEDS = [0, 1, 2]
VARIANTS = {"prefix": "E-prefix (primary: prefix poses 0..j + j)", "prefix_ego": "E-prefix+ego (variant: + |v|, |a|, command)"}
SCENES = "/root/VLA/autovla/dataset/nuplan/navtest_poc"
SEED = 20261015
torch.set_num_threads(8)
T0 = time.time()
TL = []


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def check(name, val, tol):
    ok = bool(val <= tol)
    TL.append(f"[{'PASS' if ok else 'FAIL'}] {name}: max err {val:.3e} (tol {tol:.0e})")
    print(TL[-1])
    return ok


@torch.no_grad()
def predict(model, F):
    return np.concatenate([model(torch.from_numpy(F[s:s + 65536])).numpy() for s in range(0, len(F), 65536)])


def eval_inputs():
    cb, disp, dyaw = C.codebook_geometry()
    rows = []
    for src, path in (("primary", f"{MB.P1A}/raw/records.jsonl"), ("fixedpos", f"{MB.P1A}/raw_fixedpos/records.jsonl")):
        for line in open(path):
            r = json.loads(line)
            t = r["t_p"]; ref = r["ref_tokens"]; p = r["perturbation_info"]["token_p"]
            if p == ref[t]:
                continue
            pad = np.zeros((X.N_ACT - 1 - t, 3))
            rows.append({"source": src, "family": r["family"], "group": r["group"], "unit_id": r["unit_id"], "scene_id": r["token"],
                         "log_id": r["log"], "drive_id": CB.drive_of(r["log"]), "t_p": t,
                         "pre_p": np.vstack([K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p]), pad]),
                         "pre_0": np.vstack([K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [ref[t]]), pad]),
                         "ref9": np.asarray(r["ref_traj_xyh"], float)[9, :2], "gt9": np.asarray(r["trajectory_gt"], float)[9, :2]})
    ego = {}
    for s in {x["scene_id"] for x in rows}:
        js = json.load(open(f"{SCENES}/{s}.json"))
        cmd = np.zeros(4); cmd[X.CMD_NAMES.index(js["instruction"])] = 1
        ego[s] = X.ego_vector(js["velocity"], js["acceleration"], cmd)[0]
    return rows, ego


def unit_tests(rows, S):
    ok = True
    rng = np.random.default_rng(1)
    # features read nothing after pose j (no future / suffix leakage)
    e = 0.0
    for x in rows[:500]:
        j = np.array([x["t_p"]]); a = x["pre_p"][None].copy(); b = a.copy()
        b[0, x["t_p"] + 1:] = rng.normal(0, 50, b[0, x["t_p"] + 1:].shape)
        e = max(e, float(np.abs(X.prefix_features(a, j) - X.prefix_features(b, j)).max()))
    ok &= check("X1 prefix_features invariant to poses after j (no future leakage)", e, 0)
    # to_frame / from_frame round trip
    P = np.c_[rng.normal(0, 20, (1000, 2)), rng.uniform(-3, 3, 1000)][:, None]
    ref = np.c_[rng.normal(0, 20, (1000, 2)), rng.uniform(-3, 3, 1000)]
    R = X.from_frame(X.to_frame(P, ref), ref)
    ok &= check("X2 SE(2) to_frame/from_frame round trip", float(np.abs(np.c_[R[..., :2].reshape(-1, 2) - P[..., :2].reshape(-1, 2),
                                                                           X.wrap(R[..., 2] - P[..., 2]).reshape(-1, 1)]).max()), 1e-9)
    # targets -> assemble with the true suffix reproduces the trajectory
    dec = S["dec"][:2000]; j = rng.integers(0, 9, 2000)
    tg, m = X.targets(dec, j)
    A = X.assemble(dec, j, tg)
    ok &= check("X3 targets/assemble round trip on training plans", float(np.abs(np.c_[(A - dec)[..., :2].reshape(-1, 2),
                                                                                     X.wrap(A[..., 2] - dec[..., 2]).reshape(-1, 1)]).max()), 1e-5)
    # tokenizer port exact on the 208 navtest_poc scenes (stored GT tokens)
    gt_tok = {}
    for line in open(f"{MB.P1A}/raw/records.jsonl"):
        r = json.loads(line); gt_tok[r["token"]] = r["gt"]
    toks = sorted(gt_tok)
    G = np.array([json.load(open(f"{SCENES}/{t}.json"))["gt_trajectory"] for t in toks])
    cb, _, _ = C.codebook_geometry()
    T, D = X.tokenize_gt(G, cb)
    ok &= check(f"X4 GT->token port vs stored AutoVLA GT tokens ({len(toks)} scenes): share mismatching tokens",
                float((T != np.array([gt_tok[t] for t in toks])).mean()), 0)
    dd = np.array([C.rollout_np(cb, t) for t in T])
    ok &= check("X5 tokenizer running pose == decoder rollout", float(np.abs(dd - D).max()), 1e-4)
    # ego vector at evaluation = AutoVLA prompt inputs (speed and |acc| magnitudes)
    s0 = rows[0]["scene_id"]; js = json.load(open(f"{SCENES}/{s0}.json"))
    ok &= check("X6 ego speed feature == |velocity| from the scene JSON", abs(float(np.linalg.norm(js["velocity"])) / 10 -
                                                                            float(X.ego_vector(js["velocity"], js["acceleration"], np.eye(4)[0])[0, 0])), 1e-6)
    return ok


def coverage(S, rows):
    tr = np.where(S["split"] == 0)[0]; va = np.where(S["split"] == 1)[0]
    rng = np.random.default_rng(20261016)
    Ftr, _, _, Jtr = build_rows(S, tr, "prefix"); Fva, _, _, Jva = build_rows(S, va, "prefix")
    mu = Ftr.mean(0); sd = np.maximum(Ftr.std(0), 1e-3)
    nd = 60                                                         # Q slots + increments (no mask / one-hot)
    z = lambda F: ((F - mu) / sd)[:, :nd]
    jj = np.array([x["t_p"] for x in rows])
    Fp = X.prefix_features(np.stack([x["pre_p"] for x in rows]), jj)
    F0 = X.prefix_features(np.stack([x["pre_0"] for x in rows]), jj)
    nn_p = np.full(len(rows), np.nan); nn_0 = np.full(len(rows), np.nan); thr = np.full(len(rows), np.nan)
    cov = []
    for j in range(X.N_ACT - 1):
        it = np.where(Jtr == j)[0]; it = rng.choice(it, min(100_000, len(it)), replace=False)
        iv = np.where(Jva == j)[0]; iv = rng.choice(iv, min(20_000, len(iv)), replace=False)
        tree = cKDTree(z(Ftr[it]))
        dv = tree.query(z(Fva[iv]), k=1)[0]
        q99 = float(np.quantile(dv, .99))
        ie = np.where(jj == j)[0]
        if len(ie):
            nn_p[ie] = tree.query(z(Fp[ie]), k=1)[0]; nn_0[ie] = tree.query(z(F0[ie]), k=1)[0]; thr[ie] = q99
        cov.append({"j": j, "n_train_ref": len(it), "n_val": len(iv), "val_nn_q50": float(np.median(dv)),
                    "val_nn_q90": float(np.quantile(dv, .9)), "val_nn_q99": q99, "n_eval_units": int(len(ie))})
    return pd.DataFrame(cov), nn_p, nn_0, thr


def val_accuracy(S, models):
    """descriptive: validation-drive mean position error (m) per j, learned models vs CM / CV0 (same token-domain plans)."""
    va = np.where(S["split"] == 1)[0]
    rng = np.random.default_rng(7); va = rng.choice(va, min(5000, len(va)), replace=False)
    dec = S["dec"][va]; ego = S["ego"][va]
    out = []
    for j in range(X.N_ACT - 1):
        jj = np.full(len(va), j)
        errs = {}
        for name, (m, var) in models.items():
            pr = predict(m, X.prefix_features(dec, jj, ego if var == "prefix_ego" else None))
            A = X.assemble(dec, jj, pr)
            errs[name] = float(np.linalg.norm(A[:, j + 1:, :2] - dec[:, j + 1:, :2], axis=-1).mean())
        for nm, f in (("CM", K.bl_cm), ("CV0", K.bl_cv0), ("CTRV-K3", lambda a, b: K.bl_ctrv(a, b)[0]),
                      ("CTRA-2", lambda a, b: K.bl_ctra(a, b)[0])):
            E = np.stack([f(dec[n], j) for n in range(len(va))])
            errs[nm] = float(np.linalg.norm(E[:, j + 1:, :2] - dec[:, j + 1:, :2], axis=-1).mean())
        out += [{"j": j, "method": k, "val_mean_pos_err_m": v, "n_val_plans": len(va)} for k, v in errs.items()]
    return pd.DataFrame(out)


def main():
    lock = check_lock()
    sep = json.load(open(f"{OUT}/extrap_drive_separation.json"))
    rows, ego = eval_inputs()
    eval_drives = {x["drive_id"] for x in rows}
    assert not eval_drives & (set(sep["train_drives"]) | set(sep["val_drives"])), "eval drive in train/val"
    S = dict(np.load(f"{DD}/samples.npz", allow_pickle=False))
    assert not eval_drives & set(S["drive"].tolist()), "eval drive in samples"
    assert not {x["log_id"] for x in rows} & set(S["log"].tolist()), "eval log in samples"
    tests_ok = unit_tests(rows, S)
    TL.append(f"\n{'ALL PASS' if tests_ok else 'FAILURES'}")
    open(f"{OUT}/extrap_tests_output.txt", "w").write("\n".join(TL) + "\n")
    assert tests_ok
    log("units", len(rows))
    jj = np.array([x["t_p"] for x in rows])
    Pp = np.stack([x["pre_p"] for x in rows]); P0 = np.stack([x["pre_0"] for x in rows])
    EG = np.stack([ego[x["scene_id"]] for x in rows])
    U = pd.DataFrame([{k: v for k, v in x.items() if k not in ("pre_p", "pre_0", "ref9", "gt9")} for x in rows])
    models = {}
    for v in VARIANTS:
        for s in SEEDS:
            m, ck = load(f"final_{v}_s{s}")
            models[f"{v}_s{s}"] = (m, v)
            eg = EG if v == "prefix_ego" else None
            Ap = X.assemble(Pp, jj, predict(m, X.prefix_features(Pp, jj, eg)))
            A0 = X.assemble(P0, jj, predict(m, X.prefix_features(P0, jj, eg)))
            ph = [MB.phi_pair(Ap[n], A0[n], int(jj[n]), rows[n]["ref9"], rows[n]["gt9"]) for n in range(len(rows))]
            for met in MB.METRICS:
                U[f"E_{v}_s{s}_{met}"] = [d[met] for d in ph]
    for v in VARIANTS:
        for met in MB.METRICS:
            U[f"E_{v}_{met}"] = U[[f"E_{v}_s{s}_{met}" for s in SEEDS]].mean(1)
    # model Delta from P2-A unit table (same phi, same units)
    MU = pd.read_parquet(f"{OUT}/p2_units.parquet")
    U = U.merge(MU[["source", "unit_id"] + [f"model_{m}" for m in MB.METRICS] +
                   [f"{b}_{m}" for b in MB.KIN for m in MB.METRICS]], on=["source", "unit_id"], how="left", validate="1:1")
    assert U[[f"model_{m}" for m in MB.METRICS]].notna().all().all()
    for v in VARIANTS:
        for met in MB.METRICS:
            U[f"ex_E_{v}_{met}"] = U[f"model_{met}"] - U[f"E_{v}_{met}"]
            for s in SEEDS:
                U[f"ex_E_{v}_s{s}_{met}"] = U[f"model_{met}"] - U[f"E_{v}_s{s}_{met}"]
    log("predictions done")
    cov, nn_p, nn_0, thr = coverage(S, rows)
    U["nn_pert"], U["nn_unpert"], U["nn_thr_val_q99"] = nn_p, nn_0, thr
    U["out_of_range"] = (U.nn_pert > U.nn_thr_val_q99) | (U.nn_unpert > U.nn_thr_val_q99)
    U.to_parquet(f"{OUT}/extrap_units.parquet", index=False)
    # coverage summary per family x stratum
    crow = []
    for fam in ["G", "E", "GF1", "GF3"]:
        for grp in ("A-", "A+", "all"):
            g = U[(U.family == fam) & ((U.group == grp) if grp != "all" else True)]
            q = lambda s, a: float(np.quantile(s, a))
            ratio_p = g.nn_pert / g.nn_thr_val_q99
            crow.append({"family": fam, "group": grp, "n_units": len(g),
                         "nn_pert_q50": q(g.nn_pert, .5), "nn_pert_q90": q(g.nn_pert, .9), "nn_pert_q99": q(g.nn_pert, .99), "nn_pert_max": float(g.nn_pert.max()),
                         "nn_unpert_q50": q(g.nn_unpert, .5), "nn_unpert_q90": q(g.nn_unpert, .9),
                         "ratio_pert_to_val_q99_q50": q(ratio_p, .5), "ratio_pert_to_val_q99_q90": q(ratio_p, .9),
                         "share_pert_out_of_range": float((g.nn_pert > g.nn_thr_val_q99).mean()),
                         "share_unpert_out_of_range": float((g.nn_unpert > g.nn_thr_val_q99).mean()),
                         "share_unit_out_of_range": float(g.out_of_range.mean()),
                         "n_units_in_range": int((~g.out_of_range).sum()),
                         "n_drives_in_range": int(g.loc[~g.out_of_range, "drive_id"].nunique())})
    pd.concat([cov.assign(table="per_j_reference"), pd.DataFrame(crow).assign(table="eval_by_family_stratum")]).to_csv(
        f"{OUT}/extrap_coverage.csv", index=False)
    log("coverage done")
    # ---------------------------------------------------------------- tables
    out = []; seed = SEED
    for fam in ["G", "E", "GF1", "GF3"]:
        for grp in ("A-", "A+", "all"):
            g = U[(U.family == fam) & ((U.group == grp) if grp != "all" else True)]
            for met, mdesc in MB.METRICS.items():
                block = []
                for v, vdesc in VARIANTS.items():
                    for analysis, gg in (("primary (all units, seed-mean Delta_E)", g),
                                         ("sensitivity: out-of-range units excluded", g[~g.out_of_range])):
                        r = CB.drive_row(gg, f"ex_E_{v}_{met}", margin=None, seed=seed); seed += 11
                        x = {"row_type": "excess", "family": fam, "group": grp, "metric": met, "metric_desc": mdesc,
                             "comparator": f"E:{v}", "comparator_desc": vdesc, "analysis": analysis,
                             "quantity": f"Delta_model - Delta_E[{v}]", **r}
                        if analysis.startswith("primary"):
                            ests = []
                            for s in SEEDS:
                                rs = CB.drive_row(g, f"ex_E_{v}_s{s}_{met}", margin=None, seed=seed); seed += 11
                                ests.append(rs["est_scene"])
                                out.append({"row_type": "excess_per_seed", "family": fam, "group": grp, "metric": met,
                                            "metric_desc": mdesc, "comparator": f"E:{v}", "comparator_desc": vdesc,
                                            "analysis": f"single training seed {s}", "train_seed": s,
                                            "quantity": f"Delta_model - Delta_E[{v}, seed {s}]", **rs})
                            x["seed_est_min"], x["seed_est_max"], x["seed_est_sd"] = min(ests), max(ests), float(np.std(ests, ddof=1))
                            block.append(x)
                        else:
                            out.append(x)
                adj = SC.holm([x["p_signflip"] for x in block])
                for x, a in zip(block, adj):
                    x["p_holm_m2"] = a
                    ex0 = x["ci95_lo"] > 0 or x["ci95_hi"] < 0
                    x["judgment"] = (("model > extrapolator" if x["est_scene"] > 0 else "model < extrapolator")
                                     if (a < 0.05 and ex0) else "not detected (equivalence not tested)")
                out += block
                for v in VARIANTS:                                  # level Delta_E (descriptive)
                    r = CB.drive_row(g, f"E_{v}_{met}", margin=None, seed=seed); seed += 11
                    out.append({"row_type": "level", "family": fam, "group": grp, "metric": met, "metric_desc": mdesc,
                                "comparator": f"E:{v}", "comparator_desc": VARIANTS[v], "analysis": "level Delta_E (seed mean)",
                                "quantity": f"Delta_E[{v}]", "judgment": "descriptive level", **r})
            log(fam, grp)
    T = pd.DataFrame(out)
    T.insert(0, "status", MB.STATUS)
    T.to_csv(f"{OUT}/LEARNED_EXTRAPOLATOR_RESULTS.csv", index=False)
    va = val_accuracy(S, models)
    va.to_csv(f"{OUT}/extrap_val_accuracy.csv", index=False)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump({"protocol_lock": lock, "status": MB.STATUS, "p1_gate_file_present_at_run": os.path.exists(MB.GATE),
               "n_units": len(U), "seed_base": SEED, "train_seeds": SEEDS, "bootstrap_B": CB.B,
               "models": {f"final_{v}_s{s}": sha(f"{OUT}/_extrap_models/final_{v}_s{s}.pt") for v in VARIANTS for s in SEEDS},
               "samples_npz_sha256": sha(f"{DD}/samples.npz"), "elapsed_s": time.time() - T0},
              open(f"{OUT}/extrap_run_meta.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
