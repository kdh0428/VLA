#!/usr/bin/env python
"""
Tests of analyze_p1c.py on SYNTHETIC data only (never on evaluation outcomes). CPU.
  1. exact sign-flip (meet in the middle) == brute-force enumeration
  2. sign-flip type I under a symmetric null (drive totals) ~ nominal
  3. Hajek estimator recovers the population effect under the P1-C sampling design (F all, S m=4 per segment)
  4. test-inversion CI contains theta and has p(bound) ~ alpha
  5. end-to-end: a synthetic run directory (selection / units / four mechanism blocks) with known effects
     -> analyze_p1c.main; checks estimates, Holm, decisions, masking, denominators.
  python test_analyze_p1c.py <scratch_dir>
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_p1c as A  # noqa: E402

rng = np.random.default_rng(1)
OK = []


def check(name, cond, info=""):
    OK.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)


# 1 ------------------------------------------------------------------------------------------------------------
for G in (1, 2, 5, 9, 12):
    T = rng.normal(0.1, 1, G)
    bf = np.mean([abs(np.dot(s, T)) >= abs(T.sum()) - 1e-12 for s in itertools.product([-1, 1], repeat=G)])
    check(f"signflip exact == brute force G={G}", abs(A.signflip_exact(T) - bf) < 1e-12, f"{A.signflip_exact(T):.6f} vs {bf:.6f}")
T = np.array([1.0, 0.0, 0.0, 2.0])
bf = np.mean([abs(np.dot(s, T)) >= 3 - 1e-12 for s in itertools.product([-1, 1], repeat=4)])
check("signflip with zero totals", abs(A.signflip_exact(T) - bf) < 1e-12)
check("signflip obs=0 -> 1", A.signflip_exact(np.zeros(5)) == 1.0)
import time
t0 = time.time(); p24 = A.signflip_exact(rng.normal(0, 1, 24)); check("G=24 exact is fast", time.time() - t0 < 5, f"{time.time()-t0:.2f}s p={p24:.4f}")

# 2 ------------------------------------------------------------------------------------------------------------
ps = np.array([A.signflip_exact(rng.standard_t(3, 20) * rng.choice([0.2, 1, 3], 20)) for _ in range(2000)])
check("type I at .05 under symmetric null within [.035,.065]", 0.035 <= np.mean(ps < .05) <= .065, f"{np.mean(ps < .05):.3f}")
check("type I at .0083 within [.003,.015]", 0.003 <= np.mean(ps < .05 / 6) <= .015, f"{np.mean(ps < .05/6):.4f}")

# 3 ------------------------------------------------------------------------------------------------------------
def synth_scenes(G=24, segs=(1, 4), effF=-0.4, effS=-0.03, het=0.1):
    rows = []
    for g in range(G):
        u = rng.normal(0, het)
        for s in range(rng.integers(*segs)):
            nF, NS = rng.poisson(1.5), rng.integers(0, 60)
            m = min(4, NS)
            for i in range(nF):
                rows.append(dict(scene=f"d{g}s{s}F{i}", cluster=f"d{g}", log=f"d{g}s{s}", stratum="F", w=1.0,
                                 d=np.clip(effF + u + rng.normal(0, .3), -1, 1), n_units=7))
            pop = np.clip(effS + u / 4 + rng.normal(0, .1, NS), -1, 1)
            for i in range(m):
                rows.append(dict(scene=f"d{g}s{s}S{i}", cluster=f"d{g}", log=f"d{g}s{s}", stratum="S", w=NS / m,
                                 d=pop[i], n_units=7, popmean=pop.mean(), NS=NS))
    sc = pd.DataFrame(rows)
    sc["dsum"] = sc.d * sc.n_units
    return sc


est, truth = [], []
for _ in range(300):
    sc = synth_scenes()
    F, S = sc[sc.stratum == "F"], sc[sc.stratum == "S"]
    segS = S.groupby("log").first()
    true_P = (F.d.sum() + (segS.popmean * segS.NS).sum()) / (len(F) + segS.NS.sum())
    est.append(A.estimate(sc, "scene", boot=False)["theta"]); truth.append(true_P)
bias = np.mean(np.array(est) - np.array(truth))
check("Hajek theta_P ~unbiased for the finite-population effect", abs(bias) < 0.003, f"bias {bias:+.4f}")
sc = synth_scenes()
r = A.estimate(sc[sc.stratum == "F"], "scene", with_ci_inversion=True)
check("theta_F inside sign-flip CI95 and CI90 nested", r["sf_ci95"][0] <= r["sf_ci90"][0] <= r["theta"] <= r["sf_ci90"][1] <= r["sf_ci95"][1],
      f"{r['theta']:.3f} {r['sf_ci90']} {r['sf_ci95']}")
labs, N, D = A.cluster_arrays(sc[sc.stratum == "F"], "scene")
pb = A.signflip_exact(N - r["sf_ci95"][0] * D)
check("p at the CI95 bound ~ .05", 0.03 <= pb <= 0.07, f"p={pb:.4f}")
de = A.estimate(sc, "drive_equal", boot=True)
check("drive-equal estimate finite with CIs", np.isfinite(de["theta"]) and "boot_ci95" in de)
q = A.delta_q3(sc)
check("Q3 delta = theta_F - theta_S", abs(q["delta"] - (q["theta_F"] - q["theta_S"])) < 1e-12 and q["theta_F"] < q["theta_S"])
check("holm", np.allclose(A.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
check("decision replicated", A.decision(0.001, .05, -0.3, -1, [-0.4, -0.2], -0.1) == "replicated")
check("decision wrong sign not replicated", A.decision(0.001, .05, 0.3, -1, [0.2, 0.4], -0.1) == "inconclusive")
check("decision below MME", A.decision(0.5, .05, -0.01, -1, [-0.05, 0.03], -0.1) == "not_replicated_below_MME")


# 5 ------------------------------------------------------------------------------------------------------------
def write_synthetic_run(root, G=24, effects=None):
    """records in the exact layouts of p1c_select / p1c_build_units / the four dev mechanism scripts."""
    effects = effects or {}
    os.makedirs(f"{root}/selection", exist_ok=True); os.makedirs(f"{root}/units", exist_ok=True)
    for d in A.BLOCK_DIR.values():
        os.makedirs(f"{root}/mech/{d}", exist_ok=True)
    gtraj = [[2.0 * (k + 1), 0.0, 0.0] for k in range(10)]
    good = [p[:2] for p in gtraj]
    bad = [[2.0 * (k + 1), 0.6 * (k + 1)] for k in range(10)]           # 6 m lateral at 5 s: A- and FDE > 3
    split_rows, sel, den = [], [], {}
    fu = {d: open(f"{root}/mech/{d}/records.jsonl", "w") for d in A.BLOCK_DIR.values()}
    funits = open(f"{root}/units/records.jsonl", "w")
    base_amp = {"F": 0.5, "S": 0.06}

    def amp(stratum, cond, block):
        p = base_amp[stratum] + effects.get((block, cond, stratum), 0.0)
        return rng.random() < min(max(p, 0), 1)

    for g in range(G):
        drive = f"2021.01.{g+1:02d}.10.00.00_veh-{g % 16:02d}"
        for s in range(2):
            log = f"{drive}_{s:05d}_{s+100:05d}"
            split_rows.append(dict(log=log, drive=drive, day_vehicle=f"2021.01.{g % 16 + 1:02d}_veh-{g % 16:02d}", role="eval",
                                   navhard_source=str(g % 2 == 0), dayveh_overlaps_poc=str(g % 3 == 0), shard="6"))
            nF, NS = int(rng.integers(0, 4)), int(rng.integers(5, 40))
            m = min(4, NS)
            den[log] = {"scenes": nF + NS + 50, "eligible": nF + NS, "F": nF, "S_eligible": NS, "S_sampled": m,
                        "not_eligible_by_reason": {"no_mismatch": 50}}
            for stratum, n in (("F", nF), ("S", m)):
                for i in range(n):
                    tok = f"tok{g}_{s}_{stratum}{i}"
                    w = 1.0 if stratum == "F" else NS / m
                    sel.append(dict(token=tok, log=log, stratum=stratum, pi=1 / w, weight=w, N_S_seg=NS, m_seg=m, t_star=2))
                    alts = [{"token": 100 + j} for j in range(5)]
                    funits.write(json.dumps({"token": tok, "group": "A-" if stratum == "F" else "A+", "t_star": 2, "log": log,
                                             "pred": [1] * 10, "gt": [0] * 10, "trajectory_gt": [p[:2] for p in gtraj],
                                             "alternatives": alts, "conditions": {}}) + "\n")
                    perts = ["original"] + [f"alt{j}" for j in range(5)]

                    def row(block, c):
                        return {"action_idx": [0] * 10, "trajectory_pred": bad if amp(stratum, c, block) else good,
                                "entropy_steps": [1.0] * 7, "ent": [1.0] * 7}
                    h7 = {pn: {"forced_token": 1, "conditions": {c: row("H7", c) for c in A.BLOCK_CONDS["H7"]}} for pn in perts}
                    fu["action_history_causal"].write(json.dumps({"token": tok, "perturbations": h7}) + "\n")
                    for pn in perts:
                        fu["prev_action_state_patching"].write(json.dumps({"token": tok, "perturbation": pn, "forced_token": 1,
                            "rows": {c: row("H8", c) for c in A.BLOCK_CONDS["H8"]}}) + "\n")
                        fu["temporal_feedback_window"].write(json.dumps({"token": tok, "perturbation": pn, "forced_token": 1,
                            "rows": {c: row("H9", c) for c in A.BLOCK_CONDS["H9"]}}) + "\n")
                        fu["motion_semantics_ablation"].write(json.dumps({"token": tok, "perturbation": pn, "forced_token": 1,
                            "conditions": {c: row("H32", c) for c in A.BLOCK_CONDS["H32"]}}) + "\n")
    for f in fu.values():
        f.close()
    funits.close()
    json.dump({"selected": sel, "denominators": den}, open(f"{root}/selection/shard_6.json", "w"))
    pd.DataFrame(split_rows).to_csv(f"{root}/split.csv", index=False)


scratch = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1c_test"
eff = {("H7", "recent_gt", "F"): -0.40, ("H7", "recent_gt", "S"): -0.03,
       ("H8", "reverse@emb", "F"): 0.0, ("H8", "gt_history", "F"): -0.42, ("H8", "gt_history", "S"): -0.04,
       ("H32", "dir_wrong_mag_ok", "F"): 0.0, ("H32", "dir_ok_mag_wrong", "F"): 0.0,
       ("H9", "gt_history", "F"): -0.42, ("H9", "win4", "F"): -0.41, ("H9", "win3", "F"): -0.38, ("H9", "win1", "F"): -0.2}
for tag, mask in (("unmasked", False), ("masked", True)):
    root = os.path.join(scratch, f"synthetic_run_{tag}")
    if not os.path.exists(root):
        write_synthetic_run(root, effects=eff)
    out = os.path.join(root, "analysis")
    if not os.path.exists(os.path.join(out, "paired_effects.csv")):
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_p1c.py"), "--run", root, "--split", f"{root}/split.csv",
                        "--out", out] + (["--mask-effects"] if mask else []), check=True)
    pe = pd.read_csv(os.path.join(out, "paired_effects.csv"))
    dec = json.load(open(os.path.join(out, "decisions.json")))
    if not mask:
        dd = {(d["contrast"], d["estimand"]): d for d in dec["decisions"]}
        check("synthetic RN F replicated (true -0.40)", dd[("RN", "F")]["decision"] == "replicated", f"theta={dd[('RN','F')]['theta']:.3f}")
        check("synthetic RV F replicated (true +0.42 = reverse - gt_history)", dd[("RV", "F")]["decision"] == "replicated",
              f"theta={dd[('RV','F')]['theta']:.3f}")
        check("synthetic DM F not 'replicated' (true 0)", dd[("DM", "F")]["decision"] != "replicated",
              f"theta={dd[('DM','F')]['theta']:.3f} {dd[('DM','F')]['decision']}")
        check("theta_F RN near truth", abs(dd[("RN", "F")]["theta"] + 0.40) < 0.1)
        check("Holm family size 3", len([d for d in dec["decisions"] if d["estimand"] == "F"]) == 3)
        kinds = set(pe.kind)
        need = {"primary", "sensitivity_weighting", "sensitivity_subset", "sensitivity_common_inclusion", "secondary_fde5",
                "secondary_Q3_delta", "per_drive", "per_log", "secondary_window_level", "secondary_window_gap_fraction",
                "secondary_window_equivalence", "ood_entropy_rule", "secondary_theta_S"}
        check("all output kinds present", need <= kinds, str(need - kinds))
        s3 = pe[(pe.kind == "sensitivity_subset") & (pe.variant == "S3_dayveh_cluster") & (pe.contrast == "RN") & (pe.estimand == "F")]
        check("S3 day-vehicle clustering uses 16 clusters", int(s3.n_clusters.iloc[0]) <= 16, str(s3.n_clusters.iloc[0]))
        check("forest figure written", os.path.exists(os.path.join(out, "figures", "forest_RN.png")))
    else:
        num = pe.drop(columns=["kind", "contrast", "block", "cond_a", "cond_b", "metric", "estimand", "variant", "weighting",
                               "n_scenes", "n_units", "n_clusters"], errors="ignore")
        vals = set(map(str, pd.unique(num.values.ravel())))
        check("masked: no effect values in paired_effects.csv", vals <= {"MASKED", "nan", "None", "0.05", "0.025", "-1", "1", "-1.0", "1.0",
              "-0.1", "0.1", "0.05", "-0.02", "0.02", "False", "True"} or vals <= {"MASKED", "nan", "None"} | {v for v in vals if v in
              ("0.05", "0.025", "-0.1", "0.1", "-0.02", "0.02", "1.0", "-1.0", "1", "-1")}, str(sorted(vals)[:10]))
        check("masked: decisions masked", all(d["decision"] == "MASKED" for d in dec["decisions"]))
        u = pd.read_parquet(os.path.join(out, "units.parquet"))
        check("masked: unit outcomes masked", set(u.amplification) == {"MASKED"})
        dn = pd.read_csv(os.path.join(out, "denominators.csv"))
        check("denominators per log", len(dn) == 48 and (dn.H7_scenes_analysed == dn.units_built_scenes).all())

print(f"\n{sum(o for _, o in OK)}/{len(OK)} passed")
sys.exit(0 if all(o for _, o in OK) else 1)
