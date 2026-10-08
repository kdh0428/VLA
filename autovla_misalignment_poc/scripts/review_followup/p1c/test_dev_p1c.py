#!/usr/bin/env python
"""
Dev-data execution test of analyze_p1c.py (CPU). Uses ONLY existing dev (PoC, shards 0-5) outputs:
equal_distance_perturbation (units), action_history_causal (H7) and motion_semantics_ablation (H32) records.
Dev A- scenes play the role of F (pi = 1) and dev A+ scenes the role of S with pi = 1 (the dev A+ set is t*-matched,
not the P1-C sample, so theta_S/theta_P here are NOT P1-C estimands). Check: theta_F / theta_S reproduce P0-B's
scene-weighted dev estimates (Recent-Normal A- -0.3979, A+ -0.0335; Direction-Magnitude A- +0.0985, A+ +0.0450).
  python test_dev_p1c.py <scratch_dir>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

O = f"{C.POC}/outputs"
root = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p1c_dev", "dev_run")
out = os.path.join(root, "analysis")
if not os.path.exists(os.path.join(out, "paired_effects.csv")):
    for d in ("selection", "units", "mech/action_history_causal", "mech/motion_semantics_ablation"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    ed = [json.loads(l) for l in open(f"{O}/equal_distance_perturbation/records.jsonl")]
    sel = [{"token": r["token"], "log": r["log"], "stratum": "F" if r["group"] == "A-" else "S", "pi": 1.0, "weight": 1.0,
            "N_S_seg": None, "m_seg": None, "t_star": r["t_star"]} for r in ed]
    den = {}
    for s in sel:
        d = den.setdefault(s["log"], {"scenes": 0, "eligible": 0, "F": 0, "S_eligible": 0, "S_sampled": 0, "not_eligible_by_reason": {}})
        d["scenes"] += 1; d["eligible"] += 1; d["F" if s["stratum"] == "F" else "S_eligible"] += 1
        d["S_sampled"] += s["stratum"] == "S"
    json.dump({"selected": sel, "denominators": den}, open(os.path.join(root, "selection/shard_0.json"), "w"))
    shutil.copy(f"{O}/equal_distance_perturbation/records.jsonl", os.path.join(root, "units/records.jsonl"))
    shutil.copy(f"{O}/action_history_causal/records.jsonl", os.path.join(root, "mech/action_history_causal/records.jsonl"))
    shutil.copy(f"{O}/motion_semantics_ablation/records.jsonl", os.path.join(root, "mech/motion_semantics_ablation/records.jsonl"))
    subprocess.run([sys.executable, os.path.join(HERE, "analyze_p1c.py"), "--run", root, "--out", out], check=True)
pe = pd.read_csv(os.path.join(out, "paired_effects.csv"))
ok = True
for cid, estd, ref in (("RN", "F", -0.39789377289377287), ("RN", "S", -0.033527), ("DM", "F", 0.098520), ("DM", "S", 0.045024)):
    kind = "primary" if estd == "F" else "secondary_theta_S"
    r = pe[(pe.kind == kind) & (pe.contrast == cid) & (pe.estimand == estd) & (pe.variant == "main")].iloc[0]
    good = abs(r.theta - ref) < 5e-4
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] dev {cid} theta_{estd} = {r.theta:+.6f} (P0-B {ref:+.6f}); drives={r.n_clusters} "
          f"scenes={r.n_scenes} units={r.n_units} p_signflip={r.p_signflip:.3g} sf_CI95={r.get('sf_ci95_lo')},{r.get('sf_ci95_hi')}")
dn = pd.read_csv(os.path.join(out, "denominators.csv"))
print(f"denominators: {len(dn)} logs, H7 analysed scenes {dn.H7_scenes_analysed.sum()}, excluded {dn.H7_scenes_excluded_harness.sum()}")
sys.exit(0 if ok else 1)
