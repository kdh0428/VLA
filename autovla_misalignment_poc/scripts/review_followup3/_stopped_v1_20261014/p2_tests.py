#!/usr/bin/env python
"""P2 unit tests (CPU).  Writes outputs/review_followup3/P2/tests_output.txt.  Exit code 1 on any failure.
Run:  nice -n 19 /root/miniforge3/envs/autovla/bin/python p2_tests.py"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "review_followup", "p1ab"))
import p1_common as C  # noqa: E402
import p2_kinematics as K  # noqa: E402

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P2"
P1A = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1A_gt_free_reference/run_20261008"
C1U = "/root/VLA/autovla_misalignment_poc/outputs/review_followup2/C_baseline_window/run_20261013/c1_units_counterfactual.parquet"
buf = io.StringIO()
fails = []


def check(name, val, tol):
    ok = bool(val <= tol)
    line = f"[{'PASS' if ok else 'FAIL'}] {name}: max err {val:.3e} (tol {tol:.0e})"
    print(line); buf.write(line + "\n")
    if not ok:
        fails.append(name)


def herr(a, b):
    return float(np.abs(np.c_[a[:, :2] - b[:, :2], K.wrap(a[:, 2] - b[:, 2])]).max())


rng = np.random.default_rng(20261014)
cb, disp, dyaw = C.codebook_geometry()

# T1 round trip
e = 0.0
for _ in range(500):
    inc = np.c_[rng.uniform(0, 8, 10), rng.normal(0, .3, 10), rng.normal(0, .2, 10)]
    start = (rng.normal(), rng.normal(), rng.uniform(-3, 3))
    e = max(e, float(np.abs(K.body_increments(K.compose(inc, start), start) - inc).max()))
check("T1 compose <-> body_increments round trip (random SE(2) increments)", e, 1e-9)

# T2 tokens_to_poses vs rollout_np
e = 0.0
for _ in range(500):
    acts = rng.integers(0, 2048, 10)
    e = max(e, herr(K.tokens_to_poses(disp, dyaw, acts), C.rollout_np(cb, acts)))
check("T2 tokens_to_poses (SE(2) chain) vs p1_common.rollout_np (decoder copy)", e, 1e-9)

# T3 CM reproduction on real units (all perturbed units)
recs = []
for path in (f"{P1A}/raw/records.jsonl", f"{P1A}/raw_fixedpos/records.jsonl"):
    for line in open(path):
        r = json.loads(line)
        if r["perturbation_info"]["token_p"] != r["ref_tokens"][r["t_p"]]:
            recs.append({"unit_id": r["unit_id"], "t": r["t_p"], "ref": r["ref_tokens"], "p": r["perturbation_info"]["token_p"],
                         "cm": np.asarray(r["baselines"]["CM"]["traj_xyh"], float)})
U0 = pd.read_parquet(C1U).set_index("unit_id")
e_dec = e_c1 = 0.0
for x in recs:
    t, ref, p = x["t"], x["ref"], x["p"]
    pre_p = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p])
    pre_0 = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [ref[t]])
    cmp_, cm0 = K.bl_cm(pre_p, t), K.bl_cm(pre_0, t)
    e_dec = max(e_dec, herr(cmp_, x["cm"]))
    e_c1 = max(e_c1, abs(float(np.linalg.norm(cmp_[9, :2] - cm0[9, :2])) - U0.loc[x["unit_id"], "cm_end"]))
check(f"T3a bl_cm(prefix) vs stored GPU-decoder CM trajectory ({len(recs)} units)", e_dec, 5e-5)
check("T3b CM endpoint change from bl_cm vs C1 cm_end column", e_c1, 5e-5)

# T4 CTRV-K1 (no clip) == CM ; T6 CTRA with a = 0 == CM
e4 = e6 = 0.0
for x in recs[:2000]:
    t, ref, p = x["t"], x["ref"], x["p"]
    pre = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p])
    e4 = max(e4, herr(K.bl_ctrv(pre, t, K=1, omega_clip=np.inf)[0], K.bl_cm(pre, t)))
    if t == 0:
        e6 = max(e6, herr(K.bl_ctra(pre, t, omega_clip=np.inf)[0], K.bl_cm(pre, t)))
# CTRA a = 0 for t > 0: build a prefix whose last two segments have equal speed (same token repeated)
for _ in range(300):
    a = int(rng.integers(0, 2048)); t = int(rng.integers(1, 9))
    pre = K.tokens_to_poses(disp, dyaw, list(rng.integers(0, 2048, t - 1)) + [a, a])
    if np.hypot(*disp[a]) * 2 > K.SPEED_MAX:
        continue
    e6 = max(e6, herr(K.bl_ctra(pre, t, omega_clip=np.inf)[0], K.bl_cm(pre, t)))
check("T4 CTRV with K=1 and no clip == CM (real units)", e4, 1e-9)
check("T6 CTRA with a = 0 (t=0 fallback, or equal last-two speeds) == CM", e6, 1e-9)

# T5 omega -> 0 limits
errs = []
for eps in (1e-2, 1e-4, 1e-6, 1e-8):
    inc = (1.3, 0.0, eps)
    disc = K.compose([inc] * 9, (0.4, -0.2, 0.3))
    cv = K.compose([(1.3, 0.0, 0.0)] * 9, (0.4, -0.2, 0.3))
    cont = K.ctrv_continuous((0.4, -0.2), 0.3, 2.6, eps / 0.5, 4.5)
    contcv = K.ctrv_continuous((0.4, -0.2), 0.3, 2.6, 0.0, 4.5)
    errs.append((eps, float(np.abs(disc[-1, :2] - cv[-1, :2]).max()), float(np.abs(cont[:2] - contcv[:2]).max())))
line = "T5 omega->0 (eps, discrete-CTRV vs CV endpoint err, closed-form CTRV vs CV err): " + "; ".join(f"{a:.0e}: {b:.2e}, {c:.2e}" for a, b, c in errs)
print(line); buf.write(line + "\n")
check("T5a discrete CTRV -> CV as omega -> 0 (eps = 1e-8)", errs[-1][1], 1e-6)
check("T5b closed-form CTRV -> CV as omega -> 0 (eps = 1e-8)", errs[-1][2], 1e-6)
check("T5c error decreases monotonically with eps", float(not all(errs[i][1] > errs[i + 1][1] for i in range(3))), 0)
e = 0.0
for _ in range(300):
    t = int(rng.integers(0, 9))
    inc = [(rng.uniform(0, 5), rng.normal(0, .2), rng.normal(0, .1)) for _ in range(t)] + [(rng.uniform(0, 5), rng.normal(0, .2), 0.0)]
    pre = K.compose(inc)
    e = max(e, float(np.abs(K.bl_cv0(pre, t)[:, :2] - K.bl_ctrv(pre, t, K=1, omega_clip=np.inf)[0][:, :2]).max()))
check("T5d CV0 == CTRV-K1 when the last yaw increment w_t = 0", e, 1e-9)

# T7 no-future invariance
e = 0.0
for x in recs[:500]:
    t, ref, p = x["t"], x["ref"], x["p"]
    full = K.tokens_to_poses(disp, dyaw, list(ref[:t]) + [p] + list(ref[t + 1:]))
    garb = full.copy(); garb[t + 1:] = rng.normal(0, 50, garb[t + 1:].shape)
    for f in (K.bl_cm, K.bl_cv0, lambda a, b: K.bl_ctrv(a, b)[0], lambda a, b: K.bl_ctra(a, b)[0]):
        e = max(e, float(np.abs(f(full, t) - f(garb, t)).max()))
check("T7 baselines invariant to poses after t (no future / evaluation-suffix leakage)", e, 0)

# T8 C1 reproduction flag
rep = json.load(open(f"{OUT}/c1_reproduction_check.json"))
check("T8 C1 reproduced exactly (max abs table diff)", rep["table_numeric_maxabs_diff"] if rep["reproduced_exactly"] else 1.0, 1e-9)

buf.write(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}\n")
open(f"{OUT}/tests_output.txt", "w").write(buf.getvalue())
print("ALL PASS" if not fails else f"FAILURES {fails}")
sys.exit(1 if fails else 0)
