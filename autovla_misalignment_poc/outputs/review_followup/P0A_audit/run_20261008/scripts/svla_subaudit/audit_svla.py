"""P0-A SpatialVLA audit (exp 34/35). Read-only on outputs; writes only to this scratch dir."""
import json, os, sys, collections, hashlib
import numpy as np
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/spatialvla_protection")
from analyze_svla import table
import analyze_protection as AP

S = os.path.dirname(os.path.abspath(__file__))
O = "/root/VLA/autovla_misalignment_poc/outputs"
E34 = f"{O}/cross_domain_temporal_replication"
E35 = f"{O}/spatialvla_feedback_protection_ablation"
V = table()
OUT = {}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

for p in (f"{E34}/closed_loop/episodes.jsonl", f"{E34}/token_level/units.jsonl", f"{E35}/rollouts/episodes.jsonl",
          f"{E35}/rollouts_ext/episodes.jsonl", f"{E35}/pooled_seed0_119/rollouts/episodes.jsonl"):
    OUT.setdefault("sha256", {})[p] = sha(p)

# ---------------- exp 34 closed loop -----------------
eps34 = [json.loads(l) for l in open(f"{E34}/closed_loop/episodes.jsonl")]
E = {(r["cond"], r["task"], r["seed"]): r for r in eps34}
# frame source: token-level R vs each condition's exec chunk at the frame's t
units_lines = [json.loads(l) for l in open(f"{E34}/token_level/units.jsonl")]
match = collections.Counter(); tot = 0
for f in units_lines:
    t = int(f["frame"].split("/")[1][:3]); tot += 1
    for c in sorted({r["cond"] for r in eps34}):
        ep = E[(c, f["task"], f["seed"])]
        if t < len(ep["rec"]) and np.array_equal(np.array(ep["rec"][t]["exec"]), np.array(f["R"])):
            match[c] += 1
OUT["e34_frame_source_R_equals_exec_rate"] = {c: match[c] / tot for c in sorted({r["cond"] for r in eps34})}
OUT["e34_n_frames"] = tot
OUT["e34_frames_t_values"] = sorted({int(f["frame"].split("/")[1][:3]) for f in units_lines})
# closed-loop intervention counts and achieved perturbation
T_ids = None
def norm_of(tok): return np.asarray(V[int(tok)])
iv = collections.defaultdict(list)
for r in eps34:
    c = r["cond"]
    if c.startswith("natural"):
        continue
    n_int = 0; n_eff = 0; dists = []
    for x in r["rec"]:
        if 8 <= x["t"] < 24:
            n_int += 1
            g = x["nat1"]; ex1 = x["exec"][0][0]; cx1 = (x["ctx"][0][0] if x["ctx"] else ex1)
            p = cx1 if c.startswith(("feedback", "reverse")) else ex1
            dists.append(float(np.linalg.norm(norm_of(p) - norm_of(g)))); n_eff += int(p != g)
        else:
            assert x["ctx"] is None and x["exec"][0][0] == x["nat1"], (c, x["t"])
    iv[c].append((n_int, n_eff, float(np.mean(dists)), float(np.min(dists)), float(np.max(dists))))
OUT["e34_closed_loop_interventions"] = {c: {"chunks_in_window_mean": float(np.mean([a[0] for a in v])),
                                            "effective_p_ne_g_mean": float(np.mean([a[1] for a in v])),
                                            "achieved_norm_dist_mean": float(np.mean([a[2] for a in v])),
                                            "achieved_norm_dist_min": float(min(a[3] for a in v)),
                                            "achieved_norm_dist_max": float(max(a[4] for a in v)), "n_eps": len(v)}
                                        for c, v in iv.items()}
# command proxy vs actual: exp34 records no TCP pose
OUT["e34_has_tcp_pose"] = any("pose" in x for r in eps34[:5] for x in r["rec"])
# finish order (natural_mix batching)
order = [r["cond"] for r in eps34]
OUT["e34_finish_order_blocks"] = [(c, i) for i, c in enumerate(order) if i == 0 or order[i - 1] != c]

# ---------------- exp 34 token level -----------------
ROWS = ["normal", "recent_ref", "full_ref", "win1", "reverse"]
n_units = 0; same_s3 = 0; rev4_eq_norm4 = 0; full4_eq_R4 = 0; empty_frames = 0; dropped = 0
dcount = collections.Counter()
for f in units_lines:
    if not f["units"]:
        empty_frames += 1
    dcount["n_units_kept"] += len(f["units"])
    for u in f["units"]:
        n_units += 1
        rw = u["rows"]
        s3 = [tuple(rw[k][2]) for k in ("recent_ref", "full_ref", "win1", "reverse")]
        same_s3 += len(set(s3)) == 1
        rev4_eq_norm4 += rw["reverse"][3][0] == rw["normal"][3][0]
        full4_eq_R4 += rw["full_ref"][3][0] == f["R"][3][0]
        dcount[(u["d"], u["dir"])] += 1
OUT["e34_token"] = {"n_frames": len(units_lines), "frames_without_units": empty_frames, "n_units": n_units,
                    "units_by_d_dir": {f"{k[0]}_{k[1]}": v for k, v in dcount.items() if k != "n_units_kept"},
                    "max_possible_units": 12 * len(units_lines),
                    "rate_step3_identical_across_recent_full_win1_reverse": same_s3 / n_units,
                    "rate_reverse_step4_trans_eq_normal_step4": rev4_eq_norm4 / n_units,
                    "rate_full_ref_step4_trans_eq_R_step4": full4_eq_R4 / n_units}

def row_metrics(R, S):
    r = np.array([V[int(t)] for t in np.array(R)[:, 0]]); v = np.array([V[int(t)] for t in np.array(S)[:, 0]])
    inj = np.linalg.norm(v[0] - r[0]); add = np.linalg.norm((v[1:] - r[1:]).sum(0))
    return {"D": round(float(np.linalg.norm(v[1:] - r[1:], axis=1).sum()), 5), "A": round(float(add / max(inj, 1e-9)), 5),
            "amp": int(add / max(inj, 1e-9) >= 1), "rec": int(np.array_equal(np.array(S)[1:, 0], np.array(R)[1:, 0])),
            "dev4": round(float(np.linalg.norm(v[3] - r[3])), 5)}

traces = []
pick_lines = [0, len(units_lines) // 2, len(units_lines) - 1]
for li in pick_lines:
    f = units_lines[li]
    us = [u for u in f["units"] if u["d"] == 0.3 and u["dir"] == "opposite"] or f["units"]
    if not us:
        continue
    u = us[0]; R = f["R"]; rw = u["rows"]
    S1p, own2, N3 = rw["normal"][0], rw["normal"][1], rw["normal"][2]
    A3 = rw["full_ref"][2]
    ctx = {"normal": {"step2": [S1p], "step3": [S1p, own2], "step4": [S1p, own2, N3]},
           "recent_ref": {"step2": [S1p], "step3": [S1p, R[1]], "step4": [S1p, own2, R[2]]},
           "full_ref": {"step2": [S1p], "step3": [S1p, R[1]], "step4": [S1p, R[1], R[2]]},
           "win1": {"step2": [S1p], "step3": [S1p, R[1]], "step4": [S1p, R[1], A3]},
           "reverse": {"step2": [S1p], "step3": [S1p, R[1]], "step4": [S1p, R[1], N3]}}
    for row in ROWS:
        nrep = {"normal": 0, "recent_ref": 2, "full_ref": 3, "win1": 1, "reverse": 2}[row]
        traces.append({
            "experiment": 34, "stage": "token_level", "log_id": None, "scene_id": f["frame"], "task_id": f["task"],
            "episode_id": f"{f['task']}_s{f['seed']}", "seed": f["seed"], "model": "SpatialVLA-4B (spatialvla-4b-224-sft-fractal, bf16, greedy)",
            "condition": row, "decode_j": [2, 3, 4], "control_t": int(f["frame"].split("/")[1][:3]),
            "reference_branch": "R = unperturbed greedy chunk of the same frame (model's own natural decoding; NOT ground truth)",
            "perturbed_span": "step-1 translation token (1 token of 12); step-1 rot/grip and steps 2-4 regenerated",
            "perturbation_requested": {"d_norm": u["d"], "dir": u["dir"]},
            "perturbation_achieved": {"p": u["p"], "g": R[0][0], "dist_norm": u["dist"]},
            "intervention_count": nrep,
            "intervention_count_note": "number of context STEP slots (3 tokens each) replaced by R or by the Normal-branch step 3 across the step-3 and step-4 decodes; step-1 perturbation p itself is never corrected",
            "parser_status": "n/a (token ids; out-of-range translation ids clipped by decode)", "exclusion_reason": None,
            "primary_outcome": row_metrics(R, rw[row]),
            "trace": {"R_chunk": R, "output_chunk": rw[row], "context_fed_per_step_reconstructed": ctx[row],
                      "reverse_source": ("N3 = step 3 copied from the separate Normal row (svla_offline.py:111); the reverse row's own step-3 output is A3"
                                         if row == "reverse" else None),
                      "kv": "prompt prefilled once per decode_step call; context steps re-fed token by token through a fresh cache each call (spatialvla_core.py decode_step); no stale KV of replaced tokens"},
            "raw_source": f"{E34}/token_level/units.jsonl line {li + 1}, units[{f['units'].index(u)}].rows.{row}",
            "verified_from": "raw (outputs) + code (context reconstruction from svla_offline.py:92-123; context not stored)"})
# closed-loop exp 34 traces: same episode, 3 opposite modes
for c in ("feedback_opposite", "corrected_opposite", "reverse_opposite"):
    ep = E[(c, "google_robot_pick_coke_can", 0)]; nat = E[("natural", "google_robot_pick_coke_can", 0)]
    steps = []
    for x in ep["rec"][7:11]:
        steps.append({"t": x["t"], "nat1_g": x["nat1"], "exec_step1": x["exec"][0], "ctx_step1": (x["ctx"][0] if x["ctx"] else x["exec"][0]),
                      "exec_chunk": x["exec"], "env_action": x["action"]})
    n_int = sum(1 for x in ep["rec"] if 8 <= x["t"] < 24)
    traces.append({"experiment": 34, "stage": "closed_loop", "log_id": None, "scene_id": None, "task_id": "google_robot_pick_coke_can",
                   "episode_id": "google_robot_pick_coke_can_s0", "seed": 0, "model": "SpatialVLA-4B", "condition": c,
                   "decode_j": "step-1 translation token of every chunk", "control_t": [8, 23],
                   "reference_branch": "g = this episode's own natural greedy step-1 token at the same control step (same observation, not the separate natural episode)",
                   "perturbed_span": "1 token (step-1 translation) per chunk; chunk generated every step (16 chunks in window)",
                   "perturbation_requested": {"d_norm": 0.3, "dir": "opposite"},
                   "perturbation_achieved": None, "intervention_count": n_int, "parser_status": None, "exclusion_reason": None,
                   "primary_outcome": {"success": ep["success"], "natural_success": nat["success"],
                                       "cmd_endpoint_div_m_vs_natural": float(np.linalg.norm(np.cumsum([y["action"][:3] for y in ep["rec"]], 0)[-1] - np.cumsum([y["action"][:3] for y in nat["rec"]], 0)[-1]))},
                   "trace": {"steps_t7_to_t10": steps, "note": "exp 34 stores no TCP pose; trajectory metric is cumulative commanded world_vector"},
                   "raw_source": f"{E34}/closed_loop/episodes.jsonl cond={c} task=pick_coke_can seed=0 rec[7:11]",
                   "verified_from": "raw"})
for tr in traces:
    if tr["stage"] == "closed_loop":
        g = [s["nat1_g"] for s in tr["trace"]["steps_t7_to_t10"] if s["t"] >= 8]
        pp = [(s["ctx_step1"][0] if tr["condition"].startswith(("feedback", "reverse")) else s["exec_step1"][0]) for s in tr["trace"]["steps_t7_to_t10"] if s["t"] >= 8]
        tr["perturbation_achieved"] = {"t8_10_dist_norm": [round(float(np.linalg.norm(norm_of(a) - norm_of(b))), 4) for a, b in zip(pp, g)]}

# ---------------- exp 35 -----------------
eps35 = [json.loads(l) for l in open(f"{E35}/rollouts/episodes.jsonl")]
ext = [json.loads(l) for l in open(f"{E35}/rollouts_ext/episodes.jsonl")]
pool = [json.loads(l) for l in open(f"{E35}/pooled_seed0_119/rollouts/episodes.jsonl")]
K35 = {(r["cond"], r["task"], r["seed"]): r for r in eps35}
KX = {(r["cond"], r["task"], r["seed"]): r for r in ext}
KP = {(r["cond"], r["task"], r["seed"]): r for r in pool}
same = 0; diff = 0; missing = 0
for k, r in KP.items():
    src = K35.get(k) or KX.get(k)
    if src is None:
        missing += 1
    elif src["success"] == r["success"] and src["rec"][-1]["pose"] == r["rec"][-1]["pose"]:
        same += 1
    else:
        diff += 1
OUT["e35_pooled_equals_rollouts_plus_ext"] = {"same": same, "diff": diff, "missing": missing, "n_pooled": len(KP)}
# intervention counts per config/mode, achieved dist, exposure
ivc = collections.defaultdict(list)
for r in eps35 + ext:
    cfg, mode = r["cond"].split(":")
    if mode.startswith("natural"):
        continue
    n_plan = 0; n_eff = 0; ds = []
    for x in r["rec"]:
        if "exec" in x and 8 <= x["t"] < 24:
            n_plan += 1
            g = x["nat1"]; ex1 = x["exec"][0][0]; cx1 = x["ctx"][0][0] if x["ctx"] else ex1
            p = cx1 if mode in ("feedback", "reverse") else ex1
            ds.append(float(np.linalg.norm(norm_of(p) - norm_of(g)))); n_eff += int(p != g)
    ivc[(cfg, mode)].append((n_plan, n_eff, np.mean(ds)))
OUT["e35_interventions"] = {f"{c}:{m}": {"chunks_in_window": sorted(set(a[0] for a in v)), "effective_mean": float(np.mean([a[1] for a in v])),
                                         "achieved_norm_dist_mean": float(np.mean([a[2] for a in v])), "n_eps": len(v)} for (c, m), v in sorted(ivc.items())}
# exposure (weight of age>=1 predictions), from recorded ages
expo = collections.defaultdict(list)
for r in eps35:
    cfg, mode = r["cond"].split(":")
    if mode != "natural":
        continue
    E_ = int(cfg.split("e")[1])
    for x in r["rec"]:
        a = x["ages"]; n = len(a); w = np.exp(0.8 * np.arange(n)); w /= w.sum()
        expo[cfg].append(float(sum(wi for wi, ai in zip(w, a) if ai >= 1)))
OUT["e35_exposure_all_steps_natural"] = {c: float(np.mean(v)) for c, v in expo.items()}
# command proxy vs actual TCP: natural episodes, per config
cv = collections.defaultdict(list)
for r in eps35:
    cfg, mode = r["cond"].split(":")
    if mode != "natural":
        continue
    cmd = np.cumsum([x["action"][:3] for x in r["rec"]], 0)[-1]
    tcp = np.array(r["rec"][-1]["pose"][:3]) - np.array(r["pose0"][:3])
    cv[cfg].append((float(np.linalg.norm(cmd)), float(np.linalg.norm(tcp)), float(np.linalg.norm(cmd - tcp))))
OUT["e35_cmd_endpoint_vs_tcp_displacement_natural"] = {c: {"cmd_norm_mean_m": float(np.mean([a[0] for a in v])), "tcp_disp_mean_m": float(np.mean([a[1] for a in v])),
                                                         "abs_vec_diff_mean_m": float(np.mean([a[2] for a in v]))} for c, v in cv.items()}
# paired: traj_div (command proxy) vs final TCP err, r1e4 R_vs_N
keys = sorted({(r["task"], r["seed"]) for r in eps35})
for cfg in ("r1e4", "r4e1"):
    td, fe = [], []
    for k in keys:
        m = AP.pair_metrics(K35[(f"{cfg}:reverse", *k)], K35[(f"{cfg}:natural", *k)])
        td.append(m["traj_div"]); fe.append(m["final_err"])
    OUT.setdefault("e35_cmd_vs_tcp_pair_R_vs_N", {})[cfg] = {"traj_div_cmd_mean_m": float(np.mean(td)), "final_tcp_err_mean_m": float(np.mean(fe)),
                                                            "pearson_r": float(np.corrcoef(td, fe)[0, 1])}
# exp34 reverse_opposite traj_div vs its closed-loop structure: no pose -> unverifiable
# traces exp 35
picks = [("r1e4:reverse", "google_robot_pick_coke_can", 0, K35), ("r4e1:reverse", "google_robot_move_near", 0, K35),
         ("r4e1:corrected", "google_robot_pick_coke_can", 40, KX)]
for cond, task, seed, src in picks:
    ep = src[(cond, task, seed)]; cfg, mode = cond.split(":")
    natk = (f"{cfg}:natural", task, seed); nat = src[natk]
    m = AP.pair_metrics(ep, nat)
    extra = {}
    if mode == "corrected":
        fb = src[(f"{cfg}:feedback", task, seed)]; mfc = AP.pair_metrics(fb, ep)
        extra = {"F_vs_C_mean_err_m": mfc["mean_err"], "F_vs_C_final_err_m": mfc["final_err"], "F_vs_C_realign_t": mfc["realign_t"], "feedback_success": fb["success"]}
    steps = []
    for x in ep["rec"][7:14]:
        s = {"t": x["t"], "ages": x["ages"], "preds_world": x["preds"], "env_action": x["action"], "tcp_pose": x["pose"]}
        if "exec" in x:
            s.update({"nat1_g": x["nat1"], "exec_step1": x["exec"][0], "ctx_step1": (x["ctx"][0] if x["ctx"] else x["exec"][0])})
        steps.append(s)
    n_int = sum(1 for x in ep["rec"] if "exec" in x and 8 <= x["t"] < 24)
    traces.append({"experiment": 35, "stage": "closed_loop", "log_id": None, "scene_id": None, "task_id": task, "episode_id": f"{task}_s{seed}",
                   "seed": seed, "model": "SpatialVLA-4B", "condition": cond, "decode_j": "step-1 translation token of every chunk generated at t in [8,24)",
                   "control_t": [8, 23], "reference_branch": "g = this episode's own natural greedy step-1 token at that control step; paired outcome reference = separate natural episode of same config/task/seed",
                   "perturbed_span": f"1 token per generated chunk; K={cfg[1]} -> chunks generated every {cfg[1]} step(s)",
                   "perturbation_requested": {"d_norm": 0.3, "dir": "opposite"}, "perturbation_achieved": None,
                   "intervention_count": n_int, "parser_status": None, "exclusion_reason": None,
                   "primary_outcome": {"success": ep["success"], "natural_success": nat["success"], "vs_natural_mean_tcp_err_m(t>=8)": m["mean_err"],
                                       "vs_natural_final_tcp_err_m": m["final_err"], "vs_natural_cmd_traj_div_m": m["traj_div"], "realign_t": m["realign_t"], **extra},
                   "trace": {"steps_t7_to_t13": steps},
                   "raw_source": f"{E35}/{'rollouts' if src is K35 else 'rollouts_ext'}/episodes.jsonl cond={cond} task={task} seed={seed} rec[7:14]",
                   "verified_from": "raw"})
    pp = [(s["ctx_step1"][0] if mode in ("feedback", "reverse") else s["exec_step1"][0], s["nat1_g"]) for s in steps if "nat1_g" in s and s["t"] >= 8]
    traces[-1]["perturbation_achieved"] = {"dist_norm_first_chunks": [round(float(np.linalg.norm(norm_of(a) - norm_of(b))), 4) for a, b in pp]}

with open(os.path.join(S, "intervention_trace_svla.jsonl"), "w") as fo:
    for tr in traces:
        fo.write(json.dumps(tr, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
json.dump(OUT, open(os.path.join(S, "audit_out.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(OUT, indent=1, default=str)[:6000])
