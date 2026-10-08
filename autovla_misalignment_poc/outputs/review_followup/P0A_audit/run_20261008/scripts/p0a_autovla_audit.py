#!/usr/bin/env python
"""
P0-A audit (CPU only, read-only on existing outputs). AutoVLA experiments 5-9, 32.

Reads raw jsonl that still exist on disk, recomputes the headline A- numbers with the ORIGINAL
metric functions (imported, not re-implemented), builds the denominator flow and extracts
intervention traces. Writes only into this run directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

POC = "/root/VLA/autovla_misalignment_poc"
O = os.path.join(POC, "outputs")
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(RUN, "work")
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, os.path.join(POC, "scripts"))
from analyze_action_history import unit_metrics, a_eval, AMP_FDE  # noqa: E402

N_ACT = 10


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path)]


out = {"files": {}}
for p in ["full_extract/records.jsonl", "equal_distance_perturbation/records.jsonl", "equal_distance_perturbation/rows.jsonl",
          "action_history_causal/records.jsonl", "action_history_causal/units.jsonl", "motion_semantics_ablation/records.jsonl",
          "prev_action_identity_decomposition/records.jsonl"]:
    fp = os.path.join(O, p)
    out["files"][p] = {"sha256": sha(fp), "bytes": os.path.getsize(fp), "mtime": os.path.getmtime(fp)} if os.path.exists(fp) else None

# ------------------------------------------------------------------ denominator flow: full_extract -> exp5 / exp6 sets
flow = []
fe_tot = fe_noN = fe_cot = fe_run = fe_short = fe_nomis = 0
mis = []
aminus_all, n_valid_all = [], 0
for line in open(os.path.join(O, "full_extract/records.jsonl")):
    r = json.loads(line)
    fe_tot += 1
    n = r.get("arms", {}).get("N")
    if not n or not n.get("trajectory_pred"):
        fe_noN += 1
        continue
    if n["cot_present"]:
        fe_cot += 1
        continue
    if n["runaway_action_tokens"]:
        fe_run += 1
        continue
    if len(n["trajectory_pred"]) < N_ACT:
        fe_short += 1
        continue
    pred, gt = n["pred_action_idx"][:N_ACT], r["gt_action_idx"][:N_ACT]
    A = a_eval(n["trajectory_pred"], r["trajectory_gt"])
    n_valid_all += 1
    if not A:
        aminus_all.append(r["log_name"])
    t = next((k for k in range(N_ACT) if pred[k] != gt[k]), None)
    if t is None:
        fe_nomis += 1
        continue
    mis.append({"token": r["token"], "log": r["log_name"], "t": t, "A": A})
flow.append({"experiment": "4-5 (full_extract arm N)", "stage": "records in full_extract/records.jsonl", "group": "all", "n_units": fe_tot,
             "n_clusters": None, "unit_type": "scene", "cluster_type": "log"})
flow.append({"experiment": "4-5", "stage": "excluded: no arm-N trajectory (fork failed)", "group": "all", "n_units": fe_noN, "unit_type": "scene"})
flow.append({"experiment": "4-5", "stage": "excluded: CoT present in natural arm", "group": "all", "n_units": fe_cot, "unit_type": "scene"})
flow.append({"experiment": "4-5", "stage": "excluded: runaway action tokens (>10)", "group": "all", "n_units": fe_run, "unit_type": "scene"})
flow.append({"experiment": "4-5", "stage": "excluded: <10 poses", "group": "all", "n_units": fe_short, "unit_type": "scene"})
flow.append({"experiment": "4-5", "stage": "valid natural fast scenes (A label computable)", "group": "all", "n_units": n_valid_all,
             "n_clusters": None, "unit_type": "scene"})
flow.append({"experiment": "4-5", "stage": "A- among valid (label_A false)", "group": "A-", "n_units": len(aminus_all),
             "n_clusters": len(set(aminus_all)), "unit_type": "scene", "cluster_type": "log"})
flow.append({"experiment": "4-5", "stage": "excluded: no token mismatch in 10 steps", "group": "all", "n_units": fe_nomis, "unit_type": "scene"})
for g, flag in (("A-", False), ("A+", True)):
    xs = [m for m in mis if m["A"] == flag]
    flow.append({"experiment": "5 (first_mismatch_causal)", "stage": "scenes with a first mismatch t* (= exp 5 population)", "group": g,
                 "n_units": len(xs), "n_clusters": len({m['log'] for m in xs}), "unit_type": "scene", "cluster_type": "log"})
flow.append({"experiment": "5", "stage": "total exp 5 scenes", "group": "all", "n_units": len(mis),
             "n_clusters": len({m['log'] for m in mis}), "unit_type": "scene", "cluster_type": "log"})
tdist = {g: dict(sorted(Counter(m["t"] for m in mis if m["A"] == (g == "A+")).items())) for g in ("A-", "A+")}

# ------------------------------------------------------------------ exp 6 equal distance
ed = jl(os.path.join(O, "equal_distance_perturbation/records.jsonl"))
ed_by = {r["token"]: r for r in ed}
for g in ("A-", "A+"):
    rs = [r for r in ed if r["group"] == g]
    nunits = sum(1 + len(r["alternatives"]) for r in rs)
    nvalid = sum(sum(1 for c in ["original"] + [f"alt{j}" for j in range(len(r["alternatives"]))] if r["conditions"][c]["valid"]) for r in rs)
    flow.append({"experiment": "6 (equal_distance)", "stage": "scenes picked (A-: all mismatch-A-; A+: 3 per A- at same t*, seeded shuffle)",
                 "group": g, "n_units": len(rs), "n_clusters": len({r['log'] for r in rs}), "unit_type": "scene", "cluster_type": "log"})
    flow.append({"experiment": "6", "stage": "perturbation units original+alt (= units of exp 7-11, 32)", "group": g, "n_units": nunits,
                 "n_clusters": len({r['log'] for r in rs}), "unit_type": "(scene, perturbation)", "cluster_type": "log"})
    flow.append({"experiment": "6", "stage": "of which valid 10-token rollouts in exp 6 (HF generate may leave the action span)",
                 "group": g, "n_units": nvalid, "unit_type": "(scene, perturbation)"})
    flow.append({"experiment": "6", "stage": "alternatives only (Claim 2 denominators)", "group": g,
                 "n_units": sum(len(r["alternatives"]) for r in rs), "unit_type": "(scene, alternative)"})
    nstrict = sum(sum(a["matched_strict"] for a in r["alternatives"]) for r in rs)
    flow.append({"experiment": "6", "stage": "alternatives meeting strict tolerance (matched_strict)", "group": g, "n_units": nstrict,
                 "unit_type": "(scene, alternative)"})

# ------------------------------------------------------------------ exp 7 recompute
ah = jl(os.path.join(O, "action_history_causal/records.jsonl"))
units7 = []
for r in ah:
    t, gt, gtraj = r["t_star"], r["gt"], r["trajectory_gt"]
    for pname, pr in r["perturbations"].items():
        u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": pname, "t": t}
        for c, x in pr["conditions"].items():
            u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], gt, gtraj, t)
            u[c + "_acts"] = x["action_idx"]
        u["stored_rep"] = pr["conditions"]["normal"]["action_idx"] == pr["stored_normal_action_idx"]
        # Recent-GT: number of steps at which the last context token was replaced, and how many replacements changed the token
        acts = pr["conditions"]["recent_gt"]["action_idx"]
        u["recent_applications"] = max(0, N_ACT - t - 2)
        u["recent_effective"] = sum(1 for k in range(t + 2, N_ACT) if acts[k - 1] != gt[k - 1])
        acts_g = pr["conditions"]["gt_history"]["action_idx"]
        u["gth_positions_final"] = max(0, N_ACT - t - 2)          # context positions t*+1..8 that are GT at the last step
        u["gth_effective_final"] = sum(1 for j in range(t + 1, N_ACT - 1) if acts_g[j] != gt[j])
        units7.append(u)
S7 = json.load(open(os.path.join(O, "action_history_causal/summary.json")))


def rate(us, c, m="amplification"):
    v = [float(u[c][m]) for u in us if u[c][m] is not None]
    return float(np.mean(v)) if v else None


def mcn(us, a, b, m="amplification"):
    ga = sum(bool(u[a][m]) and not bool(u[b][m]) for u in us)
    gb = sum(bool(u[b][m]) and not bool(u[a][m]) for u in us)
    return ga, gb


rc = {"exp7": {}}
for g in ("A-", "A+"):
    us = [u for u in units7 if u["group"] == g]
    d = {"n_units": len(us), "n_scenes": len({u['token'] for u in us}), "n_logs": len({u['log'] for u in us})}
    for c in ("normal", "recent_gt", "gt_history", "hist_attn_mask", "recent_attn_mask"):
        d[c] = rate(us, c)
        d[c + "_summary"] = S7["subsets"]["all perturbations"][g]["conditions"][c]["amplification"]["mean"]
    d["recent_minus_normal"] = d["recent_gt"] - d["normal"]
    d["gth_minus_normal"] = d["gt_history"] - d["normal"]
    d["recent_share_of_gth"] = d["recent_minus_normal"] / d["gth_minus_normal"]
    d["mcnemar_recent_vs_normal(cond_only,normal_only)"] = mcn(us, "recent_gt", "normal")
    d["normal_reproduces_exp6_stored_tokens"] = float(np.mean([u["stored_rep"] for u in us]))
    d["recent_applications_dist"] = dict(sorted(Counter(u["recent_applications"] for u in us).items()))
    d["recent_applications_mean"] = float(np.mean([u["recent_applications"] for u in us]))
    d["recent_effective_mean"] = float(np.mean([u["recent_effective"] for u in us]))
    d["recent_effective_zero_units"] = sum(u["recent_effective"] == 0 for u in us)
    d["gth_effective_final_mean"] = float(np.mean([u["gth_effective_final"] for u in us]))
    d["t_star_dist_units"] = dict(sorted(Counter(u["t"] for u in us).items()))
    d["stored_amplified_subset_n"] = None
    rc["exp7"][g] = d
rc["exp7"]["normal_reproduces_exp6_all"] = float(np.mean([u["stored_rep"] for u in units7]))
rc["exp7"]["summary_normal_reproduces"] = S7.get("normal_reproduces_stored_equal_distance_tokens")

# ------------------------------------------------------------------ exp 32 recompute
ms = jl(os.path.join(O, "motion_semantics_ablation/records.jsonl"))
S32 = json.load(open(os.path.join(O, "motion_semantics_ablation/summary.json")))
RC32 = json.load(open(os.path.join(O, "motion_semantics_ablation/row_contrasts.json")))
units32 = []
for r in ms:
    t = r["t_star"]
    u = {"token": r["token"], "log": r["log"], "group": r["group"], "pert": r["perturbation"], "t": t,
         "rep": r.get("reproduces_action_history")}
    for c, x in r["conditions"].items():
        u[c] = unit_metrics(x["action_idx"], x["trajectory_pred"], x["entropy_steps"], r["gt"], r["trajectory_gt"], t)
        u[c + "_acts"] = x["action_idx"]
    units32.append(u)
rc["exp32"] = {}
for g in ("A-", "A+"):
    us = [u for u in units32 if u["group"] == g]
    d = {"n_units": len(us), "n_scenes": len({u['token'] for u in us}), "n_logs": len({u['log'] for u in us})}
    for c in ("normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist", "random_mag_matched"):
        d[c] = rate(us, c)
        d[c + "_summary"] = S32["groups"][g]["conditions"][c]["amplification"]["mean"]
    d["dirwrong_minus_dirok"] = d["dir_wrong_mag_ok"] - d["dir_ok_mag_wrong"]
    d["mcnemar_dirwrong_vs_dirok"] = mcn(us, "dir_wrong_mag_ok", "dir_ok_mag_wrong")
    d["mirror_minus_normal"] = d["mirror_same_dist"] - d["normal"]
    reps = [u["rep"] for u in us if u["rep"] is not None]
    d["reproduces_exp7_normal_tokens"] = float(np.mean([x["normal"] for x in reps])) if reps else None
    d["reproduces_exp7_recent_gt_tokens"] = float(np.mean([x["recent_gt"] for x in reps])) if reps else None
    rc["exp32"][g] = d

# cross check exp7 vs exp32 normal tokens directly
ah7 = {(u["token"], u["pert"]): u["normal_acts"] for u in units7}
m32 = [ah7.get((u["token"], u["pert"])) == u["normal_acts"] for u in units32 if (u["token"], u["pert"]) in ah7]
rc["exp32"]["normal_tokens_equal_exp7_raw_all"] = float(np.mean(m32))
ah7r = {(u["token"], u["pert"]): u["recent_gt_acts"] for u in units7}
rc["exp32"]["recent_gt_tokens_equal_exp7_raw_all"] = float(np.mean([ah7r.get((u["token"], u["pert"])) == u["recent_gt_acts"] for u in units32]))
# unit-level amplification discordance of Normal between exp7 and exp32 (A-)
a7 = {(u["token"], u["pert"]): u["normal"]["amplification"] for u in units7}
dis = [(a7[(u["token"], u["pert"])], u["normal"]["amplification"]) for u in units32 if u["group"] == "A-"]
rc["exp32"]["A-_normal_amp_discordant_units_vs_exp7"] = sum(a != b for a, b in dis)
rc["row_contrasts_json_keys"] = list(RC32.keys())

for g in ("A-", "A+"):
    flow.append({"experiment": "7 (action_history_causal)", "stage": "units analysed (all 7 conditions, no unit exclusion; OOD rule is per condition)",
                 "group": g, "n_units": rc["exp7"][g]["n_units"], "n_clusters": rc["exp7"][g]["n_logs"], "unit_type": "(scene, perturbation)",
                 "cluster_type": "log", "note": f"scenes {rc['exp7'][g]['n_scenes']}"})
    flow.append({"experiment": "32 (motion_semantics_ablation)", "stage": "units analysed (6 rows, errors.jsonl empty)", "group": g,
                 "n_units": rc["exp32"][g]["n_units"], "n_clusters": rc["exp32"][g]["n_logs"], "unit_type": "(scene, perturbation)",
                 "cluster_type": "log", "note": f"scenes {rc['exp32'][g]['n_scenes']}"})

json.dump({"recompute": rc, "t_star_dist_exp5_scenes": tdist, "files": out["files"]}, open(os.path.join(WORK, "autovla_recompute.json"), "w"),
          indent=1, default=str)
json.dump(flow, open(os.path.join(WORK, "autovla_flow.json"), "w"), indent=1)

# ------------------------------------------------------------------ traces
# deterministic pick of 3 A- scenes: (1) t*=0, Normal amplified & Recent-GT not; (2) t*=1 Normal amplified & Recent-GT also amplified
# (or other case); (3) t*>=2. 'original' perturbation only, smallest token id order.
def pick():
    by = defaultdict(dict)
    for u in units7:
        if u["group"] == "A-" and u["pert"] == "original":
            by[u["token"]] = u
    cands = sorted(by.values(), key=lambda u: u["token"])
    c1 = next(u for u in cands if u["t"] == 0 and u["normal"]["amplification"] and not u["recent_gt"]["amplification"])
    c2 = next(u for u in cands if u["t"] == 1 and u["normal"]["amplification"] and u["recent_gt"]["amplification"]) \
        if any(u["t"] == 1 and u["normal"]["amplification"] and u["recent_gt"]["amplification"] for u in cands) else \
        next(u for u in cands if u["t"] == 1 and u["token"] != c1["token"])
    c3 = next(u for u in cands if u["t"] >= 2 and u["log"] not in (c1["log"], c2["log"]))
    return [c1["token"], c2["token"], c3["token"]]


picks = pick()
ah_by = {r["token"]: r for r in ah}
ms_by = {(r["token"], r["perturbation"]): r for r in ms}
traces = []


def ctx_ah(cname, t, gt, own, k):
    """Context span after t* at step k for exp 7 conditions, reconstructed with the code rule
    (action_history_causal.py:207-215). own = this condition's own outputs a_{t*+1..k-1}."""
    if cname == "gt_history":
        src = [("gt", gt[j], j) for j in range(t + 1, k)]
    elif cname == "recent_gt":
        src = [("own", own[j - t - 1], j) for j in range(t + 1, k - 1)] + ([("gt", gt[k - 1], k - 1)] if k - 1 > t else [])
    else:
        src = [("own", own[j - t - 1], j) for j in range(t + 1, k)]
    return src


base_fields = dict(task_id=None, episode_id=None, seed=0, model="AutoVLA (Qwen2.5-VL-3B, AutoVLA_PDMS_89.ckpt)", control_t=None,
                   parser_status="n/a (action tokens; non-action token impossible: sampling restricted to 2,048 action rows)",
                   exclusion_reason=None)

for tok in picks:
    r = ah_by[tok]
    t, gt, pred = r["t_star"], r["gt"], r["pred"]
    pr = r["perturbations"]["original"]
    for cname in ("normal", "recent_gt", "gt_history"):
        x = pr["conditions"][cname]
        acts = x["action_idx"]
        own = acts[t + 1:]
        steps = []
        for k in range(t + 1, N_ACT):
            src = ctx_ah(cname, t, gt, own, k)
            steps.append({"decode_j": k, "forced_at_t_star": acts[t], "context_after_t_star": [{"pos": j, "token": v, "source": s} for s, v, j in src],
                          "n_gt_in_context": sum(s == "gt" for s, _, _ in src), "output_token": acts[k], "gt_token": gt[k],
                          "entropy": x["entropy_steps"][k - t - 1]})
        um = unit_metrics(acts, x["trajectory_pred"], x["entropy_steps"], gt, r["trajectory_gt"], t)
        n_app = 0 if cname == "normal" else (max(0, N_ACT - t - 2))
        eff = None
        if cname == "recent_gt":
            eff = sum(1 for k in range(t + 2, N_ACT) if acts[k - 1] != gt[k - 1])
        elif cname == "gt_history":
            eff = sum(1 for j in range(t + 1, N_ACT - 1) if acts[j] != gt[j])
        traces.append(dict(base_fields, experiment=7, log_id=r["log"], scene_id=tok, condition=cname, decode_j=f"{t+1}..9",
                           reference_branch=("GT tokens (navsim future trajectory -> codebook contour match)" if cname != "normal" else None),
                           perturbed_span=("none" if cname == "normal" else
                                           "context position k-1 only, at every step k>=t*+2 (sliding); earlier positions revert to this branch's own tokens"
                                           if cname == "recent_gt" else "all context positions t*+1..k-1 at every step k"),
                           perturbation_requested=n_app, perturbation_achieved=eff, intervention_count=n_app,
                           primary_outcome={"amplification": um["amplification"], "recovery_A+": um["recovery"], "fde5_m": um["fde5"]},
                           trace={"t_star": t, "prefix_tokens_before_t_star(pred[:t*])": pred[:t], "forced_token_t_star": pr["forced_token"],
                                  "gt_tokens": gt, "executed_output_tokens": acts, "steps": steps,
                                  "kv": "prefix KV (prompt+stub+pred[:t*]) computed once, cropped back to P after every step; the post-t* span is re-encoded each step so no stale K/V of replaced tokens survives (action_history_causal.py:186-225)",
                                  "output_vs_context": "executed trajectory = decode(pred[:t*] + forced + this branch's own sampled tokens); GT tokens enter only the context (action_history_causal.py:236)"},
                           raw_source=f"O/action_history_causal/records.jsonl token={tok} perturbations.original.conditions.{cname}",
                           verified_from="raw (outputs) + code rule (context reconstruction)"))
    # exp 32, same unit
    r32 = ms_by[(tok, "original")]
    for cname in ("normal", "recent_gt", "dir_ok_mag_wrong", "dir_wrong_mag_ok", "mirror_same_dist"):
        x = r32["conditions"][cname]
        acts = x["action_idx"]
        subs = {s["k"]: s for s in x.get("substitutions", [])}
        steps = []
        for k in range(t + 1, N_ACT):
            ctx = [{"pos": j, "token": acts[j], "source": "own"} for j in range(t + 1, k - 1)]
            if k - 1 > t:
                if cname == "normal":
                    ctx.append({"pos": k - 1, "token": acts[k - 1], "source": "own"})
                elif cname == "recent_gt":
                    ctx.append({"pos": k - 1, "token": gt[k - 1], "source": "gt"})
                else:
                    s = subs.get(k)
                    ctx.append({"pos": k - 1, "token": s["sub"] if s else None, "source": "motion_substitute",
                                "own_replaced": s["own"] if s else None, "e_own_m": s["e_own"] if s else None,
                                "d_sub_gt_m": s["d_sub_gt"] if s else None, "dphi_deg": s["dphi_sub_gt_deg"] if s else None,
                                "dmag_m": s["dmag_sub_gt"] if s else None})
            steps.append({"decode_j": k, "context_after_t_star": ctx, "output_token": acts[k], "gt_token": gt[k],
                          "entropy": x["entropy_steps"][k - t - 1]})
        um = unit_metrics(acts, x["trajectory_pred"], x["entropy_steps"], r32["gt"], r32["trajectory_gt"], t)
        n_app = 0 if cname == "normal" else max(0, N_ACT - t - 2)
        traces.append(dict(base_fields, experiment=32, log_id=r32["log"], scene_id=tok, condition=cname, decode_j=f"{t+1}..9",
                           reference_branch=(None if cname == "normal" else "GT token g=gt[k-1]; substitute chosen relative to g and this row's own token o"),
                           perturbed_span=("none" if cname == "normal" else "context position k-1 only, every step k>=t*+2 (sliding, as Recent-GT)"),
                           perturbation_requested=n_app, perturbation_achieved=(len(subs) if cname not in ("normal", "recent_gt") else
                                                                                (sum(1 for k in range(t + 2, N_ACT) if acts[k - 1] != gt[k - 1]) if cname == "recent_gt" else 0)),
                           intervention_count=n_app,
                           primary_outcome={"amplification": um["amplification"], "recovery_A+": um["recovery"], "fde5_m": um["fde5"]},
                           trace={"t_star": t, "forced_token_t_star": r32["forced_token"], "executed_output_tokens": acts, "steps": steps,
                                  "kv": "same harness as exp 7/11 but batched (B=6 rows share one expanded prefix cache, motion_semantics_ablation.py:184-188,236-241)"},
                           raw_source=f"O/motion_semantics_ablation/records.jsonl token={tok} perturbation=original conditions.{cname}",
                           verified_from="raw"))
    # exp 9: structure only (records not on disk)
    for name, start, length in (("normal", 1, 0), ("win1", 1, 1), ("win3", 1, 3), ("gt_history", 1, 99)):
        steps = []
        for k in range(t + 1, N_ACT):
            ctx = []
            for j in range(t + 1, k):
                o_ = j - t
                use = start <= o_ < start + length
                ctx.append({"pos": j, "token": gt[j] if use else None, "source": "gt" if use else "own (not saved)"})
            steps.append({"decode_j": k, "context_after_t_star": ctx, "n_gt_in_context": sum(c["source"] == "gt" for c in ctx),
                          "output_token": None})
        npos = len([o_ for o_ in range(1, N_ACT - t - 1) if start <= o_ < start + length])
        traces.append(dict(base_fields, experiment=9, log_id=r["log"], scene_id=tok, condition=name, decode_j=f"{t+1}..9",
                           reference_branch=(None if name == "normal" else "GT tokens"),
                           perturbed_span=(f"fixed absolute context positions t*+{start}..t*+{start+length-1}, GT persists in context for all later steps" if length and length < 99 else
                                           ("all post-t* context" if length else "none")),
                           perturbation_requested=npos, perturbation_achieved=None, intervention_count=npos,
                           primary_outcome=None,
                           trace={"t_star": t, "gt_tokens": gt, "steps": steps,
                                  "note": "context mask reconstructed from temporal_feedback_window.py:180-191 with this unit's t*/GT (from exp 7 raw); model outputs not saved (records.jsonl absent) -> null"},
                           raw_source="code only (O/temporal_feedback_window has no records.jsonl)", verified_from="code_only"))

# exp 8: debug samples embedded in summary.json (sanity run, 4 scenes)
S8 = json.load(open(os.path.join(O, "prev_action_state_patching/summary.json")))
for smp in S8["sanity"]["debug"]["per_sample"][:3]:
    traces.append(dict(base_fields, experiment=8, log_id=None, scene_id=smp["token"], condition="patch_full@{emb..L35} and reverse@* (debug step)",
                       decode_j=smp["k"], reference_branch="patch_full: source row = gt_history (same batch); reverse@L: target row base = gt_history, source row = normal (separate Normal branch in the same batch)",
                       perturbed_span="hidden state of the LAST span position only (index m-1 = context position k-1), at one layer; applied at every step k>=t*+2",
                       perturbation_requested=None, perturbation_achieved=None, intervention_count=None, primary_outcome=None,
                       trace={"t_star": smp["t_star"], "span_normal(post-t* tokens incl. forced)": smp["span_normal"],
                              "span_gt_history": smp["span_gt"], "kv_logits_checks": smp.get("kv_logits"),
                              "selfpatch_js_vs_normal": smp.get("selfpatch_js_vs_normal"),
                              "emb_patch_vs_recent_gt_logits_L1": smp.get("emb_patch_vs_recent_gt_logits_L1"),
                              "L35_patch_vs_gt_history_logits_L1": smp.get("L35_patch_vs_gt_history_logits_L1"),
                              "patch_checks_emb": smp["patch_checks"].get("emb"), "patch_checks_L16": smp["patch_checks"].get("L16")},
                       raw_source="O/prev_action_state_patching/summary.json sanity.debug.per_sample (from the 4-scene --debug sanity run, sanity/run_meta.json); main-run records.jsonl absent",
                       verified_from="summary (embedded debug record)"))

with open(os.path.join(WORK, "autovla_traces.jsonl"), "w") as f:
    for x in traces:
        f.write(json.dumps(x, default=str) + "\n")
print(json.dumps(rc, indent=1, default=str))
print("picks", picks, "traces", len(traces))
print(json.dumps(flow, indent=0)[:4000])
