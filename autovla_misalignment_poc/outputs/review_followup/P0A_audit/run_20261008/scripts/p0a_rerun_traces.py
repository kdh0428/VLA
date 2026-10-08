#!/usr/bin/env python
"""Turn the tiny GPU re-runs (exp 9 and exp 8, 1 A- scene x 3 units, unchanged original scripts with
--limit 1 --max-alts 2) into intervention traces. CPU only."""
from __future__ import annotations

import json
import os
import sys

POC = "/root/VLA/autovla_misalignment_poc"
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(POC, "scripts"))
from analyze_action_history import unit_metrics  # noqa: E402

N_ACT = 10
R = os.path.join(RUN, "rerun_tiny")
out = []
base = dict(task_id=None, episode_id=None, seed=0, model="AutoVLA (Qwen2.5-VL-3B, AutoVLA_PDMS_89.ckpt)", control_t=None,
            parser_status="n/a (sampling restricted to 2,048 action rows)", exclusion_reason=None)
repro = []


def ent_key(row):
    return row.get("ent") if "ent" in row else row.get("entropy_steps")


# ---- exp 9
p9 = os.path.join(R, "exp9/records.jsonl")
if os.path.exists(p9):
    win = {"normal": (1, 0), "win1": (1, 1), "win2": (1, 2), "win3": (1, 3), "win4": (1, 4), "gt_history": (1, 99),
           "delay2_len1": (2, 1)}
    for l in open(p9):
        r = json.loads(l)
        t, gt = r["t_star"], r["gt"]
        pri = r.get("prior_action_history") or {}
        for name, (start, length) in win.items():
            row = r["rows"][name]
            acts = row["action_idx"]
            steps = []
            for k in range(t + 1, N_ACT):
                ctx = []
                for j in range(t + 1, k):
                    use = start <= j - t < start + length
                    ctx.append({"pos": j, "token": gt[j] if use else acts[j], "source": "gt" if use else "own"})
                steps.append({"decode_j": k, "context_after_t_star": ctx, "n_gt_in_context": sum(c["source"] == "gt" for c in ctx),
                              "output_token": acts[k], "gt_token": gt[k], "entropy": row["ent"][k - t - 1]})
            um = unit_metrics(acts, row["trajectory_pred"], row["ent"], gt, r["trajectory_gt"], t)
            npos = len([o for o in range(1, N_ACT - t - 1) if start <= o < start + length])
            rep = (pri.get(name) == acts) if name in pri else None
            if rep is not None:
                repro.append({"exp": 9, "token": r["token"], "pert": r["perturbation"], "row": name, "equals_exp7_raw": rep})
            out.append(dict(base, experiment=9, log_id=r["log"], scene_id=r["token"], condition=f"{name} (pert={r['perturbation']})",
                            decode_j=f"{t+1}..9", reference_branch=None if name == "normal" else "GT tokens",
                            perturbed_span=("none" if length == 0 else "all post-t* context" if length >= 99 else
                                            f"fixed context positions t*+{start}..t*+{start+length-1}; GT kept in context at all later steps"),
                            perturbation_requested=npos,
                            perturbation_achieved=sum(1 for o in range(1, N_ACT - t - 1) if start <= o < start + length and acts[t + o] != gt[t + o]),
                            intervention_count=npos,
                            primary_outcome={"amplification": um["amplification"], "recovery_A+": um["recovery"], "fde5_m": um["fde5"]},
                            trace={"t_star": t, "forced_token_t_star": r["forced_token"], "pred_prefix": r["pred_prefix"], "gt_tokens": gt,
                                   "executed_output_tokens": acts, "steps": steps,
                                   "logged_gt_context_offsets_last_step": row.get("gt_context_offsets", [None])[-1] if row.get("gt_context_offsets") else None,
                                   "equals_exp7_raw_same_condition": rep},
                            raw_source=f"rerun_tiny/exp9/records.jsonl token={r['token']} pert={r['perturbation']} rows.{name}",
                            verified_from="raw (new tiny re-run with unchanged script; NOT the original exp 9 run)"))

# ---- exp 8
p8 = os.path.join(R, "exp8/records.jsonl")
if os.path.exists(p8):
    for l in open(p8):
        r = json.loads(l)
        t, gt = r["t_star"], r["gt"]
        rows = r["rows"]
        pri = r.get("prior_action_history") or {}
        for name in ("normal", "gt_history", "recent_gt", "reverse@emb", "reverse@L16", "patch_full@emb", "patch_full@L16"):
            row = rows[name]
            acts = row["action_idx"]
            normal = rows["normal"]["action_idx"]
            base_gt = name in ("gt_history",) or name.startswith("reverse")
            steps = []
            for k in range(t + 1, N_ACT):
                if base_gt:
                    ctx = [{"pos": j, "token": gt[j], "source": "gt"} for j in range(t + 1, k)]
                elif name == "recent_gt":
                    ctx = [{"pos": j, "token": acts[j], "source": "own"} for j in range(t + 1, k - 1)] + \
                          ([{"pos": k - 1, "token": gt[k - 1], "source": "gt"}] if k - 1 > t else [])
                else:
                    ctx = [{"pos": j, "token": acts[j], "source": "own"} for j in range(t + 1, k)]
                patch = None
                if "@" in name and k >= t + 2:
                    layer = name.split("@")[1]
                    if name.startswith("reverse"):
                        patch = {"position": k - 1, "layer": layer, "source_row": "normal",
                                 "source_row_token_at_position": normal[k - 1], "this_row_context_token_at_position": gt[k - 1],
                                 "this_row_own_output_at_position": acts[k - 1]}
                    else:
                        patch = {"position": k - 1, "layer": layer, "source_row": "gt_history",
                                 "source_row_token_at_position": gt[k - 1], "this_row_context_token_at_position": acts[k - 1]}
                steps.append({"decode_j": k, "context_after_t_star": ctx, "hidden_patch": patch, "output_token": acts[k], "gt_token": gt[k],
                              "entropy": row["ent"][k - t - 1]})
            um = unit_metrics(acts, row["trajectory_pred"], row["ent"], gt, r["trajectory_gt"], t)
            n_app = max(0, N_ACT - t - 2) if name not in ("normal",) else 0
            rep = (pri.get(name) == acts) if name in pri else None
            if rep is not None:
                repro.append({"exp": 8, "token": r["token"], "pert": r["perturbation"], "row": name, "equals_exp7_raw": rep})
            ach = None
            if name.startswith("reverse"):
                ach = sum(1 for k in range(t + 2, N_ACT) if normal[k - 1] != gt[k - 1])        # steps where the injected Normal token differs from GT
            out.append(dict(base, experiment=8, log_id=r["log"], scene_id=r["token"], condition=f"{name} (pert={r['perturbation']})",
                            decode_j=f"{t+1}..9",
                            reference_branch=("separate Normal row of the same batch (source of the patched state)" if name.startswith("reverse")
                                              else "gt_history row (source)" if name.startswith("patch") else
                                              "GT tokens" if name in ("gt_history", "recent_gt") else None),
                            perturbed_span=("hidden state of span position k-1 at one layer, every step k>=t*+2" if "@" in name else
                                            "context" if name != "normal" else "none"),
                            perturbation_requested=n_app, perturbation_achieved=ach, intervention_count=n_app,
                            primary_outcome={"amplification": um["amplification"], "recovery_A+": um["recovery"], "fde5_m": um["fde5"]},
                            trace={"t_star": t, "forced_token_t_star": r["forced_token"], "pred_prefix": r["pred_prefix"], "gt_tokens": gt,
                                   "executed_output_tokens": acts, "normal_row_output_tokens": normal, "steps": steps,
                                   "steps_log_from_script": r.get("steps"), "equals_exp7_raw_same_condition": rep},
                            raw_source=f"rerun_tiny/exp8/records.jsonl token={r['token']} pert={r['perturbation']} rows.{name}",
                            verified_from="raw (new tiny re-run with unchanged script; NOT the original exp 8 run)"))

with open(os.path.join(RUN, "work", "rerun_traces.jsonl"), "w") as f:
    for x in out:
        f.write(json.dumps(x) + "\n")
json.dump(repro, open(os.path.join(RUN, "work", "rerun_repro.json"), "w"), indent=1)
print(len(out), "traces")
for x in repro:
    print(x)
for x in out:
    print(x["experiment"], x["condition"], x["primary_outcome"], x["trace"]["executed_output_tokens"])
