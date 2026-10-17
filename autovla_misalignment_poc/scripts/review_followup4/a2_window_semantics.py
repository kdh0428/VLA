#!/usr/bin/env python
"""
A2 (review_followup4): correction-window semantics audit, CPU only, read-only on all inputs.

Evidence layers
  L1  code rule        temporal_feedback_window.py:189-197 (use_gt depends on offset o = j - t* only, not on step k)
  L2  actual-forward traces (spans fed to the model, with per-position source tags) recorded by
        (a) review_followup3 P1 Stage-0 gate, path H9 (p1b_engine.context == temporal_feedback_window rule)
        (b) P1-B main run raw/intervention_trace.jsonl (persistent P_* and transient T_* rows, first 24 units)
  L3  exp 9 stored records (1,408 units): final-step GT offsets + executed tokens -> per-step contexts are
      reconstructed with the L1 rule (a reconstruction, not an observation) and cross-checked against L2.

Outputs (outputs/review_followup4/A2_A3/):
  a2_step_trace_examples.csv   per (source, unit, row, step k, context position j): executed token, context token,
                               GT token, source tag, is_gt, overwritten (context != executed own token)
  a2_exp9_unit_window.csv      per exp 9 A- unit: t*, corrected positions per window, effective overwrites,
                               steps generated after t*, post-release steps (analyzer definition)
  a2_checks.json               all assertion counts
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

POC = "/root/VLA/autovla_misalignment_poc"
OUT = os.path.join(POC, "outputs/review_followup4/A2_A3")
EXP9 = os.path.join(POC, "outputs/review_followup/R8R9_raw_recovery/run_20261008/raw/exp9")
GATE = os.path.join(POC, "outputs/review_followup3/P1/gate_stage0")
P1B_TRACE = os.path.join(POC, "outputs/review_followup/P1B_start_position/run_20261008/raw/intervention_trace.jsonl")
P1B_REC = os.path.join(POC, "outputs/review_followup/P1B_start_position/run_20261008/raw/records.jsonl")
N_ACT = 10
EX_ROWS = ("normal", "win1", "win3", "gt_history", "delay2_len1", "skipfirst2_full")


def h9_rule(row, k, t, own, gt):
    """temporal_feedback_window.py:189-197 verbatim logic -> (context tokens j=t+1..k-1, is_gt flags)"""
    ctx, mask = [], []
    for j in range(t + 1, k):
        o = j - t
        use_gt = row["start"] <= o < row["start"] + row["length"]
        ctx.append(gt[j] if use_gt else own[o - 1])
        mask.append(use_gt)
    return ctx, mask


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(EXP9, "run_meta.json")))
    ROWS = {r["name"]: r for r in meta["rows"]}
    C = defaultdict(Counter)
    ex_rows = []

    # ---------------------------------------------------------------- L3: exp 9 records
    recs = [json.loads(l) for l in open(os.path.join(EXP9, "records.jsonl"))]
    rec_by = {(r["token"], r["perturbation"]): r for r in recs}
    unit_rows = []
    n_ex = 0
    for r in recs:
        t, gt, f = r["t_star"], r["gt"], r["forced_token"]
        for name, row in ROWS.items():
            acts = r["rows"][name]["action_idx"]
            own = acts[t + 1:]
            # final-step context (k = 9) by rule; compare to the stored final-step offsets
            ctx, mask = h9_rule(row, N_ACT - 1, t, own, gt)
            offs = [o + 1 for o, g in enumerate(mask) if g]
            C["exp9"]["rows_checked"] += 1
            C["exp9"]["stored_offsets_eq_rule"] += offs == r["rows"][name]["gt_context_offsets"]
            # persistence: every GT offset present at the last step was GT at every step k where it was in context
            first_seen = {}
            persistent = True
            for k in range(t + 2, N_ACT):
                _, m = h9_rule(row, k, t, own, gt)
                for o, g in enumerate(m):
                    if o in first_seen and first_seen[o] != g:
                        persistent = False
                    first_seen.setdefault(o, g)
            C["exp9"]["gt_flag_constant_over_steps"] += persistent
            if r["group"] == "A-" and name.startswith("win") or (r["group"] == "A-" and name in ("normal", "gt_history")):
                pass
        if r["group"] == "A-":
            u = {"unit_id": f"{r['token']}|{r['perturbation']}", "token": r["token"], "log": r["log"], "t_star": t,
                 "steps_generated_after_tstar": N_ACT - 1 - t,          # a_{t*+1} .. a_9
                 "steps_window_can_affect": max(0, N_ACT - 2 - t),      # a_{t*+2} .. a_9
                 "context_positions_available": max(0, 8 - t)}          # t*+1 .. 8
            for w in (1, 2, 3, 4):
                name = f"win{w}"
                acts = r["rows"][name]["action_idx"]
                n_corr = r["rows"][name]["n_gt_context_positions"]
                eff = sum(acts[t + o] != gt[t + o] for o in range(1, n_corr + 1))  # GT replaced a different own token
                u[f"win{w}_corrected_positions"] = n_corr
                u[f"win{w}_effective_overwrites"] = eff
                u[f"win{w}_post_release_steps"] = max(0, N_ACT - (t + 1 + w + 1))  # analyzer rel = t*+w+2..9
                u[f"win{w}_equals_full"] = int(w >= 8 - t)
            unit_rows.append(u)
        # example step traces (reconstructed) for the first 2 A- units with t* = 1
        if r["group"] == "A-" and t == 1 and n_ex < 2:
            n_ex += 1
            for name in EX_ROWS:
                row = ROWS[name]; acts = r["rows"][name]["action_idx"]; own = acts[t + 1:]
                for k in range(t + 1, N_ACT):
                    ctx, mask = h9_rule(row, k, t, own, gt)
                    full = list(r["pred_prefix"]) + [f] + ctx
                    for jj, (c, g) in enumerate(zip(ctx, mask)):
                        j = t + 1 + jj
                        ex_rows.append({"source": "exp9_records_reconstructed(L3)", "unit_id": f"{r['token']}|{r['perturbation']}",
                                        "t_star": t, "row": name, "step_k": k, "full_action_prefix_0..k-1": " ".join(map(str, full)),
                                        "position_j": j, "executed_own_token": acts[j], "context_token": c, "gt_token": gt[j],
                                        "source_tag": "gt" if g else "own", "is_gt_in_context": int(g),
                                        "overwritten": int(c != acts[j])})
    C["exp9"]["n_records"] = len(recs)

    # ---------------------------------------------------------------- L2a: P1 Stage-0 gate, H9
    gate_cmp = Counter()
    for run in ("run1", "run2"):
        for l in open(os.path.join(GATE, run, "decodes.jsonl")):
            d = json.loads(l)
            if d["path"] != "H9":
                continue
            tok, pert = d["unit_id"].split("|")
            rec = rec_by.get((tok, pert))
            t = rec["t_star"] if rec else None
            for name, x in d["rows"].items():
                row = ROWS.get(name)
                if row is None or t is None:
                    continue
                own = x["own"]
                gt = rec["gt"]
                for si, (span, src) in enumerate(zip(x["span"], x["src"])):
                    k = t + 1 + si
                    ctx, mask = h9_rule(row, k, t, own, gt)
                    C["gate_H9"]["steps"] += 1
                    C["gate_H9"]["span_eq_rule"] += span == [d_ for d_ in [span[0]]] + ctx
                    C["gate_H9"]["src_eq_rule"] += src[1:] == ["gt" if g else "own" for g in mask]
                    # persistence: once 'gt' at a position, 'gt' at that position in every later step
                    if si > 0:
                        prev = x["src"][si - 1][1:]
                        C["gate_H9"]["gt_positions_retained_next_step"] += all(src[1 + i] == "gt" for i, s in enumerate(prev) if s == "gt")
                        C["gate_H9"]["gt_removed_events"] += sum(1 for i, s in enumerate(prev) if s == "gt" and src[1 + i] != "gt")
                    if run == "run1" and d["batch"] == "exp" and d["decoding"] == "t001" and name in EX_ROWS and gate_cmp["ex"] < 30 * 8:
                        for jj, (c, s) in enumerate(zip(span[1:], src[1:])):
                            j = t + 1 + jj
                            ex_rows.append({"source": "P1_gate_stage0_H9_actual_span(L2a)", "unit_id": d["unit_id"], "t_star": t,
                                            "row": name, "step_k": k,
                                            "full_action_prefix_0..k-1": " ".join(map(str, list(rec["pred_prefix"]) + span)),
                                            "position_j": j, "executed_own_token": own[jj], "context_token": c, "gt_token": gt[j],
                                            "source_tag": s, "is_gt_in_context": int(s == "gt"), "overwritten": int(c != own[jj])})
                        gate_cmp["ex"] += 1
                # gate t001 exp-batch acts vs exp 9 stored acts (same unit, same row)
                if d["batch"] == "exp" and d["decoding"] == "t001" and not d["teacher_forced"]:
                    C["gate_H9"][f"acts_eq_exp9_{run}"] += x["acts"] == rec["rows"][name]["action_idx"]
                    C["gate_H9"][f"acts_cmp_{run}"] += 1

    # ---------------------------------------------------------------- L2b: P1-B traces (persistent vs transient)
    p1b_units = 0
    for l in open(P1B_TRACE):
        d = json.loads(l)
        p1b_units += 1
        t = d["t_star"]
        hist = defaultdict(dict)   # row -> {position: [(k, tag)]}
        for st in d["steps"]:
            k = st["k"]
            for name, x in st["rows"].items():
                for jj, tag in enumerate(x["context_source"][1:]):
                    hist[name].setdefault(t + 1 + jj, []).append((k, tag))
        for name, pos in hist.items():
            if not (name.startswith("P_gt") or name.startswith("T_gt")):
                continue
            kind = "persistent" if name[0] == "P" else "transient"
            for j, seq in pos.items():
                tags = [tg for _, tg in seq]
                if "gt" in tags:
                    first = tags.index("gt")
                    retained = all(tg == "gt" for tg in tags[first:])
                    C[f"p1b_{kind}"]["gt_positions"] += 1
                    C[f"p1b_{kind}"]["gt_retained_all_later_steps"] += retained
                    C[f"p1b_{kind}"]["gt_seen_only_when_most_recent"] += (len([x for x in tags if x == "gt"]) == 1 and seq[first][0] == j + 1)
        if p1b_units <= 1:
            for st in d["steps"]:
                for name in ("P_gt_s1_w3", "T_gt_s1_w3"):
                    x = st["rows"][name]
                    for jj, (c, tag) in enumerate(zip(x["context_tokens_pos_t..k-1"][1:], x["context_source"][1:])):
                        ex_rows.append({"source": "P1B_main_intervention_trace(L2b)", "unit_id": d["unit_id"], "t_star": t,
                                        "row": name, "step_k": st["k"], "full_action_prefix_0..k-1": "",
                                        "position_j": t + 1 + jj, "executed_own_token": "", "context_token": c, "gt_token": "",
                                        "source_tag": tag, "is_gt_in_context": int(tag == "gt"), "overwritten": ""})
    C["p1b"]["trace_units"] = p1b_units

    # ---------------------------------------------------------------- 90% definition summary (A-, exp 9)
    A = [u for u in unit_rows]
    def mean(v): return sum(v) / len(v)
    s90 = {"n_units": len(A), "n_scenes": len({u["token"] for u in A}), "n_logs": len({u["log"] for u in A}),
           "t_star_dist": dict(sorted(Counter(u["t_star"] for u in A).items())),
           "win3_mean_corrected_positions": mean([u["win3_corrected_positions"] for u in A]),
           "win3_corrected_positions_dist": dict(sorted(Counter(u["win3_corrected_positions"] for u in A).items())),
           "win3_mean_effective_overwrites": mean([u["win3_effective_overwrites"] for u in A]),
           "win3_effective_overwrites_dist": dict(sorted(Counter(u["win3_effective_overwrites"] for u in A).items())),
           "win3_units_equal_full": sum(u["win3_equals_full"] for u in A),
           "mean_steps_generated_after_tstar": mean([u["steps_generated_after_tstar"] for u in A]),
           "mean_steps_window_can_affect": mean([u["steps_window_can_affect"] for u in A]),
           "win3_mean_post_release_steps": mean([u["win3_post_release_steps"] for u in A]),
           "win3_post_release_steps_dist": dict(sorted(Counter(u["win3_post_release_steps"] for u in A).items())),
           "mean_context_positions_full": mean([u["context_positions_available"] for u in A])}
    # unit-weighted fraction from stored units.jsonl (analyzer metrics, unchanged)
    U = [json.loads(l) for l in open(os.path.join(EXP9, "units.jsonl"))]
    Am = [u for u in U if u["group"] == "A-"]
    amp = {nm: sum(bool(u["m"][nm]["amplification"]) for u in Am) for nm in ("normal", "win3", "gt_history")}
    s90["unit_weighted_amp_counts"] = amp
    s90["unit_weighted_fraction_win3"] = (amp["normal"] - amp["win3"]) / (amp["normal"] - amp["gt_history"])
    sc = defaultdict(lambda: defaultdict(list))
    for u in Am:
        for nm in ("normal", "win3", "gt_history"):
            sc[u["token"]][nm].append(float(bool(u["m"][nm]["amplification"])))
    lv = {nm: mean([mean(sc[s][nm]) for s in sc]) for nm in ("normal", "win3", "gt_history")}
    s90["scene_weighted_levels"] = lv
    s90["scene_weighted_fraction_win3"] = (lv["normal"] - lv["win3"]) / (lv["normal"] - lv["gt_history"])
    fde = {nm: mean([u["m"][nm]["fde5"] for u in Am]) for nm in ("normal", "win3", "gt_history")}
    s90["unit_weighted_fde5"] = fde
    s90["unit_weighted_fde_fraction_win3"] = (fde["normal"] - fde["win3"]) / (fde["normal"] - fde["gt_history"])

    with open(os.path.join(OUT, "a2_step_trace_examples.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ex_rows[0].keys())); w.writeheader(); w.writerows(ex_rows)
    with open(os.path.join(OUT, "a2_exp9_unit_window.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(unit_rows[0].keys())); w.writeheader(); w.writerows(unit_rows)
    out = {k: dict(v) for k, v in C.items()}
    out["ninety_percent_definition"] = s90
    json.dump(out, open(os.path.join(OUT, "a2_checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
