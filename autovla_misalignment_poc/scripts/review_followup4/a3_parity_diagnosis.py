#!/usr/bin/env python
"""
A3 (review_followup4): Impromptu parity diagnosis table, CPU only, read-only on exp 33 data.

Inputs
  outputs/cross_vla_temporal_replication/exp_ab/records.jsonl, exp_cde/records.jsonl    (stored exp 33 data)
  outputs/review_followup4/A2_A3/a3_parity_run/parity_records.jsonl                    (a3_parity_run.py, 4 scenes)
  outputs/review_followup4/A2_A3/a3_exp_cde_rerun_limit1/records.jsonl                (unmodified exp_cde.py --limit 1)
Outputs
  IMPROMPTU_PARITY_DIAGNOSIS.csv   one row per comparison: what is held equal, exact-match count, first-divergence
                                   waypoint and size, final displacement, A-/amplification discordance
  a3_parity_summary.json
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, os.path.join(POC, "scripts/cross_vla_temporal"))
from analyze_temporal import metrics  # noqa: E402

REP = os.path.join(POC, "outputs/cross_vla_temporal_replication")
OUT = os.path.join(POC, "outputs/review_followup4/A2_A3")


def fd(a, b):
    a, b = np.round(np.asarray(a, float), 2), np.round(np.asarray(b, float), 2)
    d = np.abs(a - b).max(1)
    idx = np.where(d > 1e-9)[0]
    return (None, 0.0, float(np.linalg.norm(a[-1] - b[-1]))) if not len(idx) else (int(idx[0]), float(d[idx[0]]), float(np.linalg.norm(a[-1] - b[-1])))


def summarise(cid, desc, held, pairs, gts=None):
    res = [fd(a, b) for a, b in pairs]
    n = len(res); eq = sum(r[0] is None for r in res)
    first = Counter(r[0] + 1 for r in res if r[0] is not None)          # 1-based waypoint index
    mag = np.array([r[1] for r in res if r[0] is not None])
    row = {"comparison_id": cid, "description": desc, **held, "n": n, "n_exact_equal_0.01m": eq,
           "share_exact_equal": round(eq / n, 4) if n else None,
           "first_divergent_waypoint_dist(1-based)": json.dumps(dict(sorted(first.items()))),
           "first_divergence_size_eq_0.01m_share": round(float(np.mean(np.isclose(mag, 0.01))), 4) if len(mag) else None,
           "first_divergence_size_le_0.02m_share": round(float(np.mean(mag <= 0.0201)), 4) if len(mag) else None,
           "final_waypoint_displacement_median_m": round(float(np.median([r[2] for r in res])), 3) if n else None,
           "final_waypoint_displacement_mean_m": round(float(np.mean([r[2] for r in res])), 3) if n else None}
    if gts is not None:
        ma = [metrics(a, g) for (a, _), g in zip(pairs, gts)]; mb = [metrics(b, g) for (_, b), g in zip(pairs, gts)]
        row["A-_discordant_share"] = round(float(np.mean([x["a_minus"] != y["a_minus"] for x, y in zip(ma, mb)])), 4)
        row["amp_discordant_share"] = round(float(np.mean([x["amp"] != y["amp"] for x, y in zip(ma, mb)])), 4)
    return row


def main():
    rows = []
    AB = [json.loads(l) for l in open(os.path.join(REP, "exp_ab/records.jsonl"))]
    ABok = [r for r in AB if all(v["ok"] for v in r["rows"].values())]          # analyze_temporal.analyze_ab filter (2,728)
    CDE = [json.loads(l) for l in open(os.path.join(REP, "exp_cde/records.jsonl"))]
    H = lambda inp, pre, dec, batch, path: {"same_input": inp, "same_prefix_tokens": pre, "same_decoding_rule": dec,  # noqa: E731
                                            "same_batch_composition": batch, "same_forward_path": path}
    rows.append(summarise("S1", "stored exp_ab: natural (HF generate, B=1) vs natural_rep (same empty prefix, inside the 18-row "
                                "exp_ab batch, left-padded)", H("yes", "yes (empty)", "yes (greedy, rep 1.05)", "NO (B=1 vs B=18, padding)", "yes (HF generate)"),
                          [(r["natural"], r["rows"]["natural_rep"]["traj"]) for r in ABok], [r["gt"] for r in ABok]))
    rows.append(summarise("S2", "stored exp_ab: natural vs pert_m0 (w1 forced to natural w1, re-encoded as text prefix, 18-row batch)",
                          H("yes", "w1 identical; prefix re-encoded", "yes", "NO", "NO (prefill of re-encoded w1 vs incremental)"),
                          [(r["natural"], r["rows"]["pert_m0"]["traj"]) for r in ABok], [r["gt"] for r in ABok]))
    rows.append(summarise("S3", "stored exp_ab: natural_rep vs pert_m0 (same batch, different prefix path)",
                          H("yes", "different (empty vs w1 text)", "yes", "same batch, different rows", "NO"),
                          [(r["rows"]["natural_rep"]["traj"], r["rows"]["pert_m0"]["traj"]) for r in ABok], [r["gt"] for r in ABok]))
    ref = [next(u for u in r["units"] if u["dir"] is None) for r in CDE]
    rows.append(summarise("S4", "stored exp_cde: delta-0 reference unit 'normal' row (FastWaypointDecoder, waypoint by waypoint, "
                                "rows grouped by suffix length, chunks <= 32) vs natural (HF generate, B=1)",
                          H("yes", "w1 identical; later waypoints re-tokenised from text each step", "greedy+rep 1.05 (re-implemented)",
                            "NO", "NO (prompt KV + suffix prefill + incremental vs one HF generate)"),
                          [(u["rows"]["normal"], r["natural"]) for u, r in zip(ref, CDE)], [r["gt"] for r in CDE]))
    # in-run identical-context check: at k = 2 every row of a unit has the identical context [w1]
    n = eq = nr = nre = 0
    for r in CDE:
        for u in r["units"]:
            w2 = [tuple(u["rows"][x][1]) for x in u["rows"]]
            n += 1; eq += len(set(w2)) == 1
            nr += len(w2); nre += sum(x == tuple(u["rows"]["normal"][1]) for x in w2)
    rows.append({"comparison_id": "S5", "description": "stored exp_cde, same run: the 12 rows of a unit have byte-identical context "
                 "[w1] at k = 2; do their generated w2 agree? (rows differ only in batch position / chunk)",
                 **H("yes", "yes (identical suffix)", "yes", "NO (position/chunk within the step's groups)", "yes"),
                 "n": n, "n_exact_equal_0.01m": eq, "share_exact_equal": round(eq / n, 4),
                 "first_divergent_waypoint_dist(1-based)": json.dumps({"rows_equal_to_normal": f"{nre}/{nr}"})})
    # ---------------------------------------------------------------- a3_parity_run (4 scenes)
    P = [json.loads(l) for l in open(os.path.join(OUT, "a3_parity_run/parity_records.jsonl"))]
    def cnt(key):
        v = [p[key] for p in P if key in p]
        return len(v), sum(x[0] is None for x in v), json.dumps([x[0] + 1 if x[0] is not None else None for x in v])
    run_cmp = [("R1", "N1_vs_N2", "rerun: natural B=1 twice in one process", H("yes", "yes", "yes", "yes", "yes")),
               ("R2", "N1_vs_store_natural", "rerun natural B=1 vs stored natural (other process, 2 weeks earlier)", H("yes", "yes", "yes", "yes", "yes")),
               ("R3", "AB18_rep_vs_store_rep", "rerun exact exp_ab 18-row batch: natural_rep vs stored natural_rep", H("yes", "yes", "yes", "yes", "yes")),
               ("R4", "AB18_m0_vs_store_m0", "rerun exact exp_ab 18-row batch: pert_m0 vs stored pert_m0", H("yes", "yes", "yes", "yes", "yes")),
               ("R5", "AB18_rep_vs_N1", "rerun: natural_rep (18-row batch) vs natural B=1 (= S1 reproduced)", H("yes", "yes", "yes", "NO", "yes")),
               ("R6", "NOPAD18_vs_N1", "18 identical empty-prefix rows (no padding) vs B=1", H("yes", "yes", "yes", "NO (batch size only)", "yes")),
               ("R7", "PAD2_rep_vs_N1", "B=2 [empty, w1 prefix]: empty row (left-padded) vs B=1", H("yes", "yes", "yes", "NO (padding, B=2)", "yes")),
               ("R8", "M0_B1_vs_N1", "w1 re-encoded as prefix, alone (B=1) vs natural B=1", H("yes", "w1 identical, re-encoded", "yes", "yes (B=1)", "NO (prefill vs incremental)")),
               ("R9", "FREE_FAST_B1_vs_N1", "FastWaypointDecoder delta-0 normal row alone (B=1) vs natural B=1", H("yes", "re-tokenised each step", "yes (re-implemented)", "yes (B=1)", "NO")),
               ("R10", "FREE_FAST_B1_vs_store_cde_normal", "FastWaypointDecoder B=1 vs stored exp_cde normal (batched)", H("yes", "yes", "yes", "NO", "yes"))]
    for cid, key, desc, held in run_cmp:
        n_, e_, firsts = cnt(key)
        rows.append({"comparison_id": cid, "description": desc, **held, "n": n_, "n_exact_equal_0.01m": e_,
                     "share_exact_equal": round(e_ / n_, 4) if n_ else None, "first_divergent_waypoint_dist(1-based)": firsts})
    nd = sum(1 for p in P for x in p["N1_vs_N2"][:1] if x is None)
    tf = [x for p in P for x in p["TF"]]
    def size(a, b):
        return round(float(np.abs(np.round(np.asarray(a, float), 2) - np.round(np.asarray(b, float), 2)).max()), 2)
    tf_sizes = Counter(size(x["hf_one"], x["natural"]) for x in tf if not x["hf_eq_nat"])
    tf_sizes_f = Counter(size(x["fast"], x["natural"]) for x in tf if not x["fast_eq_nat"])
    for cid, key, desc, sz in (("T1", "hf_eq_nat", "teacher-forced: HF one-waypoint re-encoded path (B=1) given the exact natural prefix w1..w_{k-1} reproduces natural w_k", tf_sizes),
                               ("T2", "fast_eq_nat", "teacher-forced: FastWaypointDecoder (B=1) given the exact natural prefix reproduces natural w_k", tf_sizes_f),
                               ("T3", "fast_eq_hf", "teacher-forced: FastWaypointDecoder vs HF one-waypoint path, same prefix", None)):
        e_ = sum(bool(x[key]) for x in tf)
        rows.append({"comparison_id": cid, "description": desc, **H("yes", "yes (identical ids, RETOK checked)", "yes", "yes (B=1)", "NO"),
                     "n": len(tf), "n_exact_equal_0.01m": e_, "share_exact_equal": round(e_ / len(tf), 4),
                     "first_divergent_waypoint_dist(1-based)": json.dumps({"size_of_mismatch_m": {str(k): v for k, v in sorted(sz.items())}} if sz else {})})
    rk = [x for p in P for x in p["RETOK"]]
    rows.append({"comparison_id": "K1", "description": "re-tokenisation: tokenizer(prefix_text(parsed natural w1..wk)) == the ids the model generated "
                 "(and the rebuilt text == generated text), k = 1..9", **H("yes", "-", "-", "-", "-"), "n": len(rk),
                 "n_exact_equal_0.01m": sum(x["ids_prefix_identical"] and x["text_prefix_identical"] for x in rk),
                 "share_exact_equal": round(sum(x["ids_prefix_identical"] and x["text_prefix_identical"] for x in rk) / len(rk), 4)})
    new = json.loads(open(os.path.join(OUT, "a3_exp_cde_rerun_limit1/records.jsonl")).readline())
    old = next(r for r in CDE if r["token"] == new["token"])
    e_ = sum(un["rows"][k] == uo["rows"][k] for un, uo in zip(new["units"], old["units"]) for k in un["rows"])
    t_ = sum(len(un["rows"]) for un in new["units"])
    rows.append({"comparison_id": "R11", "description": f"unmodified exp_cde.py --limit 1 rerun ({new['token']}) vs stored exp_cde: all 9 units x 12 rows",
                 **H("yes", "yes", "yes", "yes", "yes"), "n": t_, "n_exact_equal_0.01m": e_, "share_exact_equal": round(e_ / t_, 4)})
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "IMPROMPTU_PARITY_DIAGNOSIS.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump({"rows": rows, "n_ab_all": len(AB), "n_ab_ok": len(ABok), "n_cde": len(CDE)}, open(os.path.join(OUT, "a3_parity_summary.json"), "w"), indent=1)
    for r in rows:
        print(r["comparison_id"], r["n"], r["n_exact_equal_0.01m"], r.get("share_exact_equal"), r.get("first_divergent_waypoint_dist(1-based)", "")[:120],
              r.get("first_divergence_size_eq_0.01m_share"), r.get("final_waypoint_displacement_median_m"), r.get("A-_discordant_share"), r.get("amp_discordant_share"))


if __name__ == "__main__":
    main()
