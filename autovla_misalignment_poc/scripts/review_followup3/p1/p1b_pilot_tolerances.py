#!/usr/bin/env python
"""
Summarise the control-5 pilot and FIX the Stage-0 tolerances (CPU only). The rule below was written before the pilot
output was read (PROTOCOL_P1.md §5 repeats it):

  T_exact  = max |delta logit| over c5a (same process, same batch) and c5b (separate process, same batch), all paths,
             rows, steps, decodings. Used for: c1, c2_hook (same batch slot), c2_token on H7 (B = 1), and any control
             whose computation is claimed identical. If T_exact == 0 the exact-class controls must be bitwise equal.
  T_equiv  = 2 x max |delta logit| over c5c_tf (experiment batch vs B = 1, teacher-forced on the same tokens = the
             largest difference produced by an equivalent computation with a different kernel shape), max over paths.
             Used for: c2_token on batched paths (cross-row), c3a, c3b, c4_tf.
  flips    = a next-token disagreement on a step whose inputs are identical is EXPLAINED iff the comparator's
             |delta logit| at that step <= the applicable tolerance AND (greedy) the reference top-2 margin <= that
             tolerance. Unexplained flips fail the control.
  floor    = c5c_free suffix disagreement rate per path and decoding (reported next to every free-run control).
Outputs: tolerances.json, pilot_summary.csv (one row per control x path x decoding).
usage: p1b_pilot_tolerances.py PILOT_DIR
"""
import csv
import json
import sys
from collections import defaultdict


def main():
    d = sys.argv[1]
    rows = [json.loads(l) for l in open(f"{d}/run1/comparisons.jsonl")]
    rows += [dict(json.loads(l), _run="run2") for l in open(f"{d}/run2/comparisons.jsonl")]
    c5b = [json.loads(l) for l in open(f"{d}/c5b_cross_process.jsonl")]
    agg = defaultdict(lambda: {"n_rows": 0, "n_steps": 0, "maxabs": 0.0, "meanabs": [], "bitwise_all": True,
                               "suffix_equal": 0, "argmax_agree": [], "pick_agree": [], "flip_margins": [],
                               "input_hash_equal_all": True, "pos_hash_equal_all": True, "stored_equal": 0, "save_maxabs": 0.0})
    for r in rows:
        key = (r["control"], r.get("path"), r.get("decoding"))
        a = agg[key]
        a["n_rows"] += 1
        if r["control"] == "c5d":
            a["stored_equal"] += int(r["stored_equal"])
            continue
        if r["control"] == "save_path":
            a["save_maxabs"] = max(a["save_maxabs"], r["stored_traj_vs_redecoded_maxabs"], r["numpy_rollout_vs_decoder_maxabs"])
            continue
        a["n_steps"] += r["n_steps_compared"]
        if r["logit_maxabs"] is not None:
            a["maxabs"] = max(a["maxabs"], r["logit_maxabs"]); a["meanabs"].append(r["logit_meanabs"])
        a["bitwise_all"] &= bool(r["bitwise_equal"]) if r["bitwise_equal"] is not None else True
        a["suffix_equal"] += int(r["suffix_equal"])
        if r["argmax_agree"] is not None:
            a["argmax_agree"].append(r["argmax_agree"]); a["pick_agree"].append(r["pick_agree"])
        a["flip_margins"] += r["margin_at_flips_a"]
        a["input_hash_equal_all"] &= r["input_hash_equal_all"]; a["pos_hash_equal_all"] &= r["pos_hash_equal_all"]
    for r in c5b:
        a = agg[("c5b", r["path"], r["decoding"])]
        a["n_rows"] += 1
        a["maxabs"] = max(a["maxabs"], r.get("logit_maxabs") or 0.0)
        a["bitwise_all"] &= r["logits_sha_equal_all"]
        a["suffix_equal"] += int(r["suffix_equal"])
        a["input_hash_equal_all"] &= r["span_sha_equal_all"] and r["prefix_sha_equal"] and r["pixel_sha_equal"]
        a["pos_hash_equal_all"] &= r["pos_sha_equal_all"]
    out = []
    for (c, p, dec), a in sorted(agg.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out.append({"control": c, "path": p, "decoding": dec, "n_rows": a["n_rows"], "n_steps_compared": a["n_steps"],
                    "logit_maxabs": a["maxabs"], "logit_meanabs_mean": (sum(a["meanabs"]) / len(a["meanabs"])) if a["meanabs"] else None,
                    "bitwise_all": a["bitwise_all"], "suffix_equal_rows": a["suffix_equal"],
                    "argmax_agree_mean": (sum(a["argmax_agree"]) / len(a["argmax_agree"])) if a["argmax_agree"] else None,
                    "pick_agree_mean": (sum(a["pick_agree"]) / len(a["pick_agree"])) if a["pick_agree"] else None,
                    "n_flips": len(a["flip_margins"]), "max_margin_at_flip": max(a["flip_margins"]) if a["flip_margins"] else None,
                    "input_hash_equal_all": a["input_hash_equal_all"], "pos_hash_equal_all": a["pos_hash_equal_all"],
                    "stored_equal_rows": a["stored_equal"] if c == "c5d" else None,
                    "save_path_maxabs": a["save_maxabs"] if c == "save_path" else None})
    with open(f"{d}/pilot_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    t_exact = max([o["logit_maxabs"] for o in out if o["control"] in ("c5a", "c5b")] + [0.0])
    floor_by_path = {}
    for o in out:
        if o["control"] == "c5c_tf":
            floor_by_path[o["path"]] = max(floor_by_path.get(o["path"], 0.0), o["logit_maxabs"])
    t_equiv = 2 * max(floor_by_path.values()) if floor_by_path else None
    free_floor = {f"{o['path']}|{o['decoding']}": {"suffix_equal_rows": o["suffix_equal_rows"], "n_rows": o["n_rows"]}
                  for o in out if o["control"] == "c5c_free"}
    tol = {"T_exact": t_exact, "T_equiv": t_equiv, "c5c_tf_max_by_path": floor_by_path, "c5c_free_floor": free_floor,
           "rule": __doc__, "units": "absolute logit difference over the 2,048 action rows (float64 of the bf16 forward)"}
    json.dump(tol, open(f"{d}/tolerances.json", "w"), indent=1)
    for o in out:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in o.items()})
    print("T_exact", t_exact, "T_equiv", t_equiv, floor_by_path)


if __name__ == "__main__":
    main()
