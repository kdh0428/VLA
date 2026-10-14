#!/usr/bin/env python
"""c5d extension (CPU only, diagnostic): harness H8/H9 untouched T = 0.01 tokens of a run vs the stored raw of the
full dev re-runs of exp 8 / exp 9 (outputs/review_followup/R8R9_raw_recovery/run_20261008/raw/{exp8,exp9}, unmodified
scripts, 5090). Untouched = no patch hooks, so only rows whose experiment computation has no patch are comparable
(exp 8: normal, gt_history, recent_gt; exp 9: all rows).
usage: p1b_c5d_r8r9.py RUN_DIR OUT.jsonl"""
import json
import sys

R = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/R8R9_raw_recovery/run_20261008/raw"


def main():
    run, out = sys.argv[1:3]
    dec = [json.loads(l) for l in open(f"{run}/decodes.jsonl")]
    want = {(d["unit_id"].split("|")[0], d["path"]) for d in dec if d["path"] in ("H8", "H9") and d["decoding"] == "t001"}
    stored = {}
    for path, sub in (("H8", "exp8"), ("H9", "exp9")):
        for line in open(f"{R}/{sub}/records.jsonl"):
            x = json.loads(line)
            if (x["token"], path) in want and x["perturbation"] == "alt0":
                stored[(x["token"], path)] = {n: v["action_idx"] for n, v in x["rows"].items()}
    res = []
    for d in dec:
        if d["path"] not in ("H8", "H9") or d["decoding"] != "t001" or not d["tag"].endswith("|A"):
            continue
        key = (d["unit_id"].split("|")[0], d["path"])
        if key not in stored:
            continue
        rows = ["normal", "gt_history", "recent_gt"] if d["path"] == "H8" else list(d["rows"])
        for n in rows:
            a, b = d["rows"][n]["acts"], stored[key][n]
            res.append({"control": "c5d_r8r9", "path": d["path"], "unit_id": d["unit_id"], "row": n, "stored_equal": a == b,
                        "first_divergence": next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)})
    with open(out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"c5d_r8r9: {sum(r['stored_equal'] for r in res)}/{len(res)} rows equal to the stored exp8/exp9 dev re-run raw")


if __name__ == "__main__":
    main()
