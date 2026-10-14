#!/usr/bin/env python
"""C5b: compare two harness runs made in separate processes (CPU only). Same tags -> tokens, logits (key rows), hashes.
usage: p1b_compare_runs.py RUN1 RUN2 OUT.jsonl"""
import json
import sys

import numpy as np


def load(run):
    dec = {}
    for line in open(f"{run}/decodes.jsonl"):
        x = json.loads(line)
        dec[x["tag"]] = x
    return dec, np.load(f"{run}/logits_keyrows.npz")


def main():
    r1, r2, out = sys.argv[1:4]
    d1, l1 = load(r1)
    d2, l2 = load(r2)
    rows = []
    for tag in sorted(set(d1) & set(d2)):
        a, b = d1[tag], d2[tag]
        for n in a["rows"]:
            if n not in b["rows"]:
                continue
            ra, rb = a["rows"][n], b["rows"][n]
            key = f"{tag}|{n}"
            rec = {"control": "c5b", "tag": tag, "row": n, "path": a["path"], "decoding": a["decoding"], "batch": a["batch"],
                   "prefix_sha_equal": a["prefix"]["prefix_ids_sha"] == b["prefix"]["prefix_ids_sha"],
                   "pixel_sha_equal": a["prefix"]["pixel_sha"] == b["prefix"]["pixel_sha"],
                   "span_sha_equal_all": ra["span_sha"] == rb["span_sha"], "pos_sha_equal_all": ra["pos_sha"] == rb["pos_sha"],
                   "logits_sha_equal_all": ra["lsha"] == rb["lsha"], "suffix_equal": ra["own"] == rb["own"],
                   "argmax_equal_all": ra["argmax"] == rb["argmax"]}
            if key in l1.files and key in l2.files:
                x, y = l1[key].astype(np.float64), l2[key].astype(np.float64)
                if x.shape == y.shape and x.size:
                    rec["logit_maxabs"] = float(np.abs(x - y).max())
            rows.append(rec)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    print(f"c5b rows {n}: logits_sha_equal {sum(r['logits_sha_equal_all'] for r in rows)}/{n}, "
          f"suffix_equal {sum(r['suffix_equal'] for r in rows)}/{n}, max logit diff "
          f"{max((r.get('logit_maxabs') or 0.0) for r in rows) if rows else None}")


if __name__ == "__main__":
    main()
