#!/usr/bin/env python
"""
Apply PROTOCOL_P1.md §6 to a Stage-0 gate run (CPU only). Written before the gate was run.
usage: p1b_judge.py GATE_RUN1_DIR C5B_JSONL TOLERANCES_JSON OUT_PREFIX   -> OUT_PREFIX.csv / OUT_PREFIX.md
Exact class: c1, c2_hook, c2_token on H7.  Equivalence class: c2_token on H8/H9/P1A, c3a_incremental, c3b_deep_invalidate,
c4_tf.  Reported only: c4_free, c5c_free.  c5a/c5b must stay within T_exact (they re-check the pilot).
"""
import csv
import json
import sys
from collections import defaultdict

EXACT = {"c1", "c2_hook", "c5a"}
EQUIV = {"c3a_incremental", "c3b_deep_invalidate", "c4_tf", "c5c_tf"}
REPORT = {"c4_free", "c5c_free", "c5d", "save_path"}


def klass(r):
    c = r["control"]
    if c == "c2_token":
        return "exact" if r["path"] == "H7" else "equiv"
    return "exact" if c in EXACT else "equiv" if c in EQUIV else "report"


def main():
    run, c5b_path, tol_path, outp = sys.argv[1:5]
    tol = json.load(open(tol_path))
    te, tq = float(tol["T_exact"]), float(tol["T_equiv"])
    rows = [json.loads(l) for l in open(f"{run}/comparisons.jsonl")]
    fails, summ = [], defaultdict(lambda: {"n": 0, "fail": 0, "maxabs": 0.0, "flips": 0, "suffix_equal": 0})
    for r in rows:
        k = klass(r)
        key = (r["control"], r.get("path"), r.get("decoding"))
        s = summ[key]; s["n"] += 1
        if k == "report":
            if "suffix_equal" in r:
                s["suffix_equal"] += int(r["suffix_equal"])
            continue
        why = []
        m = r.get("logit_maxabs")
        lim = te if k == "exact" else tq
        if m is not None:
            s["maxabs"] = max(s["maxabs"], m)
            if m > lim:
                why.append(f"logit {m:.4g} > {lim:.4g}")
        if not r["input_hash_equal_all"]:
            why.append("input hash")
        if not r["pos_hash_equal_all"]:
            why.append("position hash")
        if r["n_steps_compared"] < r["n_steps"] and (k == "exact" or r["control"] in EQUIV):   # teacher-forced: all steps must be comparable
            why.append(f"only {r['n_steps_compared']}/{r['n_steps']} steps comparable")
        unexpl = [i for i, mg in zip(r["flip_steps"], r["margin_at_flips_a"]) if mg > lim]
        s["flips"] += len(r["flip_steps"])
        if k == "exact" and (r["flip_steps"] or not r["suffix_equal"]):
            why.append("token difference in exact control")
        if unexpl:
            why.append(f"unexplained flips {unexpl}")
        if r["control"] in ("c2_hook", "c2_token") and any(e for e in r.get("override_effective_b", [])):
            why.append("effective override in self-patch")
        if r["control"] == "c4_tf" and r.get("pos_full_eq_cached_all") is False:
            why.append("full-recompute position ids differ")
        if r["control"] in ("c1", "c2_hook") and r.get("hooks_after", {"n_module_hooks": 0}).get("n_module_hooks", 0):
            why.append("hooks left installed")
        if why:
            s["fail"] += 1
            fails.append({**{x: r.get(x) for x in ("control", "path", "unit_id", "decoding", "row_a", "row_b")}, "class": k, "why": "; ".join(why)})
    for r in map(json.loads, open(c5b_path)):
        s = summ[("c5b", r["path"], r["decoding"])]; s["n"] += 1
        m = r.get("logit_maxabs") or 0.0
        s["maxabs"] = max(s["maxabs"], m)
        if m > te or not r["suffix_equal"] or not r["span_sha_equal_all"]:
            s["fail"] += 1
            fails.append({"control": "c5b", "path": r["path"], "unit_id": r["tag"], "decoding": r["decoding"], "row_b": r["row"],
                          "class": "exact", "why": f"cross-process logit {m:.4g} / suffix {r['suffix_equal']}"})
    paths = sorted({k[1] for k in summ if k[1]})
    verdict = {p: ("FAIL" if any(f["path"] == p for f in fails) else "PASS") for p in paths}
    with open(f"{outp}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["control", "path", "decoding", "n_rows", "n_fail", "logit_maxabs", "n_flips", "suffix_equal_rows"])
        for (c, p, d), s in sorted(summ.items(), key=lambda x: tuple(map(str, x[0]))):
            w.writerow([c, p, d, s["n"], s["fail"], s["maxabs"], s["flips"], s["suffix_equal"]])
    with open(f"{outp}.md", "w") as f:
        f.write(f"# NOOP_CACHE_AUDIT (Stage-0 gate)\n\nTolerances: T_exact = {te}, T_equiv = {tq} (from {tol_path}).\n\n")
        f.write("| path | verdict |\n|---|---|\n" + "".join(f"| {p} | {v} |\n" for p, v in verdict.items()))
        f.write(f"\nFailures ({len(fails)}):\n\n")
        for x in fails:
            f.write(f"- {json.dumps(x)}\n")
    print(verdict, len(fails), "failures")


if __name__ == "__main__":
    main()
