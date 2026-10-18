#!/usr/bin/env python
"""Combine B2 analysis outputs (p1c, dev) into B/MATCHED_PREFIX_RESULTS.csv and print markdown tables (CPU, no new estimation)."""
import csv
import json
import os

B = "/root/VLA/autovla_misalignment_poc/outputs/review_followup4/B"
SRC = {"p1c": f"{B}/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv", "dev": f"{B}/B2/dev_analysis/MATCHED_PREFIX_RESULTS.csv"}
rows = []
for coh, p in SRC.items():
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            rows.append({"cohort": coh, **r})
keys = list(rows[0].keys())
with open(f"{B}/MATCHED_PREFIX_RESULTS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)


def f(x, n=3):
    try:
        return f"{float(x):+.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def ci(s, n=3):
    try:
        a, b = json.loads(s)
        return f"[{a:+.{n}f}, {b:+.{n}f}]"
    except Exception:
        return "–"


def p(x):
    try:
        v = float(x)
        return f"{v:.2g}"
    except (TypeError, ValueError):
        return "–"


def table(coh, n_old, metric, contrasts, subset="all", ests=("F", "P", "S"), clus="drive", nd=3):
    print(f"\n#### {coh} | anchor n_old={n_old} | metric={metric} | subset={subset} | cluster={clus}\n")
    print("| contrast | estimand | scenes | units | clusters | theta | boot 95% CI | sign-flip p | Holm p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in contrasts:
        for e in ests:
            for r in rows:
                if (r["cohort"], r["n_old"], r["metric"], r["contrast"], r["subset"], r["estimand"], r["cluster"]) == \
                        (coh, str(n_old), metric, c, subset, e, clus):
                    print(f"| {c} | {e} | {r['n_scenes']} | {r['n_units']} | {r['n_clusters']} | {f(r['theta'], nd)} | "
                          f"{ci(r.get('boot_ci95'), nd)} | {p(r['p_signflip'])} | {p(r.get('holm_p'))} |")


PRIM = ["C_GT", "C_NAT", "INT", "C_PROV"]
SEC = ["O_main_REF", "O_main_NSRC", "O_main_CSRC", "C_PROV_NAT", "CSRC_vs_REF_GT"]
for coh in ("p1c", "dev"):
    table(coh, 1, "severe", PRIM)
    table(coh, 1, "severe", PRIM, clus="day_vehicle")
    table(coh, 1, "severe", ["INT"], subset="older_values_differ")
    table(coh, 1, "severe", ["C_PROV"], subset="src_values_differ")
    for m, nd in (("fde5", 2), ("tok_a_changed", 3), ("local_disp_change_m", 2), ("suffix_changed_frac", 3), ("endpoint_dist_m", 2), ("A_plus", 3)):
        table(coh, 1, m, PRIM, ests=("F", "P"), nd=nd)
    table(coh, 1, "severe", SEC, ests=("F", "P"))
    table(coh, 2, "severe", PRIM, ests=("F", "P"))
    table(coh, 2, "fde5", PRIM, ests=("F", "P"), nd=2)
