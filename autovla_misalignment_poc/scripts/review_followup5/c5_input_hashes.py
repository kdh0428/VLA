#!/usr/bin/env python
"""review_followup5: INPUT_HASHES.csv — sha256 of every input read by the review_followup5 scripts, and of every
review_followup5 output (CPU).

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c5_input_hashes.py
"""
import csv
import hashlib
import os

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
R5 = f"{O}/review_followup5"
INPUTS = [
    (f"{O}/review_followup/P1A_gt_free_reference/run_20261008/raw/records.jsonl", "b_numeric_manifest.py, b_numeric_analyze.py (units, stored S0 tokens, GT, reference)"),
    (f"{O}/review_followup3/P2/p2_units.parquet", "b_numeric_analyze.py (frozen kinematic / Geo baseline endpoint responses)"),
    (f"{O}/review_followup3/P2/extrap_units.parquet", "b_numeric_analyze.py (frozen learned-extrapolator predictions)"),
    (f"{O}/review_followup4/B/B2/p1c_analysis/unit_contrasts.parquet", "a_b2_cluster.py, c_power.py"),
    (f"{O}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv", "a_b2_cluster.py (consistency check)"),
    (f"{O}/equal_distance_perturbation/records.jsonl", "a_b2_cluster.py (dev day-vehicles)"),
    (f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet", "a_stats_coverage.py"),
    (f"{O}/review_followup/P0B_cluster_stats/run_20261008/units.parquet", "a_stats_coverage.py"),
    (f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/paired_effects.csv", "a_stats_audit.py"),
    (f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/decisions.json", "a_stats_audit.py (MME rule check)"),
    (f"{O}/review_followup2/A_autovla_cluster/run_20261013/CLUSTER_REANALYSIS.csv", "a_stats_audit.py"),
    (f"{O}/review_followup2/A_autovla_cluster/run_20261013/p1c_subsets_drive_level.csv", "a_stats_audit.py"),
    (f"{O}/review_followup2/C_baseline_window/run_20261013/c2_position_length.csv", "a_stats_audit.py"),
    (f"{O}/review_followup2/C_baseline_window/run_20261013/c1_predeclared_and_gt_drive.csv", "a_stats_audit.py"),
    (f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv", "a_stats_audit.py"),
    (f"{O}/review_followup3/P0/DEPENDENCE_SENSITIVITY.csv", "a_stats_audit.py"),
    (f"{O}/review_followup3/P0/DEPENDENCE_disjoint_subsets.csv", "a_stats_audit.py"),
    (f"{O}/review_followup3/P0/DISJOINT_LOG_INVENTORY.csv", "c_disjoint.py"),
    (f"{O}/review_followup3/P0/DISJOINT_LOG_INVENTORY_meta.json", "c_disjoint.py"),
    (f"{O}/review_followup4/A1_A4/A1_CALLSITES.csv", "A_STATS scope (call sites)"),
    ("/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt", "b_numeric_runner.py (GPU); sha256 computed 2026-10-18 = HF LFS oid (see PINS.csv); not re-hashed here"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for p, use in INPUTS:
        if p.endswith(".ckpt"):
            rows.append({"kind": "input", "file": p, "bytes": os.path.getsize(p),
                         "sha256": "58246773393da45678a3f35d354fd969eed6833ecc8ee596edc5e283d1a87473", "used_by": use})
            continue
        rows.append({"kind": "input", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": use})
    for dp, dn, fn in os.walk(R5):
        dn.sort()
        for f in sorted(fn):
            q = os.path.join(dp, f)
            if f in ("INPUT_HASHES.csv",) or f.endswith(".part"):
                continue
            rows.append({"kind": "output", "file": q, "bytes": os.path.getsize(q), "sha256": sha(q), "used_by": "review_followup5 result"})
    for dp, dn, fn in os.walk(f"{POC}/scripts/review_followup5"):
        if "__pycache__" in dp:
            continue
        for f in sorted(fn):
            q = os.path.join(dp, f)
            rows.append({"kind": "code", "file": q, "bytes": os.path.getsize(q), "sha256": sha(q), "used_by": "review_followup5 script"})
    with open(f"{R5}/INPUT_HASHES.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows))


if __name__ == "__main__":
    main()
