#!/usr/bin/env python
"""Author queries: INPUT_HASHES.csv and MANUSCRIPT_AUTHOR_ANSWERS_20261019.zip (no staging directory). CPU only.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q5_package.py
"""
import csv, glob, hashlib, os, zipfile

ROOT = "/root/VLA"
POC = f"{ROOT}/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/manuscript_author_queries_20261019"
ZIP = f"{OUT}/MANUSCRIPT_AUTHOR_ANSWERS_20261019.zip"
INPUTS = {  # path -> (used by, shipped in this ZIP?)
    f"{O}/review_followup/P0B_cluster_stats/run_20261008/units.parquet": ("q1_coverage", False),
    f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet": ("q1_coverage", False),
    f"{O}/review_followup4/B/B2/p1c_main/cells.jsonl": ("q1_coverage", False),
    f"{O}/review_followup4/B/B2/p1c_cohort_t4.json": ("q1_coverage (scene strata / drives)", True),
    f"{O}/review_followup5/A_STATS/A_METHOD_SELECTION.json": ("q1_recommendation", True),
    f"{O}/review_followup5/A_STATS/AUTOVLA_CI_AUDIT.csv": ("q3_final_numbers", True),
    f"{O}/review_followup5/A_B2_CLUSTER/B2_CLUSTER_CONTRIBUTIONS.csv": ("q2_power", True),
    f"{O}/review_followup5/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv": ("q3_final_numbers", True),
    f"{O}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv": ("q3_final_numbers", True),
    f"{O}/review_followup5/B_NUMERIC/B_NUMERIC_RESULTS.csv": ("q3_final_numbers", True),
    f"{O}/review_followup5/B_NUMERIC/B_NUMERIC_MATERIALITY.csv": ("q4_materiality_jsonci", True),
    f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv": ("q3_final_numbers", True),
    f"{O}/review_followup3/P0/WINDOW_WEIGHT_AUDIT.csv": ("q3_final_numbers", True),
    f"{O}/review_followup4/A2_A3/a2_checks.json": ("q3_final_numbers", True),
    f"{O}/review_followup4/A1_A4/CI_VALIDITY_AUDIT.csv": ("q3_final_numbers", True),
    f"{O}/review_followup5/reproducibility_followup5.zip": ("earlier package (scripts + inputs)", False),
    f"{O}/review_followup4/reproducibility_minimal.zip": ("earlier package (scripts + inputs)", False),
}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    rows = []
    for p, (use, shipped) in INPUTS.items():
        rows.append({"kind": "input", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": use, "in_this_zip": shipped})
    code = sorted(glob.glob(f"{POC}/scripts/manuscript_author_queries_20261019/*.py"))
    code.append(f"{POC}/scripts/review_followup5/b_numeric_analyze.py")
    outs = sorted(p for p in glob.glob(f"{OUT}/*") if not p.endswith((".zip", "INPUT_HASHES.csv")))
    for p in code:
        rows.append({"kind": "code", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": "", "in_this_zip": True})
    for p in outs:
        rows.append({"kind": "output", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": "", "in_this_zip": True})
    with open(f"{OUT}/INPUT_HASHES.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp = ZIP + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in outs + [f"{OUT}/INPUT_HASHES.csv"]:
            z.write(p, os.path.basename(p))
        for p in code:
            z.write(p, "code/" + os.path.relpath(p, POC))
        for p, (use, shipped) in INPUTS.items():
            if shipped:
                z.write(p, "inputs/" + os.path.relpath(p, POC))
    os.replace(tmp, ZIP)
    print(ZIP, os.path.getsize(ZIP))

if __name__ == "__main__":
    main()
