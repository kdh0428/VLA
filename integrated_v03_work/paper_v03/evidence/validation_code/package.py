#!/usr/bin/env python
"""CVPR next validation: INPUT_HASHES.csv (new inputs + new outputs only) and CVPR_NEXT_VALIDATION_20261019.zip. CPU only."""
import csv, glob, hashlib, os, zipfile

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/cvpr_next_validation_20261019"
ZIP = f"{OUT}/CVPR_NEXT_VALIDATION_20261019.zip"
INPUTS = {
    f"{O}/review_followup/P1C_new_log_replication/run_20261008/analysis/units.parquet": ("sim_weighted (E RN/RV, F RN)", False),
    f"{O}/review_followup4/B/B2/p1c_main/cells.jsonl": ("sim_weighted (E4 / F B2)", False),
    f"{O}/review_followup4/B/B2/p1c_cohort_t4.json": ("sim_weighted (E4 weights, frame)", True),
    f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv": ("paper_numbers", True),
    f"{O}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv": ("paper_numbers", True),
    f"{O}/manuscript_author_queries_20261019/COVERAGE_SUMMARY.csv": ("policy", True),
    f"{O}/review_followup5/C_DISJOINT/overlap_audit.json": ("disjoint delta reference", True),
    "/root/VLA/CVPR_20261019_source.zip": ("manuscript source (read only, listing)", False),
}
for f in sorted(glob.glob(f"{O}/review_followup/P1C_new_log_replication/run_20261008/selection/shard_*.json")):
    if os.path.basename(f).replace("shard_", "").replace(".json", "").isdigit():
        INPUTS[f] = ("sim_weighted (E frame per log)", True)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rows = [{"kind": "input", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": u, "in_zip": s} for p, (u, s) in INPUTS.items()]
code = sorted(glob.glob(f"{POC}/scripts/cvpr_next_validation_20261019/*.py"))
outs = sorted(p for p in glob.glob(f"{OUT}/*") if not p.endswith((".zip", "INPUT_HASHES.csv")))
rows += [{"kind": "code", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": "", "in_zip": True} for p in code]
rows += [{"kind": "output", "file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "used_by": "", "in_zip": True} for p in outs]
with open(f"{OUT}/INPUT_HASHES.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
tmp = ZIP + ".part"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in outs + [f"{OUT}/INPUT_HASHES.csv"]:
        z.write(p, os.path.basename(p))
    for p in code:
        z.write(p, "code/" + os.path.relpath(p, POC))
    for p, (u, s) in INPUTS.items():
        if s:
            z.write(p, "inputs/" + os.path.relpath(p, POC))
os.replace(tmp, ZIP)
print(ZIP, os.path.getsize(ZIP))
