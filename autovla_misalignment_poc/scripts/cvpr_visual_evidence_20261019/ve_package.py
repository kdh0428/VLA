#!/usr/bin/env python
"""SOURCE_MANIFEST.csv + CVPR_VISUAL_EVIDENCE_20261019.zip (relative paths only)."""
import csv
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ve_common as V  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = [
    (f"{V.B2}/p1c_main/cells.jsonl", "stored (direct)", "review_followup4 B2 main run 20261018T001424"),
    (f"{V.B2}/p1c_analysis/unit_contrasts.parquet", "stored (derived by B2 analysis)", "review_followup4 B2 analysis"),
    (f"{V.B2}/p1c_analysis/MATCHED_PREFIX_RESULTS.csv", "stored aggregate", "review_followup4 B2 analysis"),
    (f"{V.B2}/p1c_cohort_t4.json", "stored", "review_followup4 B2 cohort"),
    (f"{V.P1C}/mech/action_history_causal/records.jsonl", "stored (direct)", "P1-C run_20261008"),
    (f"{V.P1A}/raw/records.jsonl", "stored (direct)", "P1-A run_20261008"),
    (f"{V.P1A}/deviation_curves.csv", "stored aggregate", "P1-A run_20261008"),
    (f"{V.BNUM}/run/records.jsonl", "stored (direct)", "review_followup5 B_NUMERIC run"),
    (f"{V.BNUM}/B_NUMERIC_RESULTS.csv", "stored aggregate", "review_followup5 B_NUMERIC"),
    (f"{V.P2}/p2_units.parquet", "stored (derived)", "review_followup3 P2"),
    (f"{V.P2}/MOTION_BASELINE_RESULTS.csv", "stored aggregate", "review_followup3 P2"),
    (f"{V.R8R9}/condition_summary.csv", "stored aggregate", "R8R9 raw recovery run_20261008"),
    (f"{V.R8R9}/units.parquet", "stored (derived)", "R8R9 raw recovery run_20261008"),
    (f"{V.P1D}/rollouts/main/episodes.jsonl", "stored (direct)", "P1-D run_20261008 rollouts/main"),
    (f"{V.P1D}/paired_effects.csv", "stored aggregate", "P1-D run_20261008"),
    (f"{V.P1D}/condition_summary.csv", "stored aggregate", "P1-D run_20261008"),
    (f"{V.AQ}/FINAL_PAPER_NUMBERS.csv", "stored aggregate", "manuscript_author_queries_20261019"),
    (f"{V.CVPR_NEXT}/PAPER_READY_NUMBERS.csv", "stored aggregate", "cvpr_next_validation_20261019"),
    (f"{V.CVPR_NEXT}/FINAL_INTERVAL_POLICY.csv", "stored aggregate", "cvpr_next_validation_20261019"),
    (f"{V.POC}/scripts/review_followup3/p2_kinematics.py", "code (baseline definitions)", "review_followup3"),
    (f"{V.POC}/scripts/review_followup/p1ab/p1_common.py", "code (codebook geometry)", "review_followup p1ab"),
]


def main():
    rows = []
    for p, kind, rid in INPUTS:
        rows.append({"relative_path": V.rel(p), "sha256": V.sha(p), "bytes": os.path.getsize(p), "kind": kind,
                     "generated_or_reconstructed": "input (read only)", "run_id": rid})
    for d in sorted(os.listdir(f"{V.OUT}/cases")):
        for f in sorted(os.listdir(f"{V.OUT}/cases/{d}")):
            if f.endswith(".jpg"):
                p = f"{V.OUT}/cases/{d}/{f}"
                cam = f.split("_")[0] + "_" + f.split("_")[1]
                rows.append({"relative_path": f"cases/{d}/{f}", "sha256": V.sha(p), "bytes": os.path.getsize(p),
                             "kind": f"raw camera frame copy ({cam}); origin autovla/dataset/nuplan/sensor_blobs/test",
                             "generated_or_reconstructed": "copied unmodified", "run_id": "nuPlan/OpenScene sensor blob"})
    for root, _, files in os.walk(V.OUT):
        for f in sorted(files):
            p = os.path.join(root, f)
            r = os.path.relpath(p, V.OUT)
            if r.startswith("cases/") and f.endswith(".jpg") or f.endswith(".zip") or r == "SOURCE_MANIFEST.csv":
                continue
            rows.append({"relative_path": r, "sha256": V.sha(p), "bytes": os.path.getsize(p), "kind": "package output",
                         "generated_or_reconstructed": "generated (CPU, this package)", "run_id": "cvpr_visual_evidence_20261019"})
    with open(f"{V.OUT}/SOURCE_MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    z = f"{V.OUT}/CVPR_VISUAL_EVIDENCE_20261019.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, _, files in os.walk(V.OUT):
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.endswith(".zip"):
                    continue
                zf.write(p, "CVPR_VISUAL_EVIDENCE_20261019/" + os.path.relpath(p, V.OUT))
        for f in sorted(os.listdir(HERE)):
            if f.endswith(".py"):
                zf.write(os.path.join(HERE, f), f"CVPR_VISUAL_EVIDENCE_20261019/scripts/{f}")
    print(z, os.path.getsize(z) / 1e6, "MB", len(rows), "manifest rows")


if __name__ == "__main__":
    main()
