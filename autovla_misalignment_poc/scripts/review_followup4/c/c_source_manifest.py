#!/usr/bin/env python
"""review_followup4: SOURCE_MANIFEST.csv — every file in outputs/review_followup4 (and every manuscript-facing file
changed this round) with sha256, size, producing script / source data, commit and evidence link (2026-10-18). CPU only.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_source_manifest.py
"""
import csv
import hashlib
import os
import re
import subprocess

ROOT = "/root/VLA"
POC = f"{ROOT}/autovla_misalignment_poc"
R4 = f"{POC}/outputs/review_followup4"
S4 = "scripts/review_followup4"

# (regex on path relative to outputs/review_followup4, producing script, source data, evidence link / use)
RULES = [
    (r"A1_A4/(CI_VALIDITY_AUDIT\.csv|A1_JUDGMENT_CHANGES\.csv|A1_OPENVLA_2x2\.csv|A1_CALLSITES\.csv|A1_DEGENERACY_DEMO\.json|a1_audit_meta\.json)",
     f"{S4}/a1_ci_audit.py (+ a1_paired_ci.py)", "outputs/review_followup2/B_spatialvla_direct/run_20261013/*; outputs/review_followup3/P0/OPENVLA_episode_records.csv; A1_METHOD_SELECTION.json",
     "A1 post-hoc corrected CIs; CLAIM_CHANGES §3 A1; numbers.json RF4.A1.*; Table 13"),
    (r"A1_A4/(A1_METHOD_SELECTION\.json|A1_COVERAGE_SIM_.*|A1_TOST_SIZE_SIM\.csv|a1_coverage_sim\.log)",
     f"{S4}/a1_coverage_sim.py", "simulation (seed 20261017); design sizes from review_followup2 pairs.csv", "A1 method selection rule"),
    (r"A1_A4/CI_VALIDITY_AUDIT\.md", "hand-written report", "A1 tables", "A1 report"),
    (r"A1_A4/(COHORT_ESTIMAND_MAP\.(md|csv)|A4_WORDING_VIOLATIONS\.csv)", f"{S4}/a4_cohort_map.py",
     "outputs/review_followup3/P0/POPULATION_*.csv and other source tables; manuscript-facing documents", "A4 sets/definitions; numbers.json RF4.A4.*; MANUSCRIPT_REVISION §1"),
    (r"A2_A3/(a2_checks\.json|a2_step_trace_examples\.csv|a2_exp9_unit_window\.csv)", f"{S4}/a2_window_semantics.py",
     "temporal_feedback_window.py; review_followup3 P1 gate_stage0 decodes; P1-B intervention_trace; R8R9 exp9 raw", "A2 verdict; numbers.json RF4.A2.*"),
    (r"A2_A3/a2_p1b_transient_vs_persistent\.json", f"{S4}/a2_p1b_transient_vs_persistent.py",
     "outputs/review_followup/P1B_start_position/run_20261008/units.parquet", "A2 transient vs persistent (post hoc)"),
    (r"A2_A3/WINDOW_SEMANTICS_AUDIT\.md", "hand-written report", "A2 outputs", "A2 report; CLAIM_CHANGES L1"),
    (r"A2_A3/(IMPROMPTU_PARITY_DIAGNOSIS\.csv|a3_parity_summary\.json)", f"{S4}/a3_parity_diagnosis.py",
     "outputs/cross_vla_temporal_replication (exp 33 stored records); A2_A3/a3_parity_run", "A3 verdict; numbers.json RF4.A3.*"),
    (r"A2_A3/a3_parity_run/.*", f"{S4}/a3_parity_run.py (GPU)", "Impromptu 3B_AD rev 33aa6f34; 4 scenes", "A3 same-condition runs"),
    (r"A2_A3/a3_exp_cde_rerun_limit1/.*", "scripts/cross_vla_temporal/exp_cde.py --limit 1 (GPU, unmodified)", "Impromptu 3B_AD", "A3 R11 108/108"),
    (r"A2_A3/IMPROMPTU_PARITY_DIAGNOSIS\.md", "hand-written report", "A3 outputs", "A3 report; CLAIM_CHANGES §3 A3"),
    (r"B/B1/(b1_manifest.*\.json)", f"{S4}/b/b1_manifest.py", "Stage-0 manifest rule (review_followup3 p1_stage0_manifest.py)", "B1 manifest"),
    (r"B/B1/run/.*|B/B1/run_b1\.log", f"{S4}/b/run_b1.sh -> b/b1_runner.py (GPU)", "AutoVLA ckpt sha256 58246773…; B1 manifest", "B1 raw rows"),
    (r"B/B1/(NUMERICAL_SENSITIVITY\.csv|numerical_sensitivity_rows\.json)", f"{S4}/b/b1_analyze.py", "B/B1/run/*", "B1 verdict; numbers.json RF4.B1.*; Table 14"),
    (r"B/B1/.*\.md", "hand-written protocol / report", "B1", "B1 protocol (before the run) / report"),
    (r"B/B2/p1c_cohort_t4\.json", f"{S4}/b/b2_cohort.py", "P1-C selection / weights (no outcomes)", "B2 cohort and recomputed weights"),
    (r"B/B2/fetch/.*", f"{S4}/b/b2_fetch_p1c_images.sh", "OpenScene test camera shards 6-31 (network)", "B2 P1-C inputs (ledger only; images not kept in the ZIP)"),
    (r"B/B2/gate_stage0.*", f"{S4}/b/run_b2.sh gate -> b2_matched_prefix.py (GPU)", "Stage-0 manifest", "B2 gate PASS"),
    (r"B/B2/pilot_throughput.*", f"{S4}/b/run_b2.sh pilot (GPU)", "B1 dev manifest alt0", "B2 cost estimate"),
    (r"B/B2/(dev_main|p1c_main)(/.*|\.log)|B/B2/p1c_chain\.log", f"{S4}/b/run_b2.sh dev|p1c, run_b2_p1c_chain.sh, b2_p1c_input_check.py (GPU)",
     "AutoVLA ckpt; P1-C re-fetched images; equal_distance / P1-C units", "B2 bank and cell outcomes (unit level)"),
    (r"B/B2/(dev_analysis|p1c_analysis)/.*", f"{S4}/b/b2_analyze.py", "B/B2/*_main/{bank,cells}.jsonl; p1c_cohort_t4.json", "B2 contrasts; numbers.json RF4.B2.*; Table 12"),
    (r"B/MATCHED_PREFIX_RESULTS\.csv", f"{S4}/b/b2_report_tables.py", "B/B2/*_analysis/MATCHED_PREFIX_RESULTS.csv", "B2 combined table"),
    (r"B/MATCHED_PREFIX_(PROTOCOL|RESULTS)\.md", "hand-written protocol (552b3ac, before main data) / report (+ appendix from b2_report_tables.py)", "B2", "B2 claims (CLAIM_CHANGES 1b, 2)"),
    (r"FEASIBILITY\.md", "hand-written (code / metadata reading)", "openpi 215abfb, FAST ec4d7aa, WorldVLA forks (see §0)", "§6 feasibility"),
    (r"OPENVLA_GATE\.(md|csv)", "hand-written (2026-10-18)", "user decision 2026-10-17; HF cache refs", "B3 NOT_RUN"),
    (r"STATUS\.md|CLAIM_CHANGES\.md|RUN_COMMANDS\.md", "hand-written (2026-10-18)", "all review_followup4 outputs", "final status / claims / commands"),
    (r"MANUSCRIPT_REVISION\.md", "hand-written + table generated from DOC_EDITS_APPLIED.csv", "A4_WORDING_VIOLATIONS.csv; CLAIM_CHANGES.md", "manuscript sentence replacements"),
    (r"DOC_EDITS_APPLIED\.csv", f"{S4}/c/c_doc_fixes.py", "A4_WORDING_VIOLATIONS.csv; EXTRA edit list", "log of in-place document edits"),
    (r"REPRO_CHECK.*", f"{S4}/c/c_repro_record.py (after zip_text/regen.sh in the unzipped copy)", "reproducibility_minimal.zip", "A5 regeneration check"),
    (r"reproducibility_minimal\.zip", f"{S4}/c/c_build_zip.py", "see MANIFEST.csv inside", "A5 package"),
    (r"SOURCE_MANIFEST\.csv", f"{S4}/c/c_source_manifest.py", "-", "this file (its own hash is not listed)"),
]
EXTERNAL = [
    (f"{ROOT}/paper_quantitative_package/numbers.json", f"{S4}/c/c_numbers.py (base: scripts/review_followup3/cp2/cp2_numbers.py)", "review_followup4 CSV/JSON outputs", "numbers bank (+RF4 rows)"),
    (f"{ROOT}/paper_quantitative_package/numbers_sources.csv", f"{S4}/c/c_numbers.py", "numbers.json rows", "row -> source map"),
    (f"{ROOT}/paper_quantitative_package/tables.tex", f"{S4}/c/c_tables.py + c_doc_fixes.py", "numbers.json", "Tables 8-14"),
    (f"{ROOT}/paper_quantitative_package/PAPER_NUMBERS.md", f"{S4}/c/c_doc_fixes.py", "CLAIM_CHANGES.md; DOC_EDITS_APPLIED.csv", "2026-10-18 section, Claims 25-27, in-place fixes"),
    (f"{ROOT}/paper_quantitative_package/MANUSCRIPT_REVISION.md", f"{S4}/c/c_doc_fixes.py", "CLAIM_CHANGES.md", "2026-10-18 revision section, in-place fixes"),
    (f"{ROOT}/paper_quantitative_package/README.md", f"{S4}/c/c_doc_fixes.py", "-", "2026-10-18 note"),
    (f"{ROOT}/paper_quantitative_package/PAPER_EVIDENCE.md", f"{S4}/c/c_doc_fixes.py", "A4_WORDING_VIOLATIONS.csv", "in-place fixes"),
    (f"{ROOT}/paper_quantitative_package/EXP35_RESULTS.md", f"{S4}/c/c_doc_fixes.py", "A4_WORDING_VIOLATIONS.csv", "in-place fixes"),
    (f"{ROOT}/paper_quantitative_package/CONSISTENCY_AUDIT.md", f"{S4}/c/c_doc_fixes.py", "A4_WORDING_VIOLATIONS.csv", "in-place fixes"),
    (f"{ROOT}/paper_quantitative_package/SOURCE_MAP.md", f"{S4}/c/c_doc_fixes.py", "A4_WORDING_VIOLATIONS.csv", "in-place fixes"),
    (f"{ROOT}/EXPERIMENT_SUMMARY.md", f"{S4}/c/c_doc_fixes.py", "all review_followup4 outputs", "§6g + in-place fixes"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def commit_of(p):
    out = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h", "--", p], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", p], capture_output=True, text=True).stdout.strip()
    if not out:
        return "uncommitted (new, 2026-10-18)"
    return out + (" + uncommitted changes (2026-10-18)" if dirty else "")


def main():
    rows = []
    for dp, dn, fn in os.walk(R4):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, R4)
            if rel == "SOURCE_MANIFEST.csv" or rel.endswith(".part"):
                continue
            rule = next((r for r in RULES if re.fullmatch(r[0], rel)), None)
            rows.append({"file": f"outputs/review_followup4/{rel}", "bytes": os.path.getsize(p), "sha256": sha(p),
                         "produced_by": rule[1] if rule else "UNMAPPED", "source_data": rule[2] if rule else "",
                         "evidence_link": rule[3] if rule else "", "commit": commit_of(p)})
    for p, by, src, ev in EXTERNAL:
        rows.append({"file": p, "bytes": os.path.getsize(p), "sha256": sha(p), "produced_by": by, "source_data": src,
                     "evidence_link": ev, "commit": commit_of(p)})
    with open(f"{R4}/SOURCE_MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    un = [r["file"] for r in rows if r["produced_by"] == "UNMAPPED"]
    print(len(rows), "rows;", len(un), "unmapped", un[:20])


if __name__ == "__main__":
    main()
