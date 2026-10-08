#!/usr/bin/env python
"""Write manifest.json for a review-follow-up run dir: sha256 of every output file (excluding the
manifest itself), the scripts used, input files, and environment (commits, GPU env vars). CPU only."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    run = os.path.abspath(a.run_dir)
    files = {}
    for root, _, fs in os.walk(run):
        for f in sorted(fs):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, run)
            if rel == "manifest.json" or os.path.getsize(p) > 4e9:
                continue
            files[rel] = {"sha256": C.sha256_file(p), "bytes": os.path.getsize(p)}
    sdir = os.path.dirname(os.path.abspath(__file__))
    scripts = {f: C.sha256_file(os.path.join(sdir, f)) for f in sorted(os.listdir(sdir)) if f.endswith((".py", ".sh"))}
    orig = {f: C.sha256_file(os.path.join(C.SCRIPTS, f)) for f in ("action_history_causal.py", "analyze_action_history.py",
            "prev_action_state_patching.py", "temporal_feedback_window.py", "analyze_prev_action_state_patching.py",
            "analyze_temporal_feedback_window.py", "equal_distance_perturbation.py")}
    man = {"experiment": a.experiment, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": a.note,
           "repo_head_VLA": C.git_head("/root/VLA"), "repo_head_autovla": C.git_head(C.AUTOVLA_DIR),
           "inputs": {"ed_records": C.ED_RECORDS, "ed_records_sha256": C.sha256_file(C.ED_RECORDS),
                      "full_records": C.FULL_RECORDS, "action_history_records": C.AH_RECORDS,
                      "codebook_sha256": C.sha256_file(C.CODEBOOK), "config_sha256": C.sha256_file(C.CONFIG),
                      "checkpoint": C.CHECKPOINT, "checkpoint_bytes": os.path.getsize(C.CHECKPOINT)},
           "gpu_env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "device": "NVIDIA GeForce RTX 5090"},
           "scripts_review_followup_p1ab": scripts, "original_scripts_sha256": orig, "files": files}
    json.dump(man, open(os.path.join(run, "manifest.json"), "w"), indent=1)
    print(f"manifest: {len(files)} files")


if __name__ == "__main__":
    main()
