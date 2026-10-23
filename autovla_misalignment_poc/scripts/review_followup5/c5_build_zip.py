#!/usr/bin/env python
"""review_followup5: build outputs/review_followup5/reproducibility_followup5.zip (2026-10-18). CPU only.
(Derived from scripts/review_followup4/c/c_build_zip.py; only SPEC / OUT / TEXT differ.)

Files are streamed straight from their repository location into the ZIP (no staging directory). Arc names mirror the
repository under the top folder `VLA/`, so `regen.sh` (inside the ZIP) can rewrite the hard-coded `/root/VLA` prefix of
the analysis scripts to the unzipped location and re-run them on the self-contained inputs.

Excluded on purpose (size / policy): images (png/jpg/pdf/mp4), model checkpoints (*.ckpt, *.pt, *.safetensors), raw
logits (logits_*.npz), raw rollouts / episode frames, the P2-B training-data cache (_extrap_data, re-streamable) and
the P1-C / R8R9 / P1-D mechanism raw directories (hundreds of MB). See MISSING.md inside the ZIP.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python c_build_zip.py [--check-only]
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone

ROOT = "/root/VLA"
POC = f"{ROOT}/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup5/reproducibility_followup5.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(HERE, "zip_text")          # README / MISSING / regen scripts shipped inside the ZIP
MAX_TOTAL = 150 * 1024 ** 2                      # target compressed size

IMG = r"\.(png|jpg|jpeg|gif|pdf|mp4|webm)$"
HEAVY = r"(\.ckpt|\.pt|\.pth|\.safetensors|\.bin)$|logits_[^/]*\.npz$"
SKIP = r"(__pycache__|/_stopped_v1_20261014/|\.pyc$|reproducibility_minimal\.zip$|reproducibility_followup5\.zip$)"

# (root, include regex on the path relative to root, exclude regex)
SPEC = [
    # ---- code (all rounds) and labelling rules
    (f"{POC}/scripts", r".*\.(py|sh|md|json|ya?ml|txt)$", r""),
    (f"{ROOT}/pra_comparison", r"[^/]*\.py$", r""),
    (f"{POC}/src", r".*\.py$", r""),
    (f"{ROOT}/vla_misalignment_poc/analysis", r"[^/]*\.py$", r""),
    # ---- review_followup5 outputs (protocol, A_STATS, A_B2_CLUSTER, B_NUMERIC raw + analysis, C_DISJOINT, D_RELEASE, docs)
    (f"{POC}/outputs/review_followup5", r".*", r""),
    # ---- inputs of the CPU re-analyses
    (f"{POC}/outputs/review_followup4/B/B2/p1c_analysis", r".*", r""),
    (f"{POC}/outputs/review_followup4/B", r"MATCHED_PREFIX_(PROTOCOL|RESULTS)\.(md|csv)$", r""),
    (f"{POC}/outputs/review_followup4/A1_A4", r"(A1_CALLSITES|COHORT_ESTIMAND_MAP)\.csv$", r""),
    (f"{POC}/outputs/equal_distance_perturbation", r"records\.jsonl$", r""),
    (f"{POC}/outputs/review_followup", r"P1A_gt_free_reference/run_20261008/raw/records\.jsonl$", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/run_20261008/analysis/(units\.parquet|decisions\.json|paired_effects\.csv)$", r""),
    (f"{POC}/outputs/review_followup", r"P0B_cluster_stats/run_20261008/units\.parquet$", r""),
    (f"{POC}/outputs/review_followup3/P2", r"(p2_units|extrap_units)\.parquet$|MOTION_BASELINE_SPEC_LOCKED_v1\.md$|LEARNED_EXTRAPOLATOR_PROTOCOL\.md$", r""),
    (f"{POC}/outputs/review_followup3/P0", r"(POPULATION_EFFECTS|DEPENDENCE_SENSITIVITY|DEPENDENCE_disjoint_subsets|DISJOINT_LOG_INVENTORY|DISJOINT_LOG_INVENTORY_summary)\.csv$|DISJOINT_LOG_INVENTORY_meta\.json$", r""),
    (f"{POC}/outputs/review_followup2/A_autovla_cluster/run_20261013", r"(CLUSTER_REANALYSIS|cluster_reanalysis_long|p1c_subsets_drive_level)\.csv$", r""),
    (f"{POC}/outputs/review_followup2/C_baseline_window/run_20261013", r"[^/]*\.csv$", r""),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect():
    seen, files = set(), []
    for root, inc, exc in SPEC:
        for dp, dn, fn in os.walk(root):
            dn.sort()
            for f in sorted(fn):
                p = os.path.join(dp, f)
                r = os.path.relpath(p, root)
                if not re.fullmatch(inc, r) or (exc and re.search(exc, r)):
                    continue
                if re.search(IMG, p, re.I) or re.search(HEAVY, p) or re.search(SKIP, p) or os.path.islink(p):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                files.append(p)
            if root in (f"{ROOT}/pra_comparison", f"{POC}/outputs/equal_distance_perturbation",
                        f"{ROOT}/vla_misalignment_poc/analysis", f"{POC}/outputs/review_followup3/P2", f"{POC}/outputs/review_followup3/P0"):
                break   # top level only
    return files


def main():
    files = collect()
    raw = sum(os.path.getsize(p) for p in files)
    print(f"{len(files)} files, {raw / 1024 ** 2:.1f} MB uncompressed")
    if "--check-only" in sys.argv:
        big = sorted(files, key=os.path.getsize)[-15:]
        for p in big:
            print(f"{os.path.getsize(p) / 1024 ** 2:8.1f} MB  {p}")
        return
    free = shutil.disk_usage("/root").free
    assert free > 3 * raw / 4 + 300 * 1024 ** 2, f"not enough free disk ({free / 1024 ** 3:.2f} GB)"
    man = io.StringIO()
    w = csv.writer(man)
    w.writerow(["arcname", "bytes", "sha256", "source_path"])
    tmp = OUT + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            arc = "VLA/" + os.path.relpath(p, ROOT)
            z.write(p, arc)
            w.writerow([arc, os.path.getsize(p), sha(p), p])
        for f in sorted(os.listdir(TEXT)):
            src = os.path.join(TEXT, f)
            z.write(src, f)
            w.writerow([f, os.path.getsize(src), sha(src), src])
        z.writestr("MANIFEST.csv", man.getvalue())
        z.writestr("BUILD_INFO.json", json.dumps({
            "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "builder": os.path.abspath(__file__), "n_files": len(files), "uncompressed_bytes": raw,
            "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()}, indent=1))
    os.replace(tmp, OUT)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}: {size / 1024 ** 2:.1f} MB")
    assert size < MAX_TOTAL, "ZIP above the 150 MB target"


if __name__ == "__main__":
    main()
