#!/usr/bin/env python
"""review_followup4 A5: build outputs/review_followup4/reproducibility_minimal.zip (2026-10-18). CPU only.

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
OUT = f"{POC}/outputs/review_followup4/reproducibility_minimal.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(HERE, "zip_text")          # README / MISSING / regen scripts shipped inside the ZIP
MAX_TOTAL = 150 * 1024 ** 2                      # target compressed size

IMG = r"\.(png|jpg|jpeg|gif|pdf|mp4|webm)$"
HEAVY = r"(\.ckpt|\.pt|\.pth|\.safetensors|\.bin)$|logits_[^/]*\.npz$"
SKIP = r"(__pycache__|/_stopped_v1_20261014/|\.pyc$|reproducibility_minimal\.zip$)"

# (root, include regex on the path relative to root, exclude regex)
SPEC = [
    # ---- code: all analysis / statistics / intervention implementation (every round), tools, labelling rules
    (f"{POC}/scripts", r".*\.(py|sh|md|json|ya?ml|txt)$", r""),
    (f"{ROOT}/pra_comparison", r"[^/]*\.py$", r""),
    (f"{POC}/src", r".*\.py$", r""),
    (f"{ROOT}/vla_misalignment_poc/analysis", r"[^/]*\.py$", r""),
    (f"{ROOT}/tools", r"[^/]*\.(sh|py)$", r""),
    # ---- review_followup4 (this round): everything except images
    (f"{POC}/outputs/review_followup4", r".*", r""),
    # ---- review_followup3: P0 (population, OpenVLA records / per-task / physical units, SpatialVLA tables),
    #      P1 (intervention spec, protocol, manifests, tolerance pilot, judge logs, restore audit), P2 (locked specs,
    #      unit tables, results), P3 protocol, P4 power plan, P5 inventory
    (f"{POC}/outputs/review_followup3", r".*", r"(_extrap_data/|_extrap_models/)"),
    (f"{ROOT}/review_followup3", r".*\.md$", r""),
    # ---- review_followup2 (cluster statistics, SpatialVLA direct effects incl. state manifest / pairs, baseline-window)
    (f"{POC}/outputs/review_followup2", r".*", r""),
    # ---- review_followup (P0-P1): run-level tables, protocols, manifests, unit parquet; selected raw inputs
    (f"{POC}/outputs/review_followup", r"[^/]*\.md$", r""),
    (f"{POC}/outputs/review_followup", r"(P0A_audit|P0B_cluster_stats|P0C_exp35_fixed_axis)/.*", r""),
    (f"{POC}/outputs/review_followup", r"(P1A_gt_free_reference|P1B_start_position|P1D_dose_control|R8R9_raw_recovery)/run_20261008/[^/]+$", r""),
    (f"{POC}/outputs/review_followup", r"P1A_gt_free_reference/run_20261008/(raw|raw_fixedpos)/records\.jsonl$", r""),
    (f"{POC}/outputs/review_followup", r"P1A_gt_free_reference/run_20261008/reference/.*", r""),
    (f"{POC}/outputs/review_followup", r"P1B_start_position/run_20261008/raw/(intervention_trace\.jsonl|run_meta\.json|errors\.jsonl)$", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/[^/]*\.(md|json)$", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/design_20261008/.*", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/run_20261008/[^/]+$", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/run_20261008/(analysis|selection|units|stage1)/.*", r""),
    (f"{POC}/outputs/review_followup", r"P1C_new_log_replication/run_20261008/mech/action_history_causal/[^/]*\.(jsonl|json)$", r""),
    # ---- development inputs (natural pass, equal-distance mechanism set) and the AutoVLA action codebook
    (f"{POC}/outputs/full_extract", r"records\.jsonl$", r""),
    (f"{POC}/outputs/equal_distance_perturbation", r"[^/]*\.(jsonl|json|md|csv)$", r""),
    (f"{ROOT}/autovla/codebook_cache", r"agent_vocab\.pkl$", r""),
    # ---- OpenVLA exp 28 summary files (episode records / state mapping / per-task CSV are in review_followup3/P0)
    (f"{POC}/outputs/cross_vla_replication", r"[^/]*\.(md|json)$", r""),
    (f"{POC}/outputs/cross_vla_temporal_replication", r"[^/]*\.(md|json)$", r""),
    # ---- manuscript-facing documents
    (f"{ROOT}/paper_quantitative_package", r"[^/]*\.(md|json|csv|tex)$", r""),
    (f"{ROOT}", r"(EXPERIMENT_SUMMARY|CONCLUSIONS|AUTOVLA_NEXT_STEPS_20261018)\.md$", r""),
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
            if root in (ROOT, f"{ROOT}/tools", f"{ROOT}/pra_comparison", f"{ROOT}/paper_quantitative_package",
                        f"{POC}/outputs/full_extract", f"{POC}/outputs/equal_distance_perturbation",
                        f"{POC}/outputs/cross_vla_replication", f"{POC}/outputs/cross_vla_temporal_replication",
                        f"{ROOT}/vla_misalignment_poc/analysis"):
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
