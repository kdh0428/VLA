#!/usr/bin/env python
"""Restore ONLY the model-input camera files (front, front-left, front-right; 4 history frames each) of manifest scenes by
streaming OpenScene test camera shards (same source and pattern mechanism as scripts/review_followup/p1c/p1c_driver.sh).
Never overwrites or deletes existing files; extraction scratch is inside /root/VLA/autovla/dataset/nuplan/_g4_tmp only.
Usage: python g4_fetch_images.py --manifest <g4_manifest.json> --shards 14,16 --ledger <jsonl> [--min-free-gb 1.0]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

BLOBS = "/root/VLA/autovla/dataset/nuplan/sensor_blobs/test"
TMP_BASE = "/root/VLA/autovla/dataset/nuplan/_g4_tmp"
TMP = None
HF = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera"
CAMS = ("front_camera_paths", "left_camera_paths", "right_camera_paths")   # CAM_F0, CAM_L1, CAM_R1 (nuPlan model inputs)


def free_gb():
    st = os.statvfs("/root")
    return st.f_bavail * st.f_frsize / 1e9


def needed(manifest):
    M = json.load(open(manifest))
    by = {}
    for x in M["scenes"]:
        sc = json.load(open("/root/VLA/" + x["scene_json"]))
        for k in CAMS:
            for c in sc[k]:
                if not os.path.exists(f"{BLOBS}/{c}"):
                    by.setdefault(x["shard"], set()).add(c)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shards", default="")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--worker", default="0")
    a = ap.parse_args()
    global TMP
    TMP = f"{TMP_BASE}_{a.worker}"
    by = needed(a.manifest)
    shards = [s for s in a.shards.split(",") if s] or sorted(by, key=int)
    for sh in shards:
        files = sorted(by.get(sh, ()))
        if not files:
            print(f"shard {sh}: nothing missing", flush=True)
            continue
        if free_gb() < a.min_free_gb + 0.5:
            sys.exit(f"free disk {free_gb():.2f} GB too low before shard {sh}")
        if os.path.exists(TMP):
            assert TMP.startswith(TMP_BASE + "_")
            shutil.rmtree(TMP)
        os.makedirs(TMP)
        pat = os.path.join(TMP, "patterns.txt")
        open(pat, "w").write("\n".join("*/" + f for f in files) + "\n")
        t0 = time.time()
        cmd = f"wget -qO- '{HF}/openscene_sensor_test_camera_{sh}.tgz' | tar -xz -C '{TMP}' --wildcards -T '{pat}'"
        p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd])
        while p.poll() is None:
            time.sleep(5)
            if free_gb() < a.min_free_gb:
                p.kill(); subprocess.run(["pkill", "-f", f"openscene_sensor_test_camera_{sh}.tgz"])
                sys.exit(f"disk watchdog: free < {a.min_free_gb} GB during shard {sh}")
        dt = time.time() - t0
        moved, nbytes, missing = 0, 0, []
        for f in files:
            src = None
            for root in (f"{TMP}/openscene-v1.1/sensor_blobs/test", f"{TMP}/openscene-v1.1/sensor_blobs", TMP):
                if os.path.exists(f"{root}/{f}"):
                    src = f"{root}/{f}"; break
            if src is None:
                missing.append(f); continue
            dst = f"{BLOBS}/{f}"
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            nbytes += os.path.getsize(src)
            shutil.move(src, dst); moved += 1
        shutil.rmtree(TMP)
        rec = {"shard": sh, "rc": p.returncode, "stream_s": round(dt, 1), "files_needed": len(files), "files_moved": moved,
               "bytes_moved": nbytes, "files_not_found_in_shard": len(missing), "free_gb_after": round(free_gb(), 2),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(a.ledger, "a") as fl:
            fl.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if missing:
            print("missing examples:", missing[:3], flush=True)


if __name__ == "__main__":
    main()
