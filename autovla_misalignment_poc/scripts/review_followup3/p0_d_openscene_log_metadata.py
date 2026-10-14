#!/usr/bin/env python
"""
P0-D / P5 preparation (review_followup3, 2026-10-14): per-log metadata of the OpenScene v1.1 splits that define the
AutoVLA training data (trainval) and the candidate pool (test).

POST-HOC data inventory. CPU only, no model inference. Nothing is extracted to disk: the public OpenScene metadata
tarball is streamed (curl | tarfile 'r|gz') and, for each per-log pickle (meta_datas/<split>/<log_name>.pkl), only
log-level fields are kept: log_name, vehicle_name, map_location (city), number of frames, first/last timestamp,
number of frames with a non-empty route (roadblock_ids). The tarball itself is never written.

Why streamed from the source: the AutoVLA repo configs (config/training/qwen2.5-vl-3B-mix-sft.yaml,
qwen2.5-vl-3B-nuplan-grpo-cot.yaml) train on ./dataset/nuplan/trainval with scene filter
train_test_split/trainval.yaml (data_split trainval, scene_filter all_scenes, log_names = all). The trainval log
list is not on this machine; navtrain.yaml (1,192 logs) is only a subset of it.

Usage:
  nice -n 19 python p0_d_openscene_log_metadata.py test
  nice -n 19 python p0_d_openscene_log_metadata.py trainval
Output: outputs/review_followup3/P0/_external/openscene_v1.1_<split>_log_metadata.csv (+ .json provenance)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tarfile
import time

import pandas as pd

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P0/_external"
URL = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{rev}/openscene-v1.1/openscene_metadata_{split}.tgz"
REV = "a76f840b65e972bc45e56c2adced897498e9a026"     # dataset repo commit observed 2026-10-14 (x-repo-commit)


def main(split: str):
    os.makedirs(OUT, exist_ok=True)
    url = URL.format(rev=REV, split=split)
    t0 = time.time()
    p = subprocess.Popen(["curl", "-sL", "--fail", url], stdout=subprocess.PIPE)
    h = hashlib.sha256()

    class Tee:                                   # hash the compressed stream while tarfile reads it
        def read(self, n=-1):
            b = p.stdout.read(n)
            h.update(b)
            return b

    rows, n_members = [], 0
    with tarfile.open(fileobj=Tee(), mode="r|gz") as tf:
        for m in tf:
            n_members += 1
            if not (m.isfile() and m.name.endswith(".pkl")):
                continue
            f = tf.extractfile(m)
            d = pickle.load(f)
            log = os.path.basename(m.name)[:-4]
            fr = d if isinstance(d, list) else []
            first = fr[0] if fr else {}
            rows.append({"split": split, "member": m.name, "log_name": first.get("log_name", log), "file_log": log,
                         "vehicle_name": first.get("vehicle_name"), "map_location": first.get("map_location"),
                         "n_frames": len(fr), "n_frames_with_route": sum(1 for x in fr if x.get("roadblock_ids")),
                         "t_first_us": first.get("timestamp"), "t_last_us": fr[-1].get("timestamp") if fr else None,
                         "member_bytes": m.size})
            if len(rows) % 200 == 0:
                print(f"[{time.time() - t0:6.0f}s] {len(rows)} logs", flush=True)
    rc = p.wait()
    df = pd.DataFrame(rows)
    dst = os.path.join(OUT, f"openscene_v1.1_{split}_log_metadata.csv")
    df.to_csv(dst, index=False)
    prov = {"split": split, "url": url, "dataset_repo_commit": REV, "curl_returncode": rc,
            "stream_sha256": h.hexdigest(), "n_tar_members": n_members, "n_log_pickles": len(rows),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
            "note": "streamed, never written to disk; only log-level fields kept"}
    json.dump(prov, open(dst.replace(".csv", ".json"), "w"), indent=1)
    print(json.dumps(prov, indent=1))
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1])
