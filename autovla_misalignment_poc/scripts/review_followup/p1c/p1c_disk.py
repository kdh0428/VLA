#!/usr/bin/env python
"""
P1-C image-directory housekeeping with strict guards (D7c). Every deletion is appended to
outputs/review_followup/P1C_new_log_replication/DISK_CLEANUP.md (path, files, bytes, reason, how to regenerate) BEFORE
anything is removed. Only these targets can ever be deleted:
  - /root/VLA/autovla/dataset/nuplan/sensor_blobs/test/<LOG>/...  where LOG matches the log-name regex, has role
    `eval` or `pilot_harness_check` in split.csv, is NOT a PoC (shards 0-5) log, and is listed in the run's
    fetched-log ledger (i.e. this pipeline moved it in);
  - /root/VLA/autovla/dataset/nuplan/_p1c_tmp (the extraction scratch dir).
Paths are resolved with realpath and must equal the expected literal location (no symlinks, no '..').

  prune      --ledger L --keep keep_files.json --logs LOG..   delete images of LOG not needed by selected scenes
  remove     --ledger L --logs LOG..                           delete LOG's whole image dir
  clean-tmp                                                    delete the extraction scratch dir
  check-free --min-gb G                                        exit 10 if free disk < G GB
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

SENSOR = "/root/VLA/autovla/dataset/nuplan/sensor_blobs/test"
TMP = "/root/VLA/autovla/dataset/nuplan/_p1c_tmp"
LEDGER_DIR_PREFIX = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/"
CLEANUP_MD = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/DISK_CLEANUP.md"


def free_gb(path="/"):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def log_entry(path, n_files, n_bytes, reason, regen):
    new = not os.path.exists(CLEANUP_MD)
    with open(CLEANUP_MD, "a") as f:
        if new:
            f.write("# P1-C disk cleanup log\n\nEvery deletion made by `scripts/review_followup/p1c/p1c_disk.py` is written here "
                    "before it happens. Columns: UTC, path, files, bytes, reason, regeneration.\n\n"
                    "| UTC | path | files | bytes | reason | regeneration |\n|---|---|---|---|---|---|\n")
        f.write(f"| {time.strftime('%FT%TZ', time.gmtime())} | `{path}` | {n_files} | {n_bytes} | {reason} | {regen} |\n")
        f.flush(); os.fsync(f.fileno())


def guarded_log_dir(log, ledger):
    split = C.load_split()
    if not C.LOG_RE.match(log):
        raise SystemExit(f"refusing: bad log name {log!r}")
    if log in C.protected_logs():
        raise SystemExit(f"refusing: {log} is a PoC (protected) log")
    if split.get(log, {}).get("role") not in ("eval", "pilot_harness_check"):
        raise SystemExit(f"refusing: {log} is not an eval/pilot log in split.csv")
    if log not in ledger:
        raise SystemExit(f"refusing: {log} is not in the fetched-log ledger (this pipeline did not move it in)")
    d = os.path.join(SENSOR, log)
    if os.path.islink(d) or os.path.realpath(d) != d or os.path.dirname(d) != SENSOR:
        raise SystemExit(f"refusing: {d} is not the expected real directory")
    return d


def read_ledger(p):
    p = os.path.realpath(p)
    if not p.startswith(LEDGER_DIR_PREFIX) or not os.path.isfile(p):
        raise SystemExit(f"bad ledger {p}")
    led = set()
    for l in open(p):
        if l.strip():
            led.add(json.loads(l)["log"])
    return led


def dir_stats(d, only=None):
    n = b = 0
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            if only is None or p in only:
                n += 1; b += os.path.getsize(p)
    return n, b


def cmd_prune(a):
    led = read_ledger(a.ledger)
    keep_rel = set(json.load(open(a.keep)))
    shard_of = {l: r["shard"] for l, r in C.load_split().items()}
    for log in a.logs:
        d = guarded_log_dir(log, led)
        if not os.path.isdir(d):
            print(f"[prune] {log}: no dir"); continue
        victims = []
        for root, dirs, fs in os.walk(d):
            for f in fs:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, SENSOR)
                if rel not in keep_rel:
                    victims.append(p)
        kept = dir_stats(d)[0] - len(victims)
        nb = sum(os.path.getsize(p) for p in victims)
        log_entry(f"{d}/CAM_*/<{len(victims)} files not used by selected scenes>", len(victims), nb,
                  f"{a.reason}; kept {kept} files needed by the F/S-selected scenes",
                  f"`bash scripts/review_followup/p1c/p1c_driver.sh` fetch of shard {shard_of.get(log)} (CAM_F0/L1/R1 of this log)")
        for p in victims:
            assert os.path.realpath(p).startswith(d + "/") and not os.path.islink(p)
            os.remove(p)
        for root, dirs, fs in sorted(os.walk(d), key=lambda x: -len(x[0])):
            if root != d and not os.listdir(root):
                os.rmdir(root)
        print(f"[prune] {log}: deleted {len(victims)} files ({nb/1e6:.1f} MB), kept {kept}; free {free_gb():.2f} GB", flush=True)


def cmd_remove(a):
    led = read_ledger(a.ledger)
    shard_of = {l: r["shard"] for l, r in C.load_split().items()}
    for log in a.logs:
        d = guarded_log_dir(log, led)
        if not os.path.isdir(d):
            print(f"[remove] {log}: no dir"); continue
        n, b = dir_stats(d)
        log_entry(d, n, b, a.reason, f"re-fetch CAM_F0/L1/R1 of this log from navtest camera shard {shard_of.get(log)} "
                  f"(`p1c_driver.sh`)")
        shutil.rmtree(d)
        print(f"[remove] {log}: {n} files ({b/1e6:.1f} MB); free {free_gb():.2f} GB", flush=True)


def cmd_clean_tmp(a):
    d = TMP
    if os.path.islink(d):
        raise SystemExit("refusing: tmp is a symlink")
    if not os.path.exists(d):
        return
    n, b = dir_stats(d)
    log_entry(d, n, b, a.reason, "scratch extraction dir; nothing to regenerate (re-created by the next fetch)")
    shutil.rmtree(d)
    print(f"[clean-tmp] {n} files ({b/1e6:.1f} MB); free {free_gb():.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prune"); p.add_argument("--ledger", required=True); p.add_argument("--keep", required=True)
    p.add_argument("--logs", nargs="+", required=True); p.add_argument("--reason", required=True)
    p = sp.add_parser("remove"); p.add_argument("--ledger", required=True); p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--reason", required=True)
    p = sp.add_parser("clean-tmp"); p.add_argument("--reason", default="extraction scratch after moving the logs")
    p = sp.add_parser("check-free"); p.add_argument("--min-gb", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "prune":
        cmd_prune(a)
    elif a.cmd == "remove":
        cmd_remove(a)
    elif a.cmd == "clean-tmp":
        cmd_clean_tmp(a)
    else:
        g = free_gb()
        print(f"free {g:.2f} GB (min {a.min_gb})")
        sys.exit(0 if g >= a.min_gb else 10)


if __name__ == "__main__":
    main()
