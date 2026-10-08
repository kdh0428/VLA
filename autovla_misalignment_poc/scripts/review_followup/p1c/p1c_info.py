#!/usr/bin/env python
"""
Small CPU helpers for the P1-C driver (no deletion here).
  shards   --role R                       shard numbers holding logs of role R, ordered by the min eval_order_key
                                           (split.csv; sha256 of drive|log) of their logs (pilot: numeric order)
  logs     --role R --shard N             logs of role R in shard N (one per line), each validated
  tokens   --logs L.. --out F             scene tokens of these logs -> JSON list (refuses to overwrite)
  budget   --logs L..                     expected bytes of CAM_F0/L1/R1 images for these logs (all log frames x 3 cams)
  check-images --logs L..                 count image files the scenes of these logs need and how many are missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

BYTES_PER_FRAME_3CAM = 641079.0   # design_20261008/work/shard_budget.json


def frames_of_logs(logs):
    fr = set()
    for t in C.tokens_of_logs(logs):
        sc = json.load(open(os.path.join(C.SCENES, f"{t}.json")))
        fr.update(sc["front_camera_paths"])
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("shards"); p.add_argument("--role", required=True)
    p = sp.add_parser("logs"); p.add_argument("--role", required=True); p.add_argument("--shard", required=True)
    p = sp.add_parser("tokens"); p.add_argument("--logs", nargs="+", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("budget"); p.add_argument("--logs", nargs="+", required=True)
    p = sp.add_parser("check-images"); p.add_argument("--logs", nargs="+", required=True)
    a = ap.parse_args()
    split = C.load_split()
    prot = C.protected_logs()
    if a.cmd == "shards":
        by = {}
        for l, r in split.items():
            if r["role"] == a.role:
                by.setdefault(int(r["shard"]), []).append(r.get("eval_order_key") or l)
        order = sorted(by, key=lambda s: (min(by[s]), s)) if a.role == "eval" else sorted(by)
        print(" ".join(map(str, order)))
    elif a.cmd == "logs":
        for l, r in sorted(split.items()):
            if r["role"] == a.role and r["shard"] == str(int(a.shard)):
                assert C.LOG_RE.match(l) and l not in prot, l
                print(l)
    elif a.cmd == "tokens":
        if os.path.exists(a.out):
            print(f"{a.out} exists (kept)"); return
        toks = C.tokens_of_logs(a.logs)
        json.dump(toks, open(a.out, "w"))
        print(len(toks))
    elif a.cmd == "budget":
        import pickle
        n = sum(len(pickle.load(open(os.path.join(C.NUP, "navsim_logs/test", f"{l}.pkl"), "rb"))) for l in a.logs)
        print(int(n * BYTES_PER_FRAME_3CAM))
    else:
        need = set()
        for t in C.tokens_of_logs(a.logs):
            need.update(C.image_files_of_scene(json.load(open(os.path.join(C.SCENES, f"{t}.json")))))
        miss = [p for p in need if not os.path.isfile(os.path.join(C.SENSOR, p))]
        print(json.dumps({"needed": len(need), "missing": len(miss), "examples": sorted(miss)[:3]}))


if __name__ == "__main__":
    main()
