#!/usr/bin/env python
"""review_followup5 B_NUMERIC runner (RTX 5090 only; protocol outputs/review_followup5/FOLLOWUP5_PROTOCOL.md §3).

P1-A family-G units, path P1A of the P1-verified engine (scripts/review_followup3/p1/p1b_engine.py, unchanged), experiment
batch (10 rows, original order), t001 (T = 0.01), seed 0, key "original". Settings run sequentially in one process:
S0 (bf16, sdpa) -> SE (eager attention) -> SF (bf16-rounded weights upcast to fp32, sdpa). Switches reuse
scripts/review_followup4/b/b1_runner.py (set_attn / set_precision / param_checksum / cmp_rows) unchanged.

  --mode smoke : first N units: S0 twice (bitwise), S0 vs stored raw tokens, teacher-forced full recompute, prefix lock
  --mode main  : all units, settings S0,SE,SF; records.jsonl (no logit arrays) + logit_cmp.jsonl (summary vs S0)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup3/p1")
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup4/b")
import p1b_engine as E  # noqa: E402
import b1_runner as B1  # noqa: E402
C = E.C
ROWS = [r["name"] for r in E.path_rows("P1A")]


def load(manifest):
    man = json.load(open(manifest))
    units = man["units"]
    ed = {r["token"]: r for r in map(json.loads, open(C.ED_RECORDS))}
    stub_of, _, _ = C.load_stubs_and_token_freq({u["scene"] for u in units})
    cb, disp, _ = C.codebook_geometry()
    for u in units:      # the stored p must be the G-rule token of the unit's own sector (frozen; checked)
        gsel = C.select_gtfree_perturbations(disp, cb, u["ref_tokens"][u["t_p"]], 0.10, 0.03)
        s = C.SECTORS.index(u["sector"])
        assert gsel[s]["token"] == u["p_token"], (u["unit_id"], gsel[s]["token"], u["p_token"])
    return man, units, ed, stub_of


def gpu_check(eng):
    name = eng.torch.cuda.get_device_name(0)
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    assert "5090" in name, f"not the RTX 5090: {name}"
    return {"device_name": name, "nvidia_smi": uuid.strip(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER")}


def run_unit(eng, u, ed, stub, **kw):
    res = eng.decode("P1A", u, ed[u["scene"]], stub, decoding="t001", seed=0, batch="exp", **kw)
    return B1.eff_spans(B1.slim(res, ROWS))


def prefix_ok(u, rows):
    t, r, p = u["t_p"], u["ref_tokens"], u["p_token"]
    out = True
    for n, x in rows.items():
        exp_t = r[t] if n in ("U", "C1") else p
        out &= list(x["acts"][:t]) == list(r[:t]) and x["acts"][t] == exp_t and list(x["acts"][t + 1:]) == list(x["own"])
    return bool(out)


def smoke(args, eng, units, ed, stub_of):
    rep = {"units": [], "fail": 0}
    for u in units[:args.n_smoke]:
        st = stub_of[u["scene"]]
        a = run_unit(eng, u, ed, st)
        b = run_unit(eng, u, ed, st)
        rec = {"unit_id": u["unit_id"], "repeat_bitwise": all(B1.cmp_rows(a[n], b[n])["bitwise"] is not False and a[n]["own"] == b[n]["own"] for n in ROWS),
               "stored_raw_equal": {n: list(a[n]["acts"]) == list(u["stored_rows_acts"][n]) for n in ROWS},
               "prefix_lock": prefix_ok(u, a)}
        force = {n: a[n]["own"] for n in ("U", "OC")}
        full = eng.decode("P1A", u, ed[u["scene"]], st, decoding="t001", seed=0, forward="full", force=force, row_names=["U", "OC"])
        d = []
        for n in ("U", "OC"):
            lf = full["rows"][n]["logits"].astype(np.float64)
            ls = a[n]["logits"].astype(np.float64)
            d.append(float(np.abs(lf - ls).max()))
            rec[f"full_pos_eq_{n}"] = all(x for x in full["rows"][n]["pos_full_eq_cached"] if x is not None)
        rec["full_recompute_max_dlogit"] = max(d)
        ok = rec["repeat_bitwise"] and rec["prefix_lock"] and rec["full_recompute_max_dlogit"] <= 0.75 and rec["full_pos_eq_U"] and rec["full_pos_eq_OC"]
        rec["pass"] = bool(ok)
        rep["fail"] += int(not ok)
        rep["units"].append(rec)
        print(json.dumps(rec), flush=True)
    rep["stored_raw_equal_rows"] = sum(v for r in rep["units"] for v in r["stored_raw_equal"].values())
    rep["stored_raw_rows"] = sum(len(r["stored_raw_equal"]) for r in rep["units"])
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    ap.add_argument("--settings", default="S0,SE,SF")
    ap.add_argument("--n-smoke", type=int, default=4)
    ap.add_argument("--cap-min", type=float, default=180.0)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    man, units, ed, stub_of = load(args.manifest)
    eng = E.Engine()
    meta = {"script": os.path.abspath(__file__), "argv": sys.argv, "mode": args.mode, **gpu_check(eng), **eng.env(),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_units": len(units), "switches": {}, "skipped": {}}
    if args.mode == "smoke":
        rep = smoke(args, eng, units, ed, stub_of)
        meta.update(rep)
        meta["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(meta, open(os.path.join(out, "smoke_report.json"), "w"), indent=1, default=str)
        print("SMOKE fail =", rep["fail"], "stored raw rows equal", rep["stored_raw_equal_rows"], "/", rep["stored_raw_rows"])
        return
    if os.path.exists(os.path.join(out, "records.jsonl")):
        raise SystemExit("records.jsonl exists; refusing to overwrite")
    fr = open(os.path.join(out, "records.jsonl"), "w")
    fl = open(os.path.join(out, "logit_cmp.jsonl"), "w")
    s0 = {}
    t0 = time.time()
    ck0 = B1.param_checksum(eng)
    timing = []
    for setting in args.settings.split(","):
        if (time.time() - t0) / 60 > args.cap_min:
            meta["skipped"][setting] = "GPU-time cap reached"; break
        if shutil.disk_usage("/root").free < 0.5e9:
            meta["skipped"][setting] = "disk < 0.5 GB"; break
        if setting == "SE":
            meta["switches"]["SE"] = B1.set_attn(eng, "eager")
        if setting == "SF":
            free, _ = eng.torch.cuda.mem_get_info()
            if free < 17e9:
                meta["skipped"]["SF"] = f"BLOCKED: insufficient GPU memory for fp32 at B = 10 (free {free/1e9:.1f} GB)"; continue
            meta["switches"]["SF"] = B1.set_precision(eng, "fp32")
        ts = time.time()
        for u in units:
            tu = time.time()
            rows = None
            for attempt in range(21):
                try:
                    rows = run_unit(eng, u, ed, stub_of[u["scene"]]); break
                except eng.torch.OutOfMemoryError:
                    eng.torch.cuda.empty_cache(); print(f"OOM {setting} {u['unit_id']} attempt {attempt}", flush=True); time.sleep(30)
            if rows is None:
                meta["skipped"].setdefault(f"{setting}_oom_units", []).append(u["unit_id"]); continue
            for n, r in rows.items():
                fr.write(C.jdump({"setting": setting, "unit_id": u["unit_id"], "scene": u["scene"], "log": u["log"], "group": u["group"],
                                  "t_p": u["t_p"], "row": n, "B": r.get("B"), "own": r["own"], "acts": r["acts"],
                                  "poses": np.round(r["poses"], 6).tolist(), "margin": r["margin"], "ent": r["ent"],
                                  "prefix_lock": prefix_ok(u, {n: r})}) + "\n")
                if setting == "S0":
                    s0[(u["unit_id"], n)] = r
                else:
                    ref = s0.get((u["unit_id"], n))
                    if ref is not None:
                        c = B1.cmp_rows(ref, r)
                        c.pop("logit_maxabs_per_step", None)
                        fl.write(C.jdump({"setting": setting, "unit_id": u["unit_id"], "row": n, **c}) + "\n")
            fr.flush(); fl.flush()
            timing.append({"setting": setting, "unit": u["unit_id"], "sec": round(time.time() - tu, 3)})
            print(f"{setting} {u['unit_id']} {time.time()-tu:.1f}s total {(time.time()-t0)/60:.1f} min", flush=True)
        meta.setdefault("setting_min", {})[setting] = (time.time() - ts) / 60
        if setting == "SE":
            meta["switches"]["SE_restore"] = B1.set_attn(eng, "sdpa")
        if setting == "SF":
            meta["switches"]["SF_restore"] = B1.set_precision(eng, "bf16")
            meta["param_checksum_restored_equal"] = B1.param_checksum(eng) == ck0
    fr.close(); fl.close()
    meta.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_min": (time.time() - t0) / 60,
                 "timing": timing, "peak_mem_gb": eng.torch.cuda.max_memory_allocated() / 1e9, "param_checksum_start": ck0})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1, default=str)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
