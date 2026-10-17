#!/usr/bin/env python
"""
B2 AutoVLA matched-prefix factorial (RTX 5090 only; protocol outputs/review_followup4/B/MATCHED_PREFIX_PROTOCOL.md).

Unit = (scene, perturbation) with forced token f at t* (exp 7 / P1-C units: 'original' = pred[t*], 'alt<j>').
Anchor a = t* + 1 + n_old + 1 (primary n_old = 1 -> a = t*+3; sensitivity n_old = 2 -> a = t*+4).
  older prefix  = context positions t*+1 .. a-2   factor O in {GT, NAT}
  latest block  = context position  a-1           factor L in {REF (gt), NSRC (frozen Normal), CSRC (frozen Full-GT branch)}
Phase A (source bank, built for the whole cohort BEFORE any cell): the P1-verified H7 path (p1b_engine.decode, B = 1,
  t001 CRN) rows 'normal' and 'gt_history'; stored: tokens, decoded per-token motion, span hashes. Bank file sha256 frozen.
Phase B (cells): one prefix KV cache per unit (B = 1; identical to the H7 prefix), every cell decoded ALONE (B = 1) on it:
  executed tokens  = pred[:t*] + [f] + bankN[t*+1 .. a-1] + own[a .. 9]     (identical pre-anchor for every cell)
  context at k>=a  = [f] + O-source[t*+1 .. a-2] + L-source[a-1] + own[a .. k-1]  (edited prefix is KEPT for all later steps)
  rows: the 6 cells + 'noop' (no override: context = executed tokens; identical values to (NAT, NSRC) by construction).
  CRN key = the unit's perturbation (as in H7/H8), so (NAT, NSRC) must reproduce the bank Normal bitwise.
Gate mode adds: c5a repeat, full recompute (no cache, B = 1, teacher-forced), incremental KV decoding, (NAT,NSRC) vs
bank Normal bitwise, value-identity whenever two cells' spans coincide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup3/p1")
import p1b_engine as E  # noqa: E402
C = E.C

POC = "/root/VLA/autovla_misalignment_poc"
P1C_RUN = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008"
CELLS = [("GT", "REF"), ("GT", "NSRC"), ("GT", "CSRC"), ("NAT", "REF"), ("NAT", "NSRC"), ("NAT", "CSRC")]
T_EQUIV = 0.75                                    # review_followup3 P1 pilot_c5 tolerances.json (unchanged)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def disk_free_gb():
    return shutil.disk_usage("/root").free / 1e9


# ---------------------------------------------------------------------------------------------- cohorts
def cohort_units(cohort, manifest=None):
    """returns (units, ed_by_token, stub_of). Units carry drive/day-vehicle ids; no outcome is read."""
    if cohort == "stage0":
        man = json.load(open(manifest))
        ed = {r["token"]: r for r in map(json.loads, open(C.ED_RECORDS))}
        units = [dict(u, perturbation="alt0") for u in man["units"] if "H7" in u["paths"] or "H8" in u["paths"]]
        units = [{"unit_id": u["unit_id"], "scene": u["scene"], "log": u["log"], "t_star": u["t_star"], "perturbation": "alt0",
                  "forced_token": u["forced_token"], "stratum": u["group"], "weight": 1.0} for u in units]
        stub_of, _, _ = C.load_stubs_and_token_freq({u["scene"] for u in units})
        return units, ed, stub_of
    if cohort == "dev":
        ed = {r["token"]: r for r in map(json.loads, open(C.ED_RECORDS))}
        units = []
        for r in ed.values():
            perts = [("original", r["pred"][r["t_star"]])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
            for pn, f in perts:
                units.append({"unit_id": f"{r['token']}|{pn}", "scene": r["token"], "log": r["log"], "t_star": r["t_star"],
                              "perturbation": pn, "forced_token": f, "stratum": r["group"], "weight": 1.0})
        stub_of, _, _ = C.load_stubs_and_token_freq({u["scene"] for u in units})
        return units, ed, stub_of
    if cohort == "p1c":
        C.SCENES = "/root/VLA/autovla/dataset/nuplan/navtest_ext"            # P1-C scene JSONs (images must be present)
        ed = {r["token"]: r for r in map(json.loads, open(f"{P1C_RUN}/units/records.jsonl"))}
        w = json.load(open(manifest))["scene_weights"]                     # recomputed domain weights (b2_cohort.py)
        units = []
        for r in ed.values():
            if r["token"] not in w:
                continue
            perts = [("original", r["pred"][r["t_star"]])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
            for pn, f in perts:
                units.append({"unit_id": f"{r['token']}|{pn}", "scene": r["token"], "log": r["log"], "t_star": r["t_star"],
                              "perturbation": pn, "forced_token": f, **w[r["token"]]})
        stub_of = {}
        want = {u["scene"] for u in units}
        for line in open(f"{P1C_RUN}/stage1/records.jsonl"):
            x = json.loads(line)
            if x["token"] in want:
                stub_of[x["token"]] = x["arms"]["N"]["token_ids"]
        return units, ed, stub_of
    raise ValueError(cohort)


# ---------------------------------------------------------------------------------------------- phase A
def bank_unit(eng, u, e, stub, seed, cb_disp, keep_logits=False):
    res = eng.decode("H7", {"scene": u["scene"], "t_p": u["t_star"], "forced_token": u["forced_token"],
                            "perturbation": u["perturbation"], "unit_id": u["unit_id"]}, e, stub,
                     decoding="t001", seed=seed, row_names=["normal", "gt_history"])
    n, g = res["rows"]["normal"], res["rows"]["gt_history"]
    t = u["t_star"]
    rec = {"unit_id": u["unit_id"], "scene": u["scene"], "t_star": t, "perturbation": u["perturbation"], "forced_token": u["forced_token"],
           "normal_own": [int(x) for x in n["own"]], "full_own": [int(x) for x in g["own"]],
           "normal_motion": [[round(float(v), 5) for v in cb_disp[a]] for a in n["own"]],
           "full_motion": [[round(float(v), 5) for v in cb_disp[a]] for a in g["own"]],
           "normal_span_sha": n["span_sha"], "full_span_sha": g["span_sha"], "prefix_ids_sha": res["prefix"]["prefix_ids_sha"],
           "pixel_sha": res["prefix"]["pixel_sha"], "normal_acts": [int(x) for x in n["acts"]],
           "normal_poses": np.round(np.asarray(n["poses"]), 6).tolist(), "full_acts": [int(x) for x in g["acts"]],
           "full_poses": np.round(np.asarray(g["poses"]), 6).tolist()}
    mem = {"normal_logits": n["logits"], "normal_span": n["span"]} if keep_logits else None
    return rec, mem


# ---------------------------------------------------------------------------------------------- phase B
def cell_context(cell, t, a, bank, gt):
    """edited context tokens for positions t+1 .. a-1 (relative to t+1)."""
    bn = bank["normal_own"]                                  # bn[j - t - 1] = Normal token at position j
    bf = bank["full_own"]
    if cell == "noop":
        return [bn[j - t - 1] for j in range(t + 1, a)], ["own"] * (a - t - 1)
    O, L = cell
    ctx, src = [], []
    for j in range(t + 1, a - 1):
        ctx.append(gt[j] if O == "GT" else bn[j - t - 1]); src.append("gt" if O == "GT" else "bankN")
    ctx.append({"REF": gt[a - 1], "NSRC": bn[a - 1 - t - 1], "CSRC": bf[a - 1 - t - 1]}[L])
    src.append({"REF": "gt", "NSRC": "bankN", "CSRC": "bankF"}[L])
    return ctx, src


def run_cells(eng, u, e, stub, bank, a, seed, forward="span", force=None, cells=None, prefix_state=None):
    """decode every cell alone (B = 1) from step a on one shared prefix cache. forward: span | full | incremental.
    force: {cell_name: own tokens} teacher forcing. Returns {cell_name: row dict}."""
    torch, A0 = eng.torch, eng.A0
    t, f, key, token = u["t_star"], u["forced_token"], u["perturbation"], u["scene"]
    gt, pred = e["gt"], e["pred"]
    cells = cells or (CELLS + ["noop"])
    if prefix_state is None:
        cache, P, chk, info, mi, prefix = eng.build_prefix(token, stub, pred[:t], 1)
    else:
        cache, P, chk, info, mi, prefix = prefix_state
    rd = info["rope_deltas"][0][0] if isinstance(info["rope_deltas"][0], list) else info["rope_deltas"][0]
    out = {}
    exec_pre = [bank["normal_own"][j - t - 1] for j in range(t + 1, a)]
    for cell in cells:
        name = cell if isinstance(cell, str) else f"{cell[0]}_{cell[1]}"
        ctx, src = cell_context(cell, t, a, bank, gt)
        own, spans, lg_l, margin, picks, span_sha, pos_eq = [], [], [], [], [], [], []
        fed = []
        for k in range(a, C.N_ACT):
            span = [f] + ctx + own
            sp_abs = [A0 + x for x in span]
            if forward == "span":
                full = eng.fwd_span(cache, P, [sp_abs]); C.check_cache(cache, P, chk); pos_eq.append(None)
            elif forward == "full":
                full, posf = eng.fwd_full(mi, prefix, sp_abs)
                pos_eq.append(bool(np.array_equal(posf, eng.cached_positions(P, len(sp_abs), rd))))
            elif forward == "incremental":
                assert sp_abs[:len(fed)] == fed, "not append-only"
                new = sp_abs[len(fed):]
                L = P + len(fed)
                with torch.no_grad():
                    o = eng.llm(input_ids=torch.tensor([new], device="cuda:0"),
                                attention_mask=torch.ones((1, L + len(new)), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(L, L + len(new), device="cuda:0"))
                full = o.logits[:, -1].double(); del o
                fed = list(sp_abs); pos_eq.append(None)
            lg = full[:, A0:A0 + C.N_TOK]
            srt = torch.sort(lg, -1, descending=True).values
            pick = C.sample_rows(torch, lg, eng.temp, C.seed_int(seed, token, key, k))[0]
            nxt = force[name][k - a] if force is not None else pick
            own.append(int(nxt)); picks.append(int(pick)); spans.append(span); span_sha.append(E.h(span))
            lg_l.append(lg[0].float().cpu().numpy()); margin.append(float(srt[0, 0] - srt[0, 1]))
        if forward == "incremental":
            cache.crop(P)
        acts = list(pred[:t]) + [f] + exec_pre + own
        poses, _ = C.decode(torch, eng.model, A0, acts)
        out[name] = {"cell": name, "ctx_edit": ctx, "ctx_src": src, "own": own, "picks": picks, "acts": acts,
                     "poses": np.asarray(poses, np.float64), "span": spans, "span_sha": span_sha, "logits": np.stack(lg_l),
                     "margin": margin, "pos_full_eq_cached": pos_eq}
    if prefix_state is None:
        del cache
        torch.cuda.empty_cache()
    return out, info


def cmp(a, b, tf=False):
    n = min(len(a["own"]), len(b["own"]))
    steps = [i for i in range(n) if a["span"][i] == b["span"][i]]
    d = [float(np.abs(a["logits"][i].astype(np.float64) - b["logits"][i].astype(np.float64)).max()) for i in steps]
    flips = [i for i in steps if int(np.argmax(a["logits"][i])) != int(np.argmax(b["logits"][i]))]
    return {"n_steps": n, "n_cmp": len(steps), "logit_maxabs": max(d) if d else None, "bitwise": all(x == 0 for x in d) if d else None,
            "suffix_equal": a["own"] == b["own"], "flips": flips, "unexplained_flips": [i for i in flips if a["margin"][i] > T_EQUIV],
            "within_tol": (max(d) <= T_EQUIV) if d else None}


def cell_record(u, a, n_old, row, bank, cb_disp, metrics_fn, e):
    t = u["t_star"]
    m = metrics_fn(row["acts"], row["poses"][:, :2].tolist(), [], e["gt"], e["trajectory_gt"], t)
    tok_a = row["own"][0]
    return {"unit_id": u["unit_id"], "scene": u["scene"], "anchor": a, "n_old": n_old, "cell": row["cell"],
            "ctx_edit": row["ctx_edit"], "ctx_src": row["ctx_src"],
            "ctx_edit_motion": [[round(float(v), 5) for v in cb_disp[x]] for x in row["ctx_edit"]],
            "own": row["own"], "acts": row["acts"], "poses": np.round(row["poses"], 5).tolist(), "span_sha": row["span_sha"],
            "tok_a": tok_a, "tok_a_motion": [round(float(v), 5) for v in cb_disp[tok_a]],
            "severe": int(m["amplification"]), "A_plus": int(m["recovery"]), "fde5": m["fde5"],
            "endpoint": [round(float(v), 5) for v in row["poses"][-1, :2]],
            "endpoint_shift_vs_bankN_m": float(np.linalg.norm(row["poses"][-1, :2] - np.asarray(bank["normal_poses"])[-1, :2]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["stage0", "dev", "p1c"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="main", choices=["main", "gate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-star-max", type=int, default=4, help="primary eligibility: suffix after the edit >= 3 tokens")
    ap.add_argument("--limit-units", type=int, default=0)
    ap.add_argument("--timing-only", action="store_true", help="throughput pilot: cell outcomes are not written")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, C.SCRIPTS)
    from analyze_action_history import unit_metrics  # noqa: E402 (dev outcome definition, unchanged)
    units, ed, stub_of = cohort_units(args.cohort, args.manifest)
    gate = args.mode == "gate"
    if gate:
        elig = units                                                    # gate: every unit, anchor clipped to <= 9
    else:
        elig = [u for u in units if u["t_star"] <= args.t_star_max]
        # processing order (protocol section 7): F (natural A-) first, then S, each by sha256("B2-20261017|" + unit_id)
        elig.sort(key=lambda u: (u.get("stratum") not in ("F", "A-"), hashlib.sha256(("B2-20261017|" + u["unit_id"]).encode()).hexdigest()))
    if args.limit_units:
        elig = elig[:args.limit_units]
    json.dump({"cohort": args.cohort, "n_units_all": len(units), "n_units_eligible": len(elig), "t_star_max": args.t_star_max,
               "excluded_t_star": sorted({u["t_star"] for u in units} - {u["t_star"] for u in elig})},
              open(os.path.join(out, "eligibility.json"), "w"), indent=1)
    eng = E.Engine()
    cb, disp, dyaw = C.codebook_geometry()
    cb_disp = {i: (disp[i][0], disp[i][1], dyaw[i]) for i in range(C.N_TOK)}            # (forward m, left m, dyaw rad)
    meta = {"argv": sys.argv, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **eng.env(), "timing": []}
    t0 = time.time()
    # ---------------- phase A: frozen source bank
    bank_path = os.path.join(out, "bank.jsonl")
    mem = {}
    done = set()
    if os.path.exists(bank_path):
        done = {json.loads(l)["unit_id"] for l in open(bank_path)}
    if not os.path.exists(os.path.join(out, "bank_manifest.json")):
        with open(bank_path, "a") as fb:
            for u in elig:
                if u["unit_id"] in done and not gate:
                    continue
                if disk_free_gb() < 1.0:
                    raise SystemExit("disk < 1 GB free: stop")
                tu = time.time()
                rec, m = bank_unit(eng, u, ed[u["scene"]], stub_of[u["scene"]], args.seed, cb_disp, keep_logits=gate)
                fb.write(C.jdump(rec) + "\n"); fb.flush()
                if gate:
                    mem[u["unit_id"]] = m
                meta["timing"].append({"phase": "A", "unit": u["unit_id"], "sec": time.time() - tu})
                print(f"[b2:A] {u['unit_id']} {time.time()-tu:.1f}s", flush=True)
        bank_sha = sha_file(bank_path)
        json.dump({"bank": bank_path, "sha256": bank_sha, "n_units": sum(1 for _ in open(bank_path)), "frozen_utc":
                   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "units": {u["unit_id"]: {k: u.get(k) for k in ("scene", "log", "stratum", "weight", "drive", "day_vehicle")} for u in elig}},
                  open(os.path.join(out, "bank_manifest.json"), "w"), indent=1)
    bank_sha = json.load(open(os.path.join(out, "bank_manifest.json")))["sha256"]
    # ---------------- phase B: cells (bank sha asserted before the pass)
    assert sha_file(bank_path) == bank_sha, "bank file changed after freezing"
    bank = {json.loads(l)["unit_id"]: json.loads(l) for l in open(bank_path)}
    cells_path = os.path.join(out, "cells.jsonl")
    gate_path = os.path.join(out, "gate.jsonl")
    done_b = set()
    if os.path.exists(cells_path):
        done_b = {json.loads(l)["unit_id"] for l in open(cells_path)}
    fc = open(os.devnull if args.timing_only else cells_path, "a")
    fg = open(gate_path, "a") if gate else None
    for u in elig:
        if u["unit_id"] in done_b:
            continue
        if disk_free_gb() < 1.0:
            raise SystemExit("disk < 1 GB free: stop")
        e, stub, bk = ed[u["scene"]], stub_of[u["scene"]], bank[u["unit_id"]]
        t = u["t_star"]
        anchors = [(1, t + 3)] + ([(2, t + 4)] if t + 4 <= C.N_ACT - 3 or gate else [])
        if gate:
            anchors = [(n, a) for n, a in anchors if a <= C.N_ACT - 1] or [(0, t + 2)]
        tu = time.time()
        torch = eng.torch
        pstate = eng.build_prefix(u["scene"], stub, e["pred"][:t], 1)
        buf = []                                                      # a unit's cells are written together (clean resume)
        for n_old, a in anchors:
            if n_old == 0:                                            # gate-only degenerate anchor (empty older prefix)
                continue
            rows, info = run_cells(eng, u, e, stub, bk, a, args.seed, prefix_state=pstate)
            for r in rows.values():
                buf.append(C.jdump(cell_record(u, a, n_old, r, bk, cb_disp, unit_metrics, e) | {"prefix_ids_sha": info["prefix_ids_sha"],
                                                                                              "bank_sha": bank_sha}) + "\n")
            if gate:
                g = {"unit_id": u["unit_id"], "anchor": a, "n_old": n_old}
                g["prefix_eq_bank"] = info["prefix_ids_sha"] == bk["prefix_ids_sha"]
                g["noop_vs_NAT_NSRC"] = cmp(rows["noop"], rows["NAT_NSRC"])
                g["pre_anchor_exec_equal"] = len({tuple(r["acts"][:a]) for r in rows.values()}) == 1 and \
                    list(rows["noop"]["acts"][:a]) == bk["normal_acts"][:a]
                # (NAT, NSRC) vs bank Normal (same B = 1 computation, same spans) -> bitwise expected
                nm = mem[u["unit_id"]]
                k0 = a - t - 1
                bn = {"own": bk["normal_own"][k0:], "span": nm["normal_span"][k0:], "logits": nm["normal_logits"][k0:],
                      "margin": [0.0] * (C.N_ACT - a)}
                g["NAT_NSRC_vs_bankNormal"] = cmp(bn, rows["NAT_NSRC"])
                # value identity: any two cells whose spans coincide must be bitwise equal
                vi = []
                names = list(rows)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        A_, B_ = rows[names[i]], rows[names[j]]
                        if A_["span"][0] == B_["span"][0]:
                            c = cmp(A_, B_)
                            vi.append({"pair": [names[i], names[j]], "bitwise": c["bitwise"], "suffix_equal": c["suffix_equal"]})
                g["value_identity"] = vi
                # c5a repeat (same process)
                rep, _ = run_cells(eng, u, e, stub, bk, a, args.seed, prefix_state=pstate)
                g["c5a"] = {n: cmp(rows[n], rep[n]) for n in rows}
                force = {n: r["own"] for n, r in rows.items()}
                full, _ = run_cells(eng, u, e, stub, bk, a, args.seed, forward="full", force=force)
                g["c4_full_tf"] = {n: cmp(rows[n], full[n]) | {"pos_full_eq_cached_all": all(full[n]["pos_full_eq_cached"])} for n in rows}
                inc, _ = run_cells(eng, u, e, stub, bk, a, args.seed, forward="incremental", force=force, prefix_state=pstate)
                g["c3a_incremental_tf"] = {n: cmp(rows[n], inc[n]) for n in rows}
                fg.write(C.jdump(g) + "\n"); fg.flush()
        fc.write("".join(buf)); fc.flush()
        del pstate
        torch.cuda.empty_cache()
        meta["timing"].append({"phase": "B", "unit": u["unit_id"], "sec": time.time() - tu, "n_anchors": len(anchors)})
        print(f"[b2:B] {u['unit_id']} {time.time()-tu:.1f}s total {(time.time()-t0)/60:.1f} min", flush=True)
    fc.close()
    if fg:
        fg.close()
    meta.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_min": (time.time() - t0) / 60,
                 "peak_mem_gb": eng.torch.cuda.max_memory_allocated() / 1e9, "bank_sha256": bank_sha})
    json.dump(meta, open(os.path.join(out, f"run_meta_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w"), indent=1, default=str)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
