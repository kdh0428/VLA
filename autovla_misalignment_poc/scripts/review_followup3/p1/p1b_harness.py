#!/usr/bin/env python
"""
P1-B AutoVLA no-op / cache parity harness (RTX 5090 only; protocol: outputs/review_followup3/P1/PROTOCOL_P1.md).

  --controls c5           self-reproducibility pilot (fixes tolerances; the only control run in Stage-0 phase 1)
  --controls c1,c2,c3,c4,c5   the Stage-0 gate (only after the protocol is frozen)

Controls (each for decoding in {greedy, t001} = argmax / the experiments' T = 0.01 CRN sampling):
  c5a  untouched vs untouched, same process, experiment batch composition                       (exact floor)
  c5c  untouched experiment batch vs untouched B = 1, teacher-forced on the same tokens           (batch-shape floor)
  c5b  untouched, separate process (run the harness twice; p1b_compare.py)                         (process floor)
  c5d  untouched vs the stored raw of the original run (exp 7 raw, P1-A raw), tokens only          (diagnostic)
  c1   untouched vs experiment hooks installed, no edit (H7: embed hook + attention wrappers; H8: layer hooks, empty plan)
  c2   untouched vs self-patch: (i) hooks whose edit writes back the value it read (H7 mask/embedding, H8 every patch row
       sourced from itself, at its own layer and alpha) and (ii) token-level self-override (every overridden context
       position filled with the row's own token through the same rule) compared with the row whose context is all-own
  c3   untouched B = 1 span path vs (a) incremental AR KV decoding (append-only rows) and (b) stub+prefix KV invalidated
       and recomputed with unchanged tokens; teacher-forced on the same tokens
  c4   cached-edit path actually used (experiment batch, experiment hooks) vs full recompute from position 0 of the same
       final edited sequence (B = 1, no cache; H8 emb patches by token substitution), teacher-forced; plus free-run
Every comparison row stores input (span) / position / prefix / pixel hashes, logits max/mean |diff| (2,048 action rows,
float64 of the bf16 forward), next-token (argmax and sampled) agreement, suffix agreement, first divergence, decoded pose
diff, override positions and effective overrides.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1b_engine as E  # noqa: E402
C = E.C

AH_RAW = f"{E.POC}/outputs/action_history_causal/records.jsonl"
P1A_RAW = {"G": f"{E.POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008/raw/records.jsonl",
           "GF": f"{E.POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008/raw_fixedpos/records.jsonl"}
APPEND_ONLY = {"H7": ["normal", "gt_history"], "H8": ["normal", "gt_history"],
               "H9": None,  # every H9 row is append-only (fixed absolute positions)
               "P1A": ["U", "OC", "O1", "C1", "OC_ref1", "OC_refH", "O_refH", "OC_gtH"],
               "H32": ["normal"],                                                     # AMENDMENT P1-03
               "P1B": ["normal", "full_gt", "full_self", "P_gt_s1_w1", "P_self_s2_w2", "P_gt_s4_w3"]}
O0_RAW = {"H32": f"{E.POC}/outputs/motion_semantics_ablation/records.jsonl",
          "P1B": f"{E.POC}/outputs/review_followup/P1B_start_position/run_20261008/raw/records.jsonl"}


# ---------------------------------------------------------------------------------------------- comparison
def compare(a, b, rows=None, mapping=None, tol=None):
    """a, b: decode() results. mapping: {row_in_b: row_in_a}. Logits are compared only on steps whose input span
    (and position ids) hash is identical in both (later steps of a diverged free run are not comparable)."""
    out = []
    names = rows or [r for r in b["rows"] if (mapping or {}).get(r, r) in a["rows"]]
    for nb in names:
        na = (mapping or {}).get(nb, nb)
        if na not in a["rows"] or nb not in b["rows"]:
            continue
        ra, rb = a["rows"][na], b["rows"][nb]
        n = min(len(ra["span_sha"]), len(rb["span_sha"]))
        same_in = [ra["span_sha"][i] == rb["span_sha"][i] for i in range(n)]
        subs = rb.get("substituted") or []
        subst_verified = []
        for i in range(n):                       # AMENDMENT A1: C4 token substitution of H8 emb-patch rows
            if not same_in[i] and i < len(subs) and subs[i]:
                tgt_tok, src_tok = subs[i]
                ok = (ra["span"][i][:-1] == rb["span"][i][:-1] and ra["span"][i][-1] == tgt_tok and rb["span"][i][-1] == src_tok)
                subst_verified.append(ok)
                same_in[i] = ok
        same_pos = [ra["pos_sha"][i] == rb["pos_sha"][i] for i in range(n)]
        cmp_steps = [i for i in range(n) if same_in[i] and same_pos[i]]
        d = [np.abs(ra["logits"][i].astype(np.float64) - rb["logits"][i].astype(np.float64)) for i in cmp_steps]
        maxd = [float(x.max()) for x in d]
        meand = [float(x.mean()) for x in d]
        am_ag = [ra["argmax"][i] == rb["argmax"][i] for i in cmp_steps]
        pk_ag = [ra["pick"][i] == rb["pick"][i] for i in cmp_steps]
        flips = [i for i in cmp_steps if ra["argmax"][i] != rb["argmax"][i]]
        first_div = next((i for i in range(n) if ra["own"][i] != rb["own"][i]), None)
        pa, pb = np.asarray(ra["poses"]), np.asarray(rb["poses"])
        row = {"row_b": nb, "row_a": na, "n_steps": n, "n_steps_compared": len(cmp_steps),
               "input_hash_equal_all": all(same_in), "pos_hash_equal_all": all(same_pos),
               "logit_maxabs": max(maxd) if maxd else None, "logit_meanabs": float(np.mean(meand)) if meand else None,
               "logit_maxabs_per_step": maxd, "bitwise_equal": all(x == 0.0 for x in maxd) if maxd else None,
               "argmax_agree": float(np.mean(am_ag)) if am_ag else None, "pick_agree": float(np.mean(pk_ag)) if pk_ag else None,
               "suffix_equal": ra["own"] == rb["own"], "first_divergence_step_index": first_div,
               "flip_steps": flips, "margin_at_flips_a": [ra["margin"][i] for i in flips],
               "min_margin_a": float(min(ra["margin"][i] for i in cmp_steps)) if cmp_steps else None,
               "pose_maxabs": float(np.abs(pa - pb).max()) if pa.shape == pb.shape else None,
               "override_pos_b": rb["override_pos"], "override_effective_b": rb["override_effective"],
               "substituted_b": [s for s in rb.get("substituted", []) if s],
               "substitution_steps_verified": subst_verified,
               "B_a": ra.get("B"), "B_b": rb.get("B")}
        if tol is not None and maxd:
            row["within_tol"] = max(maxd) <= tol
            row["unexplained_flips"] = [i for i in flips if ra["margin"][i] > tol]
        out.append(row)
    return out


def strip(res, key_rows):
    """json-able summary of a decode result (logits only for key rows, as float32 lists are too big -> npz)."""
    rows = {}
    for n, r in res["rows"].items():
        rows[n] = {k: (v if k not in ("logits", "poses") else None) for k, v in r.items()}
        rows[n]["poses"] = np.round(np.asarray(r["poses"]), 6).tolist()
        rows[n]["logits_sha"] = E.h(r["logits"])
    return {k: v for k, v in res.items() if k != "rows"} | {"rows": rows}


# ---------------------------------------------------------------------------------------------- units
def prepare_units(manifest, only_pilot):
    man = json.load(open(manifest))
    ed = {r["token"]: r for r in map(json.loads, open(C.ED_RECORDS))}
    units = [u for u in man["units"] if (u["pilot"] or not only_pilot)]
    stub_of, _, _ = C.load_stubs_and_token_freq({u["scene"] for u in units})
    cb, disp, _ = C.codebook_geometry()
    for u in units:
        if "P1A" in u["paths"]:
            ref = u["ref_tokens"]
            gsel = C.select_gtfree_perturbations(disp, cb, ref[u["t_p"]], 0.10, 0.03)   # p1a_gt_free_reference.py:146
            s = min(gsel)                                                                # lowest available sector index (manifest rule)
            u["p_token"] = gsel[s]["token"]; u["sector_used"] = C.SECTORS[s]
            u["p1a_raw_unit_id"] = f"{u['scene']}|{u['family']}:{u['family']}_{C.SECTORS[s]}"
    return man, units, ed, stub_of


def load_stored(units):
    """original-run tokens for c5d (diagnostic): exp 7 raw (H7, T = 0.01) and P1-A raw (P1A, T = 0.01)."""
    want = {u["scene"] for u in units}
    st = {}
    for line in open(AH_RAW):
        x = json.loads(line)
        if x["token"] in want and "alt0" in x["perturbations"]:
            st[("H7", x["token"])] = {c: v["action_idx"] for c, v in x["perturbations"]["alt0"]["conditions"].items()}
    for line in open(O0_RAW["H32"]):                                       # AMENDMENT P1-03 diagnostics
        x = json.loads(line)
        if x["token"] in want and x["perturbation"] == "alt0":
            st[("H32", x["token"])] = {c: v["action_idx"] for c, v in x["conditions"].items()}
    for line in open(O0_RAW["P1B"]):
        x = json.loads(line)
        if x["token"] in want and x["perturbation"] == "alt0":
            st[("P1B", x["token"])] = {c: v["action_idx"] for c, v in x["rows"].items()}
    ids = {u["p1a_raw_unit_id"] for u in units if "P1A" in u["paths"]}
    for fam, f in P1A_RAW.items():
        for line in open(f):
            if '"unit_id"' not in line[:200]:
                continue
            x = json.loads(line)
            if x["unit_id"] in ids:
                st[("P1A", x["unit_id"])] = {n: v["action_idx"] for n, v in x["rows"].items()}
                st[("P1A_traj", x["unit_id"])] = {n: v["traj_xyh"] for n, v in x["rows"].items()}
    return st


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", default="c5")
    ap.add_argument("--paths", default="H7,H8,H9,P1A")
    ap.add_argument("--decodings", default="greedy,t001")
    ap.add_argument("--pilot-only", action="store_true", help="only the manifest's pilot units")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", default=None, help="json file with frozen tolerances (gate runs)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "comparisons.jsonl")):
        raise SystemExit(f"{out}/comparisons.jsonl exists; refusing to overwrite")
    controls = args.controls.split(",")
    paths = args.paths.split(",")
    decs = args.decodings.split(",")
    tol = json.load(open(args.tol)) if args.tol else None
    man, units, ed, stub_of = prepare_units(args.manifest, args.pilot_only)
    stored = load_stored(units) if "c5" in controls else {}

    eng = E.Engine()
    meta = {"script": os.path.abspath(__file__), "argv": sys.argv, "manifest": os.path.abspath(args.manifest),
            "controls": controls, "paths": paths, "decodings": decs, "units": [u["unit_id"] for u in units],
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **eng.env()}
    json.dump(meta, open(os.path.join(out, "run_meta_start.json"), "w"), indent=1, default=str)
    fc = open(os.path.join(out, "comparisons.jsonl"), "w")
    fr = open(os.path.join(out, "decodes.jsonl"), "w")
    logits_store = {}
    t0 = time.time()
    timing = []

    def emit(control, path, u, dec, comps, extra=None):
        for c in comps:
            fc.write(C.jdump({"control": control, "path": path, "unit_id": u["unit_id"], "scene": u["scene"], "stratum": u["stratum"],
                              "group": u["group"], "t_p": u["t_p"], "decoding": dec, **(extra or {}), **c}) + "\n")
        fc.flush()

    def keep(tag, res, path):
        fr.write(C.jdump({"tag": tag, **strip(res, E.KEY_ROWS[path])}) + "\n"); fr.flush()
        for n in E.KEY_ROWS[path]:
            if n in res["rows"]:
                logits_store[f"{tag}|{n}"] = res["rows"][n]["logits"]

    for u in units:
        for path in paths:
            if path not in u["paths"]:
                continue
            e = ed[u["scene"]]
            for dec in decs:
                tu = time.time()
                kw = dict(decoding=dec, seed=args.seed)
                tag0 = f"{u['unit_id']}|{path}|{dec}"
                A = eng.decode(path, u, e, stub_of[u["scene"]], **kw)                       # untouched, experiment batch
                keep(f"{tag0}|A", A, path)
                own_A = {n: r["own"] for n, r in A["rows"].items()}
                base_rows = [n for n in E.KEY_ROWS[path] if n in A["rows"]]
                if "c5" in controls:
                    A2 = eng.decode(path, u, e, stub_of[u["scene"]], **kw)
                    emit("c5a", path, u, dec, compare(A, A2))
                    if path != "H7":                                                         # H7's experiment batch is already 1
                        rows1 = [n for n in base_rows if not ("@" in n)]
                        B1 = eng.decode(path, u, e, stub_of[u["scene"]], batch="1", force=own_A, row_names=rows1, **kw)
                        keep(f"{tag0}|B1tf", B1, path)
                        emit("c5c_tf", path, u, dec, compare(A, B1), {"note": "teacher-forced on A tokens"})
                        B1f = eng.decode(path, u, e, stub_of[u["scene"]], batch="1", row_names=rows1, **kw)
                        emit("c5c_free", path, u, dec, compare(A, B1f))
                    if dec == "t001":
                        key = (path, u["scene"]) if path in ("H7", "H32", "P1B") else ("P1A", u.get("p1a_raw_unit_id"))
                        if key in stored:
                            for n, toks in stored[key].items():
                                if n in A["rows"]:
                                    fc.write(C.jdump({"control": "c5d", "path": path, "unit_id": u["unit_id"], "decoding": dec,
                                                      "row_b": n, "stored_equal": A["rows"][n]["acts"] == toks,
                                                      "first_divergence_step_index": next((i for i, (x, y) in enumerate(zip(A["rows"][n]["acts"], toks)) if x != y), None),
                                                      "stored_source": {"H7": "exp7 raw", "H32": "exp32 raw", "P1B": "P1-B raw"}.get(path, "P1-A raw")}) + "\n")
                            if path == "P1A":                                                # rollout-saving path (C1 input of section 6.2)
                                for n, tr in stored[("P1A_traj", u["p1a_raw_unit_id"])].items():
                                    toks = stored[key][n]
                                    poses, _ = C.decode(eng.torch, eng.model, eng.A0, toks)
                                    fc.write(C.jdump({"control": "save_path", "path": path, "unit_id": u["unit_id"], "row_b": n,
                                                      "stored_traj_vs_redecoded_maxabs": float(np.abs(np.asarray(tr) - poses).max()),
                                                      "numpy_rollout_vs_decoder_maxabs": float(np.abs(C.rollout_np(C.codebook_geometry()[0], toks) - poses).max())}) + "\n")
                if "c1" in controls and path in ("H7", "H8"):
                    R = eng.decode(path, u, e, stub_of[u["scene"]], hooks="installed", **kw)
                    emit("c1", path, u, dec, compare(A, R), {"hook_calls": R.get("hook_calls"), "hooks_after": R["hooks_after"]})
                if "c2" in controls:
                    if path in ("H7", "H8"):
                        R = eng.decode(path, u, e, stub_of[u["scene"]], hooks="selfpatch", **kw)
                        emit("c2_hook", path, u, dec, compare(A, R), {"hook_calls": R.get("hook_calls")})          # same row, same batch slot
                    R = eng.decode(path, u, e, stub_of[u["scene"]], self_override=True, **kw)
                    if path == "P1A":
                        mp = {r["name"]: ("OC" if r["ctx"] == "p" else "O1") for r in E.path_rows(path)}
                    else:
                        mp = {r["name"]: "normal" for r in E.path_rows(path)}
                    emit("c2_token", path, u, dec, compare(A, R, mapping=mp))
                if "c3" in controls:
                    rows1 = [n for n in base_rows if "@" not in n]
                    Bsp = eng.decode(path, u, e, stub_of[u["scene"]], batch="1", force=own_A, row_names=rows1, **kw)
                    ao = APPEND_ONLY[path] if APPEND_ONLY[path] is not None else rows1
                    Inc = eng.decode(path, u, e, stub_of[u["scene"]], forward="incremental", force=own_A,
                                     row_names=[n for n in rows1 if n in ao], **kw)
                    emit("c3a_incremental", path, u, dec, compare(Bsp, Inc))
                    Dp = eng.decode(path, u, e, stub_of[u["scene"]], forward="deep", force=own_A, row_names=rows1, **kw)
                    emit("c3b_deep_invalidate", path, u, dec, compare(Bsp, Dp), {"invalidated": Dp.get("deep_invalidated_range")})
                if "c4" in controls:
                    X = eng.decode(path, u, e, stub_of[u["scene"]], hooks="experiment", **kw) if path in ("H7", "H8") else A
                    if path == "H8":
                        keep(f"{tag0}|EXP", X, path)
                    own_X = {n: r["own"] for n, r in X["rows"].items()}
                    rowsF = [n for n in E.KEY_ROWS[path] if n in X["rows"]]
                    F = eng.decode(path, u, e, stub_of[u["scene"]], forward="full", force=own_X, row_names=rowsF, **kw)
                    emit("c4_tf", path, u, dec, compare(X, F), {"pos_full_eq_cached_all": all(
                        all(v for v in r["pos_full_eq_cached"]) for r in F["rows"].values())})
                    Ff = (eng.decode(path, u, e, stub_of[u["scene"]], forward="full", row_names=rowsF, **kw)
                          if path != "H8" else None)                                         # H8 patch rows need the Normal row's live tokens
                    if Ff is not None:
                        emit("c4_free", path, u, dec, compare(X, Ff))
                timing.append({"unit": u["unit_id"], "path": path, "decoding": dec, "sec": time.time() - tu})
                print(f"[p1b] {u['unit_id']} {path} {dec} {time.time()-tu:.1f}s total {(time.time()-t0)/60:.1f} min", flush=True)
    fc.close(); fr.close()
    np.savez_compressed(os.path.join(out, "logits_keyrows.npz"), **logits_store)
    meta.update({"end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_min": (time.time() - t0) / 60,
                 "timing": timing, "peak_mem_gb": eng.torch.cuda.max_memory_allocated() / 1e9})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1, default=str)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
