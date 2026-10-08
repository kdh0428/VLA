#!/usr/bin/env python
"""
P1-A: GT-free reference, coordinate and extrapolation controls (GPU, RTX 5090 only).

Reference r (fixed beforehand by p1_reference.py; sha256 checked): the model's own unperturbed
rollout of the scene. Position t_p = t* (inherited from the equal-distance set). One physical
perturbation token p replaces r[t_p]. All rows of a unit run in ONE batched forward per step with
the same per-step seed (scene-level common random numbers, key "original" = the reference's key).

OUTPUT prefix (what is executed and decoded) and CONDITIONING prefix (what the model reads) are
set separately for position t_p:
  row        output@t_p  context@t_p  context at positions j > t_p (step k reads j = t_p+1..k-1)
  U          r           r            own                                   unperturbed
  OC         p           p            own                                   output + context perturbed
  O1         p           r            own                                   output perturbed + reference context (one-shot)
  C1         r           p            own                                   reference output + context perturbed (one-shot)
  OC_ref1    p           p            j = t_p+1 -> r[j] (fixed, stays); rest own   one-shot reference correction
  OC_refR    p           p            j = k-1   -> r[j] at every step; rest own    repeated (sliding) reference correction
  OC_refH    p           p            all -> r[j]                                  reference history
  O_refH     p           r            all -> r[j]                                  output perturbed + reference context (repeated)
  OC_gtR     p           p            j = k-1   -> gt[j] (sliding)                 [GT arm, comparator only]
  OC_gtH     p           p            all -> gt[j]                                 [GT arm, comparator only]
Executed/decoded tokens of every row = r[:t_p] (= pred[:t_p] when t_p <= t*) + [output@t_p] + own samples. Context tokens
never enter the decoder (AutoVLA tokens are relative motions chained from (0,0,0)); this is checked
per unit (decoder origin, locked output-prefix poses).

Baselines without inference (decoded with the real decoder, so translation/rotation are composed in
the decoder's own order):
  Geo  = pred[:t_p] + [p] + r[t_p+1:]       reference future local actions kept, first perturbation composed
  CM   = pred[:t_p] + [p] * (10 - t_p)      pre-fixed constant-motion: the perturbed local motion repeated

Perturbation families (units):
  G (primary, GT-free): around r[t_p], per 45-degree sector of disp[p] - disp[r[t_p]] the token with
    achieved distance closest to D_TARGET = 0.10 m (per 0.5 s segment), only if |achieved - 0.10| <= 0.03 m.
  E (secondary, linkage): the existing equal-distance forced tokens (original + alternatives; these were
    distance-matched to the GT token, so their choice is GT-anchored). 'original' = r[t*] -> p == r.
SECONDARY fixed-position arm (--fixed-positions 1,3 --families G): t_p is the same for every scene and is
chosen without GT; prefix = reference tokens r[:t_p] (model's own, GT-free); family named GF{t_p}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402

D_TARGET = 0.10
D_TOL = 0.03
ROWS = [
    {"name": "U", "out": "r", "ctx": "r", "rule": "own"},
    {"name": "OC", "out": "p", "ctx": "p", "rule": "own"},
    {"name": "O1", "out": "p", "ctx": "r", "rule": "own"},
    {"name": "C1", "out": "r", "ctx": "p", "rule": "own"},
    {"name": "OC_ref1", "out": "p", "ctx": "p", "rule": "ref_once"},
    {"name": "OC_refR", "out": "p", "ctx": "p", "rule": "ref_recent"},
    {"name": "OC_refH", "out": "p", "ctx": "p", "rule": "ref_hist"},
    {"name": "O_refH", "out": "p", "ctx": "r", "rule": "ref_hist"},
    {"name": "OC_gtR", "out": "p", "ctx": "p", "rule": "gt_recent"},
    {"name": "OC_gtH", "out": "p", "ctx": "p", "rule": "gt_hist"},
]
ONE_SHOT = {"O1", "C1", "OC_ref1"}
REPEATED = {"OC_refR", "OC_refH", "O_refH", "OC_gtR", "OC_gtH"}


def context_for(row, k, t, own, ref, gt, p):
    """tokens (relative action ids) at context positions t..k-1 and the source of each position."""
    toks = [p if row["ctx"] == "p" else ref[t]]
    src = ["pert" if row["ctx"] == "p" else "ref"]
    rule = row["rule"]
    for j in range(t + 1, k):
        o = own[j - t - 1]
        if rule == "ref_hist" or (rule == "ref_recent" and j == k - 1) or (rule == "ref_once" and j == t + 1):
            toks.append(ref[j]); src.append("ref")
        elif rule == "gt_hist" or (rule == "gt_recent" and j == k - 1):
            toks.append(gt[j]); src.append("gt")
        else:
            toks.append(o); src.append("own")
    return toks, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="raw output dir (records.jsonl etc.)")
    ap.add_argument("--reference", required=True, help="dir written by p1_reference.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="G,E")
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--max-units-per-scene", type=int, default=0, help="smoke only")
    ap.add_argument("--trace-units", type=int, default=24, help="units with a full per-step trace (all rows)")
    ap.add_argument("--fixed-positions", default="", help="SECONDARY arm: comma list of t_p fixed for all scenes "
                    "(chosen without GT; prefix = reference tokens r[:t_p]); family G only, named GF{t_p}")
    args = ap.parse_args()
    C.gpu_guard()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "records.jsonl")):
        raise SystemExit(f"{out}/records.jsonl exists; refusing to overwrite")
    man = json.load(open(os.path.join(args.reference, "manifest.json")))
    ref_path = os.path.join(args.reference, "references.jsonl")
    ref_sha = C.sha256_file(ref_path)
    if man["files"]["references.jsonl"] != ref_sha:
        raise SystemExit("reference sha256 mismatch")
    REF = {x["token"]: x for x in map(json.loads, open(ref_path))}
    LP0 = np.load(os.path.join(args.reference, "prefix_logprobs.npz"))
    fams = args.families.split(",")

    ed = C.load_ed(args.limit_scenes, args.tokens.split(",") if args.tokens else None)
    ed = [r for r in ed if r["token"] in REF]
    want = {r["token"] for r in ed}
    stub_of, f_pred, f_gt = C.load_stubs_and_token_freq(want)
    cb, disp, dyaw = C.codebook_geometry()

    torch, model, llm, A0, temp, agent, vis_box, cfg = C.load_model()
    gpu = C.assert_5090(torch)
    B = len(ROWS)
    fout = open(os.path.join(out, "records.jsonl"), "w")
    ftr = open(os.path.join(out, "intervention_trace.jsonl"), "w")
    ferr = open(os.path.join(out, "errors.jsonl"), "w")
    support = []
    t0 = time.time()
    n_units = n_traced = 0
    unit_times = []
    fixed = [int(x) for x in args.fixed_positions.split(",") if x != ""]
    if fixed and fams != ["G"]:
        raise SystemExit("--fixed-positions supports family G only (--families G)")
    items = [(r, tp) for r in ed for tp in fixed] if fixed else [(r, r["t_star"]) for r in ed]
    for si, (r, t) in enumerate(items, 1):
        token, gt, pred = r["token"], r["gt"], r["pred"]
        R = REF[token]
        ref = R["ref_tokens"]
        assert ref[:r["t_star"] + 1] == pred[:r["t_star"] + 1], "reference prefix must equal the stored natural prefix"
        assert 0 <= t <= C.N_ACT - 2
        famG = f"GF{t}" if fixed else "G"
        ref_traj = np.asarray(R["ref_traj_xyh"])
        lp0 = LP0[token]
        units = []
        if "G" in fams:
            gsel = C.select_gtfree_perturbations(disp, cb, ref[t], D_TARGET, D_TOL)
            support.append({"token": token, "log": r["log"], "group": r["group"], "t_star": r["t_star"], "t_p": t, "family": famG,
                            "sectors_available": sorted(C.SECTORS[s] for s in gsel), "n_sectors": len(gsel)})
            for s in sorted(gsel):
                units.append((famG, f"{famG}_{C.SECTORS[s]}", gsel[s]["token"], gsel[s]))
        if "E" in fams:
            units.append(("E", "original", pred[t], None))
            for j, a in enumerate(r["alternatives"]):
                units.append(("E", f"alt{j}", a["token"], {"ed_alternative": a}))
        if args.max_units_per_scene:          # smoke only: first n units of EACH family
            units = [u for fm in (fams if not fixed else [famG]) for u in [x for x in units if x[0] == fm][:args.max_units_per_scene]]
        try:
            cache, P, chk, lg_prefix = C.build_prefix_cache(torch, model, llm, agent, vis_box, A0, token, stub_of[token], ref[:t], B)
        except Exception as exc:
            ferr.write(C.jdump({"token": token, "stage": "scene", "error": repr(exc)}) + "\n"); ferr.flush()
            continue
        lp_here = torch.log_softmax(lg_prefix, -1).numpy()
        ref_live, _ = C.decode(torch, model, A0, ref)          # unrounded reference poses for the lock check
        ref_round_err = float(np.abs(ref_live - ref_traj).max())
        for fam, pname, p, info in units:
            tu = time.time()
            try:
                own = [[] for _ in range(B)]
                ents = [[] for _ in range(B)]
                p_ref = [[] for _ in range(B)]
                p_gt = [[] for _ in range(B)]
                applied = [0] * B
                effective = [0] * B
                ovr_pos = [set() for _ in range(B)]
                trace_this = n_traced < args.trace_units
                steps_trace = []
                for k in range(t + 1, C.N_ACT):
                    spans, srcs = [], []
                    for i, row in enumerate(ROWS):
                        toks, src = context_for(row, k, t, own[i], ref, gt, p)
                        spans.append([A0 + a for a in toks]); srcs.append(src)
                        overr = [jj for jj, s_ in enumerate(src[1:]) if s_ != "own"]
                        if overr:
                            ovr_pos[i].update(t + 1 + jj for jj in overr)
                            applied[i] += 1
                            effective[i] += int(any(toks[1 + jj] != own[i][jj] for jj in overr))
                    full = C.forward_spans(torch, llm, cache, P, spans)
                    lg = full[:, A0:A0 + C.N_TOK]
                    lp = torch.log_softmax(lg, -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    picks = C.sample_rows(torch, lg, temp, C.seed_int(args.seed, token, "original", k))
                    for i in range(B):
                        own[i].append(picks[i])
                        ents[i].append(round(float(ent[i]), 5))
                        p_ref[i].append(round(float(lp[i, ref[k]].exp()), 6))
                        p_gt[i].append(round(float(lp[i, gt[k]].exp()), 6))
                    cc = C.check_cache(cache, P, chk)
                    if trace_this:
                        steps_trace.append({"k": k, "span_len": len(spans[0]), "cache_len_after_crop": cache.get_seq_length(),
                                            "prefix_checksum_ok": cc["prefix_checksum_ok"],
                                            "rows": {ROWS[i]["name"]: {"context_tokens_pos_t..k-1": [x - A0 for x in spans[i]],
                                                                       "context_source": srcs[i], "sampled": picks[i]}
                                                     for i in range(B)}})
                rows_out, checks = {}, {}
                prefix_poses_ref = ref_live[:t + 1]
                for i, row in enumerate(ROWS):
                    out_t = p if row["out"] == "p" else ref[t]
                    acts = list(ref[:t]) + [out_t] + own[i]
                    poses, origin = C.decode(torch, model, A0, acts)
                    locked = acts[:t + 1] == list(ref[:t]) + [out_t]
                    if row["out"] == "r":
                        pose_lock_err = float(np.abs(poses[:t + 1] - prefix_poses_ref).max())
                    else:
                        pose_lock_err = float(np.abs(poses[:t] - prefix_poses_ref[:t]).max()) if t > 0 else 0.0
                    rows_out[row["name"]] = {"action_idx": acts, "traj_xyh": np.round(poses, 6).tolist(), "ent": ents[i],
                                             "p_ref": p_ref[i], "p_gt": p_gt[i], "context_override_steps": applied[i],
                                             "context_override_positions": sorted(ovr_pos[i]),
                                             "context_override_effective_steps": effective[i]}
                    checks[row["name"]] = {"output_prefix_locked": bool(locked), "decoder_origin_zero": bool(np.allclose(origin, 0)),
                                           "prefix_pose_max_abs_err_vs_reference": pose_lock_err}
                geo = list(ref[:t]) + [p] + list(ref[t + 1:])
                cm = list(ref[:t]) + [p] * (C.N_ACT - t)
                base_out = {}
                for nm, acts in (("Geo", geo), ("CM", cm)):
                    poses, origin = C.decode(torch, model, A0, acts)
                    base_out[nm] = {"action_idx": acts, "traj_xyh": np.round(poses, 6).tolist()}
                u_sfx, o1_sfx = rows_out["U"]["action_idx"][t + 1:], rows_out["O1"]["action_idx"][t + 1:]
                checks["reference_file_rounding_maxabs"] = ref_round_err
                checks["OC_suffix_equals_C1_suffix"] = rows_out["OC"]["action_idx"][t + 1:] == rows_out["C1"]["action_idx"][t + 1:]
                checks["U_equals_reference_tokens"] = rows_out["U"]["action_idx"] == ref
                checks["O1_suffix_equals_U_suffix"] = o1_sfx == u_sfx
                checks["O1_tokens_equal_Geo_if_U_eq_ref"] = (rows_out["O1"]["action_idx"] == geo) if checks["U_equals_reference_tokens"] else None
                checks["C1_context_differs_from_U_at_tp"] = p != ref[t]
                checks["numpy_rollout_matches_decoder_maxabs"] = float(np.abs(C.rollout_np(cb, geo) - np.asarray(base_out["Geo"]["traj_xyh"])).max())
                vec = disp[p] - disp[ref[t]]
                pert = {"family": fam, "token_p": int(p), "token_ref_tp": int(ref[t]), "position_tp": t,
                        "requested_m": D_TARGET if fam.startswith("G") else None, "tolerance_m": D_TOL if fam.startswith("G") else None,
                        "achieved_m": float(np.linalg.norm(vec)), "direction_deg": float(math.degrees(math.atan2(vec[1], vec[0]))) if p != ref[t] else None,
                        "sector": C.SECTORS[C.sector_of(vec)] if p != ref[t] else None,
                        "vec_forward": float(vec[0]), "vec_left": float(vec[1]),
                        "dyaw_p_minus_ref_rad": float((dyaw[p] - dyaw[ref[t]] + math.pi) % (2 * math.pi) - math.pi),
                        "shape_d_ref": float(np.linalg.norm(cb[p] - cb[ref[t]], axis=-1).mean()),
                        "freq_natural_pred_p": int(f_pred[p]), "freq_natural_pred_ref": int(f_pred[ref[t]]),
                        "freq_gt_p": int(f_gt[p]), "freq_gt_ref": int(f_gt[ref[t]]),
                        "logp_p_given_prefix": float(lp_here[p]), "logp_ref_given_prefix": float(lp_here[ref[t]]),
                        "rank_p_given_prefix": int((lp_here > lp_here[p]).sum()) + 1,
                        "prefix_logprob_matches_reference_file": (float(np.abs(lp_here - lp0).max()) if t == r["t_star"] else None),
                        "info": info}
                rec = {"unit_id": f"{token}|{fam}:{pname}", "token": token, "log": r["log"], "group": r["group"], "t_star": r["t_star"], "t_p": t,
                       "position_rule": ("fixed (GT-free)" if fixed else "t* (inherited, GT-informed)"), "family": fam, "perturbation": pname, "seed": args.seed, "seed_key": "original", "gt": gt,
                       "trajectory_gt": r["trajectory_gt"], "ref_tokens": ref, "ref_traj_xyh": R["ref_traj_xyh"],
                       "perturbation_info": pert, "rows": rows_out, "baselines": base_out, "checks": checks, "prefix_len": P}
                fout.write(C.jdump(rec) + "\n"); fout.flush()
                if trace_this:
                    ftr.write(C.jdump({"unit_id": rec["unit_id"], "t_p": t, "p": int(p), "ref_tp": int(ref[t]), "prefix_len": P,
                                       "prefix_action_tokens": list(ref[:t]), "steps": steps_trace, "checks": checks,
                                       "output_prefix_by_row": {nm: v["action_idx"][:t + 1] for nm, v in rows_out.items()}}) + "\n")
                    ftr.flush(); n_traced += 1
                n_units += 1
                unit_times.append(time.time() - tu)
            except Exception as exc:
                import traceback; traceback.print_exc()
                ferr.write(C.jdump({"token": token, "unit": pname, "stage": "unit", "error": repr(exc)}) + "\n"); ferr.flush()
        del cache
        if si % 10 == 0 or args.limit_scenes or args.tokens:
            el = time.time() - t0
            print(f"[p1a] {si}/{len(items)} scene-positions units={n_units} {el/si:.1f}s/item eta {(len(items)-si)*el/si/60:.1f} min", flush=True)
    fout.close(); ftr.close(); ferr.close()
    with open(os.path.join(out, "perturbation_support.jsonl"), "w") as f:
        for s in support:
            f.write(C.jdump(s) + "\n")
    meta = C.env_record({"script": os.path.abspath(__file__), "rows": ROWS, "families": fams, "fixed_positions": fixed, "d_target_m": D_TARGET, "tol_m": D_TOL,
                         "n_scenes": len(ed), "n_units": n_units, "seed": args.seed, "temperature": temp, "batch_rows": B,
                         "reference_dir": os.path.abspath(args.reference), "reference_sha256": ref_sha,
                         "elapsed_min": (time.time() - t0) / 60, "sec_per_unit_mean": float(np.mean(unit_times)) if unit_times else None,
                         "gpu_checked": gpu, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
    print(f"[done] units={n_units} elapsed {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
