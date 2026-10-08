#!/usr/bin/env python
"""
P1-C equal-distance unit builder with an explicit token list (protocol §3 "Units", decision D7b). GPU 1.

Wrapper of scripts/equal_distance_perturbation.py (that file is NOT modified):
  - the alternative-token rule is the dev function itself: ED.select_alternatives (8 direction sectors, tolerance
    max(5 mm, rel * d(GT, pred)), rel 0.10 -> 0.20 -> 0.35 only if < 5 sectors qualify; codebook disp as in ED);
  - the A+/A- label is ED.a_label (the dev rule), and must agree with the stage-1 label (asserted);
  - the per-scene GPU loop (original / original_reseed / gt / alt_j continuations with llm.generate, seed
    sha256(f"{seed}:{token}") shared by all conditions except original_reseed = +7919) is a verbatim copy of ED.main's
    loop, because action_history_causal reads `conditions[*].action_idx` from these records;
  - ONLY the scene set differs: instead of "all A- + 3 A+ per A- matched on t*", the scenes are the P1-C selection
    (all eligible A- = F, plus the sha256-ordered A+ sample S per log segment), given by --selection files.
Records = ED layout + "p1c": {stratum, pi, weight, N_S_seg, m_seg}. Resumable (skips tokens already written);
existing lines are never rewritten. `--check-dev` reproduces stored dev ED records for parity testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402

sys.path.insert(0, C.SCRIPTS)
import equal_distance_perturbation as ED  # noqa: E402  (dev module, unchanged; importing it runs nothing)


def load_items(stage1_files, selection_files, dev_ed=None):
    """Build ED 'picks' for the selected scenes from stage-1 (full_extract arm-N layout) records."""
    sel = {}
    for f in selection_files:
        for s in json.load(open(f))["selected"]:
            assert s["token"] not in sel, f"token selected twice: {s['token']}"
            sel[s["token"]] = s
    if dev_ed:      # parity test: the scenes/groups of a stored dev ED run
        for l in open(dev_ed):
            r = json.loads(l)
            sel[r["token"]] = {"token": r["token"], "stratum": {"A-": "F", "A+": "S"}[r["group"]], "pi": None,
                               "weight": None, "N_S_seg": None, "m_seg": None, "t_star": r["t_star"]}
    picks = []
    seen = set()
    for f in stage1_files:
        for line in open(f):
            r = json.loads(line)
            if r["token"] not in sel or r["token"] in seen:
                continue
            seen.add(r["token"])
            n = r["arms"]["N"]
            # identical eligibility logic to ED.main (plus t* <= 7 from the selection)
            assert not (n["cot_present"] or n["runaway_action_tokens"] or len(n["trajectory_pred"]) < ED.N_ACT)
            pred, gt = n["pred_action_idx"][:ED.N_ACT], r["gt_action_idx"][:ED.N_ACT]
            t = next((k for k in range(ED.N_ACT) if pred[k] != gt[k]), None)
            s = sel[r["token"]]
            assert t is not None and t == s["t_star"], f"t* mismatch for {r['token']}"
            group = "A+" if ED.a_label(n["trajectory_pred"], r["trajectory_gt"]) else "A-"
            assert group == {"F": "A-", "S": "A+"}[s["stratum"]], f"label mismatch for {r['token']}"
            picks.append({"token": r["token"], "t_star": t, "log": r["log_name"], "pred": pred, "gt": gt,
                          "n_ids": n["token_ids"], "velocity": r["velocity"],
                          "trajectory_gt": [p[:2] for p in r["trajectory_gt"]], "map_name": r["map_name"], "group": group,
                          "p1c": {k: s.get(k) for k in ("stratum", "pi", "weight", "N_S_seg", "m_seg")}})
    missing = set(sel) - seen
    if missing:
        raise SystemExit(f"{len(missing)} selected tokens have no stage-1 record, e.g. {sorted(missing)[:3]}")
    picks.sort(key=lambda x: (x["group"] != "A-", x["token"]))
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", nargs="+", required=True, help="stage-1 records.jsonl file(s)")
    ap.add_argument("--selection", nargs="*", default=[], help="p1c_select.py output file(s)")
    ap.add_argument("--check-dev", default=None, help="stored dev ED records.jsonl: rebuild those scenes (parity test)")
    ap.add_argument("--scenes", default=C.SCENES)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=os.path.join(C.AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("pinned to the RTX 5090: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1")
    for k in ("scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)
    rec_path = os.path.join(args.output, "records.jsonl")

    cb = np.asarray(pickle.load(open(os.path.join(ED.AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl"), "rb"))["token_all"]["veh"], np.float32)
    disp = cb[:, -1].mean(1)

    picks = load_items([os.path.abspath(f) for f in args.stage1], [os.path.abspath(f) for f in args.selection], args.check_dev)
    done = set()
    if os.path.exists(rec_path):
        done = {json.loads(l)["token"] for l in open(rec_path) if l.strip()}
    picks = [p for p in picks if p["token"] not in done]
    if args.limit:
        picks = picks[:args.limit]
    print(f"[plan] {len(picks)} scenes to build (A- {sum(p['group']=='A-' for p in picks)}, "
          f"A+ {sum(p['group']=='A+' for p in picks)}); {len(done)} already written", flush=True)
    if not picks:
        print("[done] nothing to do", flush=True)
        return

    import torch
    import yaml
    os.chdir(ED.AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    t_load = time.time()
    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(ED.AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(ED.AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    gc = cfg["inference"]["sample"]
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    load_s = time.time() - t_load

    vis_box = {}
    orig_visual = llm.visual.forward

    def visual_cached(pixel_values, grid_thw=None, **kw):
        if vis_box.get("v") is None:
            vis_box["v"] = orig_visual(pixel_values, grid_thw=grid_thw, **kw)
        return vis_box["v"]
    llm.visual.forward = visual_cached

    agent = AutoVLAAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                                               interval_length=cfg["model"]["trajectory"]["interval_length"]),
        sensor_data_path=Path("dataset/nuplan/sensor_blobs/test"),
        codebook_cache_path=cfg["model"]["codebook_cache_path"], skip_model_load=True)

    fout = open(rec_path, "a")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "a")
    ftime = open(os.path.join(args.output, "timing.jsonl"), "a")
    t0 = time.time()
    n_ok = n_fail = 0
    for i, p in enumerate(picks, 1):
        ts = time.time()
        try:
            # ---- verbatim from equal_distance_perturbation.main (per-scene loop) ----
            scene = json.load(open(os.path.join(args.scenes, f"{p['token']}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0].tolist()
            fa = next(j for j, x in enumerate(p["n_ids"]) if x >= A0)
            stub = p["n_ids"][:fa]
            t, gt, pred = p["t_star"], p["gt"], p["pred"]
            alts, geo = ED.select_alternatives(disp, cb, gt[t], pred[t])
            base_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{p['token']}".encode()).digest()[:4], "little")
            conds = [("original", pred[t], base_seed), ("original_reseed", pred[t], base_seed + 7919), ("gt", gt[t], base_seed)]
            conds += [(f"alt{j}", a["token"], base_seed) for j, a in enumerate(alts)]
            remaining = ED.N_ACT - t - 1
            out = {}
            for name, tok_forced, seed in conds:
                head = stub + [A0 + a for a in pred[:t]] + [A0 + tok_forced]
                ids = torch.tensor([prompt_ids + head], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                acts = list(pred[:t]) + [tok_forced]
                ents, gtlp = [], []
                if remaining > 0:
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():
                        g = llm.generate(**m2, max_new_tokens=remaining + 4, do_sample=True, temperature=gc["temperature"],
                                         top_k=gc["top_k"], top_p=gc["top_p"], return_dict_in_generate=True, output_logits=True)
                    new = g.sequences[0][ids.shape[1]:].tolist()
                    for j, x in enumerate(new):
                        if len(acts) >= ED.N_ACT or x < A0:
                            break
                        lg = g.logits[j][0, A0:].float()
                        lp = torch.log_softmax(lg, -1)
                        ents.append(float(-(lp.exp() * lp).sum()))
                        gtlp.append(float(lp[gt[len(acts)]]))
                        acts.append(x - A0)
                    del g
                rec = {"forced_token": tok_forced, "action_idx": acts, "valid": len(acts) == ED.N_ACT,
                       "entropy_steps": ents, "gt_logprob_steps": gtlp}
                if rec["valid"]:
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
                    rec["trajectory_pred"] = np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()
                out[name] = rec
            keep = {k: p[k] for k in ("token", "group", "t_star", "log", "pred", "gt", "velocity", "trajectory_gt", "map_name")}
            # ---- end of verbatim block ----
            fout.write(json.dumps(keep | {"geometry": geo, "alternatives": alts, "conditions": out, "p1c": p["p1c"]}) + "\n")
            fout.flush()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": p["token"], "stage": "units", "error": repr(exc)}) + "\n"); ferr.flush()
        ftime.write(json.dumps({"token": p["token"], "s": round(time.time() - ts, 3)}) + "\n"); ftime.flush()
        if i % 20 == 0 or i == len(picks):
            el = time.time() - t0
            print(f"[prog] {i}/{len(picks)} ok={n_ok} fail={n_fail} {el/i:.1f}s/scene eta {(len(picks)-i)*el/i/60:.1f}min", flush=True)
    fout.close(); ferr.close(); ftime.close()
    with open(os.path.join(args.output, "run_meta.jsonl"), "a") as f:
        f.write(json.dumps({"n_ok": n_ok, "n_fail": n_fail, "gen_conf": gc, "seed": args.seed, "sectors": ED.SECTORS,
                            "rel_levels": ED.REL_LEVELS, "abs_tol_m": ED.ABS_TOL, "min_alts": ED.MIN_ALTS,
                            "selection_files": args.selection, "stage1_files": args.stage1, "check_dev": args.check_dev,
                            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "load_s": round(load_s, 1), "elapsed_min": (time.time() - t0) / 60}) + "\n")
    print(f"[done] ok={n_ok} fail={n_fail} ({(time.time()-t0)/60:.1f} min, load {load_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
