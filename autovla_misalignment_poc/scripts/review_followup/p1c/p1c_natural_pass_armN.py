#!/usr/bin/env python
"""
P1-C stage 1, ALTERNATIVE procedure (pilot candidate after check P1 failed with _planner.Planner; NOT the committed
protocol procedure - using it for the main run needs a dated protocol amendment).

Reproduces full_extract.py pass 1 (arm N) per scene: prompt (AutoVLA feature builders + model.get_prompt), forced stem
"<think>\\nThis is a", then ONE llm.generate call (do_sample, AutoVLA sample config: T = 0.01, top_k, top_p,
max_length = config max_length) that produces the stub AND the action tokens with the incremental KV cache, exactly as
full_extract arm N. Differences to full_extract: (1) full_extract seeded torch once per run and also ran arm C between
scenes, so its RNG stream cannot be reproduced for a subset; here torch.manual_seed(sha256("<seed>:<token>:armN")) per
scene; (2) arm C and the tensor dump are not run. Action tokens = the first 10 generated ids >= A0; runaway = more than
10 action ids; CoT = "complex scenario" in the text (full_extract rule).
Output layout identical to p1c_natural_pass.py (full_extract arm-N fields + "p1c" stage-1 labels). Resumable.
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
import p1c_common as C  # noqa: E402

sys.path.insert(0, C.SCRIPTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--scenes", default=C.SCENES)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("pinned to the RTX 5090: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1")
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)
    rec_path = os.path.join(args.output, "records.jsonl")
    done = {json.loads(l)["token"] for l in open(rec_path) if l.strip()} if os.path.exists(rec_path) else set()
    toks = [t for t in json.load(open(args.tokens)) if t not in done]
    if args.limit:
        toks = toks[:args.limit]
    print(f"[plan] {len(toks)} scenes ({len(done)} done)", flush=True)
    if not toks:
        return

    import torch
    from _planner import Planner, seed_of
    from analyze_action_history import a_eval

    t_load = time.time()
    pl = Planner()                      # same model build / checkpoint / feature builders as every harness
    load_s = time.time() - t_load
    llm, A0, model, gc = pl.llm, pl.A0, pl.model, pl.gen_conf
    forced = list(pl.stem)
    fout = open(rec_path, "a"); ferr = open(os.path.join(args.output, "errors.jsonl"), "a")
    ftime = open(os.path.join(args.output, "timing.jsonl"), "a")
    t0 = time.time(); n_ok = n_fail = 0
    for i, token in enumerate(toks, 1):
        ts = time.time()
        try:
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            pl._vis.clear()
            feats = {}
            for b in pl.agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt = mi["input_ids"][0].tolist()
            stem = torch.tensor([prompt + forced], device="cuda:0")
            m1 = dict(mi); m1["input_ids"] = stem; m1["attention_mask"] = torch.ones_like(stem)
            torch.manual_seed(seed_of(args.seed, token, "armN"))
            with torch.no_grad():
                g = llm.generate(**m1, max_length=gc["max_length"], do_sample=True, temperature=gc["temperature"],
                                 top_k=gc["top_k"], top_p=gc["top_p"])
            new = g[0][stem.shape[1]:].tolist()
            full_new = forced + new
            acts_all = [t - A0 for t in full_new if t >= A0]
            pred = acts_all[:10]
            text = pl.tok.decode(full_new, skip_special_tokens=False)
            cot = "complex scenario" in text
            runaway = len(acts_all) if len(acts_all) > 10 else 0
            traj = np.asarray(pl.decode(pred), np.float32)[:, :2].tolist() if pred else []
            gt = pl.tokenize(scene["gt_trajectory"])
            lab = C.stage1_labels(pred, gt, traj if len(traj) == 10 else None, scene["gt_trajectory"], cot, runaway, a_eval)
            fa = next((j for j, x in enumerate(full_new) if x >= A0), len(full_new))
            anno = next((json.load(open(p)) for p in (os.path.join(d, f"{token}.json") for d in C.ANNO_DIRS)
                         if os.path.exists(p)), {})
            rec = {"token": token, "log_name": C.log_of_scene(scene), "map_name": anno.get("map_name"),
                   "instruction": scene.get("instruction"), "velocity": scene.get("velocity"),
                   "acceleration": scene.get("acceleration"), "trajectory_gt": scene["gt_trajectory"],
                   "gt_action_idx": gt, "gt_action_token_ids": [A0 + a for a in gt],
                   "arms": {"N": {"token_ids": full_new, "action_token_ids": [A0 + a for a in acts_all],
                                  "pred_action_idx": acts_all, "cot_present": cot, "runaway_action_tokens": runaway,
                                  "truncated": bool(full_new and full_new[-1] != pl.tok.eos_token_id),
                                  "trajectory_pred": traj, "generated_text": text, "tensor_file": None}},
                   "p1c": {"stage": "stage1_natural", "procedure": "full_extract_armN_generate",
                           "key": f"{args.seed}:{token}:armN", "seed": args.seed, "stub_len": fa, **lab,
                           "runtime_s": round(time.time() - ts, 3)}}
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n_ok += 1
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": token, "stage": "stage1", "error": repr(exc)}) + "\n"); ferr.flush()
        ftime.write(json.dumps({"token": token, "s": round(time.time() - ts, 3)}) + "\n"); ftime.flush()
        if i % 25 == 0 or i == len(toks):
            el = time.time() - t0
            print(f"[prog] {i}/{len(toks)} ok={n_ok} fail={n_fail} {el/i:.2f}s/scene", flush=True)
    fout.close(); ferr.close(); ftime.close()
    with open(os.path.join(args.output, "run_meta.jsonl"), "a") as f:
        f.write(json.dumps({"n_ok": n_ok, "n_fail": n_fail, "load_s": round(load_s, 1), "decode_s": round(time.time() - t0, 1),
                            "procedure": "full_extract_armN_generate", "seed": args.seed, "gen_conf": gc,
                            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                            "utc": time.strftime("%FT%TZ", time.gmtime())}) + "\n")
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
