#!/usr/bin/env python
"""
A3 (review_followup4): Impromptu VLA 3B same-condition parity diagnosis on a few scenes. RTX 5090 only.

Uses the UNMODIFIED experiment-33 code (scripts/cross_vla_temporal/impromptu_core.py: Impromptu.continue_batch,
FastWaypointDecoder, prefix_text, parse_wps, scene_inputs). Nothing in that code is changed; this script only calls it
under controlled conditions and, for token-level checks, repeats the exact `generate` call of continue_batch to keep ids.

Per scene (stored exp 33 records give the reference values):
  N1, N2     natural decode, B = 1 (continue_batch([""])) twice in one process     -> same-condition determinism
  N_vs_store N1 vs the stored `natural` (cross-process / different day)
  AB18       the exact exp_ab batch (18 rows: "", prefix(nat w1), 16 perturbed prefixes), chunk 24 -> natural_rep, pert_m0
  NOPAD18    18 identical "" rows (batch size 18, no padding)                          -> batch size without padding
  PAD2       ["", prefix(nat w1)] (B = 2, the "" row left-padded)                      -> padding with minimal batch
  M0_B1      prefix(nat w1) alone, B = 1                                                -> re-encoded prefix path, no batch
  TF_HF      teacher-forced next waypoint, HF one_waypoint path, B = 1, prefix = natural w1..w_{k-1}   (k = 2..10)
  TF_FAST    teacher-forced next waypoint, FastWaypointDecoder, B = 1, same prefixes                  (k = 2..10)
  FREE_FAST  FastWaypointDecoder free-running delta-0 normal row, B = 1 (vs N1 and vs stored exp_cde normal)
  RETOK      natural generated ids vs tokenizer(ANSWER-side text rebuilt with prefix_text from parsed waypoints)
             and string identity of the rebuilt text vs the generated text (0.01 m formatting, no -0.0, trailing zeros)

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python a3_parity_run.py <out_dir> --tokens a,b,c
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

POC = "/root/VLA/autovla_misalignment_poc"
sys.path.insert(0, os.path.join(POC, "scripts/cross_vla_temporal"))
from impromptu_core import FastWaypointDecoder, Impromptu, parse_wps, prefix_text, scene_inputs, ANSWER_HEAD, fmt_wp  # noqa: E402

SCENES = "/root/VLA/autovla/dataset/nuplan/navtest_poc"
REP = os.path.join(POC, "outputs/cross_vla_temporal_replication")
MAGS, DIRS = (0.2, 0.5), tuple(range(0, 360, 45))


def r2(x):
    return np.round(np.asarray(x, float), 2).tolist()


def first_diff(a, b):
    a, b = np.round(np.asarray(a, float), 2), np.round(np.asarray(b, float), 2)
    n = min(len(a), len(b))
    for i in range(n):
        if np.abs(a[i] - b[i]).max() > 1e-9:
            return i, float(np.abs(a[i] - b[i]).max())
    return (None, 0.0) if len(a) == len(b) else (n, float("nan"))


@torch.no_grad()
def gen_ids(m, q, imgs, prefix=""):
    """Exactly continue_batch's generate call for one row; returns (new ids, decoded text)."""
    texts = [m._text(q, prefix)]
    enc = m.proc(text=texts, images=list(imgs), return_tensors="pt", padding=True).to("cuda:0")
    out = m.model.generate(**enc, do_sample=False, repetition_penalty=1.05, pad_token_id=m.proc.tokenizer.pad_token_id,
                           max_new_tokens=200)
    new = out[0, enc["input_ids"].shape[1]:].tolist()
    return new, m.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "parity_records.jsonl")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    AB = {}
    for l in open(os.path.join(REP, "exp_ab/records.jsonl")):
        x = json.loads(l); AB[x["token"]] = x
    CDE = {}
    for l in open(os.path.join(REP, "exp_cde/records.jsonl")):
        x = json.loads(l); CDE[x["token"]] = x
    m = Impromptu(); fd = FastWaypointDecoder(m); tok = m.proc.tokenizer
    t0 = time.time()
    with open(dst, "w") as fo:
        for t in a.tokens.split(","):
            ts = time.time()
            log = json.load(open(os.path.join(SCENES, f"{t}.json")))["front_camera_paths"][0].split("/")[0]
            q, imgs, fut = scene_inputs(log, t)
            R = {"token": t, "log": log, "in_exp_ab": t in AB, "in_exp_cde": t in CDE}
            n1 = parse_wps(m.continue_batch(q, imgs, [""])[0])[:10]
            n2 = parse_wps(m.continue_batch(q, imgs, [""])[0])[:10]
            R["N1"], R["N2"] = n1, n2
            R["N1_vs_N2"] = first_diff(n1, n2)
            if t in AB:
                R["N1_vs_store_natural"] = first_diff(n1, AB[t]["natural"])
                nat_store = AB[t]["natural"]
            elif t in CDE:
                R["N1_vs_store_natural"] = first_diff(n1, CDE[t]["natural"])
                nat_store = CDE[t]["natural"]
            nat = n1
            # ---- exact exp_ab batch (built from THIS run's natural, as exp_ab does)
            rows = [("natural_rep", None), ("pert_m0", tuple(nat[0]))]
            for mg in MAGS:
                for ang in DIRS:
                    d = (mg * math.cos(math.radians(ang)), mg * math.sin(math.radians(ang)))
                    rows.append((f"pert_m{mg}_d{ang}", (nat[0][0] + d[0], nat[0][1] + d[1])))
            outs = m.continue_batch(q, imgs, ["" if w1 is None else prefix_text([w1]) for _, w1 in rows])
            rep = parse_wps(outs[0])[:10]; m0 = [list(nat[0])] + parse_wps(outs[1])[:9]
            R["AB18_natural_rep"], R["AB18_pert_m0"] = rep, m0
            R["AB18_rep_vs_N1"] = first_diff(rep, nat)
            R["AB18_m0_vs_N1"] = first_diff(m0, nat)
            if t in AB and first_diff(nat, AB[t]["natural"])[0] is None:
                R["AB18_rep_vs_store_rep"] = first_diff(rep, AB[t]["rows"]["natural_rep"]["traj"])
                R["AB18_m0_vs_store_m0"] = first_diff(m0, AB[t]["rows"]["pert_m0"]["traj"])
            # ---- batch size without padding / padding with B = 2 / re-encoded prefix alone
            o = m.continue_batch(q, imgs, [""] * 18)
            nop = [parse_wps(x)[:10] for x in o]
            R["NOPAD18_rows_identical_to_each_other"] = all(first_diff(nop[0], x)[0] is None for x in nop)
            R["NOPAD18_vs_N1"] = first_diff(nop[0], nat)
            o = m.continue_batch(q, imgs, ["", prefix_text([nat[0]])])
            R["PAD2_rep_vs_N1"] = first_diff(parse_wps(o[0])[:10], nat)
            R["PAD2_m0_vs_N1"] = first_diff([list(nat[0])] + parse_wps(o[1])[:9], nat)
            o = m.continue_batch(q, imgs, [prefix_text([nat[0]])])
            R["M0_B1_vs_N1"] = first_diff([list(nat[0])] + parse_wps(o[0])[:9], nat)
            # ---- token-level re-tokenization and text identity
            ids, text = gen_ids(m, q, imgs, "")
            wps = parse_wps(text)[:10]
            R["gen_text"] = text
            R["gen_ids_eq_continue_batch"] = first_diff(wps, nat)[0] is None
            rebuilt = prefix_text(wps)          # " [x, y], [x, y], ... ,"
            R["rebuilt_text_is_prefix_of_generated"] = text.startswith(rebuilt) or text.startswith(rebuilt[:-1])
            per_k = []
            for k in range(1, 10):
                pt = prefix_text(wps[:k])
                re_ids = tok(pt, add_special_tokens=False)["input_ids"]
                per_k.append({"k": k, "text_prefix_identical": text.startswith(pt),
                              "ids_prefix_identical": ids[:len(re_ids)] == re_ids,
                              "n_retok": len(re_ids)})
            R["RETOK"] = per_k
            R["gen_ids_head"] = ids[:40]
            R["gen_tokens_head"] = tok.convert_ids_to_tokens(ids[:40])
            # ---- teacher-forced single-step parity: HF one_waypoint path vs FastWaypointDecoder, B = 1
            fd.set_scene(q, imgs)
            tf = []
            for k in range(2, 11):
                pt = prefix_text(nat[:k - 1])
                hf = parse_wps(m.continue_batch(q, imgs, [pt], one_waypoint=True)[0])
                fs = parse_wps(fd.next_waypoint([pt])[0])
                tf.append({"k": k, "natural": list(nat[k - 1]), "hf_one": list(hf[0]) if hf else None,
                           "fast": list(fs[0]) if fs else None,
                           "hf_eq_nat": bool(hf) and r2(hf[0]) == r2(nat[k - 1]),
                           "fast_eq_nat": bool(fs) and r2(fs[0]) == r2(nat[k - 1]),
                           "fast_eq_hf": bool(hf) and bool(fs) and r2(hf[0]) == r2(fs[0])})
            R["TF"] = tf
            # ---- free-running fast decoder, delta-0 normal row alone (B = 1)
            own = [list(nat[0])]
            for k in range(2, 11):
                w = parse_wps(fd.next_waypoint([prefix_text(own)])[0])
                own.append(list(w[0]) if w else own[-1])
            R["FREE_FAST_B1"] = own
            R["FREE_FAST_B1_vs_N1"] = first_diff(own, nat)
            if t in CDE:
                u0 = [u for u in CDE[t]["units"] if u["dir"] is None][0]
                R["store_cde_normal_vs_store_natural"] = first_diff(u0["rows"]["normal"], CDE[t]["natural"])
                R["FREE_FAST_B1_vs_store_cde_normal"] = first_diff(own, u0["rows"]["normal"])
            R["sec"] = round(time.time() - ts, 1)
            fo.write(json.dumps(R) + "\n"); fo.flush()
            print(f"[a3] {t} done in {R['sec']} s", flush=True)
    json.dump({"tokens": a.tokens.split(","), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "elapsed_min": (time.time() - t0) / 60, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9},
              open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
