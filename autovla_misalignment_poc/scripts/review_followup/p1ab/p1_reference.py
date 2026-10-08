#!/usr/bin/env python
"""
P1 reference rollouts (GPU, RTX 5090 only). Shared by P1-A and P1-B.

For every scene of the equal-distance set, the model's own unperturbed rollout is generated ONCE
and written to disk (with sha256 in manifest) before any perturbation/correction condition runs.
No outcome label is computed or read here (no GT metric, no A-/A+ selection of rollouts).

  r[:t*+1] = pred[:t*+1]        the stored natural tokens (full_extract arm N; model output, GT-free)
  r[t*+1:] = harness rollout    prefix KV (prompt + stub + pred[:t*]), span [pred[t*], r_t*+1, ...],
                                B = 1, action rows, T = 0.01, seed key "original" =
                                sha256(f"{seed}:{token}:original:{k}")  (= the exp-7 Normal 'original' row;
                                reproduction vs exp-7 raw is checked as a diagnostic, not a filter)
The perturbation position t* itself was defined in the equal-distance run from the first GT mismatch;
that selection is inherited (documented in the protocols), the reference tokens are not GT-informed.

Outputs (in --out): references.jsonl, prefix_logprobs.npz (log p over the 2,048 action rows at
position t*, for diagnostics), run_meta.json, manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--tokens", default=None, help="comma-separated scene tokens (smoke)")
    args = ap.parse_args()
    C.gpu_guard()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "references.jsonl")):
        raise SystemExit(f"{out}/references.jsonl exists; refusing to overwrite")

    ed = C.load_ed(args.limit_scenes, args.tokens.split(",") if args.tokens else None)
    want = {r["token"] for r in ed}
    stub_of, _, _ = C.load_stubs_and_token_freq(want)
    exp7 = {}
    for line in open(C.AH_RECORDS):
        x = json.loads(line)
        if x["token"] in want:
            exp7[x["token"]] = x["perturbations"]["original"]["conditions"]["normal"]["action_idx"]

    torch, model, llm, A0, temp, agent, vis_box, cfg = C.load_model()
    gpu = C.assert_5090(torch)
    fout = open(os.path.join(out, "references.jsonl"), "w")
    lp_store = {}
    t0 = time.time()
    n_ok = 0
    for si, r in enumerate(ed, 1):
        token, t, pred = r["token"], r["t_star"], r["pred"]
        cache, P, chk, lg_prefix = C.build_prefix_cache(torch, model, llm, agent, vis_box, A0, token, stub_of[token], pred[:t], 1)
        lp0 = torch.log_softmax(lg_prefix, -1).numpy()
        lp_store[token] = lp0.astype(np.float32)
        own, ents = [], []
        for k in range(t + 1, C.N_ACT):
            span = [A0 + pred[t]] + [A0 + a for a in own]
            full = C.forward_spans(torch, llm, cache, P, [span])
            lg = full[:, A0:A0 + C.N_TOK]
            lp = torch.log_softmax(lg, -1)
            ents.append(round(float(-(lp.exp() * lp).sum()), 5))
            own += C.sample_rows(torch, lg, temp, C.seed_int(args.seed, token, "original", k))
        cc = C.check_cache(cache, P, chk)
        acts = list(pred[:t]) + [pred[t]] + own
        poses, origin = C.decode(torch, model, A0, acts)
        rec = {"token": token, "log": r["log"], "group_stratum": r["group"], "t_star": t, "ref_tokens": acts,
               "ref_traj_xyh": np.round(poses, 6).tolist(), "decoder_origin": origin.tolist(), "entropy_steps": ents,
               "prefix_len": P, "harness_argmax_at_tstar": int(np.argmax(lp0)), "logp_pred_tstar": float(lp0[pred[t]]),
               "seed_key": "original", "seed": args.seed, "batch_rows": 1, "cache_check": cc,
               "diag_equals_exp7_normal_original": (exp7.get(token) == acts) if token in exp7 else None}
        fout.write(C.jdump(rec) + "\n"); fout.flush()
        n_ok += 1
        del cache
        if si % 20 == 0 or args.limit_scenes or args.tokens:
            el = time.time() - t0
            print(f"[ref] {si}/{len(ed)} {el/si:.2f}s/scene eta {(len(ed)-si)*el/si/60:.1f} min", flush=True)
    fout.close()
    np.savez_compressed(os.path.join(out, "prefix_logprobs.npz"), **lp_store)
    meta = C.env_record({"script": os.path.abspath(__file__), "n_scenes": n_ok, "seed": args.seed, "temperature": temp,
                         "elapsed_min": (time.time() - t0) / 60, "gpu_checked": gpu,
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
    man = {"files": {f: C.sha256_file(os.path.join(out, f)) for f in ("references.jsonl", "prefix_logprobs.npz", "run_meta.json")},
           "note": "reference fixed before any condition run; consumers verify references.jsonl sha256"}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"[done] {n_ok} references", flush=True)


if __name__ == "__main__":
    main()
