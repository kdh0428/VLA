#!/usr/bin/env python
"""
Layer-wise previous-action state patching (GPU 1, natural/fast only).

Question: through which layer representation does the immediately preceding action token
propagate instability to the next action?

Data: the equal-distance perturbation set, exactly as used by action_history_causal
(208 scenes: A- 52, A+ 156; unit = (scene, perturbation) with the forced token at t*).
Decoding is the action_history_causal harness: prefix KV cache (prompt + fast stub + action
tokens before t*), then at every step a forward over the post-t* span, action-row sampling
at T = 0.01 with seed sha256(f"{seed}:{token}:{perturbation}:{k}").

CAUSAL INDEXING
  At step k the span is [a_t*, a_{t*+1}, ..., a_{k-1}] and the logits at its LAST position
  produce a_k. So "the immediately preceding action token" is the last span position, which
  is also the query position for a_k. The patch always targets that position (index m-1 of
  the span; absolute index P + m - 1), tracked per step.
  At k = t*+1 the preceding token is the perturbation itself, identical in every rollout, so
  source == target and the patch is a no-op by construction. The first effective patch is at
  k = t*+2, the first step whose preceding token is a SELF-GENERATED action.

CONSEQUENCE OF THAT INDEXING (read results with it in mind)
  patch@emb   = replacing the layer-0 input of that position by the GT token's embedding,
                i.e. mathematically Recent-GT;
  patch@L35   = replacing the final residual, i.e. forcing GT-history's next-action
                distribution. Both ends recover by construction; the interior is what informs.

LAYER KEYS
  emb = input to block 0 (embedding output);  L{i} = output of decoder block i, i = 0..35.

ROWS (one batched forward per step; every row shares the prefix, seeds and numerics)
  normal                     plain autoregression on the model's own tokens
  gt_history                 context after t* is GT (perturbation kept)
  recent_gt                  only the most recent context token is GT            (reference)
  patch_full@{layer}         normal context; last position at `layer` <- gt_history row's state
  patch_delta{a}@{layer}     h + a (h_gt_history - h), a in {0.25, 0.5, 0.75}, representative layers
  selfpatch@{layer}          normal <- normal row's own state            (Check 1: must equal normal)
  identitysrc@{layer}        normal context; source = recent_gt row (GT token identity with the
                             model's own earlier context) -- separates identity from GT context
  reverse@{layer}            gt_history context; last position <- normal row's state (noising)
A patch copies across rows inside the same forward, so the source is the counterpart state of
the same scene, perturbation, semantic step and layer.

KV CACHE
  The prefix cache is never patched and is cropped back after every step; the span is
  recomputed from scratch each step, so a patched state always reaches the later layers' K/V
  and the logits. --debug verifies this numerically (hidden, K, V, logits differences).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)

N_ACT = 10
N_LAYERS = 36
ALL_LAYERS = ["emb"] + [f"L{i}" for i in range(N_LAYERS)]
REP_LAYERS = ["emb", "L0", "L4", "L8", "L12", "L16", "L20", "L24", "L28", "L35"]
ALPHAS = [0.25, 0.5, 0.75]


def build_rows():
    rows = [{"name": "normal", "base": "normal"},
            {"name": "gt_history", "base": "gt"},
            {"name": "recent_gt", "base": "recent_gt"}]
    for lk in ALL_LAYERS:
        rows.append({"name": f"patch_full@{lk}", "base": "normal", "layer": lk, "alpha": 1.0, "src": "gt_history"})
    for lk in REP_LAYERS:
        for a in ALPHAS:
            rows.append({"name": f"patch_delta{a}@{lk}", "base": "normal", "layer": lk, "alpha": a, "src": "gt_history"})
    for lk in ("emb", "L16", "L35"):
        rows.append({"name": f"selfpatch@{lk}", "base": "normal", "layer": lk, "alpha": 1.0, "src": "normal"})
    for lk in REP_LAYERS:
        rows.append({"name": f"identitysrc@{lk}", "base": "normal", "layer": lk, "alpha": 1.0, "src": "recent_gt"})
    for lk in REP_LAYERS:
        rows.append({"name": f"reverse@{lk}", "base": "gt", "layer": lk, "alpha": 1.0, "src": "normal"})
    return rows


class PATCH:
    on = False
    plan = {}          # layer key -> (tgt idx tensor, src idx tensor, alpha tensor (n,1,))
    debug = None       # dict collecting assertions for a debug step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-records", default=os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl"))
    ap.add_argument("--ah-records", default=os.path.join(POC_DIR, "outputs/action_history_causal/records.jsonl"))
    ap.add_argument("--full-records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/prev_action_state_patching"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-alts", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="numerical checks of the intervention path")
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("ed_records", "ah_records", "full_records", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(args.output, exist_ok=True)

    ROWS = build_rows()
    B = len(ROWS)
    idx = {r["name"]: i for i, r in enumerate(ROWS)}
    src_of = {"gt_history": idx["gt_history"], "normal": idx["normal"], "recent_gt": idx["recent_gt"]}

    ed = [json.loads(l) for l in open(args.ed_records)]
    ed.sort(key=lambda r: (r["group"] != "A-", r["token"]))
    if args.limit:
        a_m = [r for r in ed if r["group"] == "A-"][:max(1, args.limit // 2)]
        a_p = [r for r in ed if r["group"] == "A+"][:args.limit - len(a_m)]
        ed = a_m + a_p
    want = {r["token"] for r in ed}
    stub_of = {}
    for line in open(args.full_records):
        x = json.loads(line)
        if x["token"] in want:
            stub_of[x["token"]] = x["arms"]["N"]["token_ids"]
    prior = {}                                      # action_history_causal rows, for reproduction
    for line in open(args.ah_records):
        x = json.loads(line)
        if x["token"] in want:
            for pname, pr in x["perturbations"].items():
                prior[(x["token"], pname)] = {c: pr["conditions"][c]["action_idx"] for c in ("normal", "gt_history", "recent_gt")}
    print(f"[plan] {len(ed)} scenes (A- {sum(r['group']=='A-' for r in ed)}, A+ {sum(r['group']=='A+' for r in ed)}), "
          f"{B} rows per batched step", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    temp = float(cfg["inference"]["sample"]["temperature"])
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id
    layers = llm.model.layers

    # ---- patch plan: per layer key, vectorised (tgt, src, alpha) ------------------------------
    plan = {}
    for i, r in enumerate(ROWS):
        if "layer" in r:
            plan.setdefault(r["layer"], []).append((i, src_of[r["src"]], r["alpha"]))
    PATCH.plan = {lk: (torch.tensor([t for t, _, _ in v], device="cuda:0"),
                       torch.tensor([s for _, s, _ in v], device="cuda:0"),
                       torch.tensor([a for _, _, a in v], device="cuda:0", dtype=torch.float32)[:, None])
                  for lk, v in plan.items()}

    def apply_patch(hs, lk):
        if not PATCH.on or lk not in PATCH.plan:
            return hs
        tgt, src, alpha = PATCH.plan[lk]
        pre = hs.clone() if PATCH.debug is not None else None
        h_src = hs[src, -1].float()
        h_tgt = hs[tgt, -1].float()
        new_last = (h_tgt + alpha * (h_src - h_tgt)).to(hs.dtype)
        hs = hs.clone()
        hs[tgt, -1] = new_last
        if PATCH.debug is not None:
            full = (alpha[:, 0] == 1.0)
            d = PATCH.debug.setdefault(lk, {})
            d["max_abs_tgt_minus_src_alpha1"] = float((hs[tgt[full], -1].float() - hs[src[full], -1].float()).abs().max()) if full.any() else None
            d["other_positions_changed"] = float((hs[:, :-1].float() - pre[:, :-1].float()).abs().max()) if hs.shape[1] > 1 else 0.0
            untouched = torch.ones(hs.shape[0], dtype=torch.bool, device=hs.device)
            untouched[tgt] = False
            d["untouched_rows_changed"] = float((hs[untouched].float() - pre[untouched].float()).abs().max())
            d["mean_patch_l2"] = float((new_last.float() - pre[tgt, -1].float()).norm(dim=-1).mean())
        return hs

    def pre_hook_emb(_m, a, kw):
        if a:
            hs = apply_patch(a[0], "emb")
            return (hs,) + tuple(a[1:]), kw
        if kw.get("hidden_states") is not None:
            kw = dict(kw); kw["hidden_states"] = apply_patch(kw["hidden_states"], "emb")
        return a, kw
    layers[0].register_forward_pre_hook(pre_hook_emb, with_kwargs=True)

    def make_post(i):
        def hook(_m, _a, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs2 = apply_patch(hs, f"L{i}")
            if hs2 is hs:
                return out
            return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
        return hook
    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_post(i))

    KV = {"on": False, "k": {}, "v": {}}

    def make_kv(i, kind):
        def hook(_m, _a, out):
            if KV["on"]:
                KV[kind][i] = out.detach().float().cpu()
        return hook
    if args.debug:
        for i, layer in enumerate(layers):
            layer.self_attn.k_proj.register_forward_hook(make_kv(i, "k"))
            layer.self_attn.v_proj.register_forward_hook(make_kv(i, "v"))

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

    fout = open(os.path.join(args.output, "records.jsonl"), "w")
    ferr = open(os.path.join(args.output, "errors.jsonl"), "w")
    debug_log = []
    t0 = time.time()
    n_ok = n_fail = n_units = n_unit_fail = 0
    for si, r in enumerate(ed, 1):
        try:
            token, t, gt, pred = r["token"], r["t_star"], r["gt"], r["pred"]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            n_ids = stub_of[token]
            stub = n_ids[:next(j for j, x in enumerate(n_ids) if x >= A0)]
            prefix = mi["input_ids"][0].tolist() + stub + [A0 + a for a in pred[:t]]
            P = len(prefix)
            ids = torch.tensor([prefix], device="cuda:0")
            llm.rope_deltas = None
            PATCH.on = False
            with torch.no_grad():
                out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                          video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"))
            cache = out.past_key_values
            del out
            for li in range(len(cache.key_cache)):                  # one prefix copy per row
                cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
                cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
            prefix_checksum = float(cache.key_cache[0][0, :, :P].float().abs().sum())
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            ferr.write(json.dumps({"token": r.get("token"), "stage": "scene", "error": repr(exc)}) + "\n"); ferr.flush()
            continue

        perts = [("original", pred[t])] + [(f"alt{j}", a["token"]) for j, a in enumerate(r["alternatives"])]
        if args.max_alts:
            perts = perts[:1 + args.max_alts]
        for pname, ptok in perts:
            try:
                choices = [[] for _ in range(B)]
                stats = {name: {"ent": [], "p_gt": [], "p_normal_choice": [], "js_to_normal": [], "kl_normal_to_row": [], "js_to_gt_history": []}
                         for name in idx}
                steps_log = []
                for k in range(t + 1, N_ACT):
                    spans = []
                    for i, row in enumerate(ROWS):
                        if row["base"] == "gt":
                            ctx = list(gt[t + 1:k])
                        elif row["base"] == "recent_gt":
                            ctx = choices[i][:-1] + [gt[k - 1]] if choices[i] else []
                        else:
                            ctx = choices[i]
                        spans.append([A0 + ptok] + [A0 + a for a in ctx])
                    m = len(spans[0])
                    assert all(len(s) == m for s in spans), "span lengths must match across rows"
                    steps_log.append({"k": k, "prev_position_abs": P + m - 1, "span_len": m,
                                      "normal_prev_token": spans[idx["normal"]][-1] - A0,
                                      "gt_prev_token": spans[idx["gt_history"]][-1] - A0,
                                      "prev_is_perturbation": m == 1})
                    sids = torch.tensor(spans, device="cuda:0")
                    debug_step = args.debug and k == t + 2 and pname == "original"
                    PATCH.debug = {} if debug_step else None
                    KV["on"] = debug_step
                    PATCH.on = True
                    with torch.no_grad():
                        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
                    PATCH.on = False
                    KV["on"] = False
                    cache.crop(P)
                    lg = o.logits[:, -1, A0:A0 + 2048].double()
                    del o
                    lp = torch.log_softmax(lg, -1)
                    pr = lp.exp()
                    ent = -(pr * lp).sum(-1)
                    # sampling: identical seed for every row (as action_history_causal)
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}:{pname}:{k}".encode()).digest()[:4], "little")
                    probs_t = torch.softmax(lg / temp, -1).cpu()
                    gen = torch.Generator()
                    picks = []
                    for i in range(B):
                        gen.manual_seed(seed)
                        picks.append(int(torch.multinomial(probs_t[i], 1, generator=gen)))
                    n_choice = picks[idx["normal"]]
                    pn, pg = pr[idx["normal"]], pr[idx["gt_history"]]

                    def js(p, q):
                        mm = 0.5 * (p + q)
                        return 0.5 * (p * (torch.log(p + 1e-30) - torch.log(mm + 1e-30))).sum(-1) + \
                            0.5 * (q * (torch.log(q + 1e-30) - torch.log(mm + 1e-30))).sum(-1)
                    js_n = js(pr, pn.expand_as(pr))
                    js_g = js(pr, pg.expand_as(pr))
                    kl_n = (pn * (torch.log(pn + 1e-30) - torch.log(pr + 1e-30))).sum(-1)
                    for name, i in idx.items():
                        s = stats[name]
                        s["ent"].append(round(float(ent[i]), 5)); s["p_gt"].append(round(float(pr[i, gt[k]]), 6))
                        s["p_normal_choice"].append(round(float(pr[i, n_choice]), 6)); s["js_to_normal"].append(round(float(js_n[i]), 6))
                        s["kl_normal_to_row"].append(round(float(kl_n[i]), 6)); s["js_to_gt_history"].append(round(float(js_g[i]), 6))
                    if debug_step:
                        dbg = {"token": token, "k": k, "t_star": t, "patch_checks": PATCH.debug,
                               "span_normal": [x - A0 for x in spans[idx["normal"]]], "span_gt": [x - A0 for x in spans[idx["gt_history"]]]}
                        # K/V and logits differences: normal vs patch_full@L{i}; must be 0 at layers <= i
                        kvd = {}
                        for lk in ("emb", "L4", "L16", "L28", "L35"):
                            ti = idx[f"patch_full@{lk}"]
                            cut = -1 if lk == "emb" else int(lk[1:])
                            dk = [float((KV["k"][li][ti, -1] - KV["k"][li][idx["normal"], -1]).abs().max()) for li in range(N_LAYERS)]
                            dv = [float((KV["v"][li][ti, -1] - KV["v"][li][idx["normal"], -1]).abs().max()) for li in range(N_LAYERS)]
                            kvd[lk] = {"max_K_diff_at_or_before_patch": max(dk[:cut + 1]) if cut >= 0 else None,
                                       "max_V_diff_at_or_before_patch": max(dv[:cut + 1]) if cut >= 0 else None,
                                       "max_K_diff_after_patch": max(dk[cut + 1:]) if cut + 1 < N_LAYERS else None,
                                       "max_V_diff_after_patch": max(dv[cut + 1:]) if cut + 1 < N_LAYERS else None,
                                       "logits_L1_vs_normal": float((lg[ti] - lg[idx["normal"]]).abs().sum()),
                                       "js_vs_normal": float(js_n[ti]), "js_vs_gt_history": float(js_g[ti])}
                        dbg["kv_logits"] = kvd
                        dbg["selfpatch_js_vs_normal"] = {lk: float(js_n[idx[f"selfpatch@{lk}"]]) for lk in ("emb", "L16", "L35")}
                        dbg["emb_patch_vs_recent_gt_logits_L1"] = float((lg[idx["patch_full@emb"]] - lg[idx["recent_gt"]]).abs().sum())
                        dbg["L35_patch_vs_gt_history_logits_L1"] = float((lg[idx["patch_full@L35"]] - lg[idx["gt_history"]]).abs().sum())
                        dbg["normal_vs_gt_history_js"] = float(js_g[idx["normal"]])
                        debug_log.append(dbg)
                        KV["k"].clear(); KV["v"].clear()
                    PATCH.debug = None
                    for i in range(B):
                        choices[i].append(picks[i])
                # prefix cache must be untouched by the whole rollout
                chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
                assert abs(chk - prefix_checksum) < 1e-3 * max(1.0, prefix_checksum), "prefix cache was modified"
                assert cache.get_seq_length() == P, "cache not cropped back to the prefix"

                rows_out = {}
                acts_all = [list(pred[:t]) + [ptok] + choices[i] for i in range(B)]
                trajs = []
                for i in range(B):
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts_all[i]]))
                    trajs.append(np.asarray(tr[0, 1:], np.float32)[:, :2].tolist())
                for name, i in idx.items():
                    rows_out[name] = {"action_idx": acts_all[i], "trajectory_pred": trajs[i], **stats[name]}
                pri = prior.get((token, pname))
                rec = {"token": token, "group": r["group"], "t_star": t, "log": r["log"], "perturbation": pname,
                       "forced_token": ptok, "gt": gt, "pred_prefix": pred[:t], "trajectory_gt": r["trajectory_gt"],
                       "steps": steps_log, "rows": rows_out,
                       "prior_action_history": pri,
                       "check_prefix_identical_all_rows": all(a[:t + 1] == acts_all[0][:t + 1] for a in acts_all)}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                n_units += 1
            except Exception as exc:
                n_unit_fail += 1
                import traceback; traceback.print_exc()
                ferr.write(json.dumps({"token": r["token"], "perturbation": pname, "stage": "unit", "error": repr(exc)}) + "\n"); ferr.flush()
        del cache
        n_ok += 1
        if si % 10 == 0 or args.debug:
            el = time.time() - t0
            print(f"[prog] {si}/{len(ed)} scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail} "
                  f"{el/si:.1f}s/scene eta {(len(ed)-si)*el/si/60:.1f}min", flush=True)
    fout.close(); ferr.close()
    if debug_log:
        json.dump(debug_log, open(os.path.join(args.output, "debug_checks.json"), "w"), indent=1)
    json.dump({"n_scenes_ok": n_ok, "n_scene_fail": n_fail, "n_units": n_units, "n_unit_fail": n_unit_fail,
               "rows": ROWS, "layer_keys": ALL_LAYERS, "rep_layers": REP_LAYERS, "alphas": ALPHAS, "temperature": temp,
               "seed": args.seed, "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] scenes ok={n_ok} fail={n_fail} units={n_units} unit_fail={n_unit_fail}", flush=True)


if __name__ == "__main__":
    main()
