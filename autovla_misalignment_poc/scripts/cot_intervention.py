#!/usr/bin/env python
"""
Is AutoVLA's CoT a causal driver of its action tokens?

Same image / ego state / prompt; only the reasoning text before the action tokens changes.
Every condition teacher-forces a text prefix that ends at "The final output action is: " and
then generates the action tokens FREELY with the original decoding config (do_sample,
T=0.01, top_k=0, top_p=1). One seed per scene, shared by every condition, so conditions differ
only in the prefix.

Conditions
  natural_nocot       the model's own fast-thinking stub (stored arm N ids)  -- NATURAL mode
  original            the model's own forced CoT (stored arm C ids)           -- FORCED mode
  template_original   Reasoning on Intent + Best Driving Action rewritten by template, still
                      declaring the ORIGINAL decision -> control for "any rewrite at all"
  corrected_decision  only the declared decision phrase replaced by the GT action
  corrected_full      template reasoning + decision for the GT action
  counter_decision    only the declared decision phrase replaced by the opposite action
  counter_full        template reasoning + decision for the opposite action

Scene Description and Critical Object Description are never edited: the perception text is
held fixed, only reasoning / decision changes.

The GT action text is the record's `gt_action_instruction`, i.e. AutoVLA's own parser applied
to the GT trajectory, so the corrected CoT speaks the model's own action vocabulary.

Nothing existing is modified; outputs go to outputs/cot_intervention/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRA_DIR = "/root/VLA/pra_comparison"
sys.path.insert(0, AUTOVLA_DIR)
sys.path.insert(0, os.path.join(AUTOVLA_DIR, "navsim"))
sys.path.insert(0, POC_DIR)
sys.path.insert(0, PRA_DIR)

import pra_labels as PL                                    # noqa: E402
from src.labeling.labels import _LON_MAP, parse_action_instruction  # noqa: E402

CONDITIONS = ["natural_nocot", "original", "template_original", "corrected_decision",
              "corrected_full", "counter_decision", "counter_full"]
N_ACT = 10

LON_WORD = {"STOP": "a deceleration to zero", "DECELERATE": "a deceleration",
            "MAINTAIN": "a constant speed", "ACCELERATE": "an acceleration"}
CF_LON = {"STOP": "ACCELERATE", "DECELERATE": "ACCELERATE",
          "MAINTAIN": "STOP", "ACCELERATE": "STOP"}
CF_BEH = {"turn left": "turn right", "turn right": "turn left",
          "change lane to left": "change lane to right", "change lane to right": "change lane to left",
          "move forward": "move forward"}
LAT_OF_BEH = {"move forward": "STRAIGHT", "turn left": "LEFT", "change lane to left": "LEFT",
              "turn right": "RIGHT", "change lane to right": "RIGHT"}
LON_REASON = {
    "STOP": "Given the surrounding situation, the ego vehicle should slow down, come to a complete stop and wait before proceeding.",
    "DECELERATE": "Given the surrounding situation, the ego vehicle should reduce its speed to keep a safe margin.",
    "MAINTAIN": "The path ahead is clear enough for the ego vehicle to keep its current speed.",
    "ACCELERATE": "The path ahead is clear, so the ego vehicle should increase its speed to match the traffic flow.",
}
LAT_REASON = {
    "STRAIGHT": "The ego vehicle should keep following its current lane.",
    "LEFT": "The ego vehicle should move to the left as required by the route.",
    "RIGHT": "The ego vehicle should move to the right as required by the route.",
}
JUSTIFY = {
    "STOP": "Stopping is the safest choice here, so the vehicle should come to a complete stop.",
    "DECELERATE": "Decelerating keeps a safe distance and gives the vehicle time to react to its surroundings.",
    "MAINTAIN": "Maintaining a constant speed ensures smooth and safe progress.",
    "ACCELERATE": "Accelerating lets the vehicle proceed efficiently while remaining safe.",
}
_SEC = re.compile(r"#+\s*\**\s*(reasoning on intent|best driving action)[^\n]*", re.I)


# --------------------------------------------------------------------------------------
# text construction
# --------------------------------------------------------------------------------------
def split_instruction(s: str) -> tuple[str, str]:
    s = (s or "").strip().lower().rstrip(".")
    if s == "stop":
        return "move forward", "a deceleration to zero"
    if " with " in s:
        beh, spd = s.split(" with ", 1)
    else:
        beh, spd = s, "a constant speed"
    beh = beh.strip().replace("to the ", "to ")
    return (beh if beh in LAT_OF_BEH else "move forward"), (spd.strip() if spd.strip() in _LON_MAP else "a constant speed")


def phrase(beh: str, spd: str, cap: bool = True) -> str:
    p = f"{beh} with {spd}"
    return p[0].upper() + p[1:] if cap else p


def template(beh: str, spd: str, speed: float, instruction: str) -> str:
    lon, lat = _LON_MAP[spd], LAT_OF_BEH[beh]
    return ("### Reasoning on Intent:\n"
            f"- The ego vehicle is currently travelling at {speed:.1f} m/s, and the driving instruction is to \"{instruction}\".\n"
            f"- {LON_REASON[lon]}\n- {LAT_REASON[lat]}\n\n"
            "### Best Driving Action:\n"
            f"**{phrase(beh, spd)}**\n\n{JUSTIFY[lon]}")


def _section(P: str, name: str):
    hits = [h for h in _SEC.finditer(P) if h.group(1).lower() == name]
    return hits[0] if hits else None


def declared_in(P: str) -> tuple[str, str] | None:
    """(behaviour, speed phrase) declared in the Best Driving Action section, or None."""
    bda = _section(P, "best driving action")
    end = P.find("</think>")
    if bda is None or end < 0:
        return None
    seg = P[bda.end():end].lower()
    m = PL._DECL.search(seg)
    if m:
        return m.group(1).replace("to the ", "to "), m.group(2)
    if re.search(r"\bstop\b", seg):
        return "move forward", "a deceleration to zero"
    return None


def edit_decision(P: str, beh: str, spd: str) -> str | None:
    """Replace ONLY the declared decision phrase; the surrounding reasoning is untouched."""
    bda = _section(P, "best driving action")
    end = P.find("</think>")
    if bda is None or end < 0:
        return None
    seg = P[bda.end():end]
    m = PL._DECL.search(seg.lower())
    if m is None:
        m = re.search(r"\bstop\b", seg, re.I)
        if m is None:
            return None
    old = seg[m.start():m.end()]
    new = phrase(beh, spd, cap=old[:1].isupper())
    return P[:bda.end()] + seg[:m.start()] + new + seg[m.end():] + P[end:]


def edit_full(P: str, beh: str, spd: str, speed: float, instruction: str) -> str | None:
    """Replace Reasoning on Intent + Best Driving Action (everything up to </think>)."""
    start = _section(P, "reasoning on intent") or _section(P, "best driving action")
    end = P.find("</think>")
    if start is None or end < 0:
        return None
    return P[:start.start()] + template(beh, spd, speed, instruction) + "\n" + P[end:]


# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(POC_DIR, "outputs/full_extract/records.jsonl"))
    ap.add_argument("--tensors", default=os.path.join(POC_DIR, "outputs/full_extract/tensors"))
    ap.add_argument("--subsets", default=os.path.join(PRA_DIR, "outputs/subsets/AutoVLA"))
    ap.add_argument("--scenes", default=os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc"))
    ap.add_argument("--output", default=os.path.join(POC_DIR, "outputs/cot_intervention"))
    ap.add_argument("--config", default=os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--checkpoint", default="/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--control-n", type=int, default=64, help="P+R+A+ control scenes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--sanity", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("pinned to GPU 1: launch with CUDA_VISIBLE_DEVICES=1")
    for k in ("records", "tensors", "subsets", "scenes", "output", "config"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    os.makedirs(os.path.join(args.output, "logits"), exist_ok=True)

    # ---- sample selection (from the P/R/A comparison subsets) ------------------------
    picks = []
    for g, f in (("P+R-A-", "PpRmAm.json"), ("P+R+A-", "PpRpAm.json")):
        picks += [(s["sample_id"], g) for s in json.load(open(os.path.join(args.subsets, f)))["samples"]]
    ctrl = [s["sample_id"] for s in json.load(open(os.path.join(args.subsets, "PpRpAp.json")))["samples"]]
    random.Random(args.seed).shuffle(ctrl)
    picks += [(t, "P+R+A+ (control)") for t in ctrl[:args.control_n]]
    if args.limit:
        picks = picks[:args.limit]
    want = {t for t, _ in picks}
    recs = {}
    for line in open(args.records):
        r = json.loads(line)
        if r["token"] in want:
            recs[r["token"]] = r
    print(f"[plan] {len(picks)} scenes x {len(CONDITIONS)} conditions", flush=True)

    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
    gen_conf = cfg["inference"]["sample"]
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0, tok = model.vlm, model.action_start_id, model.processor.tokenizer

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
    t0 = time.time()
    n_ok = n_fail = 0
    for i, (token, group) in enumerate(picks, 1):
        try:
            r = recs[token]
            scene = json.load(open(os.path.join(args.scenes, f"{token}.json")))
            vis_box.clear()
            feats = {}
            for b in agent.get_feature_builders():
                feats.update(b.compute_features(scene))
            mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
            prompt_ids = mi["input_ids"][0].tolist()

            c_ids, n_ids = r["arms"]["C"]["token_ids"], r["arms"]["N"]["token_ids"]
            fa_c = next(j for j, t in enumerate(c_ids) if t >= A0)
            fa_n = next(j for j, t in enumerate(n_ids) if t >= A0)
            P = tok.decode(c_ids[:fa_c], skip_special_tokens=False)
            speed = float(np.linalg.norm(r["velocity"][:2]))
            instr = r.get("instruction") or "keep forward"

            gt_beh, gt_spd = split_instruction(r["gt_action_instruction"])
            cf_beh, cf_spd = CF_BEH[gt_beh], LON_WORD[CF_LON[_LON_MAP[gt_spd]]]
            decl = declared_in(P)

            texts = {
                "template_original": edit_full(P, *decl, speed, instr) if decl else None,
                "corrected_decision": edit_decision(P, gt_beh, gt_spd),
                "corrected_full": edit_full(P, gt_beh, gt_spd, speed, instr),
                "counter_decision": edit_decision(P, cf_beh, cf_spd),
                "counter_full": edit_full(P, cf_beh, cf_spd, speed, instr),
            }
            prefixes = {"natural_nocot": n_ids[:fa_n], "original": c_ids[:fa_c]}
            for name, txt in texts.items():
                prefixes[name] = tok.encode(txt, add_special_tokens=False) if txt else None

            seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{token}".encode()).digest()[:4], "little")
            logits_store = np.full((len(CONDITIONS), N_ACT, 2048), np.nan, dtype=np.float16)
            out_conds = {}
            for ci, name in enumerate(CONDITIONS):
                pids = prefixes.get(name)
                if pids is None:
                    out_conds[name] = {"skipped": True}
                    continue
                ids = torch.tensor([prompt_ids + list(pids)], device="cuda:0")
                m2 = dict(mi); m2["input_ids"] = ids; m2["attention_mask"] = torch.ones_like(ids)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                with torch.no_grad():
                    g = llm.generate(**m2, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=gen_conf["temperature"], top_k=gen_conf["top_k"],
                                     top_p=gen_conf["top_p"], return_dict_in_generate=True, output_logits=True)
                new = g.sequences[0][ids.shape[1]:].tolist()
                first_is_action = bool(new) and new[0] >= A0
                act = []
                for t in new:
                    if t >= A0 and len(act) < N_ACT:
                        act.append(t)
                    elif act:
                        break
                for s_ in range(min(N_ACT, len(g.logits))):
                    logits_store[ci, s_] = g.logits[s_][0, A0:].float().cpu().numpy().astype(np.float16)
                step0 = g.logits[0][0].float()
                lp0 = torch.log_softmax(step0, -1)
                top_v, top_i = lp0.topk(5)
                traj = None
                if len(act) == N_ACT:
                    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor(act))
                    traj = np.asarray(tr[0, 1:], np.float32)[:, :2].tolist()
                out_conds[name] = {
                    "skipped": False, "prefix_len": len(pids),
                    "declared": PL.autovla_decision(PL.autovla_sections(tok.decode(pids))
                                                    .get("best driving action", "")) if name != "natural_nocot" else [None, None],
                    "generated_ids": new, "first_is_action": first_is_action,
                    "action_idx": [t - A0 for t in act], "n_action": len(act),
                    "step0_top5_ids": top_i.tolist(), "step0_top5_logprob": [round(float(x), 4) for x in top_v],
                    "trajectory_pred": traj,
                    "prefix_text": tok.decode(pids) if name not in ("natural_nocot", "original") else None,
                }
            np.savez_compressed(os.path.join(args.output, "logits", f"{token}.npz"),
                                logits=logits_store, conditions=np.array(CONDITIONS))
            rec = {"token": token, "group": group, "log_name": r["log_name"], "velocity": r["velocity"],
                   "instruction": instr, "gt_action_instruction": r["gt_action_instruction"],
                   "gt_target": [gt_beh, gt_spd], "cf_target": [cf_beh, cf_spd],
                   "original_declared": list(decl) if decl else None,
                   "gt_action_idx": r["gt_action_idx"], "trajectory_gt": [p[:2] for p in r["trajectory_gt"]],
                   "stored_armC_action_idx": r["arms"]["C"]["pred_action_idx"][:N_ACT],
                   "stored_armN_action_idx": r["arms"]["N"]["pred_action_idx"][:N_ACT],
                   "stored_armC_traj": r["arms"]["C"]["trajectory_pred"],
                   "stored_armN_traj": r["arms"]["N"]["trajectory_pred"],
                   "conditions": out_conds}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n_ok += 1

            if args.sanity:
                st = np.load(os.path.join(args.tensors, f"{token}_C.npz"))
                ref = st["action_logits"][-1, 0].astype(np.float32)
                ours = logits_store[CONDITIONS.index("original"), 0].astype(np.float32)
                oc, on = out_conds["original"], out_conds["natural_nocot"]
                print(f"  [{token} {group}] stored-vs-rerun step0 action-logit maxdiff={np.abs(ref-ours).max():.3f} "
                      f"argmax {int(ref.argmax())}/{int(ours.argmax())} | armC reproduce {oc['action_idx']==rec['stored_armC_action_idx']} "
                      f"| armN reproduce {on['action_idx']==rec['stored_armN_action_idx']}", flush=True)
                for name in CONDITIONS:
                    c = out_conds[name]
                    if c.get("skipped"):
                        print(f"     {name:20s} SKIPPED"); continue
                    print(f"     {name:20s} len={c['prefix_len']:4d} decl={c['declared']} first={c['action_idx'][:3]} "
                          f"n_act={c['n_action']} first_is_action={c['first_is_action']}")
                if i == 1:
                    print("---- corrected_full tail ----\n" + out_conds["corrected_full"]["prefix_text"][-700:])
                    print("---- counter_decision tail ----\n" + out_conds["counter_decision"]["prefix_text"][-500:]
                          if out_conds["counter_decision"].get("prefix_text") else "counter_decision skipped")
        except Exception as exc:
            n_fail += 1
            import traceback; traceback.print_exc()
            print(f"[fail] {token}: {exc}", flush=True)
        if i % 10 == 0 or args.sanity:
            el = time.time() - t0
            print(f"[prog] {i}/{len(picks)} ok={n_ok} fail={n_fail} {el/i:.1f}s/scene eta {(len(picks)-i)*el/i/60:.1f}min", flush=True)
    fout.close()
    json.dump({"n_ok": n_ok, "n_fail": n_fail, "conditions": CONDITIONS, "gen_conf": gen_conf,
               "seed": args.seed, "control_n": args.control_n,
               "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "elapsed_min": (time.time() - t0) / 60}, open(os.path.join(args.output, "run_meta.json"), "w"), indent=2)
    print(f"[done] ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
