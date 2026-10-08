#!/usr/bin/env python
"""
Shared helpers for the P1-A / P1-B review-follow-up runs (AutoVLA, natural/fast path).

Everything model-related is copied from the existing harness (action_history_causal.py,
temporal_feedback_window.py) without behavioural changes:
  - checkpoint loaded with torch.load(mmap=True), model in bf16 (AutoVLA default)
  - prefix = prompt + stored fast stub (full_extract arm N) + model tokens before t*
  - prefix KV computed once per scene and expanded to B rows; at every step the post-t* span
    is recomputed from scratch and the cache is cropped back to the prefix (no stale K/V)
  - logits restricted to the 2,048 action rows, sampling at T = cfg temperature (0.01) with a
    torch.Generator seeded from sha256(key) and re-seeded for every row (common random numbers)
  - decoded with model.action_tokenizer.decode_token_ids_to_trajectory (the real decoder)
No existing file is modified; this module only imports from the existing scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AUTOVLA_DIR = "/root/VLA/autovla"
POC_DIR = "/root/VLA/autovla_misalignment_poc"
SCRIPTS = os.path.join(POC_DIR, "scripts")
OUT_ROOT = os.path.join(POC_DIR, "outputs/review_followup")
ED_RECORDS = os.path.join(POC_DIR, "outputs/equal_distance_perturbation/records.jsonl")
FULL_RECORDS = os.path.join(POC_DIR, "outputs/full_extract/records.jsonl")
AH_RECORDS = os.path.join(POC_DIR, "outputs/action_history_causal/records.jsonl")
SCENES = os.path.join(AUTOVLA_DIR, "dataset/nuplan/navtest_poc")
CONFIG = os.path.join(AUTOVLA_DIR, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml")
CHECKPOINT = "/root/VLA/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt"
CODEBOOK = os.path.join(AUTOVLA_DIR, "codebook_cache/agent_vocab.pkl")
N_ACT = 10
N_TOK = 2048
SECTORS = ["faster", "faster-left", "left", "slower-left", "slower", "slower-right", "right", "faster-right"]

for p in (AUTOVLA_DIR, os.path.join(AUTOVLA_DIR, "navsim"), POC_DIR, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------------------------- environment
def gpu_guard() -> None:
    """Only the RTX 5090 (PCI bus order index 1)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("launch with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (RTX 5090 only)")


def assert_5090(torch) -> str:
    name = torch.cuda.get_device_name(0)
    if "5090" not in name:
        raise SystemExit(f"expected RTX 5090, got {name}")
    return name


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def env_record(extra: dict | None = None) -> dict:
    import torch
    import transformers
    rec = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "repo_head_VLA": git_head("/root/VLA"), "repo_head_autovla": git_head(AUTOVLA_DIR),
        "checkpoint": CHECKPOINT, "checkpoint_size": os.path.getsize(CHECKPOINT),
        "codebook": CODEBOOK, "codebook_sha256": sha256_file(CODEBOOK),
        "config": CONFIG, "config_sha256": sha256_file(CONFIG),
        "ed_records_sha256": sha256_file(ED_RECORDS),
        "precision": "bf16 model weights (AutoVLA default, autovla/models/autovla.py torch_dtype=bfloat16); "
                     "logits cast to float64 before softmax/sampling",
        "kv_cache": "prefix KV once per scene, expanded to B rows; post-t* span recomputed each step; cache.crop(P)",
    }
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------------------------- data
def load_ed(limit_scenes: int = 0, tokens: list[str] | None = None) -> list[dict]:
    ed = [json.loads(l) for l in open(ED_RECORDS)]
    ed.sort(key=lambda r: (r["group"] != "A-", r["token"]))           # same order as existing harness
    if tokens:
        keep = set(tokens)
        ed = [r for r in ed if r["token"] in keep]
    if limit_scenes:
        ed = ed[:limit_scenes]
    return ed


def load_stubs_and_token_freq(want: set[str]):
    """fast stub per scene + token frequencies over ALL valid natural rollouts (diagnostic only)."""
    stub_of, f_pred, f_gt = {}, np.zeros(N_TOK, np.int64), np.zeros(N_TOK, np.int64)
    for line in open(FULL_RECORDS):
        x = json.loads(line)
        n = x["arms"]["N"]
        if x["token"] in want:
            stub_of[x["token"]] = n["token_ids"]
        if n.get("cot_present") or n.get("runaway_action_tokens") or len(n.get("trajectory_pred") or []) < N_ACT:
            continue
        for a in n["pred_action_idx"][:N_ACT]:
            f_pred[a] += 1
        for a in x["gt_action_idx"][:N_ACT]:
            f_gt[a] += 1
    return stub_of, f_pred, f_gt


# ---------------------------------------------------------------------------------------------- codebook geometry
def codebook_geometry():
    """disp[a] = (forward, left) end displacement of token a in the frame of its own start (m / 0.5 s);
    dyaw[a] = heading change of token a, computed with the decoder's own formula
    (atan2 of corner0 - corner3 of the last sub-step box, action_tokenizer.rollout)."""
    cb = np.asarray(pickle.load(open(CODEBOOK, "rb"))["token_all"]["veh"], np.float64)   # (2048, 6, 4, 2)
    disp = cb[:, -1].mean(1)
    dv = cb[:, -1, 0] - cb[:, -1, 3]
    dyaw = np.arctan2(dv[:, 1], dv[:, 0])
    return cb, disp, dyaw


def sector_of(v) -> int:
    ang = math.atan2(float(v[1]), float(v[0]))
    return int(((ang + math.pi / 8) % (2 * math.pi)) // (math.pi / 4)) % 8


def rollout_np(cb: np.ndarray, acts) -> np.ndarray:
    """numpy re-implementation of ActionTokenizer.rollout (for verification only; poses 1..n, (x, y, heading))."""
    pos = np.zeros(2); head = 0.0; out = []
    for a in acts:
        c, s = math.cos(head), math.sin(head)
        R = np.array([[c, s], [-s, c]])                     # same row-vector convention: p_local @ R
        g = cb[a].reshape(-1, 2) @ R + pos
        g = g.reshape(cb[a].shape)
        pos = g[-1].mean(0)
        d = g[-1, 0] - g[-1, 3]
        head = math.atan2(d[1], d[0])
        out.append([pos[0], pos[1], head])
    return np.asarray(out)


def select_gtfree_perturbations(disp, cb, ref_tok: int, d_target: float, tol: float):
    """GT-free perturbation set around the REFERENCE token (no GT used): per 45-degree sector of
    (disp[p] - disp[ref]) the token with achieved distance closest to d_target, among tokens with
    |achieved - d_target| <= tol. Ties -> smaller angular distance to sector centre -> smaller id.
    Returns {sector_index: info}; sectors with no candidate are absent (common-support rule)."""
    v = disp - disp[ref_tok]
    dd = np.linalg.norm(v, axis=1)
    out = {}
    for a in np.where(np.abs(dd - d_target) <= tol)[0]:
        a = int(a)
        if a == ref_tok:
            continue
        s = sector_of(v[a])
        ang = math.atan2(v[a, 1], v[a, 0])
        centre = s * math.pi / 4
        ang_err = abs((ang - centre + math.pi) % (2 * math.pi) - math.pi)
        key = (abs(dd[a] - d_target), ang_err, a)
        if s not in out or key < out[s]["_key"]:
            out[s] = {"_key": key, "token": a, "sector": SECTORS[s], "achieved_m": float(dd[a]),
                      "direction_deg": float(math.degrees(ang)), "angle_err_to_sector_centre_deg": float(math.degrees(ang_err)),
                      "quantization_residual_m": float(abs(dd[a] - d_target)),
                      "vec_forward": float(v[a, 0]), "vec_left": float(v[a, 1]),
                      "shape_d_ref": float(np.linalg.norm(cb[a] - cb[ref_tok], axis=-1).mean())}
    for s in out:
        out[s].pop("_key")
    return out


# ---------------------------------------------------------------------------------------------- model
def load_model():
    """identical to the existing harness: returns (torch, model, llm, A0, temp, agent, vis_box, cfg)."""
    import torch
    import yaml
    os.chdir(AUTOVLA_DIR)
    from models.autovla import AutoVLA
    from navsim.agents.autovla_agent import AutoVLAAgent
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cfg = yaml.safe_load(open(CONFIG))
    cfg["model"]["pretrained_model_path"] = os.path.join(AUTOVLA_DIR, "Qwen2.5-VL-3B-Instruct")
    cfg["model"]["codebook_cache_path"] = CODEBOOK
    temp = float(cfg["inference"]["sample"]["temperature"])
    model = AutoVLA(cfg, inference=True, device="cuda:0")
    sd = torch.load(CHECKPOINT, map_location="cpu", mmap=True)["state_dict"]
    model.load_state_dict({k[len("autovla."):]: v for k, v in sd.items() if k.startswith("autovla.")}, strict=False)
    model.to("cuda:0").eval()
    del sd
    llm, A0 = model.vlm, model.action_start_id

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
    return torch, model, llm, A0, temp, agent, vis_box, cfg


def build_prefix_cache(torch, model, llm, agent, vis_box, A0, token, stub_ids, prefix_actions, B):
    """prefix = prompt + stub + prefix_actions; returns cache (B rows), P, checksum, logits at the
    prefix's last position (action rows, float64, row 0) = the distribution of the token at t*."""
    scene = json.load(open(os.path.join(SCENES, f"{token}.json")))
    vis_box.clear()
    feats = {}
    for b in agent.get_feature_builders():
        feats.update(b.compute_features(scene))
    mi = {k: v.to("cuda:0") for k, v in model.get_prompt(feats).items() if isinstance(v, torch.Tensor)}
    stub = stub_ids[:next(j for j, x in enumerate(stub_ids) if x >= A0)]
    prefix = mi["input_ids"][0].tolist() + stub + [A0 + a for a in prefix_actions]
    P = len(prefix)
    ids = torch.tensor([prefix], device="cuda:0")
    llm.rope_deltas = None
    with torch.no_grad():
        out = llm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=mi.get("pixel_values_videos"),
                  video_grid_thw=mi.get("video_grid_thw"), use_cache=True, cache_position=torch.arange(P, device="cuda:0"))
    cache = out.past_key_values
    lg_prefix = out.logits[0, -1, A0:A0 + N_TOK].double().cpu()
    del out
    if B > 1:
        for li in range(len(cache.key_cache)):
            cache.key_cache[li] = cache.key_cache[li].expand(B, -1, -1, -1).contiguous()
            cache.value_cache[li] = cache.value_cache[li].expand(B, -1, -1, -1).contiguous()
    checksum = float(cache.key_cache[0][0, :, :P].float().abs().sum())
    return cache, P, checksum, lg_prefix


def check_cache(cache, P, checksum):
    chk = float(cache.key_cache[0][0, :, :P].float().abs().sum())
    ok_sum = abs(chk - checksum) < 1e-3 * max(1.0, checksum)
    ok_len = cache.get_seq_length() == P
    assert ok_sum, "prefix cache was modified"
    assert ok_len, "cache not cropped back to the prefix"
    return {"prefix_checksum_ok": ok_sum, "cache_len_eq_P": ok_len, "checksum": chk}


def forward_spans(torch, llm, cache, P, spans_abs):
    """one batched forward over equal-length spans (absolute token ids); returns float64 logits over
    action rows (B, 2048) and the full-vocab action mass; crops the cache back to P."""
    B = len(spans_abs)
    m = len(spans_abs[0])
    assert all(len(s) == m for s in spans_abs), "span lengths must match across rows"
    sids = torch.tensor(spans_abs, device="cuda:0")
    with torch.no_grad():
        o = llm(input_ids=sids, attention_mask=torch.ones((B, P + m), device="cuda:0", dtype=torch.long),
                past_key_values=cache, use_cache=True, cache_position=torch.arange(P, P + m, device="cuda:0"))
    cache.crop(P)
    full = o.logits[:, -1].double()
    del o
    return full


def seed_int(seed: int, token: str, key: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{token}:{key}:{k}".encode()).digest()[:4], "little")


def sample_rows(torch, lg_act, temp, sid):
    """identical to the existing harness: every row re-seeds the same generator (common random numbers)."""
    probs = torch.softmax(lg_act / temp, -1).cpu()
    gen = torch.Generator()
    picks = []
    for i in range(probs.shape[0]):
        gen.manual_seed(sid)
        picks.append(int(torch.multinomial(probs[i], 1, generator=gen)))
    return picks


def decode(torch, model, A0, acts):
    """real decoder; returns poses after each token (n, 3) = x forward, y left, heading (rad), plus origin."""
    tr = model.action_tokenizer.decode_token_ids_to_trajectory(torch.tensor([A0 + a for a in acts]))
    arr = np.asarray(tr[0], np.float64)
    return arr[1:], arr[0]


def jdump(obj) -> str:
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
