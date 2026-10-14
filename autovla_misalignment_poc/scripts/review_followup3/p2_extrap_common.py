#!/usr/bin/env python
"""
P2-B (review_followup3, 2026-10-14): shared helpers for the learned non-AR motion extrapolator.  numpy (+ torch only in
p2_extrap_train.py).  Spec: outputs/review_followup3/P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md.

GT -> token projection = numpy port of AutoVLA navsim/navsim/agents/autovla_agent.py TokenProcessor._match_agent_token
(num_k = 1, i.e. the training-target path TrajectoryTargetBuilder): vehicle box width 2.0 m, length 4.8 m; at step i the
token whose last-sub-step contour (codebook cb[:, -1], placed at the previous MATCHED pose) has the smallest summed
corner distance to the GT contour is chosen; the next start pose is the matched contour's mean / heading
atan2(corner0 - corner3) -- identical to the decoder pose (ActionTokenizer.rollout / p1_common.rollout_np).

Frames: decoder frame = ego frame at plan time (x forward, y left).  Pose index k = 0..9 = pose after token k.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "review_followup", "p1ab"))
import p1_common as C  # noqa: E402

N_ACT = 10
WIDTH, LENGTH = 2.0, 4.8
CMD_NAMES = ["turn left", "keep forward", "turn right"]          # driving_command one-hot index 0, 1, 2 (verified)


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def contour(pos, head):
    """AutoVLA cal_polygon_contour: (N,2),(N,) -> (N,4,2) corners [left_front, right_front, right_back, left_back]."""
    x, y = pos[:, 0], pos[:, 1]
    hc, hs = 0.5 * np.cos(head), 0.5 * np.sin(head)
    lc, ls, wc, ws = LENGTH * hc, LENGTH * hs, WIDTH * hc, WIDTH * hs
    lf = np.stack([x + lc - ws, y + ls + wc], -1)
    rf = np.stack([x + lc + ws, y + ls - wc], -1)
    rb = np.stack([x - lc + ws, y - ls - wc], -1)
    lb = np.stack([x - lc - ws, y - ls + wc], -1)
    return np.stack([lf, rf, rb, lb], 1)


def tokenize_gt(poses: np.ndarray, cb: np.ndarray, chunk: int = 1024):
    """poses (N, 10, 3) GT (x, y, heading) in the plan frame -> tokens (N, 10) int, decoded poses (N, 10, 3)."""
    tok = cb[:, -1].astype(np.float32)                       # (2048, 4, 2) local contours of the last sub-step
    N = len(poses)
    out_t = np.zeros((N, N_ACT), np.int64)
    out_p = np.zeros((N, N_ACT, 3))
    for s in range(0, N, chunk):
        P = poses[s:s + chunk]
        n = len(P)
        pos = np.zeros((n, 2)); head = np.zeros(n)
        for i in range(N_ACT):
            c, sn = np.cos(head), np.sin(head)
            R = np.stack([np.stack([c, sn], -1), np.stack([-sn, c], -1)], 1)          # row-vector rot (transform_to_global)
            world = np.einsum("tkd,nde->ntke", tok, R.astype(np.float32)) + pos[:, None, None, :].astype(np.float32)
            gt = contour(P[:, i, :2], P[:, i, 2]).astype(np.float32)                  # (n,4,2)
            dist = np.linalg.norm(world - gt[:, None], axis=-1).sum(-1)               # (n, 2048)
            idx = dist.argmin(1)
            m = world[np.arange(n), idx].astype(np.float64)                          # (n,4,2)
            pos = m.mean(1)
            d = m[:, 0] - m[:, 3]
            head = np.arctan2(d[:, 1], d[:, 0])
            out_t[s:s + n, i] = idx
            out_p[s:s + n, i] = np.c_[pos, head]
    return out_t, out_p


# ------------------------------------------------------------------------------------------------ SE(2) helpers (batched)
def to_frame(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """poses P (..., K, 3) expressed in the frame of pose ref (..., 3)."""
    c, s = np.cos(ref[..., 2])[..., None], np.sin(ref[..., 2])[..., None]
    dx, dy = P[..., 0] - ref[..., None, 0], P[..., 1] - ref[..., None, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(P[..., 2] - ref[..., None, 2])], -1)


def from_frame(Q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    c, s = np.cos(ref[..., 2])[..., None], np.sin(ref[..., 2])[..., None]
    return np.stack([ref[..., None, 0] + c * Q[..., 0] - s * Q[..., 1], ref[..., None, 1] + s * Q[..., 0] + c * Q[..., 1],
                     wrap(Q[..., 2] + ref[..., None, 2])], -1)


def body_incs(P: np.ndarray) -> np.ndarray:
    """(N, K, 3) poses after steps 0..K-1 (start = origin) -> body increments (N, K, 3)."""
    prev = np.concatenate([np.zeros_like(P[:, :1]), P[:, :-1]], 1)
    c, s = np.cos(prev[..., 2]), np.sin(prev[..., 2])
    dx, dy = P[..., 0] - prev[..., 0], P[..., 1] - prev[..., 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(P[..., 2] - prev[..., 2])], -1)


# ------------------------------------------------------------------------------------------------ features (fixed spec)
POS_SCALE = 10.0       # m -> network units
N_SLOT = N_ACT         # prefix slots k = -1 (origin), 0..8  -> 10 slots
OUT_SLOTS = N_ACT - 1  # suffix slots m = 0..8  (pose j+1+m)


def prefix_features(Ppre: np.ndarray, j: np.ndarray, ego: np.ndarray | None = None) -> np.ndarray:
    """Ppre (N, 10, 3): decoded plan poses (only k <= j are read; the rest is ignored and may be garbage);
    j (N,) int in 0..8; ego (N, 5) optional [speed m/s, |acc| m/s^2, cmd_left, cmd_straight, cmd_right].
    Returns float32 (N, D):  [Q slots (10x3), body increments (10x3), slot mask (10), one-hot j (9)] (+ ego (5)).
      Q slot 0 = origin, slot k+1 = pose k, all in the frame of pose j;  increments: slot k = increment of step k."""
    N = len(Ppre)
    ar = np.arange(N_ACT)
    mask = (ar[None, :] <= j[:, None])                                       # pose k valid iff k <= j
    P = np.where(mask[..., None], Ppre, 0.0)
    ref = P[np.arange(N), j]                                                 # pose j
    full = np.concatenate([np.zeros((N, 1, 3)), P], 1)[:, :N_SLOT]           # origin + poses 0..8
    Q = to_frame(full, ref)
    m_slot = np.concatenate([np.ones((N, 1), bool), mask[:, :N_SLOT - 1]], 1)
    Q = np.where(m_slot[..., None], Q, 0.0)
    inc = body_incs(P)                                                       # step k increment
    inc = np.where(mask[..., None], inc, 0.0)
    Q = Q.copy(); Q[..., :2] /= POS_SCALE
    inc = inc.copy(); inc[..., :2] /= POS_SCALE
    oh = np.zeros((N, N_ACT - 1)); oh[np.arange(N), j] = 1.0
    parts = [Q.reshape(N, -1), inc.reshape(N, -1), mask.astype(float), oh]
    if ego is not None:
        parts.append(ego)
    return np.concatenate(parts, 1).astype(np.float32)


def targets(Pfull: np.ndarray, j: np.ndarray):
    """suffix poses j+1..9 in the frame of pose j -> (N, 9, 3) scaled, mask (N, 9)."""
    N = len(Pfull)
    ref = Pfull[np.arange(N), j]
    idx = np.clip(j[:, None] + 1 + np.arange(OUT_SLOTS)[None], 0, N_ACT - 1)
    S = np.take_along_axis(Pfull, idx[..., None].repeat(3, -1), 1)
    Q = to_frame(S, ref)
    m = (j[:, None] + 1 + np.arange(OUT_SLOTS)[None]) <= N_ACT - 1
    Q = np.where(m[..., None], Q, 0.0); Q[..., :2] /= POS_SCALE
    return Q.astype(np.float32), m


def assemble(Ppre: np.ndarray, j: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """pred (N, 9, 3) scaled suffix in frame of pose j -> full plan poses (N, 10, 3) = prefix 0..j + predicted j+1..9."""
    N = len(Ppre)
    ref = Ppre[np.arange(N), j]
    Q = pred.astype(np.float64).copy(); Q[..., :2] *= POS_SCALE
    S = from_frame(Q, ref)
    out = Ppre.copy()
    for n in range(N):
        jj = int(j[n]); out[n, jj + 1:] = S[n, :N_ACT - 1 - jj]
    return out


def ego_vector(vel_xy, acc_xy, cmd_onehot) -> np.ndarray:
    """exactly what AutoVLA's prompt reads (models/autovla.py): speed = |v|, acc = |a| (magnitudes), command."""
    v = np.linalg.norm(np.asarray(vel_xy, float).reshape(-1, 2), axis=1)
    a = np.linalg.norm(np.asarray(acc_xy, float).reshape(-1, 2), axis=1)
    c = np.asarray(cmd_onehot, float).reshape(-1, 4)[:, :3]
    return np.c_[v / 10.0, a / 2.0, c].astype(np.float32)


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
