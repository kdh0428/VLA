#!/usr/bin/env python
"""
P2 (review_followup3, 2026-10-14): kinematic helpers for the motion-baseline audit.  Pure numpy, CPU only.

Frame convention (identical to AutoVLA ActionTokenizer.rollout / p1_common.rollout_np):
  decoder frame = ego frame at t = 0 (x forward, y left), pose index k = 0..9 is the pose AFTER token k (0.5 s each),
  pose "-1" = origin (0, 0, 0).  A token a is an SE(2) increment in the frame of its own start pose:
      x_k = x_{k-1} + Rot(h_{k-1}) @ disp[a],   h_k = wrap(h_{k-1} + dyaw[a])
  (rollout_np: pos_new = g[-1].mean(0) with g = cb[a] @ R + pos, R = [[c, s], [-s, c]] -> row-vector rotation by +h).

Body increment of segment k (any trajectory):  ell_k = Rot(-h_{k-1}) (x_k - x_{k-1}),  w_k = wrap(h_k - h_{k-1}).

Baselines (spec: outputs/review_followup3/P2/MOTION_BASELINE_AUDIT.md, section MOTION_BASELINE_SPEC).  Every baseline
reads ONLY the decoded executed prefix poses 0..t (prefix tokens r[:t] + the token at t, perturbed p or unperturbed r_t)
and the origin; never r[t+1:], the model output, or GT.
"""
from __future__ import annotations

import math

import numpy as np

N_ACT = 10
DT = 0.5                       # s per token / segment

# ---------------------------------------------------------------- fixed hyper-parameters (MOTION_BASELINE_SPEC)
CTRV_K = 3                     # CTRV-K3 fit window: last 3 segments (1.5 s), truncated at the origin
OMEGA_CLIP = 0.5               # rad per segment (= 1.0 rad/s) clip of fitted yaw increment (CTRV-K3, CTRA-2)
ACC_CLIP = 1.0                 # m/seg per segment (= 4.0 m/s^2) clip of fitted speed change (CTRA-2)
SPEED_MAX = 10.0               # m per segment (= 20 m/s) cap on extrapolated speed (CTRA-2)


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def rot(h: float) -> np.ndarray:
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s], [s, c]])          # column-vector rotation by +h


def compose(increments, start=(0.0, 0.0, 0.0)) -> np.ndarray:
    """chain SE(2) body increments [(fwd, left, dyaw), ...] from start pose -> poses (n, 3) (x, y, heading wrapped)."""
    x = np.array(start[:2], float); h = float(start[2]); out = []
    for f, l, w in increments:
        x = x + rot(h) @ np.array([f, l], float)
        h = float(wrap(h + w))
        out.append([x[0], x[1], h])
    return np.asarray(out).reshape(-1, 3)


def tokens_to_poses(disp, dyaw, acts) -> np.ndarray:
    return compose([(disp[a][0], disp[a][1], dyaw[a]) for a in acts])


def body_increments(poses: np.ndarray, start=(0.0, 0.0, 0.0)) -> np.ndarray:
    """inverse of compose: poses (n, 3) -> (n, 3) body increments (fwd, left, dyaw)."""
    prev = np.vstack([np.asarray(start, float)[None], poses[:-1]])
    out = []
    for p0, p1 in zip(prev, poses):
        ell = rot(-p0[2]) @ (p1[:2] - p0[:2])
        out.append([ell[0], ell[1], float(wrap(p1[2] - p0[2]))])
    return np.asarray(out)


# ---------------------------------------------------------------- baselines: prefix poses 0..t -> poses 0..9
def bl_cm(pre: np.ndarray, t: int) -> np.ndarray:
    """existing CM re-expressed: repeat the LAST body increment (ell_t, w_t) (= token at t repeated)."""
    inc = body_increments(pre[:t + 1])
    return np.vstack([pre[:t + 1], compose([tuple(inc[t])] * (N_ACT - 1 - t), pre[t])]) if t < N_ACT - 1 else pre.copy()


def bl_cv0(pre: np.ndarray, t: int) -> np.ndarray:
    """CV0: zero-turn global constant-velocity continuation.  v = x_t - x_{t-1} (global/decoder frame, x_{-1} = origin);
    x_k = x_t + (k - t) v; heading held at h_t (only used for the local-motion metric)."""
    xm1 = np.zeros(2) if t == 0 else pre[t - 1, :2]
    v = pre[t, :2] - xm1
    fut = [[pre[t, 0] + j * v[0], pre[t, 1] + j * v[1], pre[t, 2]] for j in range(1, N_ACT - t)]
    return np.vstack([pre[:t + 1], np.asarray(fut).reshape(-1, 3)])


def fit_ctrv(pre: np.ndarray, t: int, K: int = CTRV_K, omega_clip: float = OMEGA_CLIP):
    """mean body twist over the last min(K, t+1) segments; yaw increment clipped to +-omega_clip."""
    inc = body_increments(pre[:t + 1])
    win = inc[max(0, t + 1 - K):t + 1]
    f, l, w = win.mean(0)
    wc = float(np.clip(w, -omega_clip, omega_clip))
    return (float(f), float(l), wc), {"k_eff": len(win), "omega_raw": float(w), "omega_clipped": bool(wc != w)}


def bl_ctrv(pre: np.ndarray, t: int, K: int = CTRV_K, omega_clip: float = OMEGA_CLIP):
    twist, info = fit_ctrv(pre, t, K, omega_clip)
    return np.vstack([pre[:t + 1], compose([twist] * (N_ACT - 1 - t), pre[t])]), info


def bl_ctra(pre: np.ndarray, t: int, acc_clip: float = ACC_CLIP, omega_clip: float = OMEGA_CLIP, smax: float = SPEED_MAX):
    """CTRA-2: speed s_t = |ell_t|, acceleration a = s_t - s_{t-1} (per segment; a = 0 if t = 0), clipped to +-acc_clip;
    yaw increment w = w_t (latest, clipped); chord angle beta = atan2(ell_t) held.  s_k = clip(s_t + (k-t) a, 0, smax);
    once s_k = 0 the vehicle stops (no further translation or rotation)."""
    inc = body_increments(pre[:t + 1])
    s_t = float(np.hypot(*inc[t, :2]))
    a_raw = 0.0 if t == 0 else s_t - float(np.hypot(*inc[t - 1, :2]))
    a = float(np.clip(a_raw, -acc_clip, acc_clip))
    w = float(np.clip(inc[t, 2], -omega_clip, omega_clip))
    beta = math.atan2(inc[t, 1], inc[t, 0]) if s_t > 0 else 0.0
    incs, stopped = [], False
    for j in range(1, N_ACT - t):
        s = float(np.clip(s_t + j * a, 0.0, smax))
        if stopped or s == 0.0:
            stopped = True
            incs.append((0.0, 0.0, 0.0))
        else:
            incs.append((s * math.cos(beta), s * math.sin(beta), w))
    info = {"a_raw": a_raw, "acc_clipped": bool(a != a_raw), "omega_clipped": bool(w != inc[t, 2]), "stopped": stopped}
    return np.vstack([pre[:t + 1], compose(incs, pre[t]).reshape(-1, 3)]), info


def ctrv_continuous(x0, h0, v, omega, T):
    """closed-form continuous CTRV (zero lateral velocity) after time T; used only for the omega -> 0 unit test."""
    if abs(omega) < 1e-12:
        return np.array([x0[0] + v * T * math.cos(h0), x0[1] + v * T * math.sin(h0), h0])
    return np.array([x0[0] + v / omega * (math.sin(h0 + omega * T) - math.sin(h0)),
                     x0[1] - v / omega * (math.cos(h0 + omega * T) - math.cos(h0)), h0 + omega * T])
