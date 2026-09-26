"""
Trajectory metrics and coarse action semantics.

Coordinate/format facts, verified in the ORION source (not assumed):
  * `ego_fut_trajs` (GT) and `ego_fut_preds` (pred) are per-step DELTAS, not absolute
    waypoints. `orion.py:932-937` cumsums BOTH before computing L2.
  * Frame is the current LiDAR frame: +y forward, +x lateral. Built in
    `b2d_orion_dataset.get_ego_trajs`.
  * 6 steps at `sample_interval=5` frames @10 Hz => 0.5 s per step, 3 s horizon.

Every threshold is a module-level constant and is mirrored into the run config, so the
taxonomy can be recomputed under different settings for sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DT = 0.5              # seconds per trajectory step
HORIZON = 6           # steps (3 s)

# --- action-semantics thresholds ------------------------------------------------------
STOP_TOTAL_DIST_M = 1.0    # total path length below this over 3 s => STOP
STOP_FINAL_SPEED = 0.5     # m/s, speed in the last step for a STOP
ACCEL_RATIO = 1.25         # v_end / v_start above this => ACCELERATE
DECEL_RATIO = 0.75         # v_end / v_start below this => DECELERATE
MIN_SPEED_FOR_RATIO = 0.5  # m/s; below this the ratio is meaningless, use absolute deltas
ACCEL_ABS = 0.8            # m/s speed change treated as significant when slow
LATERAL_TURN_M = 2.0       # |final lateral offset| above this => LEFT/RIGHT
LATERAL_HEADING_DEG = 15.0 # final heading change above this => LEFT/RIGHT

# --- action-correctness thresholds ----------------------------------------------------
ADE_THRESHOLD_M = 1.5      # default; swept in the sensitivity analysis
FDE_THRESHOLD_M = 3.0


def cumulative(traj_delta: np.ndarray) -> np.ndarray:
    """(T,2) per-step deltas -> (T,2) positions relative to the ego origin."""
    return np.cumsum(np.asarray(traj_delta, dtype=np.float64), axis=-2)


def ade(pred_delta, gt_delta, mask=None) -> float:
    """Average displacement error over cumsum'd positions (matches ORION's own L2)."""
    p, g = cumulative(pred_delta), cumulative(gt_delta)
    n = min(len(p), len(g))
    d = np.linalg.norm(p[:n] - g[:n], axis=-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)[:n]
        if not m.any():
            return float("nan")
        return float(d[m].mean())
    return float(d.mean())


def fde(pred_delta, gt_delta, mask=None) -> float:
    """Final displacement error at the last valid step."""
    p, g = cumulative(pred_delta), cumulative(gt_delta)
    n = min(len(p), len(g))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)[:n]
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return float("nan")
        last = idx[-1]
    else:
        last = n - 1
    return float(np.linalg.norm(p[last] - g[last]))


def l2_at(pred_delta, gt_delta, seconds: float) -> float:
    """Displacement error at a horizon in seconds (1s/2s/3s, as ORION reports)."""
    step = int(round(seconds / DT)) - 1
    p, g = cumulative(pred_delta), cumulative(gt_delta)
    if step < 0 or step >= min(len(p), len(g)):
        return float("nan")
    return float(np.linalg.norm(p[step] - g[step]))


def speed_profile(traj_delta) -> np.ndarray:
    """Per-step speed (m/s) from the deltas."""
    d = np.asarray(traj_delta, dtype=np.float64)
    return np.linalg.norm(d, axis=-1) / DT


@dataclass
class ActionSemantics:
    longitudinal: str        # STOP / DECELERATE / MAINTAIN / ACCELERATE
    lateral: str             # STRAIGHT / LEFT / RIGHT
    v_start: float
    v_end: float
    total_distance: float
    final_lateral: float
    heading_change_deg: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return self.longitudinal if self.lateral == "STRAIGHT" else f"{self.longitudinal}+{self.lateral}"


def action_semantics(traj_delta) -> ActionSemantics:
    """
    Coarse driving action implied by a trajectory.

    Longitudinal decision uses the speed at the start vs the end of the 3 s horizon; a
    ratio is used when the vehicle is actually moving and an absolute delta when it is
    nearly stationary (a ratio is unstable near zero speed).
    """
    d = np.asarray(traj_delta, dtype=np.float64).reshape(-1, 2)
    pos = cumulative(d)
    v = speed_profile(d)
    total = float(np.linalg.norm(d, axis=-1).sum())

    # Average the first/last two steps to reduce single-step noise.
    v_start = float(v[:2].mean()) if len(v) >= 2 else float(v[0])
    v_end = float(v[-2:].mean()) if len(v) >= 2 else float(v[-1])

    if total < STOP_TOTAL_DIST_M and v_end < STOP_FINAL_SPEED:
        lon = "STOP"
    elif v_start < MIN_SPEED_FOR_RATIO:
        # Starting from ~rest: only an absolute increase counts as accelerating.
        lon = "ACCELERATE" if (v_end - v_start) > ACCEL_ABS else "MAINTAIN"
    else:
        r = v_end / v_start
        if r >= ACCEL_RATIO:
            lon = "ACCELERATE"
        elif r <= DECEL_RATIO:
            lon = "DECELERATE" if v_end >= STOP_FINAL_SPEED else "STOP"
        else:
            lon = "MAINTAIN"

    final_lat = float(pos[-1, 0])
    # Heading of the final step relative to straight-ahead (+y).
    last = d[-1]
    heading = float(np.degrees(np.arctan2(last[0], max(last[1], 1e-6)))) if np.linalg.norm(last) > 1e-3 else 0.0

    if abs(final_lat) > LATERAL_TURN_M or abs(heading) > LATERAL_HEADING_DEG:
        lat = "RIGHT" if final_lat > 0 else "LEFT"
    else:
        lat = "STRAIGHT"

    return ActionSemantics(lon, lat, v_start, v_end, total, final_lat, heading)


# --------------------------------------------------------------------------------------
# Safety-conditioned decisions
# --------------------------------------------------------------------------------------

SAFE_LON = {"STOP", "DECELERATE"}


def safety_requirement(perception: dict) -> str | None:
    """
    The longitudinal behaviour the scene demands, from GT perception alone.

    Returns None when the scene imposes no hard requirement (the common case) so that
    safety-conditioned accuracy is only measured where it is actually defined.
    """
    if perception.get("traffic_light") == "red":
        return "STOP_OR_DECEL"
    if perception.get("pedestrian"):
        d = perception.get("ped_distance")
        if d is not None and d <= 25.0:
            return "STOP_OR_DECEL"
    if perception.get("lead_vehicle"):
        d = perception.get("lead_distance")
        if d is not None and d <= 15.0:
            return "STOP_OR_DECEL"
    return None


def satisfies_safety(requirement: str | None, sem: ActionSemantics) -> bool | None:
    """None when no requirement applies."""
    if requirement is None:
        return None
    if requirement == "STOP_OR_DECEL":
        return sem.longitudinal in SAFE_LON
    return None


# --------------------------------------------------------------------------------------
# Action correctness (A label)
# --------------------------------------------------------------------------------------

@dataclass
class ActionVerdict:
    correct: bool
    ade: float
    fde: float
    reason: str
    sem_pred: str
    sem_gt: str
    safety_required: str | None
    safety_ok_pred: bool | None
    safety_ok_gt: bool | None


def action_correct(pred_delta, gt_delta, perception_gt: dict, mask=None,
                   ade_thr: float = ADE_THRESHOLD_M,
                   fde_thr: float = FDE_THRESHOLD_M) -> ActionVerdict:
    """
    Decide A+/A- from geometry AND semantics.

    A- if ANY of:
      * ADE above threshold, or FDE above threshold  (geometric deviation)
      * the coarse longitudinal action disagrees with the GT trajectory's
      * the scene imposes a safety requirement that the GT trajectory satisfies but the
        prediction does not  (an unsafe action even when L2 happens to look acceptable)

    The last clause is what stops a purely-L2 definition from calling a
    "drove through a red light by 1.4 m" case correct.
    """
    a = ade(pred_delta, gt_delta, mask)
    f = fde(pred_delta, gt_delta, mask)
    sp = action_semantics(pred_delta)
    sg = action_semantics(gt_delta)

    req = safety_requirement(perception_gt or {})
    ok_p = satisfies_safety(req, sp)
    ok_g = satisfies_safety(req, sg)

    reasons = []
    if not np.isnan(a) and a > ade_thr:
        reasons.append(f"ADE>{ade_thr}")
    if not np.isnan(f) and f > fde_thr:
        reasons.append(f"FDE>{fde_thr}")
    if sp.longitudinal != sg.longitudinal:
        reasons.append(f"lon:{sg.longitudinal}->{sp.longitudinal}")
    if req is not None and ok_g and not ok_p:
        reasons.append(f"unsafe:{req}")

    return ActionVerdict(
        correct=not reasons, ade=a, fde=f, reason=",".join(reasons) or "ok",
        sem_pred=sp.label, sem_gt=sg.label,
        safety_required=req, safety_ok_pred=ok_p, safety_ok_gt=ok_g,
    )
