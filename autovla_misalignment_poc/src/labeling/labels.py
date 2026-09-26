"""
P (perception) and A (action) labels for the AutoVLA PoC.

Design goal: keep the OPERATIONAL DEFINITIONS as close to the ORION PoC as the two
datasets allow, so the two numbers can sit in one table.

P — perception
    Taken from navsim ground-truth `Annotations` (boxes / names / velocity_3d), NOT from
    model text and NOT from a hidden probe. Same five variables as ORION where nuPlan
    supports them:
        lead_vehicle, critical_motion, critical_side, pedestrian
    nuPlan/OpenScene annotations carry no traffic-light state, so that ORION variable has
    no counterpart here and is simply absent (recorded as None) rather than faked.

    Box layout is navsim's: (x, y, z, length, width, height, yaw) in the ego frame,
    x forward / y left  -- note this differs from ORION/Chat-B2D where x was lateral.
    We convert to the ORION convention (lat = +right, lon = +forward) so the side/motion
    definitions and thresholds carry over unchanged.

A — action
    Coarse semantics of the trajectory, using AutoVLA's OWN official parser
    (`get_action_instruction`, dataset_utils/preprocessing/nuplan_dataset.py:255-289) so
    the label space matches what the model was trained to talk about, then mapped onto the
    ORION longitudinal/lateral taxonomy.

    A- is decided primarily by SEMANTIC DISAGREEMENT, not by raw L2: a prediction that
    stops in slightly the wrong place is still A+. Geometric error only forces A- when it
    is gross (see `action_correct`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# Thresholds. Kept identical to the ORION PoC wherever the quantity is the same, so the
# two studies' numbers are comparable. All are swept in the sensitivity analysis.
# --------------------------------------------------------------------------------------
LATERAL_CENTER_M = 2.5      # |lat| <= this counts as "in ego lane" / "center"
STATIC_SPEED_MPS = 0.5      # |speed| below this counts as static
LEAD_MAX_DIST_M = 40.0      # a leading vehicle must be within this range
PED_HAZARD_LAT_M = 6.0      # pedestrian hazard box, lateral half-width
PED_HAZARD_LON_M = 25.0     # ... and longitudinal range

DT = 0.5                    # s between trajectory poses
ORION_HORIZON = 6           # poses compared against ORION (3 s), matching PlanningMetric

# Gross-geometry backstop (NOT the primary A criterion).
ADE_GROSS_M = 4.0
FDE_GROSS_M = 8.0

VEHICLE_NAMES = {"vehicle"}
PEDESTRIAN_NAMES = {"pedestrian"}
# OpenScene lumps cyclists into "bicycle" when present; keep the set tolerant.
VRU_NAMES = PEDESTRIAN_NAMES | {"bicycle", "cyclist"}


# --------------------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------------------

@dataclass
class PerceptionLabels:
    lead_vehicle: bool | None = None
    lead_distance: float | None = None
    pedestrian: bool | None = None
    ped_distance: float | None = None
    critical_side: str | None = None      # left / center / right
    critical_motion: str | None = None    # static / moving
    traffic_light: None = None            # no counterpart in nuPlan annotations
    n_objects: int = 0
    n_vehicles: int = 0
    n_pedestrians: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _objects(anno: dict) -> list[dict]:
    """
    navsim boxes -> objects in the ORION convention.

    navsim ego frame: x forward, y left. ORION/Chat-B2D used x = lateral (+right),
    y = longitudinal (+forward). Convert so all downstream thresholds transfer.
    """
    boxes = np.asarray(anno.get("boxes", []), dtype=np.float64).reshape(-1, 7) \
        if anno.get("boxes") else np.zeros((0, 7))
    names = anno.get("names", []) or []
    vel = np.asarray(anno.get("velocity_3d", []), dtype=np.float64).reshape(-1, 3) \
        if anno.get("velocity_3d") else np.zeros((len(boxes), 3))

    out = []
    for i in range(len(boxes)):
        x_fwd, y_left = float(boxes[i, 0]), float(boxes[i, 1])
        speed = float(np.linalg.norm(vel[i, :2])) if i < len(vel) else 0.0
        out.append({
            "name": (names[i] if i < len(names) else "unknown"),
            "lat": -y_left,      # +right
            "lon": x_fwd,        # +forward
            "speed": speed,
        })
    return out


def build_perception_labels(anno: dict) -> tuple[PerceptionLabels, list[dict]]:
    objs = _objects(anno)
    p = PerceptionLabels()
    p.n_objects = len(objs)
    p.n_vehicles = sum(1 for o in objs if o["name"] in VEHICLE_NAMES)
    p.n_pedestrians = sum(1 for o in objs if o["name"] in PEDESTRIAN_NAMES)

    lead = [o for o in objs
            if o["name"] in VEHICLE_NAMES
            and abs(o["lat"]) <= LATERAL_CENTER_M and 0 < o["lon"] <= LEAD_MAX_DIST_M]
    lead = min(lead, key=lambda o: o["lon"]) if lead else None
    p.lead_vehicle = lead is not None
    p.lead_distance = lead["lon"] if lead else None

    ped = [o for o in objs
           if o["name"] in VRU_NAMES
           and abs(o["lat"]) <= PED_HAZARD_LAT_M and 0 < o["lon"] <= PED_HAZARD_LON_M]
    ped = min(ped, key=lambda o: o["lon"]) if ped else None
    p.pedestrian = ped is not None
    p.ped_distance = ped["lon"] if ped else None

    # The "critical object": nearest dynamic-capable agent ahead, as in ORION.
    dyn = [o for o in objs if o["name"] in (VEHICLE_NAMES | VRU_NAMES) and o["lon"] > 0]
    if dyn:
        key = min(dyn, key=lambda o: o["lon"])
        p.critical_side = ("center" if abs(key["lat"]) <= LATERAL_CENTER_M
                           else ("right" if key["lat"] > 0 else "left"))
        p.critical_motion = "static" if key["speed"] < STATIC_SPEED_MPS else "moving"
    return p, objs


# --------------------------------------------------------------------------------------
# Action
# --------------------------------------------------------------------------------------

#: AutoVLA's speed_meta strings -> the ORION longitudinal taxonomy.
_LON_MAP = {
    "stop": "STOP",
    "a deceleration to zero": "STOP",
    "a quick deceleration": "DECELERATE",
    "a deceleration": "DECELERATE",
    "a constant speed": "MAINTAIN",
    "an acceleration": "ACCELERATE",
    "a quick acceleration": "ACCELERATE",
}
_LAT_MAP = {
    "move forward": "STRAIGHT",
    "turn left": "LEFT",
    "change lane to left": "LEFT",
    "turn right": "RIGHT",
    "change lane to right": "RIGHT",
}


def parse_action_instruction(s: str) -> tuple[str | None, str | None]:
    """'move forward with a quick deceleration' -> ('DECELERATE', 'STRAIGHT')."""
    if not s:
        return None, None
    s = s.strip()
    if s == "STOP":                       # the parser short-circuits to bare "STOP"
        return "STOP", "STRAIGHT"
    if " with " in s:
        beh, speed = s.split(" with ", 1)
    else:
        beh, speed = s, ""
    return _LON_MAP.get(speed.strip()), _LAT_MAP.get(beh.strip())


@dataclass
class ActionSemantics:
    longitudinal: str | None
    lateral: str | None
    raw: str

    @property
    def label(self) -> str:
        lon = self.longitudinal or "?"
        lat = self.lateral or "?"
        return lon if lat == "STRAIGHT" else f"{lon}+{lat}"

    def to_dict(self) -> dict:
        return asdict(self)


def action_semantics(traj_xy, get_action_instruction) -> ActionSemantics:
    """
    Coarse action of a trajectory, via AutoVLA's own parser.

    `get_action_instruction` is injected so this module does not import the AutoVLA repo
    (keeps it importable for offline analysis); the runner passes the real function.
    """
    p = np.asarray(traj_xy, dtype=np.float64)[:, :2]
    v = np.diff(p, axis=0) / DT
    v = np.concatenate([v, v[-1:]], axis=0) if len(v) else np.zeros((1, 2))
    raw = get_action_instruction(p, v)
    lon, lat = parse_action_instruction(raw)
    return ActionSemantics(lon, lat, raw)


def ade(pred_xy, gt_xy, n: int = ORION_HORIZON) -> float:
    p = np.asarray(pred_xy, dtype=np.float64)[:n, :2]
    g = np.asarray(gt_xy, dtype=np.float64)[:n, :2]
    m = min(len(p), len(g))
    return float(np.linalg.norm(p[:m] - g[:m], axis=-1).mean()) if m else float("nan")


def fde(pred_xy, gt_xy, n: int = ORION_HORIZON) -> float:
    p = np.asarray(pred_xy, dtype=np.float64)[:n, :2]
    g = np.asarray(gt_xy, dtype=np.float64)[:n, :2]
    m = min(len(p), len(g))
    return float(np.linalg.norm(p[m - 1] - g[m - 1])) if m else float("nan")


@dataclass
class ActionVerdict:
    correct: bool
    ade: float
    fde: float
    reason: str
    sem_pred: str
    sem_gt: str
    lon_pred: str | None
    lon_gt: str | None
    lat_pred: str | None
    lat_gt: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def action_correct(sem_pred: ActionSemantics, sem_gt: ActionSemantics,
                   pred_xy, gt_xy,
                   ade_gross: float = ADE_GROSS_M,
                   fde_gross: float = FDE_GROSS_M) -> ActionVerdict:
    """
    A+/A- by SEMANTIC disagreement first, with a gross-geometry backstop.

    Per the spec: "GT와 조금 다르다는 이유만으로 A-로 하지 않는다" -- a correct STOP that
    stops a metre early stays A+. Geometry only intervenes when the trajectory is wildly
    off, which catches cases the coarse labels happen to agree on.
    """
    a, f = ade(pred_xy, gt_xy), fde(pred_xy, gt_xy)
    reasons = []
    if sem_pred.longitudinal and sem_gt.longitudinal and \
            sem_pred.longitudinal != sem_gt.longitudinal:
        reasons.append(f"lon:{sem_gt.longitudinal}->{sem_pred.longitudinal}")
    if sem_pred.lateral and sem_gt.lateral and sem_pred.lateral != sem_gt.lateral:
        reasons.append(f"lat:{sem_gt.lateral}->{sem_pred.lateral}")
    if np.isfinite(a) and a > ade_gross:
        reasons.append(f"ADE>{ade_gross}")
    if np.isfinite(f) and f > fde_gross:
        reasons.append(f"FDE>{fde_gross}")

    return ActionVerdict(
        correct=not reasons, ade=a, fde=f, reason=",".join(reasons) or "ok",
        sem_pred=sem_pred.label, sem_gt=sem_gt.label,
        lon_pred=sem_pred.longitudinal, lon_gt=sem_gt.longitudinal,
        lat_pred=sem_pred.lateral, lat_gt=sem_gt.lateral,
    )


# --------------------------------------------------------------------------------------
# R (observable reasoning) -- secondary in this study
# --------------------------------------------------------------------------------------

_COT_ABSENT = ("chain-of-thought is not needed", "a direct decision can be made")

#: The system prompt fixes the allowed final-action vocabulary
#: (models/autovla.py:653-654), so the declared intent is parseable.
_DECL_LON = [
    ("STOP", ("deceleration to zero", "stop")),
    ("DECELERATE", ("quick deceleration", "deceleration")),
    ("ACCELERATE", ("quick acceleration", "acceleration")),
    ("MAINTAIN", ("maintain constant speed", "constant speed")),
]
_DECL_LAT = [
    ("LEFT", ("change lane to left", "turn left")),
    ("RIGHT", ("change lane to right", "turn right")),
    ("STRAIGHT", ("move forward",)),
]


def parse_declared_action(text: str) -> tuple[str | None, str | None, bool]:
    """
    Read the model's declared Final Action Decision out of its CoT.

    Returns (longitudinal, lateral, cot_present). AutoVLA switches thinking modes, so a
    sample may legitimately carry no reasoning at all -- that is reported, not imputed.
    """
    if not text:
        return None, None, False
    low = text.lower()
    cot = not any(m in low for m in _COT_ABSENT)
    body = low.split("<answer>")[-1] if "<answer>" in low else low
    lon = next((n for n, keys in _DECL_LON if any(k in body for k in keys)), None)
    lat = next((n for n, keys in _DECL_LAT if any(k in body for k in keys)), None)
    return lon, lat, cot
