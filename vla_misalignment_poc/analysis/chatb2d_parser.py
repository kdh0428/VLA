"""
Symmetric parser for Chat-B2D annotations AND ORION model answers.

Both sides use the *same* functions on purpose: ORION was trained on Chat-B2D, so its
answers follow the same surface form. Using one parser for GT and prediction avoids
introducing an asymmetric extraction bias that would masquerade as a perception error.

Chat-B2D val annotation format (verified on 12,806 val frames):
    [[{"from":"human","value":Q}, {"from":"gpt","value":A}], ...]

Answer forms this parser relies on (all verified against the real data):
  critical objects : "Car at <0.32, 10.82> directly in front of ego vehicle, stationary, ..."
                     "Traffic light at <2.49, 26.87> ... showing red, requiring a stop.."
  traffic light    : "Yes, there is a traffic light ... The color is red."  /  "No, there are no ..."
  behavior         : "lanefollow." | "turn left." | "go straight." | "changelaneleft." | ...
  speed change     : "Accelerating." | "Decelerating." | "No change."
  driving reason   : "You should accelerate and follow the lane. ..."
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------------------
# Question-round routing.
#
# Chat-B2D randomises question phrasing (12 scene templates, 5 critical-object templates,
# many traffic-light paraphrases), so rounds are routed by semantic signature, never by
# exact string match. Order matters: the "difference/change" variants must be rejected
# before the plain critical-object match, otherwise the temporal-diff round is mistaken
# for the current-frame round.
# --------------------------------------------------------------------------------------

_DIFF_MARKERS = ("differ", "distinction", "changes have occurred", "differences",
                 "compared to the previous", "past scene", "previous scene")
_PAST_MARKERS = ("past few frames", "recent frames", "last few frames", "previous frame",
                 "past frames")


def classify_question(q: str) -> str:
    """Map a question string to a canonical round name."""
    ql = q.lower().strip()

    is_diff = any(m in ql for m in _DIFF_MARKERS)
    is_past = any(m in ql for m in _PAST_MARKERS)

    # "critical objects" / "significant objects", plus the inverted paraphrase
    # "Which objects in the scene are critical, and what effects ...".
    if ("critical object" in ql or "significant object" in ql
            or ("object" in ql and "are critical" in ql)):
        return "critical_objects_diff" if is_diff else "critical_objects"

    if "traffic light" in ql or "traffic signal" in ql or "traffic control light" in ql:
        # "has the driving strategy been affected ... in the past few frames" is a history
        # round and must not be used as the current-frame traffic-light label.
        return "traffic_light_past" if (is_past or is_diff) else "traffic_light"

    if "speed" in ql and ("changed" in ql or "change" in ql):
        return "speed_change"

    if "driving behavior" in ql and ("previous frame" in ql or "was my" in ql):
        return "behavior_prev"

    if "current behavior of the vehicle" in ql:
        return "behavior_current"

    if "describe your driving behavior" in ql and "reason" in ql:
        return "driving_reason"

    if "planning trajectory" in ql:
        return "planning"

    # The 12 scene-description paraphrases.
    if any(k in ql for k in ("driving condition", "panoramic image", "driving scenario",
                             "weather condition", "overall condition", "overall environment",
                             "images show")):
        return "scene_description"

    return "other"


def rounds_from_annotation(anno: list) -> dict[str, str]:
    """Chat-B2D json -> {round_name: answer}. Later duplicates do not clobber earlier ones."""
    out: dict[str, str] = {}
    for pair in anno:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        q, a = pair[0].get("value", ""), pair[1].get("value", "")
        name = classify_question(q)
        out.setdefault(name, a)
    return out


# --------------------------------------------------------------------------------------
# Critical objects
# --------------------------------------------------------------------------------------

# "Car at <0.32, 10.82> ..." / "Traffic light at <2.49, 26.87> ..."
# Coordinates are ego/LiDAR frame: x = lateral (+right), y = longitudinal (+forward).
_OBJ_RE = re.compile(r"([A-Za-z][A-Za-z_ ]{0,24}?)\s+at\s*<\s*(-?\d+(?:\.\d+)?)\s*,\s*"
                     r"(-?\d+(?:\.\d+)?)\s*>")

# Colour adjectives that Chat-B2D prefixes onto vehicles ("blue car", "red car").
_COLOR_PREFIX = re.compile(r"^(blue|red|black|white|green|yellow|orange|silver|grey|gray|dark|light)\s+")

_CLASS_CANON = {
    "car": "car", "cars": "car", "vehicle": "car", "van": "car", "truck": "car",
    "bus": "car", "suv": "car", "taxi": "car", "police car": "car", "ambulance": "car",
    "firetruck": "car", "motorcycle": "bicycle", "bike": "bicycle",
    "bicycle": "bicycle", "cyclist": "bicycle", "rider": "bicycle",
    "pedestrian": "pedestrian", "pedestrians": "pedestrian", "person": "pedestrian",
    "walker": "pedestrian", "people": "pedestrian",
    "traffic light": "traffic_light", "traffic lights": "traffic_light",
    "trafficlight": "traffic_light",
    "traffic sign": "traffic_sign", "traffic signs": "traffic_sign",
    "stop sign": "traffic_sign", "speed limit sign": "traffic_sign",
    "traffic cone": "traffic_cone", "cone": "traffic_cone", "cones": "traffic_cone",
    "construction cone": "traffic_cone", "warning sign": "traffic_sign",
}


def canon_class(raw: str) -> str:
    c = raw.strip().lower()
    c = _COLOR_PREFIX.sub("", c)
    c = re.sub(r"\s+", " ", c).strip()
    return _CLASS_CANON.get(c, c)


@dataclass
class CriticalObject:
    cls: str
    x: float            # lateral, + = right of ego
    y: float            # longitudinal, + = ahead of ego
    text: str = ""      # the clause this object was parsed from
    speed: float | None = None
    state: str | None = None   # traffic-light colour, when the object is a light

    @property
    def side(self) -> str:
        """left / center / right using a 2.5 m half-lane band."""
        if abs(self.x) <= LATERAL_CENTER_M:
            return "center"
        return "right" if self.x > 0 else "left"

    @property
    def motion(self) -> str:
        """static / moving / unknown, from the clause wording and the stated speed."""
        t = (self.text or "").lower()
        if "stationary" in t or "parked" in t or "static" in t or "not moving" in t:
            return "static"
        if "crossing" in t:
            return "moving"
        if any(k in t for k in ("moving", "traveling", "travelling", "driving", "approaching",
                                "walking", "running", "accelerat", "slow speed", "m/s")):
            if self.speed is not None and abs(self.speed) < STATIC_SPEED_MPS:
                return "static"
            return "moving"
        if self.speed is not None:
            return "static" if abs(self.speed) < STATIC_SPEED_MPS else "moving"
        return "unknown"


# Thresholds — all label thresholds live here so they can be swept.
LATERAL_CENTER_M = 2.5      # |x| <= this counts as "in ego lane" / "center"
STATIC_SPEED_MPS = 0.5      # |speed| below this counts as static
LEAD_MAX_DIST_M = 40.0      # a leading vehicle must be within this range
LEAD_CLOSE_M = 15.0         # "close" leading vehicle, used for safety-conditioned checks
PED_HAZARD_LAT_M = 6.0      # pedestrian within this lateral band counts as a hazard
PED_HAZARD_LON_M = 25.0     # ... and within this longitudinal range

_SPEED_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*m/s")
_LIGHT_COLOR_RE = re.compile(r"\b(red|green|yellow|amber)\b")


def _split_clauses(answer: str) -> list[str]:
    """Chat-B2D separates objects with '..' or '. '. Keep clauses that name an object."""
    parts = re.split(r"\.\.|(?<=[a-z0-9\)\>])\.\s+", answer)
    return [p.strip() for p in parts if p.strip()]


def parse_critical_objects(answer: str) -> list[CriticalObject]:
    """Extract every '<class> at <x, y>' object with its clause-local attributes."""
    if not answer:
        return []
    objs: list[CriticalObject] = []
    for clause in _split_clauses(answer):
        for m in _OBJ_RE.finditer(clause):
            cls = canon_class(m.group(1))
            if not cls or cls in ("scene", "object", "ego", "it"):
                continue
            try:
                x, y = float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            sp = _SPEED_RE.search(clause)
            speed = float(sp.group(1)) if sp else None
            state = None
            if cls == "traffic_light":
                c = _LIGHT_COLOR_RE.search(clause.lower())
                if c:
                    state = "yellow" if c.group(1) == "amber" else c.group(1)
            objs.append(CriticalObject(cls=cls, x=x, y=y, text=clause,
                                       speed=speed, state=state))
    return objs


# --------------------------------------------------------------------------------------
# Traffic light
# --------------------------------------------------------------------------------------

def parse_traffic_light(tl_answer: str, objs: list[CriticalObject] | None = None) -> str:
    """
    Ego-relevant traffic-light state: 'none' | 'red' | 'green' | 'yellow' | 'unknown'.

    The dedicated traffic-light round is authoritative; the critical-object list is only
    consulted to recover a colour when the round says "Yes" without naming one.
    """
    if tl_answer:
        low = tl_answer.strip().lower()
        if low.startswith("no") or "no traffic light" in low or "there are no traffic" in low:
            return "none"
        if low.startswith("yes") or "there is a traffic light" in low:
            c = _LIGHT_COLOR_RE.search(low)
            if c:
                return "yellow" if c.group(1) == "amber" else c.group(1)
            for o in (objs or []):
                if o.cls == "traffic_light" and o.state:
                    return o.state
            return "unknown"
    # No dedicated round -> fall back to the critical-object list.
    for o in (objs or []):
        if o.cls == "traffic_light" and o.state:
            return o.state
    return "unknown"


# --------------------------------------------------------------------------------------
# Derived perception variables
# --------------------------------------------------------------------------------------

def leading_vehicle(objs: list[CriticalObject]) -> CriticalObject | None:
    """Nearest in-lane vehicle ahead of ego, within LEAD_MAX_DIST_M."""
    cands = [o for o in objs
             if o.cls == "car" and abs(o.x) <= LATERAL_CENTER_M and 0 < o.y <= LEAD_MAX_DIST_M]
    return min(cands, key=lambda o: o.y) if cands else None


def pedestrian_hazard(objs: list[CriticalObject]) -> CriticalObject | None:
    """Nearest pedestrian/cyclist inside the hazard box in front of ego."""
    cands = [o for o in objs
             if o.cls in ("pedestrian", "bicycle")
             and abs(o.x) <= PED_HAZARD_LAT_M and 0 < o.y <= PED_HAZARD_LON_M]
    return min(cands, key=lambda o: o.y) if cands else None


def traffic_light_from_objects(objs: list[CriticalObject], have_objects: bool) -> str:
    """
    Ego-relevant traffic-light state inferred from the critical-object list ALONE.

    This exists for symmetry. ORION's CoT config asks only three questions (scene
    description, critical objects, driving reasons) -- there is no dedicated traffic-light
    round at inference time, so the model's state can only come from its object list.
    Chat-B2D lists ego-relevant lights as critical objects with their colour
    ("Traffic light at <2.49, 26.87> ... showing red"), so applying this same rule to the
    GT text gives a like-for-like comparison instead of pitting a dedicated GT round
    against an inferred model value.

    Returns 'unknown' only when there was no object answer at all to reason from.
    """
    if not have_objects:
        return "unknown"
    lights = [o for o in objs if o.cls == "traffic_light"]
    if not lights:
        return "none"                      # an object answer that names no light
    nearest = min(lights, key=lambda o: abs(o.y))
    for o in sorted(lights, key=lambda o: abs(o.y)):
        if o.state:
            return o.state
    return "present_unknown_color" if nearest else "none"


@dataclass
class PerceptionLabels:
    """The 5 perception variables, each True/False/str or None when not determinable."""
    traffic_light: str = "unknown"              # none/red/green/yellow/unknown
    traffic_light_obj: str = "unknown"          # same, derived from objects only (symmetric)
    lead_vehicle: bool | None = None
    lead_distance: float | None = None
    pedestrian: bool | None = None
    ped_distance: float | None = None
    critical_side: str | None = None            # left/center/right of the most critical object
    critical_motion: str | None = None          # static/moving
    n_objects: int = 0
    classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_perception_labels(rounds: dict[str, str]) -> tuple[PerceptionLabels, list[CriticalObject]]:
    """Turn a round dict (GT or model) into the comparable perception label set."""
    co_text = rounds.get("critical_objects", "") or ""
    objs = parse_critical_objects(co_text)
    p = PerceptionLabels()
    p.n_objects = len(objs)
    p.classes = sorted({o.cls for o in objs})
    p.traffic_light = parse_traffic_light(rounds.get("traffic_light", ""), objs)
    p.traffic_light_obj = traffic_light_from_objects(objs, have_objects=bool(co_text.strip()))

    lead = leading_vehicle(objs)
    p.lead_vehicle = lead is not None
    p.lead_distance = lead.y if lead else None

    ped = pedestrian_hazard(objs)
    p.pedestrian = ped is not None
    p.ped_distance = ped.y if ped else None

    # "The" critical object = the nearest non-light, non-sign object ahead; that is the one
    # whose side/motion actually constrains the ego action.
    dyn = [o for o in objs if o.cls in ("car", "pedestrian", "bicycle") and o.y > 0]
    if dyn:
        key = min(dyn, key=lambda o: o.y)
        p.critical_side = key.side
        m = key.motion
        p.critical_motion = m if m != "unknown" else None
    return p, objs


# --------------------------------------------------------------------------------------
# Reasoning / behaviour
# --------------------------------------------------------------------------------------

# Canonical high-level actions, shared with trajectory_metrics.py.
ACTIONS = ["STOP", "DECELERATE", "MAINTAIN", "ACCELERATE"]
LATERAL = ["STRAIGHT", "LEFT", "RIGHT"]

_BEHAV_CANON = {
    "lanefollow": "STRAIGHT", "lane follow": "STRAIGHT", "go straight": "STRAIGHT",
    "straight": "STRAIGHT",
    "turn left": "LEFT", "turnleft": "LEFT", "changelaneleft": "LEFT",
    "change lane left": "LEFT", "left": "LEFT",
    "turn right": "RIGHT", "turnright": "RIGHT", "changelaneright": "RIGHT",
    "change lane right": "RIGHT", "right": "RIGHT",
}


def parse_behavior(answer: str) -> str | None:
    """'lanefollow.' -> STRAIGHT, 'turn left.' -> LEFT, ..."""
    if not answer:
        return None
    a = answer.strip().lower().rstrip(".").strip()
    return _BEHAV_CANON.get(a) or _BEHAV_CANON.get(a.replace("_", " "))


def parse_speed_change(answer: str) -> str | None:
    """'Accelerating.' -> ACCELERATE, 'Decelerating.' -> DECELERATE, 'No change.' -> MAINTAIN."""
    if not answer:
        return None
    a = answer.strip().lower()
    if a.startswith("acceler"):
        return "ACCELERATE"
    if a.startswith("deceler") or a.startswith("brak") or a.startswith("slow"):
        return "DECELERATE"
    if "no change" in a or a.startswith("maintain") or a.startswith("constant"):
        return "MAINTAIN"
    if a.startswith("stop") or "stationary" in a:
        return "STOP"
    return None


# Intent extracted from the free-form "describe your driving behavior and explain the
# reasons" answer. Ordered: the first matching pattern wins, so STOP beats DECELERATE.
_INTENT_PATTERNS = [
    ("STOP", re.compile(r"\b(should |must |will |need to )?(come to a )?(full |complete )?stop\b"
                        r"|\bremain stopped\b|\bstay stopped\b|\bhalt\b|\bstopping\b")),
    ("DECELERATE", re.compile(r"\bdeceler|\bslow down\b|\bslow(ing)? \b|\breduce (the )?speed\b"
                              r"|\bbrak(e|ing)\b|\byield\b|\bcaution\b")),
    ("ACCELERATE", re.compile(r"\bacceler|\bspeed up\b|\bincrease (the )?speed\b")),
    ("MAINTAIN", re.compile(r"\bmaintain\b|\bkeep (the )?(current )?speed\b|\bcontinue\b"
                            r"|\bproceed\b|\bconstant speed\b|\bsteady\b")),
]

_LATERAL_PATTERNS = [
    ("LEFT", re.compile(r"\bturn left\b|\bleft turn\b|\bchange lanes? to the left\b|\bmerge left\b")),
    ("RIGHT", re.compile(r"\bturn right\b|\bright turn\b|\bchange lanes? to the right\b|\bmerge right\b")),
    ("STRAIGHT", re.compile(r"\bfollow the lane\b|\bgo straight\b|\bstraight\b|\bstay in\b|\blane follow\b")),
]


def parse_reasoning_intent(answer: str) -> tuple[str | None, str | None]:
    """Free-form reasoning text -> (longitudinal intent, lateral intent)."""
    if not answer:
        return None, None
    a = answer.strip().lower()
    lon = next((name for name, rx in _INTENT_PATTERNS if rx.search(a)), None)
    lat = next((name for name, rx in _LATERAL_PATTERNS if rx.search(a)), None)
    return lon, lat


@dataclass
class ReasoningLabels:
    intent_lon: str | None = None    # STOP/DECELERATE/MAINTAIN/ACCELERATE
    intent_lat: str | None = None    # STRAIGHT/LEFT/RIGHT
    behavior: str | None = None      # from the explicit behaviour round
    speed_change: str | None = None  # from the explicit speed round
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_reasoning_labels(rounds: dict[str, str]) -> ReasoningLabels:
    text = rounds.get("driving_reason", "")
    lon, lat = parse_reasoning_intent(text)
    return ReasoningLabels(
        intent_lon=lon,
        intent_lat=lat,
        behavior=parse_behavior(rounds.get("behavior_current", "")),
        speed_change=parse_speed_change(rounds.get("speed_change", "")),
        text=text,
    )


def load_annotation(path: str) -> dict[str, str]:
    with open(path) as f:
        return rounds_from_annotation(json.load(f))
