"""The gesture vocabulary: one place that names every pose and what it does.

Design rules (see docs/DESIGN.md for the reasoning):
* six poses + "none". Every pose is chosen to be easy for MediaPipe to see
  (no "three fingers vs four fingers" counting, which self-occlusion breaks).
* poses map to **modes**, the hand's motion is the **control**. Users learn
  six shapes, not thirty commands.
* anything that changes data you can't get back (reset, screenshot) needs a
  deliberate hold with a progress ring.
"""
from __future__ import annotations

from dataclasses import dataclass

NONE = "none"
POSES = ("open_palm", "fist", "point", "peace", "thumbs_up", "pinch")


@dataclass(frozen=True)
class Mode:
    name: str
    pose: str
    arm_s: float  # how long the pose must be held before the mode engages
    still: bool  # must the hand also be still while arming?
    label: str  # HUD text
    hint: str  # one-line "how to use" text for the HUD / cheat-sheet
    icon: str
    fire: str | None = None  # one-shot command emitted when the mode engages
    toast: str | None = None  # HUD message shown when it fires
    two: bool = False  # needs both hands; never chosen by a single hand's pose


MODES = {
    "orbit": Mode("orbit", "fist", 0.12, False, "Rotate", "Make a fist and drag - like grabbing the model", "✊"),
    "zoom": Mode("zoom", "pinch", 0.12, False, "Zoom", "Pinch, then move up to zoom in, down to zoom out", "🤏"),
    "grab": Mode("grab", "pinch", 0.1, False, "Two-hand zoom / rotate",
                 "Pinch with both hands: spread or bring them together to zoom, move them together to rotate",
                 "🤏🤏", two=True),
    "section": Mode(
        "section", "open_palm", 0.5, True, "Section cut",
        "Hold a flat palm still: a straight cut along the axis it faces. Slide the palm to move the cut. "
        "Show your other palm to tilt the cut freely", "✋",
    ),
    "probe": Mode("probe", "point", 0.12, False, "Probe",
                  "Point to read stress; pinch with your other hand (or hold still) to pin the value", "☝️"),
    "field": Mode("field", "peace", 0.2, False, "Field", "Show a V and swipe left/right to change the result field", "✌️"),
    "snapshot": Mode("snapshot", "thumbs_up", 1.0, True, "Snapshot", "Hold a thumbs-up to save a screenshot", "👍",
                     fire="snapshot", toast="Snapshot saved"),
}

# The Workbench runs solves, which cost seconds to minutes and change what the
# user is looking at, so the thumbs-up there means "run the solve" and needs a
# longer, still hold. Pipeline stages (geometry -> mesh) are pre-processing and
# stay on menus/keys: gestures are for looking at results.
WORKBENCH_MODES = {
    **{k: v for k, v in MODES.items() if k != "snapshot"},
    "solve": Mode("solve", "thumbs_up", 1.2, True, "Run solve", "Hold a thumbs-up still to run the solve", "👍",
                  fire="solve", toast="Solve started"),
}
PROFILES = {"viewer": MODES, "workbench": WORKBENCH_MODES}
POSE_TO_MODE = {m.pose: m.name for m in MODES.values() if not m.two}


def pose_to_mode(modes: dict) -> dict:
    return {m.pose: m.name for m in modes.values() if not m.two}

# two-hand gesture, handled separately by the intent engine
RESET_HINT = "Show both open palms for a second to reset the view"


def cheat_sheet(modes: dict | None = None) -> list[dict]:
    rows = [{"icon": m.icon, "label": m.label, "hint": m.hint, "pose": m.pose, "mode": m.name}
            for m in (modes or MODES).values()]
    rows.append({"icon": "🙌", "label": "Reset", "hint": RESET_HINT, "pose": "open_palm x2", "mode": "reset"})
    return rows
