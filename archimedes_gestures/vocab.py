"""The gesture vocabulary: one place that names every pose and what it does.

Design rules (see docs/DESIGN.md for the reasoning):
* poses that MediaPipe tells apart reliably: fist, flat palm, point, V, thumbs-up, pinch (+ "none").
* ONE hand selects and points; TWO hands manipulate, as in VR: the two-hand gestures are the big moves (zoom, pan,
  turn), the one-hand gestures are the precise ones (probe, section, field). A pose never means two things in the
  same phase: a lone pinch does nothing in the results view, so pinch and point can never be confused there.
* anything that changes what you look at needs the pose held for a moment; anything you cannot get back (reset,
  solve, snapshot) needs a still hold with a progress ring.
"""
from __future__ import annotations

from dataclasses import dataclass

NONE = "none"
POSES = ("open_palm", "fist", "point", "peace", "thumbs_up", "pinch")


@dataclass(frozen=True)
class Mode:
    name: str
    pose: str  # the pose (on both hands for a two-hand mode)
    arm_s: float  # how long the pose must be held before the mode engages
    still: bool  # must the hand also be still while arming?
    label: str  # HUD text
    hint: str  # one-line "how to use" text for the HUD / cheat-sheet
    icon: str
    fire: str | None = None  # one-shot command emitted when the mode engages
    toast: str | None = None  # HUD message shown when it fires
    two: bool = False  # needs both hands showing the pose; never chosen by a single hand


MODES = {
    # ---- one hand: precise --------------------------------------------------------------------------------
    "orbit": Mode("orbit", "fist", 0.2, False, "Rotate", "Make a fist and drag, like grabbing the model", "✊"),
    "probe": Mode("probe", "point", 0.2, False, "Probe",
                  "Point: the ring on the model is your cursor. Pinch with your other hand (or hold still) to pin", "☝️"),
    "section": Mode("section", "open_palm", 0.6, True, "Section cut",
                    "One flat palm, held still: a straight cut across the axis your palm faces. Slide it to move the cut",
                    "✋"),
    "field": Mode("field", "peace", 0.25, False, "Change field",
                  "Show a V, then move the hand right / left: one field per step (the strip shows where you are)", "✌️"),
    "snapshot": Mode("snapshot", "thumbs_up", 1.0, True, "Snapshot", "Hold a thumbs-up still to save a screenshot", "👍",
                     fire="snapshot", toast="Snapshot saved"),
    # ---- two hands: big moves -------------------------------------------------------------------------------
    "grab": Mode("grab", "pinch", 0.15, False, "Zoom / pan",
                 "Pinch with both hands: spread them to zoom in, bring them together to zoom out, move both to pan",
                 "🤏🤏", two=True),
    "turn": Mode("turn", "fist", 0.15, False, "Turn",
                 "Two fists, like holding a ball: move them to rotate, twist them like a steering wheel to spin", "✊✊",
                 two=True),
    "dual": Mode("dual", "point", 0.2, False, "Compare two points",
                 "Point with both hands: two cursors and the difference between their values", "☝️☝️", two=True),
    "reset": Mode("reset", "thumbs_up", 0.8, True, "Reset view", "Both thumbs up, held: back to the start view",
                  "👍👍", fire="reset", toast="View reset", two=True),
}

# The Workbench runs solves, which cost seconds to minutes and change what the user is looking at, so the thumbs-up
# there means "run the solve" and needs a longer, still hold. Pipeline stages (geometry -> mesh) are pre-processing
# and stay on menus/keys: gestures are for looking at results.
WORKBENCH_MODES = {
    **{k: v for k, v in MODES.items() if k != "snapshot"},
    "solve": Mode("solve", "thumbs_up", 1.2, True, "Run solve", "Hold a thumbs-up still to run the solve", "👍",
                  fire="solve", toast="Solve started"),
}
PROFILES = {"viewer": MODES, "workbench": WORKBENCH_MODES}
POSE_TO_MODE = {m.pose: m.name for m in MODES.values() if not m.two}


def pose_to_mode(modes: dict) -> dict:
    return {m.pose: m.name for m in modes.values() if not m.two}


def two_hand_modes(modes: dict) -> dict:
    """pose shown by both hands -> mode name"""
    return {m.pose: m.name for m in modes.values() if m.two}


def cheat_sheet(modes: dict | None = None) -> list[dict]:
    return [{"icon": m.icon, "label": m.label, "hint": m.hint, "pose": m.pose + (" x2" if m.two else ""), "mode": m.name,
             "two": m.two} for m in (modes or MODES).values()]
