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


MODES = {
    "orbit": Mode("orbit", "fist", 0.25, False, "Rotate", "Make a fist and drag - like grabbing the model", "✊"),
    "zoom": Mode("zoom", "pinch", 0.25, False, "Zoom", "Pinch, then move up to zoom in, down to zoom out", "🤏"),
    "section": Mode(
        "section", "open_palm", 0.8, True, "Section cut",
        "Hold a flat palm still - your palm becomes the cutting plane", "✋",
    ),
    "probe": Mode("probe", "point", 0.3, False, "Probe", "Point to read stress; hold still to pin a value", "☝️"),
    "field": Mode("field", "peace", 0.3, False, "Field", "Show a V and swipe left/right to change the result field", "✌️"),
    "snapshot": Mode("snapshot", "thumbs_up", 1.0, True, "Snapshot", "Hold a thumbs-up to save a screenshot", "👍"),
}
POSE_TO_MODE = {m.pose: m.name for m in MODES.values()}

# two-hand gesture, handled separately by the intent engine
RESET_HINT = "Show both open palms for a second to reset the view"


def cheat_sheet() -> list[dict]:
    rows = [{"icon": m.icon, "label": m.label, "hint": m.hint, "pose": m.pose} for m in MODES.values()]
    rows.append({"icon": "🙌", "label": "Reset", "hint": RESET_HINT, "pose": "open_palm x2"})
    return rows
