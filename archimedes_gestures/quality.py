"""Tracking-quality checks that turn into plain-language hints for the user.

The most common reason a gesture "doesn't work" is not the classifier, it is
the input: hand half out of frame, too far away, a dark room, motion blur.
Telling the user *which* of these is happening fixes most failures in one
step, so every check here returns a short, actionable sentence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .landmarks import FrameInput, HandFrame


@dataclass
class QualityConfig:
    min_brightness: float = 60.0  # mean luma (0-255)
    max_brightness: float = 230.0
    min_sharpness: float = 40.0  # variance of Laplacian; lower = blurry
    min_palm: float = 0.06  # palm size as a fraction of image height
    max_palm: float = 0.30
    edge_margin: float = 0.03
    min_score: float = 0.6
    max_speed: float = 4.0  # palm sizes per second; faster than this, MediaPipe smears


@dataclass
class Hint:
    code: str
    text: str
    severity: int  # 2 = blocks input, 1 = degrades it, 0 = info


def check_frame(f: FrameInput, cfg: QualityConfig) -> list[Hint]:
    hints = []
    if f.brightness is not None:
        if f.brightness < cfg.min_brightness:
            hints.append(Hint("dark", "It's too dark - turn on a light or face a window", 1))
        elif f.brightness > cfg.max_brightness:
            hints.append(Hint("bright", "Too much light behind you - avoid sitting in front of a window", 1))
    if f.sharpness is not None and f.sharpness < cfg.min_sharpness:
        hints.append(Hint("blur", "Image is blurry - slow down or clean the camera lens", 1))
    return hints


def check_hand(h: HandFrame, cfg: QualityConfig, speed: float = 0.0) -> list[Hint]:
    hints = []
    x0, y0, x1, y1 = h.bbox
    m = cfg.edge_margin
    if x0 < m or y0 < m or x1 > 1 - m or y1 > 1 - m:
        hints.append(Hint("edge", "Hand is at the edge of the picture - move it towards the centre", 2))
    if h.palm_size < cfg.min_palm:
        hints.append(Hint("far", "Hand is too far away - bring it closer to the camera", 2))
    elif h.palm_size > cfg.max_palm:
        hints.append(Hint("near", "Hand is too close - move it back a little", 1))
    if h.score < cfg.min_score:
        hints.append(Hint("unsure", "Tracking is unsure - keep your whole hand visible", 1))
    if speed > cfg.max_speed:
        hints.append(Hint("fast", "Moving too fast for the camera - slow down", 1))
    return hints
