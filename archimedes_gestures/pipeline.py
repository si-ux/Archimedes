"""GesturePipeline: one object that front-ends (desktop, browser, replay) talk to.

    raw landmarks --> HandFrame (selfie view, real handedness)
                  --> pose classifier (rules or trained) + personal adapter
                  --> PoseSmoother per hand
                  --> IntentEngine --> Commands --> ViewState
"""
from __future__ import annotations

import time
from pathlib import Path

from .calibration import Calibrator, Profile
from .classifier import PersonalAdapter, PoseClassifier, PoseSmoother, RulePoseClassifier, SklearnPoseClassifier
from .intent import EngineConfig, IntentEngine
from .landmarks import FrameInput, HandFrame
from .recorder import ClipRecorder
from .view_state import Command, ViewState
from .vocab import cheat_sheet


def load_base_classifier(model_path: str | Path | None) -> PoseClassifier:
    if model_path and Path(model_path).exists():
        return SklearnPoseClassifier.load(model_path)
    return RulePoseClassifier()


class GesturePipeline:
    def __init__(
        self,
        model_path: str | Path | None = "models/static_pose.joblib",
        profile_path: str | Path = "data/profile.json",
        personal_path: str | Path = "data/personal_samples.npz",
        clips_root: str | Path = "data/clips",
    ):
        self.profile_path, self.personal_path = Path(profile_path), Path(personal_path)
        self.profile = Profile.load(self.profile_path) if self.profile_path.exists() else Profile()
        self.base = load_base_classifier(model_path)
        self.classifier = PersonalAdapter(self.base)
        if self.personal_path.exists():
            self.classifier.load(self.personal_path)
        self.engine = IntentEngine(self._engine_cfg())
        self.view = ViewState()
        self.smoothers: dict[str, PoseSmoother] = {}
        self.calibrator: Calibrator | None = None
        self.recorder = ClipRecorder(clips_root)
        self.last_latency_ms = 0.0

    @property
    def classifier_name(self) -> str:
        return "trained" if isinstance(self.base, SklearnPoseClassifier) else "rules (untrained)"

    def _engine_cfg(self) -> EngineConfig:
        return EngineConfig(dominant=self.profile.dominant, box=self.profile.box)

    # ---- input -------------------------------------------------------
    def make_frame(self, hands_raw: list[dict], timestamp: float | None = None, mirrored: bool = True,
                   brightness: float | None = None, sharpness: float | None = None) -> FrameInput:
        """hands_raw: [{"landmarks": [[x,y,z]*21], "handedness": "Left", "score": 0.97}, ...]"""
        t = time.monotonic() if timestamp is None else timestamp
        prof = self.calibrator.profile if self.calibrator else self.profile
        hands = [
            HandFrame.from_raw(h["landmarks"], h.get("handedness", "Right"), h.get("score", 1.0), t,
                               mirrored=mirrored, swap_handedness=prof.swap_handedness)
            for h in hands_raw
        ]
        # if MediaPipe labels two hands the same, keep only the more confident one
        seen: dict[str, HandFrame] = {}
        for h in hands:
            if h.handedness not in seen or h.score > seen[h.handedness].score:
                seen[h.handedness] = h
        return FrameInput(t, list(seen.values()), brightness, sharpness)

    # ---- main step ---------------------------------------------------
    def process(self, f: FrameInput) -> dict:
        t0 = time.perf_counter()
        self.recorder.add(f)
        if self.calibrator is not None:
            status = self.calibrator.update(f)
            if self.calibrator.done:
                self._finish_calibration()
            return {"calibration": status, "hud": None, "view": self.view.to_dict(), "commands": []}

        poses = {}
        for h in f.hands:
            sm = self.smoothers.setdefault(h.handedness, PoseSmoother())
            poses[h.handedness] = sm.update(self.classifier.predict_proba(h.landmarks, h.handedness))
        for k in list(self.smoothers):
            if f.hand(k) is None:
                self.smoothers[k].reset()

        cmds, hud = self.engine.update(f, poses)
        for c in cmds:
            self.view.apply(c)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        hud["latency_ms"] = round(self.last_latency_ms, 2)
        hud["classifier"] = self.classifier_name
        hud["personal_samples"] = len(self.classifier.y)
        return {"hud": hud, "view": self.view.to_dict(), "commands": [c.__dict__ for c in cmds]}

    def command(self, kind: str, **data) -> None:
        """Mouse/keyboard fallback and UI buttons go through the same path as gestures."""
        self.view.apply(Command(kind, data))

    # ---- calibration -------------------------------------------------
    def start_calibration(self, with_poses: bool = True) -> dict:
        cal = Calibrator(with_poses=with_poses)
        cal.profile = Profile(**{**self.profile.__dict__})
        self.calibrator = cal
        return cal.status(0.0)

    def cancel_calibration(self) -> None:
        self.calibrator = None

    def _finish_calibration(self) -> None:
        cal = self.calibrator
        self.calibrator = None
        self.profile = cal.profile
        if cal.samples:
            for label in {s[0] for s in cal.samples}:
                self.classifier.clear(label)
            for label, lm, hand in cal.samples:
                self.classifier.add_sample(lm, label, hand)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile.save(self.profile_path)
        if self.classifier.y:
            self.classifier.save(self.personal_path)
        self.engine = IntentEngine(self._engine_cfg())
        self.smoothers.clear()

    def set_dominant(self, hand: str) -> None:
        self.profile.dominant = hand
        self.engine.cfg.dominant = hand

    def info(self) -> dict:
        return {
            "cheat_sheet": cheat_sheet(),
            "profile": self.profile.__dict__,
            "classifier": self.classifier_name,
            "personal_samples": self.classifier.counts(),
        }
