"""MediaPipe Hand Landmarker wrapper for Python front-ends (desktop app, dataset extraction).

Uses the MediaPipe Tasks API (``mediapipe>=0.10``). The model file is
downloaded once to ``models/hand_landmarker.task``.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(path: str | Path = "models/hand_landmarker.task") -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Thin wrapper returning plain dicts that ``GesturePipeline.make_frame`` accepts."""

    def __init__(self, mode: str = "video", num_hands: int = 2, model_path: str | Path = "models/hand_landmarker.task",
                 min_conf: float = 0.6):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        running = {"video": vision.RunningMode.VIDEO, "image": vision.RunningMode.IMAGE}[mode]
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=running,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self.mode = mode

    def _convert(self, res) -> list[dict]:
        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            out.append({
                "landmarks": [[p.x, p.y, p.z] for p in lms],
                "handedness": hd[0].category_name,
                "score": float(hd[0].score),
            })
        return out

    def detect_rgb(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> list[dict]:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            return self._convert(self._lm.detect_for_video(img, int(timestamp_ms)))
        return self._convert(self._lm.detect(img))

    def detect_file(self, path: str | Path) -> list[dict]:
        return self._convert(self._lm.detect(self._mp.Image.create_from_file(str(path))))

    def close(self) -> None:
        self._lm.close()
