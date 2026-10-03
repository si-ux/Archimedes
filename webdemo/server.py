"""Browser demo server.

The browser is the *sensor*: it opens your webcam, runs MediaPipe Hand
Landmarker locally (WebAssembly) and sends only the 21x3 landmarks over a
WebSocket. Python is the *brain*: the same GesturePipeline used on the
desktop classifies, filters and turns gestures into ViewState, which the
page renders with three.js.

Why split it this way: a GitHub Codespace has no webcam, but your browser
does. Landmarks are ~1 KB per frame, so this stays real-time even through
Codespaces port forwarding.

Run:  uvicorn webdemo.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from archimedes_gestures import synthetic
from archimedes_gestures.landmarks import FrameInput
from archimedes_gestures.pipeline import GesturePipeline

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Archimedes gesture demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health():
    return {"ok": True}


def new_pipeline() -> GesturePipeline:
    return GesturePipeline(
        model_path=ROOT / "models" / "static_pose.joblib",
        profile_path=ROOT / "data" / "profile.json",
        personal_path=ROOT / "data" / "personal_samples.npz",
        clips_root=ROOT / "data" / "clips",
    )


async def run_tour(ws: WebSocket, pipe: GesturePipeline, stop: asyncio.Event) -> None:
    pipe.engine.reset()
    pipe.command("reset")
    for t, hand, caption in synthetic.scripted_tour():
        if stop.is_set():
            break
        out = pipe.process(FrameInput(t, [hand]))
        out.update(type="state", demo=caption, landmarks=[hand.landmarks.tolist()])
        await ws.send_text(json.dumps(out))
        await asyncio.sleep(1 / 30)
    await ws.send_text(json.dumps({"type": "demo_done"}))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    pipe = new_pipeline()
    await ws.send_text(json.dumps({"type": "info", **pipe.info()}))
    tour_stop = asyncio.Event()
    tour_task: asyncio.Task | None = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "frame":
                if tour_task and not tour_task.done():
                    continue  # the demo tour owns the pipeline until it ends
                f = pipe.make_frame(
                    msg.get("hands", []),
                    timestamp=msg["t"] / 1000.0,
                    mirrored=msg.get("mirrored", False),
                    brightness=msg.get("brightness"),
                    sharpness=msg.get("sharpness"),
                )
                out = pipe.process(f)
                out["type"] = "state"
                out["seq"] = msg.get("seq")
                await ws.send_text(json.dumps(out))
            elif kind == "command":  # mouse / keyboard fallback
                pipe.command(msg["kind"], **msg.get("data", {}))
                await ws.send_text(json.dumps({"type": "view", "view": pipe.view.to_dict()}))
            elif kind == "calibrate":
                status = pipe.start_calibration(with_poses=msg.get("with_poses", True))
                await ws.send_text(json.dumps({"type": "state", "calibration": status, "hud": None,
                                               "view": pipe.view.to_dict()}))
            elif kind == "calibrate_cancel":
                pipe.cancel_calibration()
            elif kind == "set_dominant":
                pipe.set_dominant(msg["hand"])
                await ws.send_text(json.dumps({"type": "info", **pipe.info()}))
            elif kind == "record_start":
                pipe.recorder.start(msg["label"], msg.get("subject", "anon"), pipe.profile.dominant)
            elif kind == "record_stop":
                path = pipe.recorder.stop()
                await ws.send_text(json.dumps({"type": "recorded", "path": str(path.relative_to(ROOT)) if path else None}))
            elif kind == "demo":
                tour_stop.clear()
                tour_task = asyncio.create_task(run_tour(ws, pipe, tour_stop))
            elif kind == "demo_stop":
                tour_stop.set()
            elif kind == "info":
                await ws.send_text(json.dumps({"type": "info", **pipe.info()}))
    except WebSocketDisconnect:
        tour_stop.set()
