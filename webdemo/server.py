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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from archimedes_fe.export import to_glb
from archimedes_fe.session import ModelSession
from archimedes_gestures import synthetic
from archimedes_gestures.landmarks import FrameInput
from archimedes_gestures.pipeline import GesturePipeline
from archimedes_gestures.vocab import PROFILES

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
WORKBENCH = ROOT / "Archimedes"

app = FastAPI(title="Archimedes gesture demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/workbench")
def workbench_redirect():
    return RedirectResponse("/workbench/")


@app.get("/workbench/")
def workbench():
    """The Archimedes Workbench, served same-origin so its gesture client can reach /ws."""
    return FileResponse(WORKBENCH / "Archimedes Workbench.dc.html", media_type="text/html")


@app.get("/workbench/support.js")
def workbench_support():
    return FileResponse(WORKBENCH / "support.js", media_type="application/javascript")


# ---- VR / AR ---------------------------------------------------------------
# The most recently solved model, shared with /xr (a headset opens that page
# separately, so it can't use the editing session's WebSocket).
SCENE: dict = {"results": None, "version": 0}


@app.get("/xr")
def xr_page():
    """WebXR viewer: immersive VR (Quest etc.), AR on Android Chrome, plain 3D elsewhere."""
    return FileResponse(STATIC / "xr.html")


@app.get("/api/scene.json")
def scene_json():
    res = SCENE["results"]
    if res is None:
        return JSONResponse({"version": 0, "ready": False})
    fields = {k: {"min": float(v.min()), "max": float(v.max())} for k, v in res.fields.items()}
    for i, md in enumerate(res.modes):
        fields[f"mode{i + 1}"] = {"min": 0.0, "max": 1.0, "freq": md["f"]}
    return {"version": SCENE["version"], "ready": True, "model": res.model.to_dict(), "stats": res.stats,
            "fields": fields}


@app.get("/api/scene.glb")
def scene_glb(field: str = "von_mises", deform: float | None = None):
    res = SCENE["results"]
    if res is None:
        return JSONResponse({"error": "nothing solved yet"}, status_code=404)
    return Response(to_glb(res, field, deform), media_type="model/gltf-binary",
                    headers={"Content-Disposition": f'inline; filename="archimedes-{field}.glb"',
                             "Cache-Control": "no-store"})


@app.get("/healthz")
def health():
    return {"ok": True}


def new_pipeline(profile: str = "viewer") -> GesturePipeline:
    return GesturePipeline(
        profile=profile if profile in PROFILES else "viewer",
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


# Gesture commands "support", "point_load", "udl" and "trap" carry screen positions.
# They reach the page inside the "state" message; the page maps them onto the
# member (it owns the camera) and answers with a "model_action" message.


async def send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj))


async def send_model(ws: WebSocket, pipe: GesturePipeline, session: ModelSession) -> None:
    pipe.set_prompt(session.prompt)
    if pipe.phase != session.phase:
        pipe.set_phase(session.phase)
    await send(ws, {"type": "model", **session.state()})


async def run_solve(ws: WebSocket, pipe: GesturePipeline, session: ModelSession) -> None:
    await send(ws, {"type": "solving"})
    res = await asyncio.get_running_loop().run_in_executor(None, session.solve)
    if res is not None:
        payload = res.payload()
        pipe.view.fields = payload["field_order"]
        pipe.view.field_index = 0
        SCENE["results"] = res
        SCENE["version"] += 1
        await send(ws, {"type": "results", **payload, "model": session.model.to_dict()})
        await send(ws, {"type": "view", "view": pipe.view.to_dict()})  # field list changed: back to the first
    await send_model(ws, pipe, session)


async def handle_gesture_commands(ws, pipe, session, cmds) -> None:
    changed = False
    for c in cmds:
        k, d = c["kind"], c.get("data", {})
        if k == "extrude":
            session.begin("new_rect", orientation=d["orientation"], length_est=d.get("length_est"))
            changed = True
        elif k == "circle":
            session.begin("new_circle")
            changed = True
        elif k == "number_enter":
            session.answer(d.get("value"))
            changed = True
        elif k == "number_cancel":
            session.cancel()
            changed = True
        elif k == "solve":
            await run_solve(ws, pipe, session)
    if changed:
        await send_model(ws, pipe, session)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # ?profile=workbench gives the Workbench vocabulary (thumbs-up = run solve)
    profile = ws.query_params.get("profile", "viewer")
    pipe = new_pipeline(profile)
    await send(ws, {"type": "info", **pipe.info()})
    session = ModelSession() if profile != "workbench" else None
    if session is not None:
        await run_solve(ws, pipe, session)  # open on a solved demo
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
                await send(ws, out)
                if session is not None and out["commands"]:
                    await handle_gesture_commands(ws, pipe, session, out["commands"])
            elif kind == "command":  # mouse / keyboard fallback for the view
                pipe.command(msg["kind"], **msg.get("data", {}))
                await send(ws, {"type": "view", "view": pipe.view.to_dict()})
            elif session is not None and kind in ("model_action", "number", "prompt_cancel", "set_axis", "set_phase",
                                                  "load_demo", "solve", "undo_model", "clear"):
                if kind == "model_action":
                    data = dict(msg["action"])
                    try:
                        session.begin(data.pop("op"), **data)
                    except (KeyError, ValueError) as exc:
                        session.error = f"Could not do that: {exc}"
                elif kind == "number":
                    session.answer(msg.get("value"))
                elif kind == "prompt_cancel":
                    session.cancel()
                elif kind == "set_axis":
                    session.set_direction(msg["axis"], msg.get("sign", -1))
                elif kind == "set_phase":
                    session.set_phase(msg["phase"])
                elif kind == "undo_model":
                    session.undo()
                elif kind == "clear":
                    session.clear(msg.get("what", "loads"))
                if kind == "load_demo":
                    session.load_demo(msg["name"])
                    await run_solve(ws, pipe, session)
                elif kind == "solve":
                    await run_solve(ws, pipe, session)
                else:
                    await send_model(ws, pipe, session)
            elif kind == "calibrate":
                status = pipe.start_calibration(with_poses=msg.get("with_poses", True))
                await send(ws, {"type": "state", "calibration": status, "hud": None, "view": pipe.view.to_dict()})
            elif kind == "calibrate_cancel":
                pipe.cancel_calibration()
            elif kind == "set_dominant":
                pipe.set_dominant(msg["hand"])
                await send(ws, {"type": "info", **pipe.info()})
            elif kind == "record_start":
                pipe.recorder.start(msg["label"], msg.get("subject", "anon"), pipe.profile.dominant)
            elif kind == "record_stop":
                path = pipe.recorder.stop()
                await send(ws, {"type": "recorded", "path": str(path.relative_to(ROOT)) if path else None})
            elif kind == "demo":
                if session is not None:
                    session.set_phase("results")
                    pipe.set_phase("results")
                tour_stop.clear()
                tour_task = asyncio.create_task(run_tour(ws, pipe, tour_stop))
            elif kind == "demo_stop":
                tour_stop.set()
            elif kind == "info":
                await send(ws, {"type": "info", **pipe.info()})
    except WebSocketDisconnect:
        tour_stop.set()
