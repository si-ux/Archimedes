"""Desktop webcam demo (run on your own computer, not in a Codespace - it needs a camera).

  python apps/desktop_demo.py                 # OpenCV window with skeleton + HUD
  python apps/desktop_demo.py --pyvista mesh.vtu --field VonMises
                                              # also drive a PyVista window showing your result file

Keys in the camera window: c = calibrate, u = undo, r = reset, q / Esc = quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archimedes_gestures.landmarks import HAND_CONNECTIONS  # noqa: E402
from archimedes_gestures.pipeline import GesturePipeline  # noqa: E402
from archimedes_gestures.tracker import HandTracker  # noqa: E402

STATE_BGR = {"active": (142, 207, 62), "arming": (61, 184, 245), "paused": (107, 107, 255)}


def draw(frame, hands, out, fps):
    import cv2

    h, w = frame.shape[:2]
    hud = out.get("hud")
    cal = out.get("calibration")
    color = STATE_BGR.get(hud["state"] if hud else "", (220, 220, 220))
    for hd in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hd.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(frame, p, 3, color, -1)
        if hud and hud["progress"] > 0 and not hud["mode"]:
            cv2.ellipse(frame, pts[0], (28, 28), -90, 0, int(360 * hud["progress"]), (61, 184, 245), 4)

    def text(s, y, scale=0.6, c=(240, 240, 240)):
        cv2.putText(frame, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)

    if cal:
        text(f"Calibration {cal['step_index'] + 1}/{cal['step_count']}", 30, 0.7)
        text(cal["instruction"].encode("ascii", "ignore").decode(), 60)
        cv2.rectangle(frame, (12, 75), (12 + int(300 * cal["progress"]), 85), (142, 207, 62), -1)
        return
    if not hud:
        return
    title = hud["mode_label"] or (f"{hud['progress_label']}..." if hud["progress_label"] else hud["state"])
    text(f"{title}   pose: {hud['pose']} {hud['confidence'] * 100:.0f}%", 30, 0.7, color)
    for i, hint in enumerate(hud["hints"][:2]):
        text(hint, 60 + 26 * i, 0.6, (107, 107, 255))
    text(hud["tip"].encode("ascii", "ignore").decode(), h - 16, 0.5)
    text(f"{fps:.0f} fps  engine {hud['latency_ms']:.1f} ms  model: {hud['classifier']}", h - 40, 0.5)


def main():
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pyvista", help="result mesh file (.vtu/.vtk/.xdmf) to show in a PyVista window")
    ap.add_argument("--field", action="append", default=[], help="point/cell array name(s), in field order")
    args = ap.parse_args()

    pipe = GesturePipeline()
    tracker = HandTracker(mode="video")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("No camera found. In a Codespace use the browser demo instead: uvicorn webdemo.server:app")

    bridge = plotter = None
    if args.pyvista:
        import pyvista as pv

        from archimedes_gestures.pyvista_bridge import PyVistaBridge
        from archimedes_gestures.view_state import FIELDS

        mesh = pv.read(args.pyvista)
        names = args.field or list(mesh.array_names)
        plotter = pv.Plotter(title="Archimedes result")
        bridge = PyVistaBridge(plotter, mesh, dict(zip(FIELDS, names)))
        bridge.apply(pipe.view)
        plotter.show(interactive_update=True)

    t_fps, n, fps = time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie view: what the user expects to see
        ts = int(time.monotonic() * 1000)
        hands_raw = tracker.detect_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        f = pipe.make_frame(hands_raw, ts / 1000, mirrored=True, brightness=float(gray.mean()),
                            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        out = pipe.process(f)
        if bridge:
            bridge.apply(pipe.view)
            plotter.update()
        n += 1
        if time.time() - t_fps > 1:
            fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
        draw(frame, f.hands, out, fps)
        cv2.imshow("Archimedes gestures (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            pipe.start_calibration()
        elif key == ord("u"):
            pipe.command("undo")
        elif key == ord("r"):
            pipe.command("reset")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
