# ruff: noqa: E702, E731
"""Record the gesture demos for the README and a portfolio: a real browser, scripted hands, captions.

    python scripts/record_demo.py --out docs/media

What it does
    1. starts the demo server (or uses --url),
    2. opens the Workbench / the gesture modeller in headless Chromium with video recording,
    3. feeds scripted two-hand landmark frames through the same WebSocket path a webcam frame takes (the engine,
       the filters and the UI are the real ones; only the camera is replaced by synthetic hands),
    4. writes .mp4 and a small .gif per demo with ffmpeg.

Needs: playwright (python -m playwright install chromium) and ffmpeg. The synthetic hand is a kinematic model, so this
is a demonstration of the interaction, not a measurement of recognition accuracy on real hands.

Offline sandboxes: --assets DIR maps the CDN packages (three, @mediapipe/tasks-vision, react, react-dom,
@babel/standalone) from local folders named like the npm packages, so the pages load without internet.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import gesture_bench as gb  # noqa: E402

FPS = 30
SIZE = {"width": 1280, "height": 760}

CAPTION_JS = """
(text) => {
  let el = document.getElementById('demo-cap');
  if (!el) {
    el = document.createElement('div'); el.id = 'demo-cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:99999;max-width:78%;' +
      'padding:10px 20px;border-radius:10px;background:rgba(10,13,17,.88);border:1px solid rgba(94,207,224,.55);' +
      "color:#eaf3f7;font:600 19px/1.3 'IBM Plex Sans',system-ui,sans-serif;text-align:center;letter-spacing:.01em;" +
      'box-shadow:0 8px 30px rgba(0,0,0,.5);transition:opacity .2s';
    document.body.appendChild(el);
  }
  el.style.opacity = text ? 1 : 0; el.textContent = text || '';
}
"""


# ----------------------------------------------------------------------------- scripted hands
class Script:
    """Builds a list of (seconds, caption, fn(a) -> [HandFrame]) scenes; a runs 0 -> 1 over the scene."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.scenes = []

    def h(self, pose=None, fingers=None, c=(0.5, 0.55), hand="Right", size=0.12, **kw):
        return gb.hand(pose, fingers, center=c, size=size, handedness=hand, t=self.t, rng=self.rng, **kw)

    def scene(self, secs, caption, fn):
        self.scenes.append((secs, caption, fn))


def lerp(a, b, k):
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def workbench_script() -> Script:
    s = Script(1)
    L = "Left"
    s.scene(1.2, "Hands are tracked in your browser, only landmarks reach Python",
            lambda a: [s.h("none", c=(0.5, 0.55))])
    # two pinches: spread to zoom in, then bring together
    s.scene(0.5, "Two-hand zoom: pinch with both hands", lambda a: [s.h("pinch", c=(0.36, 0.55), hand=L), s.h("pinch", c=(0.64, 0.55))])
    s.scene(1.8, "...spread to zoom in", lambda a: [s.h("pinch", c=(0.36 - 0.12 * a, 0.55), hand=L), s.h("pinch", c=(0.64 + 0.12 * a, 0.55))])
    s.scene(1.2, "...bring together to zoom out", lambda a: [s.h("pinch", c=(0.24 + 0.1 * a, 0.55), hand=L), s.h("pinch", c=(0.76 - 0.1 * a, 0.55))])
    s.scene(1.8, "Move both pinched hands to rotate",
            lambda a: [s.h("pinch", c=(0.34 + 0.16 * a, 0.55 - 0.04 * a), hand=L), s.h("pinch", c=(0.66 + 0.16 * a, 0.55 - 0.04 * a))])
    s.scene(0.8, "", lambda a: [s.h("none", c=(0.5, 0.55))])
    # straight section cut
    s.scene(0.9, "Flat palm, held still: a straight cut along the axis it faces",
            lambda a: [s.h("open_palm", c=(0.4, 0.55), yaw=0.18, pitch=-0.1)])
    s.scene(2.4, "Slide the palm to move the cut: it stays straight, even with a tilted hand",
            lambda a: [s.h("open_palm", c=lerp((0.4, 0.55), (0.62, 0.55), a), yaw=0.3 * np.sin(a * 9), pitch=-0.15 + 0.1 * np.cos(a * 7))])
    s.scene(2.4, "Show your other palm to tilt the cut freely",
            lambda a: [s.h("open_palm", c=(0.62, 0.55), yaw=0.15 + 0.8 * a, pitch=-0.2 * a),
                       s.h("open_palm", c=(0.18, 0.6), hand=L)])
    s.scene(1.0, "", lambda a: [s.h("none", c=(0.5, 0.55))])
    s.scene(1.6, "Both palms, held: reset the view", lambda a: [s.h("open_palm", c=(0.32, 0.55), hand=L), s.h("open_palm", c=(0.68, 0.55))])
    return s


def tip_offset() -> np.ndarray:
    """Where the index fingertip lies relative to the palm centre for a pointing hand (image units)."""
    h = gb.hand("point", center=(0.5, 0.5), rng=np.random.default_rng(0), noise=0.0)
    return h.landmarks[8, :2] - np.array([0.5, 0.5])


def probe_script(a_img, b_img) -> Script:
    """Point from a_img to b_img (fingertip positions, image coords), then pinch with the other hand to pin."""
    s = Script(3)
    off = tip_offset()
    c = lambda p: (p[0] - off[0], p[1] - off[1])  # noqa: E731
    s.scene(0.5, "", lambda a: [s.h("none", c=c(a_img))])
    s.scene(2.8, "Point to probe: the fingertip is the cursor, the value follows it", lambda a: [s.h("point", c=c(lerp(a_img, b_img, a)))])
    s.scene(0.8, "Point to probe: the fingertip is the cursor, the value follows it", lambda a: [s.h("point", c=c(b_img))])
    s.scene(0.7, "Pinch with the other hand to pin the value", lambda a: [s.h("point", c=c(b_img)), s.h("pinch", c=(0.2, 0.6), hand="Left")])
    s.scene(1.6, "Pinch with the other hand to pin the value", lambda a: [s.h("point", c=c(b_img))])
    s.scene(1.0, "", lambda a: [s.h("none", c=(0.5, 0.55))])
    return s


def modeller_script() -> Script:
    s = Script(2)
    nothing = lambda a: []  # noqa: E731
    s.scene(0.8, "Setup: choose the load increment by holding up fingers", nothing)
    s.scene(1.6, "3 fingers: 100 mm length increment", lambda a: [s.h(fingers="imr", c=(0.5, 0.55))])
    s.scene(0.9, "", nothing)
    s.scene(1.6, "2 fingers: 10 mm section increment", lambda a: [s.h(fingers="im", c=(0.5, 0.55))])
    s.scene(0.9, "", nothing)
    s.scene(1.6, "4 fingers: 10 kN load increment: no accept step needed", lambda a: [s.h(fingers="imrp", c=(0.5, 0.55))])
    s.scene(0.9, "", nothing)
    s.scene(1.6, "Digits: hold up fingers, one digit at a time (3)", lambda a: [s.h(fingers="imr", c=(0.5, 0.55))])
    s.scene(0.7, "", nothing)
    s.scene(1.6, "(5)", lambda a: [s.h(fingers="timrp", c=(0.5, 0.55))])
    s.scene(0.7, "", nothing)
    s.scene(1.6, "(0): a fist", lambda a: [s.h(fingers="", c=(0.5, 0.55))])
    s.scene(0.8, "", nothing)
    s.scene(1.5, "Pinch and hold: delete the last digit", lambda a: [s.h("pinch", c=(0.5, 0.55))])
    s.scene(1.2, "...keep holding to delete more", lambda a: [s.h("pinch", c=(0.5, 0.55))])
    s.scene(0.8, "", nothing)
    return s


# ----------------------------------------------------------------------------- driving a page
def payload(hands):
    return [{"landmarks": h.landmarks.tolist(), "handedness": h.handedness, "score": 0.97} for h in hands]


async def play(page, script: Script, feed_js: str, label: str, hud_js: str | None = None, t: float = 0.0):
    for secs, caption, fn in script.scenes:
        await page.evaluate(CAPTION_JS, caption)
        n = max(int(secs * FPS), 1)
        for i in range(n):
            script.t = t
            hands = fn(i / max(n - 1, 1))
            await page.evaluate(feed_js, [payload(hands), t * 1000])
            await page.wait_for_timeout(int(1000 / FPS))
            t += 1 / FPS
        if hud_js:
            print(f"    {caption[:58]:58s} -> {await page.evaluate(hud_js)}")
    await page.evaluate(CAPTION_JS, "")
    print(f"  {label}: {t:.1f} s of scripted hands")
    return t


def assets_router(assets: Path | None):
    pk = {"three@0.160.0/": "three", "tasks-vision@0.10.14/": "mp", "react@18.3.1/": "react-18.3.1",
          "react-dom@18.3.1/": "react-dom-18.3.1", "@babel/standalone@7.29.0/": "babel-standalone-7.29.0"}

    async def route(r):
        url = r.request.url.split("?")[0]
        if assets is not None:
            for k, d in pk.items():
                if k in url and (assets / d / "package").exists():
                    f = assets / d / "package" / url.split(k)[1]
                    ctype = "application/wasm" if f.suffix == ".wasm" else "application/javascript"
                    return await r.fulfill(path=str(f), content_type=ctype)
            if "hand_landmarker.task" in url and (assets / "hand_landmarker.task").exists():
                return await r.fulfill(path=str(assets / "hand_landmarker.task"))
        await r.continue_()
    return route


def encode(webm: Path, out_dir: Path, name: str, start: float = 0.0, gif_width=760, gif_fps=9):
    mp4, gif = out_dir / f"{name}.mp4", out_dir / f"{name}.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    pal = out_dir / f"{name}.palette.png"
    vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(webm), "-vf", vf + ",palettegen=max_colors=96:stats_mode=diff",
                    str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(webm), "-i", str(pal), "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif)], check=True)
    pal.unlink()
    print(f"  wrote {mp4.name} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif.name} ({gif.stat().st_size / 1e6:.1f} MB)")


async def record(args):
    from playwright.async_api import async_playwright
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_video"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    server = None
    url = args.url
    if url is None:
        port = args.port
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "webdemo.server:app", "--port", str(port)], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://localhost:{port}"
        for _ in range(60):
            try:
                import urllib.request
                urllib.request.urlopen(url + "/healthz", timeout=1); break
            except Exception:
                time.sleep(0.5)
    try:
        async with async_playwright() as p:
            launch = {"args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
            if args.chromium:
                launch["executable_path"] = args.chromium
            browser = await p.chromium.launch(**launch)
            route = assets_router(Path(args.assets) if args.assets else None)

            async def new_page():
                ctx = await browser.new_context(viewport=SIZE, record_video_dir=str(tmp), record_video_size=SIZE)
                page = await ctx.new_page()
                page.on("pageerror", lambda e: print("  page error:", e))
                await page.route("https://cdn.jsdelivr.net/**", route)
                await page.route("https://unpkg.com/**", route)
                await page.route("https://storage.googleapis.com/**", route)
                return ctx, page

            # ---- Workbench: results of a solved bracket, explored with two hands -----------------------
            if "workbench" in args.only:
                print("Workbench demo")
                ctx, page = await new_page()
                t_ctx = time.monotonic()
                await page.goto(url + "/workbench/?tour")
                await page.wait_for_function("window.__workbench && window.__workbench.mesh", timeout=60000)
                await page.wait_for_function("window.__workbench.live && window.__workbench.live.ok", timeout=60000)
                # the ?tour page replays its own scripted hand; switch that off so only ours reach the engine
                await page.evaluate("() => { const L = window.__workbench.live; L.tour = false; L.ws.send(JSON.stringify({type:'demo_stop'})); }")
                await page.wait_for_timeout(1500)
                await page.evaluate("""async () => {
                    const w = window.__workbench; w.newPart('bracket'); w.setState({ engine: 'auto', analysis: 'static' });
                    await new Promise(r => setTimeout(r, 1200)); w.startSolve();
                    for (let i = 0; i < 200 && w.state.solving; i++) await new Promise(r => setTimeout(r, 300));
                    await new Promise(r => setTimeout(r, 400));
                    w.setState({ stage: 6, field: 'vm' }); w.resetView();
                }""")
                await page.wait_for_timeout(1500)
                feed = """([hands, t]) => { const L = window.__workbench.live; if (!L || !L.ws || L.ws.readyState !== 1) return;
                    L.hands = hands.map(h => h.landmarks.map(p => [p[0], p[1]]));
                    L.ws.send(JSON.stringify({ type: 'frame', t, seq: (L.seq = (L.seq || 0) + 1), hands, mirrored: true })); }"""
                t_play = time.monotonic()
                hud = "() => { const h = window.__workbench.live.hud; return h ? [h.mode, h.state, h.hands.map(x => x.pose)].join(' ') : null; }"
                t_end = await play(page, workbench_script(), feed, "workbench", hud)
                # aim the probe at the model: from the tip of the arm to the highest stress, as image coordinates of the fingertip
                pts = await page.evaluate("""() => { const w = window.__workbench, L = w.mesh, R = w.res, box = [0.2, 0.15, 0.8, 0.75];
                    const at = n => { const p = w.proj([L.px[n], L.py[n], L.pz[n]]); return [p[0] / w.w, p[1] / w.h]; };
                    return [R.argmax.umag, R.argmax.vm].map(at).map(([u, v]) => [box[0] + u * (box[2] - box[0]), box[1] + v * (box[3] - box[1])]); }""")
                print("  probe path (image coords):", [[round(v, 2) for v in q] for q in pts])
                await play(page, probe_script(pts[0], pts[1]), feed, "probe", hud, t=t_end + 0.5)
                await page.wait_for_timeout(500)
                await ctx.close()
                vid = sorted(tmp.glob("*.webm"), key=lambda f: f.stat().st_mtime)[-1]
                encode(vid, out, "workbench-gestures", start=t_play - t_ctx - 0.4)

            # ---- gesture modeller: finger digits, choices and backspace ----------------------------------
            if "modeller" in args.only:
                print("Modeller demo")
                ctx, page = await new_page()
                t_ctx = time.monotonic()
                await page.goto(url + "/")
                await page.wait_for_timeout(2500)
                await page.click("#btn-skip")
                await page.wait_for_timeout(500)
                await page.evaluate("() => window.__archimedes.send({type:'model_action', action:{op:'new_rect'}})")
                await page.wait_for_timeout(1000)
                t_play = time.monotonic()
                await play(page, modeller_script(), "([hands, t]) => window.__archimedes.feed(hands, t)", "modeller",
                           "() => document.getElementById('prompt-label').textContent + ' | ' + document.getElementById('prompt-input').value")
                await page.wait_for_timeout(500)
                await ctx.close()
                vid = sorted(tmp.glob("*.webm"), key=lambda f: f.stat().st_mtime)[-1]
                encode(vid, out, "modeller-finger-input", start=t_play - t_ctx - 0.4)
            await browser.close()
    finally:
        if server:
            server.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"))
    ap.add_argument("--url", help="use a running server instead of starting one")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--only", nargs="+", default=["workbench", "modeller"], choices=["workbench", "modeller"])
    ap.add_argument("--assets", help="folder with local copies of the CDN packages (offline machines)")
    ap.add_argument("--chromium", help="path to a Chromium binary (default: Playwright's)")
    asyncio.run(record(ap.parse_args()))


if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required to write the videos")
    main()
