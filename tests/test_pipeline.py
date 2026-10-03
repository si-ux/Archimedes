import numpy as np

from archimedes_gestures import synthetic as S
from archimedes_gestures.calibration import Calibrator
from archimedes_gestures.landmarks import FrameInput, HandFrame
from archimedes_gestures.pipeline import GesturePipeline
from archimedes_gestures.recorder import ClipRecorder, load_clips
from archimedes_gestures.view_state import Command, ViewState

FPS = 30


def scene(steps, hand="Right", rng=None):
    """steps: list of (pose, seconds, (x0,y0) -> (x1,y1)). Yields FrameInputs at 30 fps."""
    rng = rng or np.random.default_rng(0)
    t = 0.0
    for pose, dur, (p0, p1), *extra in steps:
        n = int(dur * FPS)
        canon = S.canonical_hand(pose, rng) if pose else None
        for i in range(n):
            a = i / max(n - 1, 1)
            c = (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a)
            hands = []
            if pose:
                kw = extra[0] if extra else {}
                lm = S.to_image(canon, center=c, palm_size=0.12, handedness=hand, noise=0.004, rng=rng, **kw)
                hands.append(HandFrame(lm, hand, 0.95, t))
            yield FrameInput(t, hands)
            t += 1 / FPS


def run(steps, **kw):
    pipe = GesturePipeline(model_path=None, profile_path="/nonexistent/p.json", personal_path="/nonexistent/s.npz")
    outs = [pipe.process(f) for f in scene(steps, **kw)]
    return pipe, outs


STILL = ((0.5, 0.5), (0.5, 0.5))


def test_relaxed_hand_never_moves_the_view():
    pipe, outs = run([("none", 2.0, ((0.4, 0.5), (0.6, 0.4)))])
    assert pipe.view.azimuth == ViewState().azimuth
    assert all(not o["commands"] for o in outs)


def test_fist_drag_orbits_then_release_stops():
    pipe, outs = run([
        ("fist", 0.5, STILL),
        ("fist", 0.7, ((0.5, 0.5), (0.65, 0.5))),  # drag right
        ("open_palm", 0.4, ((0.65, 0.5), (0.65, 0.5))),  # release (too short to arm section)
        ("open_palm", 0.2, ((0.65, 0.5), (0.45, 0.5))),  # move back while released
    ])
    states = [o["hud"]["state"] for o in outs]
    assert "arming" in states and "active" in states
    az_after_drag = None
    for o in outs:
        if o["hud"]["mode"] == "orbit":
            az_after_drag = o["view"]["azimuth"]
    assert az_after_drag is not None
    turned = (az_after_drag - ViewState().azimuth + 180) % 360 - 180
    assert -120 < turned < -20  # dragged right -> camera azimuth decreases
    assert pipe.view.azimuth == az_after_drag  # moving the open hand afterwards changed nothing


def test_section_needs_still_hold_and_locks_plane():
    pipe, outs = run([
        ("open_palm", 1.2, ((0.3, 0.5), (0.7, 0.5))),  # moving palm: must NOT arm a section
    ])
    assert not pipe.view.section_on
    pipe, outs = run([
        ("open_palm", 1.2, STILL),
        ("open_palm", 0.5, STILL, {"yaw": np.radians(70)}),  # palm turned sideways
        ("fist", 0.3, STILL),
    ])
    assert pipe.view.section_on and pipe.view.section_locked
    n = np.array(pipe.view.plane_normal)
    right, _, _ = pipe.view.camera_basis()
    assert abs(n @ right) > 0.6  # sideways palm -> plane normal along the screen's x axis


def test_two_open_palms_reset():
    pipe = GesturePipeline(model_path=None, profile_path="/x/p.json", personal_path="/x/s.npz")
    pipe.view.azimuth = 123.0
    rng = np.random.default_rng(0)
    canon = S.canonical_hand("open_palm", rng)
    for i in range(int(1.4 * FPS)):
        t = i / FPS
        hands = [HandFrame(S.to_image(canon, center=(0.3, 0.5), handedness="Right"), "Right", 0.9, t),
                 HandFrame(S.to_image(canon, center=(0.7, 0.5), handedness="Left"), "Left", 0.9, t)]
        out = pipe.process(FrameInput(t, hands))
    assert pipe.view.azimuth == ViewState().azimuth
    assert out["view"]["azimuth"] == ViewState().azimuth


def test_hand_at_edge_pauses_with_hint():
    pipe, outs = run([("fist", 1.0, ((0.02, 0.5), (0.02, 0.5)))])
    hud = outs[-1]["hud"]
    assert hud["state"] == "paused" and any("edge" in h for h in hud["hints"])


def test_probe_pins_after_hold():
    pipe, outs = run([("point", 0.6, STILL), ("point", 1.2, STILL)])
    assert pipe.view.probe_cursor is not None
    assert len(pipe.view.probe_pins) == 1


def test_view_state_undo_and_field_cycle():
    v = ViewState()
    v.apply(Command("field", {"step": 1}))
    assert v.field_name == "displacement"
    v.apply(Command("undo"))
    assert v.field_name == "von_mises"


def test_calibration_detects_swapped_handedness():
    cal = Calibrator(with_poses=False, comfort_s=1.0)
    rng = np.random.default_rng(0)
    canon = S.canonical_hand("open_palm", rng)
    t = 0.0
    while not cal.done and t < 10:
        # the user raised their right hand but the camera path calls it "Left"
        lm = S.to_image(canon, center=(0.5 + 0.1 * np.sin(t * 3), 0.5), handedness="Right")
        cal.update(FrameInput(t, [HandFrame(lm, "Left", 0.9, t)]))
        t += 1 / FPS
    assert cal.done and cal.profile.swap_handedness


def test_pipeline_applies_swap_from_calibration(tmp_path):
    pipe = GesturePipeline(model_path=None, profile_path=tmp_path / "p.json", personal_path=tmp_path / "s.npz")
    pipe.start_calibration(with_poses=True)
    rng = np.random.default_rng(0)
    t = 0.0
    while pipe.calibrator is not None and t < 30:
        step = pipe.calibrator.step
        pose = step.split(":")[1] if step.startswith("pose:") else "open_palm"
        lm = S.to_image(S.canonical_hand(pose, rng), center=(0.5 + 0.1 * np.sin(t), 0.5))
        f = pipe.make_frame([{"landmarks": lm.tolist(), "handedness": "Left"}], timestamp=t)
        pipe.process(f)
        t += 1 / FPS
    assert pipe.calibrator is None
    assert pipe.profile.swap_handedness
    assert (tmp_path / "p.json").exists() and (tmp_path / "s.npz").exists()
    assert set(pipe.classifier.counts()) == set(S.POSES)


def test_un_mirrored_input_is_flipped():
    pipe = GesturePipeline(model_path=None, profile_path="/x/p.json", personal_path="/x/s.npz")
    lm = S.to_image(S.canonical_hand("fist"), center=(0.2, 0.5))
    f = pipe.make_frame([{"landmarks": lm.tolist(), "handedness": "Right"}], timestamp=0, mirrored=False)
    assert abs(f.hands[0].palm_center[0] - 0.8) < 0.05


def test_recorder_roundtrip(tmp_path):
    rec = ClipRecorder(tmp_path)
    rec.start("fist", "alice")
    for f in scene([("fist", 1.0, STILL)]):
        rec.add(f)
    path = rec.stop()
    assert path is not None and path.exists()
    lms, hands, labels, subjects, clips = load_clips(tmp_path)
    assert lms.shape[1:] == (21, 3) and set(labels) == {"fist"} and set(subjects) == {"alice"}
    assert len(lms) < 30  # settle time dropped


def test_engine_handles_hand_loss_gracefully():
    pipe, outs = run([("fist", 0.8, STILL), (None, 0.6, STILL)])
    assert outs[-1]["hud"]["state"] == "no_hand"
    assert pipe.engine.mode is None


def test_scripted_tour_exercises_every_mode():
    pipe = GesturePipeline(model_path=None, profile_path="/x/p.json", personal_path="/x/s.npz")
    modes = set()
    for t, hand, _ in S.scripted_tour():
        out = pipe.process(FrameInput(t, [hand]))
        if out["hud"]["mode"]:
            modes.add(out["hud"]["mode"])
    assert {"orbit", "zoom", "section", "probe", "field"} <= modes
    v = pipe.view
    assert v.distance < 1.0 and v.section_locked and v.probe_pins and v.field_index != 0
