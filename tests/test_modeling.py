"""Modelling gestures, finger-count numbers and the model session, on synthetic hands."""
import numpy as np

from archimedes_gestures import synthetic as S
from archimedes_gestures.landmarks import FrameInput, HandFrame
from archimedes_gestures.pipeline import GesturePipeline

FPS = 30


def pipe(phase="loads"):
    p = GesturePipeline(model_path=None, profile_path="/x/p.json", personal_path="/x/s.npz")
    p.set_phase(phase)
    return p


def hand(canon, center, handedness="Right", t=0.0, rng=None, **kw):
    lm = S.to_image(canon, center=center, palm_size=0.12, handedness=handedness, noise=0.003,
                    rng=rng or np.random.default_rng(0), **kw)
    return HandFrame(lm, handedness, 0.95, t)


def run(p, frames):
    cmds = []
    for f in frames:
        cmds += p.process(f)["commands"]
    return cmds


def clip(builder, seconds, t0=0.0):
    """builder(alpha, t) -> list[HandFrame]; returns frames and the end time."""
    n = int(seconds * FPS)
    frames = [FrameInput(t0 + i / FPS, builder(i / max(n - 1, 1), t0 + i / FPS)) for i in range(n)]
    return frames, t0 + n / FPS


def kinds(cmds):
    return [c["kind"] for c in cmds]


# ---- number entry -------------------------------------------------------------
def count_hand(n, center=(0.5, 0.55), handedness="Right", t=0.0):
    return hand(S.canonical_fingers(S.COUNT_FINGERS[n]), center, handedness, t)


def test_finger_counting_types_300_and_confirms():
    p = pipe()
    p.set_prompt({"action": "new_rect", "key": "b", "step": 1})
    t, cmds = 0.0, []
    for n, dur in ((3, 1.1), (0, 1.1), (5, 0.3), (0, 1.1)):
        frames, t = clip(lambda a, tt, n=n: [count_hand(n, t=tt)], dur, t)
        cmds += run(p, frames)
    frames, t = clip(lambda a, tt: [count_hand(5, (0.3, 0.55), "Right", tt), count_hand(5, (0.7, 0.55), "Left", tt)], 1.1, t)
    cmds += run(p, frames)
    digits = [c["data"]["digit"] for c in cmds if c["kind"] == "number_digit"]
    assert digits == [3, 0, 0]
    enter = [c for c in cmds if c["kind"] == "number_enter"]
    assert enter and enter[0]["data"]["value"] == 300


def test_two_hands_make_digits_above_five_and_two_fists_cancel():
    p = pipe()
    p.set_prompt({"action": "point_load", "key": "w0", "step": 1})
    frames, t = clip(lambda a, tt: [count_hand(5, (0.3, 0.55), "Right", tt), count_hand(3, (0.7, 0.55), "Left", tt)], 1.2)
    cmds = run(p, frames)
    assert [c["data"]["digit"] for c in cmds if c["kind"] == "number_digit"] == [8]
    frames, t = clip(lambda a, tt: [count_hand(0, (0.3, 0.55), "Right", tt), count_hand(0, (0.7, 0.55), "Left", tt)], 1.3, t)
    assert "number_cancel" in kinds(run(p, frames))


# ---- MODEL phase ------------------------------------------------------------------
def test_two_hand_pinch_pull_extrudes_a_beam():
    p = pipe("model")
    pin = S.canonical_hand("pinch")

    def pull(a, tt, gap):
        return [hand(pin, (0.5 - gap / 2, 0.55), "Right", tt), hand(pin, (0.5 + gap / 2, 0.55), "Left", tt)]

    f1, t = clip(lambda a, tt: pull(a, tt, 0.12), 0.4)
    f2, t = clip(lambda a, tt: pull(a, tt, 0.12 + 0.4 * a), 0.8, t)
    f3, t = clip(lambda a, tt: pull(a, tt, 0.52), 1.0, t)
    cmds = run(p, f1 + f2 + f3)
    ex = [c for c in cmds if c["kind"] == "extrude"]
    assert len(ex) == 1 and ex[0]["data"]["orientation"] == "horizontal"
    assert 1500 <= ex[0]["data"]["length_est"] <= 6000


def test_vertical_pull_extrudes_a_column():
    p = pipe("model")
    pin = S.canonical_hand("pinch")

    def pull(a, tt, gap):
        return [hand(pin, (0.45, 0.6 - gap / 2), "Right", tt), hand(pin, (0.55, 0.6 + gap / 2), "Left", tt)]

    frames = clip(lambda a, tt: pull(a, tt, 0.1), 0.4)[0] + clip(lambda a, tt: pull(a, tt, 0.1 + 0.25 * a), 0.8, 0.4)[0] \
        + clip(lambda a, tt: pull(a, tt, 0.35), 1.0, 1.2)[0]
    ex = [c for c in run(p, frames) if c["kind"] == "extrude"]
    assert ex and ex[0]["data"]["orientation"] == "vertical"


def test_drawing_a_circle_with_the_index_finger():
    p = pipe("model")
    pt = S.canonical_hand("point")
    still, t = clip(lambda a, tt: [hand(pt, (0.5, 0.6), t=tt)], 0.5)
    circ, t = clip(lambda a, tt: [hand(pt, (0.5 + 0.1 * np.sin(2 * np.pi * a), 0.6 - 0.1 * np.cos(2 * np.pi * a)), t=tt)],
                   1.6, t)
    cmds = run(p, still + circ)
    assert "circle" in kinds(cmds)
    # pointing still is not a circle
    assert "circle" not in kinds(run(pipe("model"), clip(lambda a, tt: [hand(pt, (0.5, 0.6), t=tt)], 3.0)[0]))


# ---- LOADS phase --------------------------------------------------------------------
def test_held_still_fist_places_a_fixed_support_but_dragging_rotates():
    p = pipe()
    fist = S.canonical_hand("fist")
    cmds = run(p, clip(lambda a, tt: [hand(fist, (0.4, 0.55), t=tt)], 1.2)[0])
    sup = [c for c in cmds if c["kind"] == "support"]
    assert len(sup) == 1 and sup[0]["data"]["op"] == "support_fixed"
    p = pipe()
    cmds = run(p, clip(lambda a, tt: [hand(fist, (0.3 + 0.3 * a, 0.55), t=tt)], 1.2)[0])
    assert "support" not in kinds(cmds) and "orbit" in kinds(cmds)


def test_pinch_and_v_place_pinned_and_roller():
    for pose, op in (("pinch", "support_pinned"), ("peace", "support_roller")):
        cmds = run(pipe(), clip(lambda a, tt, pose=pose: [hand(S.canonical_hand(pose), (0.5, 0.55), t=tt)], 1.2)[0])
        assert [c["data"]["op"] for c in cmds if c["kind"] == "support"] == [op]


def test_point_then_flick_down_gives_a_point_load():
    p = pipe()
    pt = S.canonical_hand("point")
    hold, t = clip(lambda a, tt: [hand(pt, (0.5, 0.5), t=tt)], 0.9)
    flick, t = clip(lambda a, tt: [hand(pt, (0.5, 0.5 + 0.2 * a), t=tt)], 0.3, t)
    pl = [c for c in run(p, hold + flick) if c["kind"] == "point_load"]
    assert len(pl) == 1 and pl[0]["data"]["flick"][1] > 0  # image y down = downward flick


def test_palm_sweep_gives_a_uniform_load_and_two_palms_a_trapezoid():
    p = pipe()
    palm = S.canonical_hand("open_palm")
    down = {"pitch": 1.2}  # palm turned to face the floor
    sweep, t = clip(lambda a, tt: [hand(palm, (0.3 + 0.4 * a, 0.55), t=tt, **down)], 0.8)
    stop, t = clip(lambda a, tt: [hand(palm, (0.7, 0.55), t=tt, **down)], 0.9, t)
    udl = [c for c in run(p, sweep + stop) if c["kind"] == "udl"]
    assert len(udl) == 1
    assert udl[0]["data"]["xy1"][0] - udl[0]["data"]["xy0"][0] > 0.3
    assert udl[0]["data"]["normal_view"][1] < -0.5  # palm faces down in view space

    p = pipe()
    two, _ = clip(lambda a, tt: [hand(palm, (0.3, 0.55), "Right", tt), hand(palm, (0.7, 0.5), "Left", tt)], 1.2)
    tr = [c for c in run(p, two) if c["kind"] == "trap"]
    assert len(tr) == 1 and tr[0]["data"]["xy0"][0] < tr[0]["data"]["xy1"][0]


def test_thumbs_up_hold_solves_once():
    cmds = run(pipe(), clip(lambda a, tt: [hand(S.canonical_hand("thumbs_up"), (0.5, 0.55), t=tt)], 2.5)[0])
    assert kinds(cmds).count("solve") == 1


# ---- session ---------------------------------------------------------------------------
def test_session_builds_a_beam_from_prompts_and_solves():
    from archimedes_fe.session import ModelSession

    s = ModelSession()
    s.begin("new_rect", orientation="horizontal", length_est=3500)
    assert s.prompt["key"] == "inc_length"  # increments are chosen before the first sketch
    for n in (3, 3, 2):  # 100 mm, 25 mm, 1 kN
        s.answer(n)
    assert s.setup_done and s.settings == {"length": 100.0, "section": 25.0, "load": 1.0}
    assert s.prompt["key"] == "b" and s.prompt["increment"] == 25.0
    s.answer(200)
    s.answer(400)
    assert s.prompt["default"] == 3500
    s.answer(None)  # accept the gesture's estimate
    m = s.model.member
    assert (m.section.b, m.section.h, m.length, m.orientation) == (200, 400, 3500, "horizontal")
    assert s.model.issues()
    s.begin("support_fixed", t=0.02)  # snaps to the end
    assert s.model.supports[0].t == 0.0
    s.begin("uniform_load", t0=0.1, t1=0.97, axis="z", sign=-1)
    s.answer(12)
    assert s.model.loads[0].t1 == 1.0 and s.model.loads[0].w0 == 12
    assert not s.model.issues()
    assert s.solve() is not None and s.phase == "results"
    s.undo()
    assert not s.model.loads


def test_session_circle_column_and_bad_answers():
    from archimedes_fe.session import ModelSession

    s = ModelSession()
    s.setup_done = True
    s.begin("new_circle")
    s.answer(3)  # not a choice
    assert s.error and s.prompt["key"] == "orientation"
    s.answer(1)
    s.answer(250)
    s.answer(4000)
    assert s.model.member.orientation == "vertical" and s.model.member.section.d == 250
    s.begin("point_load", t=1.0, axis="z", sign=-1)
    s.set_direction("x", 1)
    s.answer(5)
    assert s.model.loads[0].axis == "x"
    assert any("supports" in i for i in s.model.issues())
    s.begin("support_pinned", t=1.0)
    assert any("base" in i for i in s.model.issues())


def test_values_snap_to_the_chosen_increment():
    from archimedes_fe.session import ModelSession

    s = ModelSession()
    s.set_phase("model")  # entering Model asks for increments first
    assert s.prompt["action"] == "setup"
    for n in (4, 4, 3):  # 250 mm, 50 mm, 5 kN
        s.answer(n)
    assert s.prompt is None and "250" in s.message
    s.begin("new_rect", orientation="horizontal", length_est=3337)
    s.answer(212)   # -> 200 (50 mm steps)
    s.answer(480)   # -> 500
    s.answer(None)  # 3337 -> 3250 (250 mm steps)
    sec = s.model.member.section
    assert (sec.b, sec.h, s.model.member.length) == (200, 500, 3250)
    s.begin("point_load", t=1.0, axis="z", sign=-1)
    s.answer(12)    # -> 10 (5 kN steps)
    assert s.model.loads[-1].w0 == 10


def test_pinch_nudges_the_prompt_value_by_increments():
    p = pipe()
    p.set_prompt({"action": "new_rect", "key": "L", "step": 3, "default": 4000.0, "increment": 250.0, "kind": "length"})
    pin = S.canonical_hand("pinch")
    hold, t = clip(lambda a, tt: [hand(pin, (0.5, 0.65), t=tt)], 0.5)
    up, t = clip(lambda a, tt: [hand(pin, (0.5, 0.65 - 0.25 * a), t=tt)], 1.0, t)
    cmds = run(p, hold + up)
    sets = [c["data"]["digits"] for c in cmds if c["kind"] == "number_set"]
    assert sets and float(sets[-1]) > 4000 and float(sets[-1]) % 250 == 0
    assert "number_digit" not in kinds(cmds)  # a pinch never types a digit


def test_finger_states_are_reported_for_diagnostics():
    p = pipe()
    out = p.process(FrameInput(0.0, [count_hand(2)]))
    assert out["hud"]["fingers"]["Right"] == [False, True, True, False, False]
