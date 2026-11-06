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


# ---- MODEL wizard: plane -> shape -> two-hand size -> extrude -> confirm ----------------------------------------------
def fingers(spec, center=(0.5, 0.55), handedness="Right"):
    return lambda a, tt: [hand(S.canonical_fingers(spec), center, handedness, tt)]


def pose(name, center=(0.5, 0.55), handedness="Right", **kw):
    canon = S.canonical_hand(name)
    return lambda a, tt: [hand(canon, center, handedness, tt, **kw)]


def pull(dx, dy, s=0.12):
    """Both hands pinched, dx / dy palm widths apart (pinch points), held."""
    cl, cr = S.canonical_hand("pinch"), S.canonical_hand("pinch")
    return lambda a, tt: [hand(cl, (0.5 - dx * s / 2, 0.55 - dy * s / 2), "Left", tt),
                          hand(cr, (0.5 + dx * s / 2, 0.55 + dy * s / 2), "Right", tt)]


NOHANDS = lambda a, tt: []  # noqa: E731


def drive(p, steps):
    t, cmds = 0.0, []
    for fn, dur in steps:
        frames, t = clip(fn, dur, t)
        cmds += run(p, frames)
    return cmds


def test_model_wizard_side_plane_rectangle_then_extrude_then_confirm():
    p = pipe("model")
    p.set_prompt(None)
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("im"), 1.2), (NOHANDS, 0.2), (fingers("i"), 1.2), (NOHANDS, 0.2),
                     (pull(3, 2), 2.5), (NOHANDS, 0.3), (pull(5, 1), 2.5), (NOHANDS, 0.3), (pose("thumbs_up"), 1.5)])
    assert [c["data"]["plane"] for c in cmds if c["kind"] == "wiz_plane"] == ["side"]
    steps = [c["data"]["step"] for c in cmds if c["kind"] == "wiz_step"]
    assert steps[:4] == ["shape", "draw_rect", "extrude", "confirm"]
    made = [c["data"] for c in cmds if c["kind"] == "create_member"]
    assert len(made) == 1
    m = made[0]
    assert m["kind"] == "rect" and m["orientation"] == "horizontal"
    assert m["b"] % 25 == 0 and m["h"] % 25 == 0 and m["L"] % 100 == 0 and m["L"] > 2000
    assert p.modeling.step == "plane"                   # ready for the next member


def test_a_sizing_step_needs_a_fresh_pinch():
    p = pipe("model")
    p.modeling.step = "draw_rect"
    p.modeling._step_reset()
    drive(p, [(pull(3, 2), 0.3)])                     # pinching when the step starts (carried over from the last one)
    p.modeling._step_reset()
    cmds = drive(p, [(pull(3, 2), 1.5)])
    assert "wiz_size" not in kinds(cmds)              # nothing happens until the fingers open
    cmds = drive(p, [(NOHANDS, 0.2), (pull(3, 2), 2.5)])
    assert [c for c in cmds if c["kind"] == "wiz_size" and c["data"]["locked"]]


def test_wider_pull_gives_a_bigger_section_on_the_increment():
    sizes = []
    for dx in (2.0, 4.0):
        p = pipe("model")
        p.modeling.step = "draw_circle"
        p.modeling._step_reset()
        cmds = drive(p, [(NOHANDS, 0.2), (pull(dx, 0), 2.5)])
        locked = [c["data"]["d"] for c in cmds if c["kind"] == "wiz_size" and c["data"]["locked"]]
        assert locked and locked[0] % 25 == 0
        sizes.append(locked[0])
    assert sizes[1] > sizes[0] * 1.5


def test_front_plane_line_then_circle_makes_a_round_column():
    p = pipe("model")
    p.set_prompt(None)
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("imr"), 1.2), (NOHANDS, 0.2),
                     (pose("point", (0.45, 0.7)), 1.5), (NOHANDS, 0.2), (pose("point", (0.45, 0.35)), 1.5)])
    picks = [c["data"]["slot"] for c in cmds if c["kind"] == "wiz_pick"]
    assert picks == ["p0", "p1"] and p.modeling.step == "shape"
    p.modeling.set_value("L", 3000.0)                   # the page measured the line on its grid
    p.modeling.set_value("orientation", "vertical")
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("im"), 1.2), (NOHANDS, 0.2), (pull(3, 0), 2.5), (NOHANDS, 0.3),
                     (pose("thumbs_up"), 1.5)])
    made = [c["data"] for c in cmds if c["kind"] == "create_member"]
    assert made and made[0]["kind"] == "circle" and made[0]["orientation"] == "vertical" and made[0]["L"] == 3000.0


def test_two_fists_go_back_one_step():
    p = pipe("model")
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("im"), 1.2)])
    assert p.modeling.step == "shape"
    fists = lambda a, tt: [hand(S.canonical_hand("fist"), (0.3, 0.55), "Left", tt),  # noqa: E731
                           hand(S.canonical_hand("fist"), (0.7, 0.55), "Right", tt)]
    cmds = drive(p, [(fists, 1.0)])
    assert p.modeling.step == "plane" and [c for c in cmds if c["kind"] == "wiz_step" and c["data"].get("back")]


def test_a_count_held_from_before_is_not_read_again():
    p = pipe("model")
    drive(p, [(NOHANDS, 0.2), (fingers("i"), 1.2)])    # 1 = plan plane
    assert p.modeling.step == "shape"
    drive(p, [(fingers("i"), 1.5)])                     # still 1: must not pick the rectangle by itself
    assert p.modeling.step == "shape"
    drive(p, [(NOHANDS, 0.2), (fingers("i"), 1.2)])
    assert p.modeling.step == "draw_rect"


# ---- LOADS wizard: what -> start -> (end) -> direction -> value ------------------------------------------------------
def test_support_is_placed_after_the_start_point():
    p = pipe()
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("i"), 1.2), (NOHANDS, 0.2), (pose("point", (0.4, 0.5)), 1.6)])
    places = [c["data"] for c in cmds if c["kind"] == "wiz_place"]
    assert places == [{"op": "support_fixed"}] and p.modeling.step == "menu"
    assert [c["data"]["slot"] for c in cmds if c["kind"] == "wiz_pick"] == ["t0"]


def test_pinching_the_other_hand_confirms_a_point_at_once():
    p = pipe()
    drive(p, [(NOHANDS, 0.2), (fingers("im"), 1.2), (NOHANDS, 0.2)])
    assert p.modeling.step == "start"
    pt, pi = S.canonical_hand("point"), S.canonical_hand("pinch")
    moving = lambda a, tt: [hand(pt, (0.3 + 0.3 * a, 0.5), "Right", tt)]  # noqa: E731
    click = lambda a, tt: [hand(pt, (0.6, 0.5), "Right", tt), hand(pi, (0.2, 0.6), "Left", tt)]  # noqa: E731
    cmds = drive(p, [(moving, 0.6), (click, 0.3)])
    assert [c["data"]["op"] for c in cmds if c["kind"] == "wiz_place"] == ["support_pinned"]


def test_point_load_needs_start_and_direction():
    p = pipe()
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("imrp"), 1.2), (NOHANDS, 0.2), (pose("point", (0.6, 0.5)), 1.6),
                     (NOHANDS, 0.2), (pose("point", (0.6, 0.5), roll=np.pi), 1.8)])
    steps = [c["data"]["step"] for c in cmds if c["kind"] == "wiz_step"]
    assert steps[:2] == ["start", "dir"]
    d = [c["data"]["vec"] for c in cmds if c["kind"] == "wiz_pick" and c["data"]["slot"] == "dir"]
    assert d and d[0][1] > 0.8                          # pointing down the screen
    assert [c["data"]["op"] for c in cmds if c["kind"] == "wiz_place"] == ["point_load"]


def test_uniform_load_has_a_start_and_an_end():
    p = pipe()
    cmds = drive(p, [(NOHANDS, 0.2), (fingers("timrp"), 1.2), (NOHANDS, 0.2), (pose("point", (0.35, 0.5)), 1.6),
                     (NOHANDS, 0.2), (pose("point", (0.65, 0.5)), 1.6), (NOHANDS, 0.2),
                     (pose("point", (0.6, 0.5), roll=np.pi), 1.8)])
    assert [c["data"]["slot"] for c in cmds if c["kind"] == "wiz_pick"] == ["t0", "t1", "dir"]
    assert [c["data"]["op"] for c in cmds if c["kind"] == "wiz_place"] == ["uniform_load"]


def test_thumbs_up_hold_solves_once():
    cmds = run(pipe(), clip(lambda a, tt: [hand(S.canonical_hand("thumbs_up"), (0.5, 0.55), t=tt)], 2.5)[0])
    assert kinds(cmds).count("solve") == 1


# ---- session ---------------------------------------------------------------------------
def test_session_creates_a_member_sized_by_the_wizard():
    from archimedes_fe.session import ModelSession

    s = ModelSession()
    s.settings.update(length=250.0, section=50.0)
    s.create_member("rect", "vertical", L=3337, b=212, h=480)
    m = s.model.member
    assert (m.section.b, m.section.h, m.length, m.orientation) == (200, 500, 3250, "vertical")
    assert s.phase == "loads" and "column(b=200, h=500, L=3250)" in s.journal[-1]["cmd"]
    s.create_member("circle", "horizontal", L=4000, d=330)
    assert s.model.member.section.d == 350 and s.model.member.orientation == "horizontal"
    s.undo()
    assert s.model.member.section.kind == "rect"


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
