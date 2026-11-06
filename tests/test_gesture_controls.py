"""Gesture controls on synthetic hands: two-hand zoom / pan / turn / reset / compare, straight section cuts, the probe
click, the field dial, finger digits and delete. Uses the scenarios of scripts/gesture_bench.py, so what the benchmark prints is what is asserted here."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gesture_bench as gb  # noqa: E402
from archimedes_gestures.counting import NumberEntry  # noqa: E402
from archimedes_gestures.landmarks import FrameInput  # noqa: E402

FPS = gb.FPS


def feed(pipe, seq, t0=0.0):
    """seq: [(seconds, hands_fn(t) -> [HandFrame])]; returns (results, end time)."""
    out, t = [], t0
    for dur, fn in seq:
        for _ in range(int(dur * FPS)):
            out.append((t, pipe.process(FrameInput(t, fn(t)))))
            t += 1 / FPS
    return out, t


def test_controls_respond_quickly_and_track_the_hand():
    z = gb.bench_zoom()
    assert z["one_hand_zoom_commands"] == 0                            # a lone pinch never zooms: no pinch/point mix-up
    assert z["two_hand_first_response_ms"] <= 250
    assert abs(z["two_hand_total_zoom"] - z["two_hand_expected_zoom"]) < 0.05   # 40 % spread -> 1/1.4 distance
    p = gb.bench_probe()
    assert p["first_response_ms"] <= 200 and p["lag_ms"] <= 40 and p["mean_error_pct"] < 2.5
    assert p["still_jitter_pct"] < 0.3


def test_two_pinching_hands_zoom_and_pan_two_fists_turn():
    rng = np.random.default_rng(1)
    pipe = gb.make_pipeline()
    spread = lambda t: [gb.hand("pinch", center=(0.5 - 0.15 - 0.1 * min(t, 1), 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                        gb.hand("pinch", center=(0.5 + 0.15 + 0.1 * min(t, 1), 0.55), t=t, rng=rng)]
    res, _ = feed(pipe, [(1.6, spread)])
    kinds = [c["kind"] for _, r in res for c in r["commands"]]
    assert "zoom" in kinds and any(r["hud"]["mode"] == "grab" for _, r in res)
    assert pipe.view.distance < 0.8                                    # spreading zooms in
    drag = lambda t: [gb.hand("pinch", center=(0.35 + 0.15 * min(t, 1), 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                      gb.hand("pinch", center=(0.65 + 0.15 * min(t, 1), 0.55), t=t, rng=rng)]
    pipe2 = gb.make_pipeline()
    az0 = pipe2.view.azimuth
    feed(pipe2, [(1.6, drag)])
    assert abs(pipe2.view.focal_point[0]) + abs(pipe2.view.focal_point[1]) > 0.05   # moving both pinches pans
    assert pipe2.view.azimuth == az0 and abs(np.log(pipe2.view.distance)) < 0.1
    fists = lambda t: [gb.hand("fist", center=(0.35 + 0.15 * min(t, 1), 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                       gb.hand("fist", center=(0.65 + 0.15 * min(t, 1), 0.55), t=t, rng=rng)]
    pipe3 = gb.make_pipeline()
    res, _ = feed(pipe3, [(1.6, fists)])
    assert any(r["hud"]["mode"] == "turn" for _, r in res) and abs(pipe3.view.azimuth - az0) > 5


def test_both_thumbs_up_resets_and_one_thumb_does_not():
    rng = np.random.default_rng(6)
    pipe = gb.make_pipeline()
    pipe.view.azimuth, pipe.view.distance = 10.0, 0.5
    one = lambda t: [gb.hand("thumbs_up", center=(0.6, 0.55), t=t, rng=rng)]  # noqa: E731
    both = lambda t: [gb.hand("thumbs_up", center=(0.35, 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                      gb.hand("thumbs_up", center=(0.65, 0.55), t=t, rng=rng)]
    res, _ = feed(pipe, [(0.3, lambda t: []), (1.5, both)])
    resets = [c for _, r in res for c in r["commands"] if c["kind"] == "reset"]
    assert len(resets) == 1 and pipe.view.distance == 1.0              # fires once while held
    res, _ = feed(gb.make_pipeline(), [(0.3, lambda t: []), (1.5, one)])
    kinds = [c["kind"] for _, r in res for c in r["commands"]]
    assert "reset" not in kinds and "snapshot" in kinds                # one thumb is the snapshot


def test_section_cut_is_always_straight_and_never_armed_by_two_raised_hands():
    s = gb.bench_section()
    assert s["section_frames"] > 0
    assert s["off_axis_deg_p95"] < 0.5 and s["plane_wobble_deg_p95"] < 0.5 and s["axis_flips"] == 0
    rng = np.random.default_rng(2)
    pipe = gb.make_pipeline()
    tilted = lambda t: [gb.hand("open_palm", center=(0.5, 0.55), t=t, rng=rng, yaw=0.5, pitch=0.25)]  # noqa: E731
    feed(pipe, [(0.3, lambda t: [gb.hand("none", center=(0.5, 0.55), t=t, rng=rng)]), (2.0, tilted)])
    n = np.abs(pipe.view.plane_normal)
    assert pipe.view.section_on and np.isclose(n.max(), 1.0)          # a tilted palm still cuts straight
    pipe = gb.make_pipeline()
    both = lambda t: [gb.hand("open_palm", center=(0.5, 0.55), t=t, rng=rng),  # noqa: E731
                      gb.hand("point", center=(0.85, 0.55), handedness="Left", t=t, rng=rng)]
    res, _ = feed(pipe, [(0.3, lambda t: []), (2.0, both)])
    assert not pipe.view.section_on                                    # the other hand is raised: no cut


def test_two_pointing_hands_compare_two_points():
    rng = np.random.default_rng(7)
    pipe = gb.make_pipeline()
    dual = lambda t: [gb.hand("point", center=(0.35, 0.5), handedness="Left", t=t, rng=rng),  # noqa: E731
                      gb.hand("point", center=(0.65, 0.5), t=t, rng=rng)]
    res, _ = feed(pipe, [(0.3, lambda t: []), (1.0, dual)])
    assert any(r["hud"]["mode"] == "dual" for _, r in res)
    assert pipe.view.probe_cursor is not None and pipe.view.probe_cursor2 is not None
    assert abs(pipe.view.probe_cursor[0] - pipe.view.probe_cursor2[0]) > 0.2


def test_field_dial_steps_one_field_per_move_and_shows_the_strip():
    rng = np.random.default_rng(8)
    pipe = gb.make_pipeline()
    pipe.view.fields = ["von_mises", "displacement", "sxx", "szz"]
    still = lambda t: [gb.hand("peace", center=(0.45, 0.55), t=t, rng=rng)]  # noqa: E731
    res, t = feed(pipe, [(0.3, lambda t: []), (0.6, still)])
    assert res[-1][1]["hud"].get("field_strip", {}).get("index") == 0
    move = lambda tt: [gb.hand("peace", center=(0.45 + 0.12 * 1.0 * min((tt - t) / 0.6, 1), 0.55), t=tt, rng=rng)]  # noqa: E731
    res, _ = feed(pipe, [(1.0, move)], t)
    assert pipe.view.field_index % 4 == 1                              # one palm width: one step (0.7), not two


def test_pinching_the_other_hand_pins_the_probe_value():
    rng = np.random.default_rng(3)
    pipe = gb.make_pipeline()
    point = lambda t: [gb.hand("point", center=(0.5, 0.5), t=t, rng=rng)]  # noqa: E731
    click = lambda t: [gb.hand("point", center=(0.5, 0.5), t=t, rng=rng),  # noqa: E731
                       gb.hand("pinch", center=(0.2, 0.6), handedness="Left", t=t, rng=rng)]
    res, _ = feed(pipe, [(0.5, point), (0.6, click), (0.5, point), (0.6, click)])
    pins = [c for _, r in res for c in r["commands"] if c["kind"] == "probe_pin"]
    assert len(pins) == 2                                              # one pin per click, not one per frame


def test_finger_count_answers_a_choice_prompt_without_an_accept_step():
    rng = np.random.default_rng(4)
    pipe = gb.make_pipeline("model")
    pipe.set_prompt({"action": "setup", "key": "inc_load", "step": 3, "label": "Load increment", "unit": "", "default": 2,
                     "choices": {1: "0.5 kN", 2: "1 kN", 3: "5 kN", 4: "10 kN", 5: "50 kN"}, "increment": None})
    res, _ = feed(pipe, [(1.5, lambda t: [gb.hand(fingers="imr", center=(0.5, 0.55), t=t, rng=rng)])])
    enters = [c for _, r in res for c in r["commands"] if c["kind"] == "number_enter"]
    assert len(enters) == 1 and enters[0]["data"]["value"] == 3.0


def test_keyboard_text_is_passed_to_the_gesture_digits():
    n = NumberEntry()
    n.digits = "305"
    n.set_text("30")                                                   # the keyboard's Backspace
    assert n.digits == "30" and n.value == 30.0
    n.set_text("1a2.5.7")
    assert n.digits == "12.57"


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_finger_counts_become_digits_in_about_half_a_second(count):
    rng = np.random.default_rng(count)
    pipe = gb.make_pipeline("model")
    pipe.set_prompt({"action": "x", "key": "k", "step": 1, "label": "Load", "unit": "kN", "default": None,
                     "choices": None, "increment": None})
    res, _ = feed(pipe, [(1.5, lambda t: [gb.hand(fingers=gb.COUNT_FINGERS[count], center=(0.5, 0.55), t=t, rng=rng)])])
    digs = [(t, c["data"]["digit"]) for t, r in res for c in r["commands"] if c["kind"] == "number_digit"]
    assert digs and digs[0][1] == count and digs[0][0] < 0.8


def test_pinch_and_hold_deletes_digits_and_never_types_one():
    r = gb.bench_pinch_delete()
    assert r["first_delete_ms"] is not None and r["first_delete_ms"] < 900
    assert r["deletes"] >= 2 and r["left"] == "3"        # held on: repeats; the pinch itself added no digit


def test_losing_one_hand_during_a_two_hand_grab_does_not_crash_or_jump():
    rng = np.random.default_rng(5)
    pipe = gb.make_pipeline()
    both = lambda t: [gb.hand("pinch", center=(0.35, 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                      gb.hand("pinch", center=(0.65, 0.55), t=t, rng=rng)]
    one = lambda t: [gb.hand("pinch", center=(0.65, 0.55), t=t, rng=rng)]  # noqa: E731
    res, t = feed(pipe, [(0.8, both), (0.1, one), (0.6, both)])
    d0 = pipe.view.distance
    feed(pipe, [(0.3, one), (0.6, lambda t: [gb.hand("none", center=(0.5, 0.55), t=t, rng=rng)])], t)
    assert abs(pipe.view.distance - d0) < 0.01                        # no jump when one hand left (a lone pinch zooms by itself)
