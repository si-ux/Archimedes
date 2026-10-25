"""Gesture controls on synthetic hands: two-hand zoom, straight section cuts, the probe click, finger digits and
backspace. Uses the scenarios of scripts/gesture_bench.py, so what the benchmark prints is what is asserted here."""
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
    assert z["one_hand_first_response_ms"] <= 250 and z["two_hand_first_response_ms"] <= 250
    assert abs(z["two_hand_total_zoom"] - z["two_hand_expected_zoom"]) < 0.05   # 40 % spread -> 1/1.4 distance
    p = gb.bench_probe()
    assert p["first_response_ms"] <= 200 and p["lag_ms"] <= 40 and p["mean_error_pct"] < 2.5
    assert p["still_jitter_pct"] < 0.3


def test_two_pinching_hands_zoom_and_rotate():
    rng = np.random.default_rng(1)
    pipe = gb.make_pipeline()
    spread = lambda t: [gb.hand("pinch", center=(0.5 - 0.15 - 0.1 * min(t, 1), 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                        gb.hand("pinch", center=(0.5 + 0.15 + 0.1 * min(t, 1), 0.55), t=t, rng=rng)]
    res, _ = feed(pipe, [(1.6, spread)])
    kinds = [c["kind"] for _, r in res for c in r["commands"]]
    assert "zoom" in kinds and any(r["hud"]["mode"] == "grab" for _, r in res)
    assert pipe.view.distance < 0.8                                    # spreading zooms in
    az0 = pipe.view.azimuth
    drag = lambda t: [gb.hand("pinch", center=(0.35 + 0.15 * min(t, 1), 0.55), handedness="Left", t=t, rng=rng),  # noqa: E731
                      gb.hand("pinch", center=(0.65 + 0.15 * min(t, 1), 0.55), t=t, rng=rng)]
    pipe2 = gb.make_pipeline()
    feed(pipe2, [(1.6, drag)])
    assert abs(pipe2.view.azimuth - az0) > 5 and abs(np.log(pipe2.view.distance)) < 0.2   # moving together rotates


def test_section_cut_is_straight_unless_the_other_palm_is_shown():
    s = gb.bench_section()
    assert s["off_axis_deg_p95"] < 0.5 and s["plane_wobble_deg_p95"] < 0.5 and s["axis_flips"] == 0
    rng = np.random.default_rng(2)
    pipe = gb.make_pipeline()
    tilted = lambda t: [gb.hand("open_palm", center=(0.5, 0.55), t=t, rng=rng, yaw=0.5, pitch=0.25)]  # noqa: E731
    feed(pipe, [(0.3, lambda t: [gb.hand("none", center=(0.5, 0.55), t=t, rng=rng)]), (2.0, tilted)])
    n = np.abs(pipe.view.plane_normal)
    assert pipe.view.section_on and not pipe.view.section_free and np.isclose(n.max(), 1.0)
    pipe = gb.make_pipeline()
    both = lambda t: [gb.hand("open_palm", center=(0.5, 0.55), t=t, rng=rng, yaw=0.5, pitch=0.25),  # noqa: E731
                      gb.hand("open_palm", center=(0.85, 0.55), handedness="Left", t=t, rng=rng)]
    feed(pipe, [(0.3, lambda t: [gb.hand("none", center=(0.5, 0.55), t=t, rng=rng)]), (0.9, tilted), (1.5, both)])
    assert pipe.view.section_free and np.abs(pipe.view.plane_normal).max() < 0.99   # tilted with the palm


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


def test_backspace_by_a_relaxed_swipe_once_per_swipe_and_by_keyboard_text():
    b = gb.bench_backspace()
    assert b["relaxed swipe"] == 1 and b["quick flick"] == 1 and b["small wave"] >= 1
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
