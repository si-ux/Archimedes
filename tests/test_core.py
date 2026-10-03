import numpy as np
import pytest

from archimedes_gestures import synthetic as S
from archimedes_gestures.classifier import PersonalAdapter, PoseSmoother, RulePoseClassifier
from archimedes_gestures.dynamic import DTWTemplateMatcher, SwipeDetector, dtw_distance
from archimedes_gestures.features import FEATURE_DIM, extract
from archimedes_gestures.filters import OneEuroFilter
from archimedes_gestures.geometry import palm_normal
from archimedes_gestures.landmarks import HandFrame, normalize


def test_normalize_invariant_to_position_scale_roll_and_hand():
    canon = S.canonical_hand("peace", np.random.default_rng(0))
    a = S.to_image(canon, center=(0.3, 0.4), palm_size=0.1, roll=0.0)
    b = S.to_image(canon, center=(0.7, 0.6), palm_size=0.2, roll=0.7)
    c = S.to_image(canon, center=(0.5, 0.5), palm_size=0.15, roll=-0.4, handedness="Left")
    na, nb, nc = normalize(a), normalize(b), normalize(c, "Left")
    assert np.allclose(na, nb, atol=1e-6)
    assert np.allclose(na, nc, atol=1e-6)
    assert np.allclose(na[9], [0, -1, 0], atol=1e-6)  # middle MCP straight "up"


def test_feature_dim():
    h = S.random_hand("fist", np.random.default_rng(0))
    assert extract(h.landmarks, h.handedness).shape == (FEATURE_DIM,)


def test_rule_classifier_on_synthetic_hands():
    rng = np.random.default_rng(3)
    clf = RulePoseClassifier()
    for pose in S.POSES:
        hits = sum(clf.predict(h.landmarks, h.handedness)[0] == pose
                   for h in (S.random_hand(pose, rng) for _ in range(60)))
        assert hits / 60 > 0.9, pose


def test_palm_normal_points_at_camera_for_both_hands():
    for hand in ("Right", "Left"):
        lm = S.to_image(S.canonical_hand("open_palm"), handedness=hand)
        n = palm_normal(HandFrame(lm, hand))
        assert n[2] < -0.9  # MediaPipe z negative = towards camera


def test_palm_normal_follows_yaw():
    lm = S.to_image(S.canonical_hand("open_palm"), yaw=np.radians(60))
    n = palm_normal(HandFrame(lm, "Right"))
    assert abs(n[0]) > 0.7


def test_one_euro_reduces_jitter_and_tracks_motion():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(1.0, 0.05)
    t = np.arange(0, 2, 1 / 30)
    noisy = 0.5 + rng.normal(0, 0.01, len(t))
    out = np.array([f(x, ti) for x, ti in zip(noisy, t)])
    assert out[15:].std() < noisy[15:].std() / 2
    f.reset()
    ramp = [f(ti, ti) for ti in t]
    assert abs(ramp[-1] - t[-1]) < 0.15


def test_smoother_hysteresis_ignores_single_bad_frames():
    sm = PoseSmoother()
    fist = {"fist": 0.9, "none": 0.1}
    palm = {"open_palm": 0.9, "none": 0.1}
    seq = [fist] * 8 + [palm] + [fist] * 3
    out = [sm.update(p)[0] for p in seq]
    assert out[5] == "fist" and all(o == "fist" for o in out[5:])
    for _ in range(8):
        cur, _ = sm.update(palm)
    assert cur == "open_palm"


def test_personal_adapter_learns_a_custom_hand():
    """A user whose 'point' looks odd to the rules gets fixed by a few samples."""
    rng = np.random.default_rng(1)
    base = RulePoseClassifier(finger_thr=1.5)  # deliberately bad: thinks extended fingers are curled
    weird = [S.random_hand("point", rng, "Right") for _ in range(40)]
    assert sum(base.predict(h.landmarks)[0] == "point" for h in weird[30:]) < 5
    ad = PersonalAdapter(base)
    for h in weird[:20]:
        ad.add_sample(h.landmarks, "point")
    for pose in ("fist", "open_palm"):
        for _ in range(20):
            ad.add_sample(S.random_hand(pose, rng, "Right").landmarks, pose)
    assert sum(ad.predict(h.landmarks)[0] == "point" for h in weird[30:]) >= 8


def test_swipe_detector_fires_once_per_swipe():
    sd = SwipeDetector()
    fires = []
    t = 0.0
    for i in range(40):  # still
        fires.append(sd.update(t, np.array([0.5, 0.5]), 0.1))
        t += 1 / 30
    for i in range(10):  # fast swipe right: 0.4 image widths in 1/3 s
        fires.append(sd.update(t, np.array([0.5 + 0.04 * i, 0.5]), 0.1))
        t += 1 / 30
    for i in range(20):
        fires.append(sd.update(t, np.array([0.9, 0.5]), 0.1))
        t += 1 / 30
    assert [f for f in fires if f] == [1]


def test_dtw_matcher():
    t = np.linspace(0, 1, 20)
    circle = np.c_[np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]
    line = np.c_[t * 2, t * 0]
    m = DTWTemplateMatcher(threshold=0.5)
    m.add("circle", circle)
    m.add("swipe", line)
    t2 = np.linspace(0, 1, 27)
    assert m.classify(np.c_[np.cos(2 * np.pi * t2), np.sin(2 * np.pi * t2)])[0] == "circle"
    assert dtw_distance(line, line) == pytest.approx(0)
