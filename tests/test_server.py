import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from archimedes_gestures import synthetic as S  # noqa: E402
from webdemo import server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)  # keep profile/clips out of the repo
    return TestClient(server.app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Archimedes" in r.text


def test_websocket_frames_drive_the_view(client):
    with client.websocket_connect("/ws") as ws:
        info = json.loads(ws.receive_text())
        assert info["type"] == "info" and len(info["cheat_sheet"]) == 7
        modes = set()
        for t, hand, _ in S.scripted_tour():
            lm = hand.landmarks.copy()
            lm[:, 0] = 1 - lm[:, 0]  # the browser sends un-mirrored frames
            ws.send_text(json.dumps({"type": "frame", "t": t * 1000, "mirrored": False,
                                     "hands": [{"landmarks": lm.tolist(), "handedness": "Right", "score": 0.9}]}))
            out = json.loads(ws.receive_text())
            modes.add(out["hud"]["mode"])
        assert {"orbit", "zoom", "section", "probe", "field"} <= modes
        ws.send_text(json.dumps({"type": "command", "kind": "undo"}))
        assert json.loads(ws.receive_text())["type"] == "view"


def test_record_clip(client, tmp_path):
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "record_start", "label": "fist", "subject": "p01"}))
        for i, (t, hand, _) in enumerate(S.scripted_tour()):
            if i > 20:
                break
            ws.send_text(json.dumps({"type": "frame", "t": t * 1000, "mirrored": True,
                                     "hands": [{"landmarks": hand.landmarks.tolist(), "handedness": "Right"}]}))
            ws.receive_text()
        ws.send_text(json.dumps({"type": "record_stop"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "recorded" and msg["path"].startswith("data/clips/p01/fist/")


def test_workbench_served_same_origin(client):
    r = client.get("/workbench/")
    assert r.status_code == 200 and "GESTURE_BY_MODE" in r.text
    assert client.get("/workbench/support.js").status_code == 200


def test_websocket_workbench_profile(client):
    with client.websocket_connect("/ws?profile=workbench") as ws:
        info = json.loads(ws.receive_text())
        assert "solve" in {row["mode"] for row in info["cheat_sheet"]}
