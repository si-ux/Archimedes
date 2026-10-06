import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from archimedes_gestures import synthetic as S  # noqa: E402
from webdemo import server  # noqa: E402


def recv(ws, kind):
    """Receive messages until one of the given type arrives."""
    for _ in range(50):
        m = json.loads(ws.receive_text())
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind} message")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)  # keep profile/clips out of the repo
    monkeypatch.setattr(server, "SESSIONS", {})  # every test starts with fresh modeller sessions
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
            out = recv(ws, "state")
            modes.add(out["hud"]["mode"])
        assert {"orbit", "zoom", "section", "probe", "field"} <= modes
        ws.send_text(json.dumps({"type": "command", "kind": "undo"}))
        assert recv(ws, "view")["type"] == "view"


def test_record_clip(client, tmp_path):
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "record_start", "label": "fist", "subject": "p01"}))
        for i, (t, hand, _) in enumerate(S.scripted_tour()):
            if i > 20:
                break
            ws.send_text(json.dumps({"type": "frame", "t": t * 1000, "mirrored": True,
                                     "hands": [{"landmarks": hand.landmarks.tolist(), "handedness": "Right"}]}))
            recv(ws, "state")
        ws.send_text(json.dumps({"type": "record_stop"}))
        msg = recv(ws, "recorded")
        assert msg["type"] == "recorded" and msg["path"].startswith("data/clips/p01/fist/")


def test_workbench_served_same_origin(client):
    r = client.get("/workbench/")
    assert r.status_code == 200 and "GESTURE_BY_MODE" in r.text
    assert client.get("/workbench/support.js").status_code == 200


def test_websocket_workbench_profile(client):
    with client.websocket_connect("/ws?profile=workbench") as ws:
        info = json.loads(ws.receive_text())
        assert "solve" in {row["mode"] for row in info["cheat_sheet"]}


def test_modelling_over_the_websocket_and_xr_endpoints(client):
    with client.websocket_connect("/ws") as ws:
        res = recv(ws, "results")  # opens on a solved demo
        assert "von_mises" in res["fields"] and res["stats"]["dof"] > 0
        assert recv(ws, "model")["phase"] == "results"
        ws.send_text(json.dumps({"type": "model_action", "action": {"op": "new_rect", "orientation": "vertical"}}))
        m = recv(ws, "model")
        assert m["prompt"]["key"] == "inc_length" and m["phase"] == "model"
        for n in (3, 3, 2):  # increments: 100 mm, 25 mm, 1 kN
            ws.send_text(json.dumps({"type": "number", "value": n}))
            m = recv(ws, "model")
        assert m["prompt"]["key"] == "b"
        for v in (300, 300, 3000):
            ws.send_text(json.dumps({"type": "number", "value": v}))
            m = recv(ws, "model")
        assert m["model"]["member"]["orientation"] == "vertical" and m["prompt"] is None and m["phase"] == "loads"
        ws.send_text(json.dumps({"type": "model_action", "action": {"op": "support_fixed", "t": 0.0}}))
        recv(ws, "model")
        ws.send_text(json.dumps({"type": "model_action", "action": {"op": "point_load", "t": 1.0, "axis": "z", "sign": -1}}))
        assert recv(ws, "model")["prompt"]["unit"] == "kN"
        ws.send_text(json.dumps({"type": "number", "value": 100}))
        recv(ws, "model")
        ws.send_text(json.dumps({"type": "solve"}))
        res = recv(ws, "results")
        assert res["model"]["member"]["orientation"] == "vertical"
        assert abs(res["reactions"]["total_kN"][2] - 100) < 1e-6
    assert client.get("/api/scene.json").json()["ready"]
    glb = client.get("/api/scene.glb?field=U3")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert client.get("/xr").status_code == 200


def test_model_survives_a_trip_to_the_workbench(client):
    with client.websocket_connect("/ws?cid=abc") as ws:
        recv(ws, "results")
        recv(ws, "model")
        ws.send_text(json.dumps({"type": "load_demo", "name": "short_column"}))
        recv(ws, "results")
    with client.websocket_connect("/ws?cid=abc") as ws:  # back from /workbench/
        res = recv(ws, "results")
        assert res["model"]["name"].startswith("Short column")
    with client.websocket_connect("/ws?cid=other") as ws:  # another browser gets its own
        assert recv(ws, "results")["model"]["name"].startswith("Cantilever")


def test_console_over_the_websocket(client):
    with client.websocket_connect("/ws?cid=console-test") as ws:
        recv(ws, "model")
        ws.send_text(json.dumps({"type": "console", "line": "beam(b=200, h=400, L=5000); fixed(at=0); point(8)"}))
        out = recv(ws, "console_out")
        assert not out["error"] and "Point 8 kN" in out["output"]
        m = recv(ws, "model")
        assert m["model"]["member"]["length"] == 5000 and m["journal"][-1]["cmd"] == "point(8, at=1, dir='-z')"
        ws.send_text(json.dumps({"type": "console", "line": "solve(); show('U3')"}))
        assert not recv(ws, "console_out")["error"]
        res = recv(ws, "results")
        assert res["model"]["member"]["length"] == 5000
        assert recv(ws, "view")["view"]["field_name"] == "U3"
        ws.send_text(json.dumps({"type": "console_script", "name": "s.py", "code": "clear('loads')\nprint(len(model.loads))"}))
        out = recv(ws, "console_out")
        assert out["output"].strip() == "0"


def test_console_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(server.scripting, "ENABLED", False)
    with client.websocket_connect("/ws?cid=off") as ws:
        recv(ws, "model")
        ws.send_text(json.dumps({"type": "console", "line": "import os"}))
        out = recv(ws, "console_out")
        assert out["error"] and "ARCHIMEDES_CONSOLE=0" in out["output"]
