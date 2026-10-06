"""Runs the CAD kernel's JavaScript unit tests (tests/cadkernel_test.js) when Node is available."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_cad_kernel():
    r = subprocess.run(["node", str(ROOT / "tests" / "cadkernel_test.js")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "kernel tests passed" in r.stdout


def test_workbench_serves_the_kernel():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from webdemo import server
    c = TestClient(server.app)
    r = c.get("/workbench/cadkernel.js")
    assert r.status_code == 200 and "CADK" in r.text
    assert "cadkernel.js" in c.get("/workbench/").text
