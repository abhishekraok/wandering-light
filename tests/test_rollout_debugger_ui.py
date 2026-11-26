"""Drive the debugger page headlessly, so a broken UI fails here not in a demo.

The page is one static file with no build step and no framework, which leaves
nothing to type-check it. ``tests/ui/render_check.mjs`` loads the real page's
script under a DOM stub, feeds it payloads this test generates from the real
endpoints, and presses the keys a person would press. It catches the failure
mode that matters -- a render path that throws on a field that moved or went
away -- which the Python tests cannot see.

Skipped when node is not installed; nothing else in the project needs it.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_rollout_debugger_server import make_context
from wandering_light.rollout_debugger.server import create_app

PAGE = (
    Path(__file__).parent.parent
    / "wandering_light"
    / "rollout_debugger"
    / "static"
    / "index.html"
)
HARNESS = Path(__file__).parent / "ui" / "render_check.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the page"
)


def test_the_page_renders_and_answers_every_key(tmp_path):
    context, record = make_context()
    http = TestClient(create_app(context), base_url="http://127.0.0.1")

    payloads = {
        "session": http.get(
            "/api/session", params={"task_id": record.task_id, "beam_width": 4}
        ),
        "drive": http.get("/api/drive", params={"task_id": record.task_id}),
        "logits": http.get("/api/logits", params={"task_id": record.task_id}),
        "encoding": http.get("/api/encoding", params={"task_id": record.task_id}),
    }
    paths = []
    for name, response in payloads.items():
        assert response.status_code == 200, name
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(response.json()), encoding="utf-8")
        paths.append(str(path))

    result = subprocess.run(
        ["node", str(HARNESS), str(PAGE), *paths],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout, result.stdout
    # The harness prints one line per check; a silent pass would mean it ran
    # nothing at all.
    assert result.stdout.count("ok  ") >= 15, result.stdout
