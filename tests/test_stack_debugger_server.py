"""The server's endpoints against a tiny untrained policy (stack-rollout-debugger, #115).

What could drift silently: that a driven state is replayed through the same
step the search takes, that an illegal transition is refused rather than
displayed, and that the op table ranks with the beam's mask and normalization.
"""

import math
import random

import pytest
from fastapi.testclient import TestClient

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_debugger.context import DebugContext
from wandering_light.stack_debugger.server import create_app
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import PolicyScorer, StackPairPolicy

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 3, held_out=6))


@pytest.fixture
def client():
    import torch

    torch.manual_seed(0)
    context = DebugContext(
        tasks=[task_for("add_pair", 6), task_for("sum"), task_for("min_pair", 2)],
        scorer=PolicyScorer(StackPairPolicy(TINY), "cpu"),
        provenance={"policy": {"source": "test", "arm": "tiny", "seed": 0}},
    )
    return TestClient(create_app(context))


def test_context_lists_behaviours(client):
    body = client.get("/api/context").json()
    assert body["tasks"] == 3
    assert [b["name"] for b in body["behaviours"]] == ["add_pair", "sum", "min_pair"]
    assert body["ops"][-2:] == ["[", "]"]


def test_session_runs_the_search_and_ships_the_trace(client):
    body = client.get(
        "/api/session", params={"task": 0, "beam_width": 8, "max_depth": 3}
    ).json()
    assert body["result"]["program"] == ["add"] and body["result"]["solved"] is True
    assert body["trace"]["layers"][0]["nodes"][0]["states"][0].startswith("[ ")
    assert body["map"]["points"][1]["kind"] == "target"
    assert body["provenance"]["selection"] == {"mode": "task", "task": 0}


def test_session_refuses_unknown_filters(client):
    assert client.get("/api/session", params={"task": 9}).status_code == 404
    assert client.get("/api/session", params={"behaviour": "nope"}).status_code == 400


def test_state_replays_the_path_and_ranks_with_the_beams_mask(client):
    body = client.get(
        "/api/state", params={"task": 1, "path": "[,add", "max_depth": 4}
    ).json()
    assert body["path"] == ["[", "add"]
    assert "MARK add" in body["states"][0]
    assert body["remaining"] == 4 - 2 - 1
    legal = [o for o in body["ops"] if o["probability"] is not None]
    assert len(legal) == body["legal"]
    assert math.isclose(sum(o["probability"] for o in legal), 1.0, abs_tol=1e-4)
    ranks = sorted(o["rank"] for o in legal)
    assert ranks == list(range(1, len(legal) + 1))
    captured = next(o for o in body["ops"] if o["op"] == "mul")
    assert captured["outcome"] == "captured"
    assert body["encoding"]["items"][0]["state_tokens"][-2:] == ["MARK", "add"]
    assert len(body["progress"]["points"]) == 3
    assert len(body["trail"]) == 3 and body["trail"][-1] == body["states"][0]


def test_state_refuses_an_illegal_path(client):
    response = client.get("/api/state", params={"task": 1, "path": "add"})
    assert response.status_code == 422
    assert "refused" in response.json()["detail"]
    assert (
        client.get("/api/state", params={"task": 1, "path": "bogus"}).status_code == 400
    )


def test_play_witness_and_custom_program(client):
    body = client.get("/api/play", params={"task": 1, "program": "witness"}).json()
    assert body["solved_shown"] is True and body["label"].startswith("witness")
    body = client.get("/api/play", params={"task": 2, "program": "swap,drop"}).json()
    assert body["label"] == "program" and body["refused"] is None
    assert [s["op"] for s in body["steps"]] == [None, "swap", "drop"]
    assert (
        client.get("/api/play", params={"task": 1, "program": "solution"}).status_code
        == 400
    )
