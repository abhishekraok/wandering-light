"""The server's endpoints, against a stub context rather than a checkpoint.

The debugger's correctness claim is that it shows the production search. These
tests pin the parts that could drift from it without failing loudly: that a
path is replayed through the real executor, that drive-mode ranking uses the
same mask and normalization the rollout uses, and that an illegal transition is
refused rather than displayed.
"""

import math

import pytest
from fastapi.testclient import TestClient

from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.rollout_debugger.context import DebugContext
from wandering_light.rollout_debugger.server import create_app
from wandering_light.solver import ActionScorer
from wandering_light.typed_list import TypedList


class StubScorer(ActionScorer):
    def __init__(self, table=None):
        self.table = table or {}
        self.calls = 0

    def score_actions_batch(self, problems, available_functions):
        self.calls += 1
        rows = []
        for state, _target in problems:
            row = self.table.get(tuple(state.items))
            rows.append(
                dict(row)
                if row
                else {function.name: 0.0 for function in available_functions}
            )
        return rows


def make_context(scorer=None):
    increment = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    double = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )
    stringify = FunctionDef(
        name="stringify",
        input_type="builtins.int",
        output_type="builtins.str",
        code="return str(x)",
    )
    shout = FunctionDef(
        name="shout",
        input_type="builtins.str",
        output_type="builtins.str",
        code="return x + '!'",
    )
    functions = FunctionDefSet([increment, double, stringify, shout])
    record = BasisTaskRecord.create(
        split="test",
        input_value=TypedList([2]),
        output_value=TypedList([6]),
        witness_function_ids=("a", "b"),
        witness_function_names=("increment", "double"),
        basis_set_id="stub",
        basis_set_digest="sha256:stub",
        generator="test",
        seed=0,
        source_index=0,
        metadata={"certified_distance": 2},
    )
    return DebugContext(
        records=[record],
        basis=None,
        available_functions=functions,
        policy=scorer or StubScorer(),
        provenance={"corpus": {"split": "test"}, "policy": {"source": "stub"}},
    ), record


@pytest.fixture
def client():
    context, record = make_context()
    return TestClient(create_app(context), base_url="http://127.0.0.1"), record


def test_session_runs_the_search_and_returns_a_navigable_trace(client):
    http, record = client
    response = http.get(
        "/api/session",
        params={"task_id": record.task_id, "beam_width": 4, "max_depth": 3},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["task_id"] == record.task_id
    assert payload["termination"] == "solved"
    assert payload["solution"] == ["increment", "double"]
    assert payload["witness"] == ["increment", "double"]
    # The layers carry what the UI steps through.
    assert payload["trace"]["layers"]
    first = payload["trace"]["layers"][0]
    assert first["depth"] == 1
    assert first["nodes"][0]["path"] == []
    assert {"function", "logit", "log_probability"} == set(
        first["nodes"][0]["compatible"][0]
    )


def test_the_witness_overlay_is_diagnosis_only_and_tracks_survival(client):
    http, record = client
    payload = http.get(
        "/api/session", params={"task_id": record.task_id, "beam_width": 4}
    ).json()
    overlay = payload["witness_overlay"]
    assert overlay[0]["prefix"] == ["increment"]
    assert overlay[0]["survives"] is True


def test_drive_ranks_actions_with_the_searchs_own_mask_and_normalization():
    scorer = StubScorer(
        {(2,): {"increment": 3.0, "double": 1.0, "stringify": 0.0, "shout": -5.0}}
    )
    context, record = make_context(scorer)
    http = TestClient(create_app(context), base_url="http://127.0.0.1")

    payload = http.get("/api/drive", params={"task_id": record.task_id}).json()

    # shout takes a str and the state is int, so the mask admits three of four.
    assert payload["mask_size"] == 3
    assert payload["basis_size"] == 4
    assert [entry["function"] for entry in payload["actions"]] == [
        "increment",
        "double",
        "stringify",
    ]
    assert [entry["rank"] for entry in payload["actions"]] == [1, 2, 3]
    assert math.isclose(
        sum(entry["probability"] for entry in payload["actions"]), 1.0, abs_tol=1e-6
    )
    assert payload["solved"] is False


def test_drive_replays_a_path_and_reports_reaching_the_target(client):
    http, record = client
    payload = http.get(
        "/api/drive",
        params={"task_id": record.task_id, "path": "increment,double"},
    ).json()

    assert payload["state"]["value"] == "[6]"
    assert payload["solved"] is True
    assert payload["path"] == ["increment", "double"]


def test_a_type_incompatible_step_is_refused_not_displayed(client):
    http, record = client
    response = http.get(
        "/api/drive", params={"task_id": record.task_id, "path": "shout"}
    )
    assert response.status_code == 400
    assert "does not accept" in response.json()["detail"]


def test_logits_return_the_whole_basis_and_mark_the_mask(client):
    http, record = client
    payload = http.get("/api/logits", params={"task_id": record.task_id}).json()

    assert len(payload["logits"]) == 4
    by_name = {row["function"]: row for row in payload["logits"]}
    assert by_name["shout"]["compatible"] is False
    assert by_name["increment"]["compatible"] is True


def test_encoding_is_the_production_encoder_not_a_description(client):
    http, record = client
    payload = http.get("/api/encoding", params={"task_id": record.task_id}).json()

    assert payload["tensor_shapes"]["sequences"][0] == 1
    assert payload["separator_token"] == 257
    assert payload["pad_token"] == 256
    assert payload["aligned"][0]["state_text"] == "2"
    assert payload["aligned"][0]["target_text"] == "6"


def test_an_unknown_task_is_a_404_not_a_wrong_answer(client):
    http, _ = client
    assert http.get("/api/session", params={"task_id": "nope"}).status_code == 404
    assert http.get("/api/drive", params={"task_id": "nope"}).status_code == 404


def test_context_lists_what_can_be_filtered_on(client):
    http, _ = client
    payload = http.get("/api/context").json()
    assert payload["distances"] == [2]
    assert payload["input_types"] == ["int"]
    assert payload["tasks"] == 1
