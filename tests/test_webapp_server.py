"""The explorer's HTTP surface."""

import pytest
from fastapi.testclient import TestClient

from wandering_light.evals import corpus_view
from wandering_light.webapp.server import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _corpus_available() -> tuple[str, str] | None:
    for ref in corpus_view.discover_corpora():
        manifest = corpus_view.load_manifest(ref)
        present = [
            split
            for split in manifest["splits"]
            if split not in corpus_view.missing_splits(ref, manifest)
        ]
        if present:
            return ref.name, present[0]
    return None


def test_basis_lists_functions_and_alternatives(client):
    payload = client.get("/api/basis").json()
    assert payload["basis_set_id"] == "wl-core-v1"
    assert payload["digest"].startswith("sha256:")
    assert len(payload["functions"]) == 118
    assert {"wl-core-v1", "wl-core-pyhash-v1"} <= {
        e["id"] for e in payload["available"]
    }
    inc = next(f for f in payload["functions"] if f["name"] == "inc")
    assert (inc["input_type"], inc["output_type"]) == ("int", "int")


def test_basis_rejects_an_unregistered_id(client):
    assert (
        client.get("/api/basis", params={"basis_set_id": "nope-v9"}).status_code == 400
    )


def test_state_round_trips_through_both_text_forms(client):
    first = client.post("/api/state", json={"state": "TL<int>([1, 2, 3])"}).json()
    second = client.post("/api/state", json={"state": first["wire"]}).json()
    assert first == second


def test_state_reports_a_parse_failure_as_a_client_error(client):
    response = client.post("/api/state", json={"state": "TL<int>(oops"})
    assert response.status_code == 400
    assert "could not parse state" in response.json()["detail"]


def test_trajectory_returns_every_intermediate_state(client):
    payload = client.post(
        "/api/trajectory",
        json={"state": "TL<int>([1, 2, 3])", "steps": ["double", "inc"]},
    ).json()
    assert payload["root"]["label"] == "TL<int>([1, 2, 3])"
    assert [s["function"] for s in payload["steps"]] == ["double", "inc"]
    assert payload["steps"][-1]["state"]["label"] == "TL<int>([3, 5, 7])"


def test_successors_flag_dead_ends_and_no_ops(client):
    payload = client.post(
        "/api/successors",
        json={"state": "TL<int>([0, 1])", "functions": ["mod2", "inc"]},
    ).json()
    by_name = {item["function"]: item for item in payload["successors"]}
    assert by_name["mod2"]["self_loop"] is True
    assert by_name["inc"]["self_loop"] is False
    assert payload["dead_end"] is False
    assert payload["applicable"] == 2


def test_successors_reject_a_function_outside_the_basis(client):
    response = client.post(
        "/api/successors", json={"state": "TL<int>([1])", "functions": ["teleport"]}
    )
    assert response.status_code == 400
    assert "teleport" in response.json()["detail"]


def test_expand_returns_a_drawable_graph(client):
    payload = client.post(
        "/api/expand",
        json={
            "state": "TL<int>([1, 2, 3])",
            "functions": ["inc", "double"],
            "max_depth": 2,
        },
    ).json()
    assert payload["stats"]["complete"] is True
    assert payload["stats"]["nodes"] == len(payload["nodes"])
    assert payload["function_edges"] == {"inc": 3, "double": 3}
    assert payload["idle_functions"] == []


def test_expand_bounds_are_enforced_by_the_schema(client):
    response = client.post(
        "/api/expand", json={"state": "TL<int>([1])", "max_depth": 99}
    )
    assert response.status_code == 422


def test_solve_reports_the_attempt_and_its_cost(client):
    payload = client.post(
        "/api/solve",
        json={
            "state": "TL<int>([1, 2, 3])",
            "target": "TL<int>([3, 5, 7])",
            "max_depth": 2,
            "budget": 5000,
        },
    ).json()
    assert payload["success"] is True
    assert payload["functions"] == ["double", "inc"]
    assert payload["elapsed_seconds"] >= 0


def test_solve_rejects_an_unknown_solver(client):
    response = client.post(
        "/api/solve",
        json={
            "state": "TL<int>([1])",
            "target": "TL<int>([2])",
            "solver": "oracle",
        },
    )
    assert response.status_code == 400


def test_root_serves_the_build_or_says_how_to_make_it(client):
    payload = client.get("/")
    assert payload.status_code == 200
    body = payload.text
    assert "<!doctype html>" in body.lower() or "npm run build" in body


def test_corpora_summarise_local_manifests(client):
    payload = client.get("/api/corpora").json()["corpora"]
    for entry in payload:
        assert entry["basis_set_id"]
        assert entry["splits"]
        assert set(entry["missing_splits"]) <= set(entry["splits"])


def test_corpus_tasks_are_addressable_by_split_and_distance(client):
    available = _corpus_available()
    if available is None:
        pytest.skip("no corpus payload downloaded")
    name, split = available
    payload = client.get(
        f"/api/corpora/{name}/tasks", params={"split": split, "limit": 5}
    ).json()
    assert len(payload["tasks"]) <= 5
    task = payload["tasks"][0]
    assert task["witness"] and task["distance"] >= 1
    # The input is returned in wire form so it can be posted straight back.
    echoed = client.post("/api/state", json={"state": task["input"]}).json()
    assert echoed["label"] == task["input_label"]

    filtered = client.get(
        f"/api/corpora/{name}/tasks",
        params={"split": split, "limit": 5, "distance": 6},
    ).json()["tasks"]
    assert all(item["distance"] == 6 for item in filtered)


def test_corpus_tasks_404_on_an_unknown_corpus_or_split(client):
    assert (
        client.get("/api/corpora/nope/tasks", params={"split": "test"}).status_code
        == 404
    )
    available = _corpus_available()
    if available is None:
        pytest.skip("no corpus payload downloaded")
    name, _ = available
    assert (
        client.get(f"/api/corpora/{name}/tasks", params={"split": "nope"}).status_code
        == 404
    )
