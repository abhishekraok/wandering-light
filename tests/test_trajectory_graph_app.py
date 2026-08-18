import asyncio
import gzip
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from wandering_light.evals.trajectory_graph_app import (
    CorpusGraphService,
    _bounded_graph_call,
    _run_blocking,
    create_app,
)
from wandering_light.typed_list import TypedList


def _legacy_row(index: int, functions: list[str], output: int) -> dict:
    return {
        "schema_version": 1,
        "split": "eval",
        "source_index": index,
        "input": TypedList([index], item_type=int).to_string(),
        "output": TypedList([output], item_type=int).to_string(),
        "original_functions": functions,
        "relabeled_functions": functions,
        "original_length": len(functions),
        "relabeled_length": len(functions),
        "lower_bound": len(functions),
        "upper_bound": len(functions),
        "certified": True,
        "method": "bfs",
    }


def _write_corpus(path: Path) -> Path:
    rows = [
        _legacy_row(1, ["inc"], 2),
        _legacy_row(2, ["inc", "double"], 6),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


@asynccontextmanager
async def _app_client(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path / "tiny.jsonl.gz")
    app = create_app(roots=(corpus_path,), cache_dir=tmp_path / "cache")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Origin": "http://test"},
        ) as client,
    ):
        yield client, app


@asynccontextmanager
async def _ready_client(tmp_path: Path):
    async with _app_client(tmp_path) as (client, app):
        source_id = (await client.get("/api/v1/sources")).json()["sources"][0]["id"]
        response = await client.post(f"/api/v1/sources/{source_id}/index")
        assert response.status_code in (200, 202)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            source = (await client.get("/api/v1/sources")).json()["sources"][0]
            if source["index"]["status"] == "ready":
                yield client, source_id, app
                return
            await asyncio.sleep(0.01)
        raise AssertionError("tiny corpus index did not become ready")


@pytest.mark.asyncio
async def test_owned_executor_does_not_depend_on_threadsafe_loop_wakeup(tmp_path):
    app = create_app(roots=(tmp_path / "missing",), cache_dir=tmp_path / "cache")

    def delayed_result() -> str:
        time.sleep(0.03)
        return "ready"

    started = time.monotonic()
    async with app.router.lifespan_context(app):
        result = await asyncio.wait_for(_run_blocking(app, delayed_result), timeout=2)

    assert result == "ready"
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_cancelled_graph_holds_capacity_until_worker_finishes(tmp_path):
    app = create_app(roots=(tmp_path / "missing",), cache_dir=tmp_path / "cache")
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_graph() -> str:
        started.set()
        release.wait(timeout=2)
        finished.set()
        return "done"

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(_bounded_graph_call(app, slow_graph))
        try:
            deadline = time.monotonic() + 1
            while not started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert started.is_set()

            task.cancel()
            await asyncio.sleep(0.03)
            assert not task.done()
            assert app.state.graph_semaphore._value == 1
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert app.state.graph_semaphore._value == 2


@pytest.mark.asyncio
async def test_source_indexing_is_background_and_single_flight(tmp_path):
    async with _app_client(tmp_path) as (client, app):
        sources = (await client.get("/api/v1/sources")).json()["sources"]
        assert len(sources) == 1
        source_id = sources[0]["id"]
        assert sources[0]["index"]["status"] == "idle"

        first = await client.post(f"/api/v1/sources/{source_id}/index")
        second = await client.post(f"/api/v1/sources/{source_id}/index")

        assert first.status_code in (200, 202)
        assert second.status_code in (200, 202)
        service = app.state.corpus_graph_service
        assert len(service._jobs) == 1


def test_source_signature_change_waits_for_active_index_worker(tmp_path, monkeypatch):
    corpus_path = _write_corpus(tmp_path / "changing.jsonl.gz")
    service = CorpusGraphService(
        roots=(corpus_path,),
        cache_dir=tmp_path / "cache",
    )
    started = threading.Event()
    release = threading.Event()
    calls = 0
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    class FakeIndex:
        def is_valid(self, _signature=None):
            return False

    def controlled_index(*_args, **_kwargs):
        nonlocal active, calls, maximum_active
        with guard:
            calls += 1
            active += 1
            maximum_active = max(maximum_active, active)
        started.set()
        release.wait(timeout=2)
        with guard:
            active -= 1
        return FakeIndex()

    monkeypatch.setattr(
        "wandering_light.evals.trajectory_graph_app.ensure_corpus_index",
        controlled_index,
    )
    source_id = service.sources()[0][0]
    try:
        first = service.start_index(source_id)
        assert first["status"] == "indexing"
        assert started.wait(timeout=1)

        # A harmless trailing byte changes the source signature while the old
        # build is still blocked.
        with corpus_path.open("ab") as handle:
            handle.write(b"\n")
        second = service.start_index(source_id)
        assert second["status"] == "indexing"
        assert calls == 1
        assert maximum_active == 1

        release.set()
        service._jobs[source_id].future.result(timeout=2)
        release.clear()
        started.clear()

        refreshed = service.start_index(source_id)
        assert refreshed["status"] == "indexing"
        assert started.wait(timeout=1)
        assert calls == 2
        assert maximum_active == 1
    finally:
        release.set()
        service.close()


@pytest.mark.asyncio
async def test_cursor_tasks_facets_and_detail(tmp_path):
    async with _ready_client(tmp_path) as (client, source_id, _app):
        facets = await client.get(f"/api/v1/sources/{source_id}/facets")
        assert facets.status_code == 200
        assert facets.json()["stats"]["records"] == 2
        inc = next(
            function
            for function in facets.json()["functions"]
            if function["function_name"] == "inc"
        )

        first = (
            await client.get(f"/api/v1/sources/{source_id}/tasks", params={"limit": 1})
        ).json()
        assert first["total"] == 2
        assert first["items"][0]["distance"] == 2
        assert first["next_cursor"]

        second = (
            await client.get(
                f"/api/v1/sources/{source_id}/tasks",
                params={"limit": 1, "cursor": first["next_cursor"]},
            )
        ).json()
        assert second["items"][0]["distance"] == 1
        assert second["next_cursor"] is None

        filtered = (
            await client.get(
                f"/api/v1/sources/{source_id}/tasks",
                params={"function_key": inc["function_key"]},
            )
        ).json()
        assert filtered["total"] == 2

        row_id = first["items"][0]["row_id"]
        detail = (
            await client.get(f"/api/v1/sources/{source_id}/tasks/{row_id}")
        ).json()
        assert detail["witness_function_names"] == ["inc", "double"]
        assert detail["basis"]["functions"]
        assert isinstance(detail["raw_json"], str)

        invalid_cursor = await client.get(
            f"/api/v1/sources/{source_id}/tasks", params={"cursor": "bad"}
        )
        assert invalid_cursor.status_code == 422


@pytest.mark.asyncio
async def test_witness_and_expansion_graph_endpoints(tmp_path):
    async with _ready_client(tmp_path) as (client, source_id, _app):
        task = (
            await client.get(f"/api/v1/sources/{source_id}/tasks", params={"limit": 1})
        ).json()["items"][0]
        row_id = task["row_id"]
        detail = (
            await client.get(f"/api/v1/sources/{source_id}/tasks/{row_id}")
        ).json()

        witness = await client.post(
            f"/api/v1/sources/{source_id}/graphs/witnesses",
            json={"row_id": row_id},
        )
        assert witness.status_code == 200
        graph = witness.json()["graph"]
        assert graph["total_nodes"] == 3
        assert graph["total_edges"] == 2
        assert graph["root_ids"]
        assert all(node["value"] for node in graph["nodes"])

        inc_id = next(
            function["id"]
            for function in detail["basis"]["functions"]
            if function["name"] == "inc"
        )
        expansion = await client.post(
            f"/api/v1/sources/{source_id}/graphs/expand",
            json={
                "row_id": row_id,
                "function_ids": [inc_id],
                "max_depth": 2,
                "max_states": 20,
                "max_transitions": 20,
            },
        )
        assert expansion.status_code == 200
        body = expansion.json()
        assert body["certified_depth"] == 2
        assert body["tasks"]
        assert body["tasks"][0]["output_serialized"]

        unknown = await client.post(
            f"/api/v1/sources/{source_id}/graphs/expand",
            json={"row_id": row_id, "function_ids": ["unknown"]},
        )
        assert unknown.status_code == 422

        range_list_id = next(
            function["id"]
            for function in detail["basis"]["functions"]
            if function["name"] == "range_list"
        )
        aggregate_range = TypedList(
            [range(10_000)] * 1_000,
            item_type=range,
        ).to_string()
        unsafe_expansion = await client.post(
            f"/api/v1/sources/{source_id}/graphs/expand",
            json={
                "row_id": row_id,
                "input_serialized": aggregate_range,
                "function_ids": [range_list_id],
            },
        )
        assert unsafe_expansion.status_code == 422
        assert "expand beyond" in unsafe_expansion.json()["detail"]


@pytest.mark.asyncio
async def test_graph_reports_reproducible_runtime_configuration_error(
    tmp_path,
    monkeypatch,
):
    def reject_runtime(_basis) -> None:
        raise RuntimeError(
            "Basis requires reproducible hashing; relaunch with PYTHONHASHSEED=0."
        )

    monkeypatch.setattr(
        "wandering_light.evals.trajectory_graph_app.require_reproducible_basis_runtime",
        reject_runtime,
    )
    async with _ready_client(tmp_path) as (client, source_id, _app):
        task = (
            await client.get(f"/api/v1/sources/{source_id}/tasks", params={"limit": 1})
        ).json()["items"][0]
        response = await client.post(
            f"/api/v1/sources/{source_id}/graphs/witnesses",
            json={"row_id": task["row_id"]},
        )

    assert response.status_code == 409
    assert "PYTHONHASHSEED=0" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_rejects_unknown_or_unindexed_sources(tmp_path):
    async with _app_client(tmp_path) as (client, _app):
        source_id = (await client.get("/api/v1/sources")).json()["sources"][0]["id"]
        assert (
            await client.get("/api/v1/sources/not-a-source/facets")
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/sources/{source_id}/facets")
        ).status_code == 409


@pytest.mark.asyncio
async def test_fetch_uses_writable_cache_and_preserves_manifest_source_id(
    tmp_path,
    monkeypatch,
):
    payload = _write_corpus(tmp_path / "remote.jsonl.gz")
    catalog = tmp_path / "catalog" / "deep"
    catalog.mkdir(parents=True)
    manifest_path = catalog / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "eval": {
                        "path": "eval.jsonl.gz",
                        "sha256": "sha256:"
                        + hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "size": 2,
                    }
                },
                "global_task_count": 2,
                "hub": {
                    "repo_id": "example/corpus",
                    "revision": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch(source_manifest, *, destination=None, **_kwargs):
        destination_path = Path(destination)
        destination_path.mkdir(parents=True)
        (destination_path / "manifest.json").write_bytes(
            Path(source_manifest).read_bytes()
        )
        (destination_path / "eval.jsonl.gz").write_bytes(payload.read_bytes())
        return destination_path

    monkeypatch.setattr("wandering_light.corpus_hub.fetch_corpus", fake_fetch)
    app = create_app(roots=(catalog,), cache_dir=tmp_path / "cache")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Origin": "http://test"},
        ) as client,
    ):
        before = (await client.get("/api/v1/sources")).json()["sources"]
        assert len(before) == 1
        source_id = before[0]["id"]
        assert not before[0]["ready"]

        fetched = await client.post(f"/api/v1/sources/{source_id}/fetch")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == source_id
        assert fetched.json()["ready"]

        after = (await client.get("/api/v1/sources")).json()["sources"]
        assert [(source["id"], source["ready"]) for source in after] == [
            (source_id, True)
        ]
        service = app.state.corpus_graph_service
        assert service.source(source_id).path.is_relative_to(service.payload_dir)
        assert service.source(source_id).path.name == source_id


@pytest.mark.asyncio
async def test_fetch_staging_is_not_discoverable_until_atomic_publish(
    tmp_path,
    monkeypatch,
):
    payload = _write_corpus(tmp_path / "remote.jsonl.gz")
    catalog = tmp_path / "catalog" / "deep"
    catalog.mkdir(parents=True)
    manifest_path = catalog / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "eval": {
                        "path": "eval.jsonl.gz",
                        "sha256": "sha256:"
                        + hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "size": 2,
                    }
                },
                "global_task_count": 2,
                "hub": {"repo_id": "example/corpus", "revision": "abc123"},
            }
        ),
        encoding="utf-8",
    )
    staged = threading.Event()
    release = threading.Event()

    def blocked_fetch(source_manifest, *, destination=None, **_kwargs):
        destination_path = Path(destination)
        destination_path.mkdir(parents=True)
        (destination_path / "manifest.json").write_bytes(
            Path(source_manifest).read_bytes()
        )
        (destination_path / "eval.jsonl.gz").write_bytes(payload.read_bytes())
        staged.set()
        release.wait(timeout=2)
        return destination_path

    monkeypatch.setattr("wandering_light.corpus_hub.fetch_corpus", blocked_fetch)
    app = create_app(roots=(catalog,), cache_dir=tmp_path / "cache")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Origin": "http://test"},
        ) as client,
    ):
        source_id = (await client.get("/api/v1/sources")).json()["sources"][0]["id"]
        request = asyncio.create_task(client.post(f"/api/v1/sources/{source_id}/fetch"))
        try:
            deadline = time.monotonic() + 1
            while not staged.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert staged.is_set()

            during = (await client.get("/api/v1/sources")).json()["sources"]
            assert [(item["id"], item["ready"]) for item in during] == [
                (source_id, False)
            ]
            assert not (app.state.corpus_graph_service.payload_dir / source_id).exists()
        finally:
            release.set()

        published = await request
        assert published.status_code == 200
        assert published.json()["ready"]


@pytest.mark.asyncio
async def test_fetch_maps_hub_failure_to_actionable_bad_gateway(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog" / "deep"
    catalog.mkdir(parents=True)
    (catalog / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "eval": {
                        "path": "eval.jsonl.gz",
                        "sha256": "sha256:" + "0" * 64,
                        "size": 2,
                    }
                },
                "global_task_count": 2,
                "hub": {"repo_id": "example/corpus", "revision": "abc123"},
            }
        ),
        encoding="utf-8",
    )

    def failing_fetch(source_manifest, *, destination=None, **_kwargs):
        destination_path = Path(destination)
        destination_path.mkdir(parents=True)
        (destination_path / "manifest.json").write_bytes(
            Path(source_manifest).read_bytes()
        )
        (destination_path / "eval.jsonl.gz").write_bytes(b"corrupt")
        raise OSError("network unavailable")

    monkeypatch.setattr("wandering_light.corpus_hub.fetch_corpus", failing_fetch)
    app = create_app(roots=(catalog,), cache_dir=tmp_path / "cache")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Origin": "http://test"},
        ) as client,
    ):
        source_id = (await client.get("/api/v1/sources")).json()["sources"][0]["id"]
        response = await client.post(f"/api/v1/sources/{source_id}/fetch")
        assert not (app.state.corpus_graph_service.payload_dir / source_id).exists()
        source = (await client.get("/api/v1/sources")).json()["sources"][0]
        assert not source["ready"]

    assert response.status_code == 502
    assert "check network access" in response.json()["detail"]
    assert "network unavailable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_invalid_ready_index_is_cleared_and_can_be_rebuilt(tmp_path):
    async with _ready_client(tmp_path) as (client, source_id, app):
        service = app.state.corpus_graph_service
        index = service._indices[source_id][1]
        index.database_path.unlink()

        source = (await client.get("/api/v1/sources")).json()["sources"][0]
        assert source["index"]["status"] == "idle"
        assert source_id not in service._indices
        assert source_id not in service._jobs

        started = await client.post(f"/api/v1/sources/{source_id}/index")
        assert started.status_code in (200, 202)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            source = (await client.get("/api/v1/sources")).json()["sources"][0]
            if source["index"]["status"] == "ready":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("invalid index was not rebuilt")

        facets = await client.get(f"/api/v1/sources/{source_id}/facets")
        assert facets.status_code == 200
        assert facets.json()["stats"]["records"] == 2


@pytest.mark.asyncio
async def test_loopback_mode_rejects_untrusted_host_and_cross_origin_posts(tmp_path):
    app = create_app(roots=(tmp_path / "missing",), cache_dir=tmp_path / "cache")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        trusted = await client.get("/api/v1/sources")
        assert trusted.status_code == 200

        untrusted = await client.get(
            "/api/v1/sources", headers={"Host": "attacker.example"}
        )
        assert untrusted.status_code == 400

        missing_origin = await client.post("/api/v1/sources/unknown/index")
        assert missing_origin.status_code == 403
        cross_origin = await client.post(
            "/api/v1/sources/unknown/index",
            headers={"Origin": "https://attacker.example"},
        )
        assert cross_origin.status_code == 403
        same_origin = await client.post(
            "/api/v1/sources/unknown/index",
            headers={"Origin": "http://test"},
        )
        assert same_origin.status_code == 404


def test_non_loopback_mode_requires_strong_token(tmp_path):
    with pytest.raises(RuntimeError, match="Refusing non-loopback"):
        create_app(
            roots=(tmp_path / "missing",),
            cache_dir=tmp_path / "cache",
            bind_host="0.0.0.0",
            auth_token="",
        )
    with pytest.raises(RuntimeError, match="at least 32"):
        create_app(
            roots=(tmp_path / "missing",),
            cache_dir=tmp_path / "cache",
            bind_host="0.0.0.0",
            auth_token="too-short",
        )


@pytest.mark.asyncio
async def test_non_loopback_mode_authenticates_reads_and_mutations(tmp_path):
    token = "review-token-" + "x" * 32
    app = create_app(
        roots=(tmp_path / "missing",),
        cache_dir=tmp_path / "cache",
        bind_host="0.0.0.0",
        auth_token=token,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://graph.example",
        ) as client,
    ):
        unauthenticated = await client.get("/api/v1/sources")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["www-authenticate"].startswith("Basic")

        bearer_headers = {"Authorization": f"Bearer {token}"}
        authenticated = await client.get("/api/v1/sources", headers=bearer_headers)
        assert authenticated.status_code == 200
        bearer_mutation = await client.post(
            "/api/v1/sources/unknown/index", headers=bearer_headers
        )
        assert bearer_mutation.status_code == 404

        basic = httpx.BasicAuth("graph", token)
        basic_without_origin = await client.post(
            "/api/v1/sources/unknown/index", auth=basic
        )
        assert basic_without_origin.status_code == 403
        basic_same_origin = await client.post(
            "/api/v1/sources/unknown/index",
            auth=basic,
            headers={"Origin": "http://graph.example"},
        )
        assert basic_same_origin.status_code == 404


@pytest.mark.asyncio
async def test_stale_cached_manifest_is_not_an_independent_source(tmp_path):
    payload = _write_corpus(tmp_path / "remote.jsonl.gz")
    payload_digest = "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()
    cache_dir = tmp_path / "cache"
    stale = cache_dir / "payloads" / "deep"
    stale.mkdir(parents=True)
    (stale / "eval.jsonl.gz").write_bytes(payload.read_bytes())
    (stale / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "eval": {
                        "path": "eval.jsonl.gz",
                        "sha256": payload_digest,
                        "size": 2,
                    }
                },
                "global_task_count": 2,
                "hub": {"repo_id": "example/old", "revision": "old"},
            }
        ),
        encoding="utf-8",
    )

    current = tmp_path / "catalog" / "deep"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "splits": {
                    "eval": {
                        "path": "eval.jsonl.gz",
                        "sha256": "sha256:" + "0" * 64,
                        "size": 2,
                    }
                },
                "global_task_count": 2,
                "hub": {"repo_id": "example/current", "revision": "current"},
            }
        ),
        encoding="utf-8",
    )

    app = create_app(roots=(tmp_path / "catalog",), cache_dir=cache_dir)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        sources = (await client.get("/api/v1/sources")).json()["sources"]

    assert len(sources) == 1
    assert sources[0]["name"] == "deep"
    assert not sources[0]["ready"]
    assert sources[0]["hub_revision"] == "current"
