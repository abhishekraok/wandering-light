"""Fast, graph-first browser for trajectory corpora.

The browser client owns viewport interaction, selection, and path highlighting.
Python is only involved when indexing/querying a corpus or building a bounded
``TrajectoryGraph`` projection.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import math
import mimetypes
import os
import secrets
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from wandering_light.basis_dataset import typed_list_from_builtin_str
from wandering_light.basis_set import (
    BasisSetDigestMismatchError,
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.evals.corpus_index import (
    FUNCTION_ROLES,
    CorpusFilters,
    CorpusIndex,
    CorpusSource,
    RecordDetail,
    RecordSummary,
    default_index_cache_dir,
    discover_corpus_sources,
    ensure_corpus_index,
    source_signature,
)
from wandering_light.evals.explorer_graph import (
    GraphView,
    build_local_expansion,
    build_witness_projection,
    validate_typed_list_workload,
)
from wandering_light.function_def import FunctionDefSet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOTS = (
    PACKAGE_ROOT / "training" / "data",
    Path(__file__).with_name("data"),
)
WEB_ROOT = Path(__file__).with_name("trajectory_graph_web")
GRAPH_HOST_ENV = "WANDERING_LIGHT_GRAPH_HOST"
GRAPH_TOKEN_ENV = "WANDERING_LIGHT_GRAPH_TOKEN"
DEFAULT_GRAPH_HOST = "127.0.0.1"
MIN_GRAPH_TOKEN_LENGTH = 32
_TEST_HOSTS = frozenset({"test", "testserver"})
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_VERIFIED_CACHE_MARKER = ".wandering-light-verified"


def _configured_graph_host() -> str:
    return os.environ.get(GRAPH_HOST_ENV, DEFAULT_GRAPH_HOST)


def _normalized_hostname(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.rstrip(".").lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return normalized or None


def _is_loopback_hostname(value: str | None, *, allow_test: bool = False) -> bool:
    hostname = _normalized_hostname(value)
    if hostname is None:
        return False
    if hostname == "localhost" or (allow_test and hostname in _TEST_HOSTS):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped is not None and mapped.is_loopback


def _validated_graph_token(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if (
        len(value) < MIN_GRAPH_TOKEN_LENGTH
        or value != value.strip()
        or any(ord(character) < 0x21 or ord(character) == 0x7F for character in value)
    ):
        raise RuntimeError(
            f"{GRAPH_TOKEN_ENV} must be at least {MIN_GRAPH_TOKEN_LENGTH} visible "
            "characters with no surrounding whitespace"
        )
    return value


def _authorization_kind(header: str | None, token: str | None) -> str | None:
    """Return the valid authentication scheme without leaking token timing."""
    if token is None or not header:
        return None
    scheme, separator, credentials = header.partition(" ")
    if not separator:
        return None
    if scheme.casefold() == "bearer":
        supplied = credentials.strip().encode("utf-8")
        return (
            "bearer"
            if secrets.compare_digest(supplied, token.encode("utf-8"))
            else None
        )
    if scheme.casefold() != "basic":
        return None
    try:
        decoded = base64.b64decode(credentials, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    _username, separator, password = decoded.partition(":")
    if not separator:
        return None
    return (
        "basic"
        if secrets.compare_digest(password.encode("utf-8"), token.encode("utf-8"))
        else None
    )


def _origin_tuple(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    hostname = _normalized_hostname(parsed.hostname)
    if (
        parsed.scheme not in {"http", "https"}
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.scheme, hostname, port or (443 if parsed.scheme == "https" else 80)


def _request_origin(request: Request) -> tuple[str, str, int] | None:
    try:
        hostname = _normalized_hostname(request.url.hostname)
        port = request.url.port
    except ValueError:
        return None
    if request.url.scheme not in {"http", "https"} or hostname is None:
        return None
    return (
        request.url.scheme,
        hostname,
        port or (443 if request.url.scheme == "https" else 80),
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WitnessGraphRequest(StrictModel):
    row_id: int = Field(ge=1)
    scope: Literal["selected", "root"] = "selected"
    max_records: int = Field(default=250, ge=1, le=500)
    max_nodes: int = Field(default=600, ge=10, le=2_000)
    max_edges: int = Field(default=1_200, ge=10, le=4_000)


class ExpansionGraphRequest(StrictModel):
    row_id: int = Field(ge=1)
    input_serialized: str | None = Field(default=None, max_length=65_536)
    function_ids: list[str] = Field(min_length=1, max_length=256)
    max_depth: int = Field(default=2, ge=1, le=3)
    max_states: int = Field(default=250, ge=10, le=1_000)
    max_transitions: int = Field(default=2_500, ge=10, le=5_000)
    include_self_loops: bool = True
    max_nodes: int = Field(default=600, ge=10, le=2_000)
    max_edges: int = Field(default=1_200, ge=10, le=4_000)


@dataclass(slots=True)
class IndexJob:
    signature: str
    status: Literal["indexing", "ready", "error"] = "indexing"
    records_indexed: int = 0
    file_index: int = 0
    file_count: int = 0
    message: str = "Preparing index"
    error: str | None = None
    future: Future[CorpusIndex] | None = None


class CorpusFetchError(RuntimeError):
    """An operational failure while downloading a configured corpus."""


class CorpusGraphService:
    """Thread-safe source registry and single-flight index builder."""

    def __init__(
        self,
        *,
        roots: Sequence[str | Path] | None = None,
        cache_dir: str | Path | None = None,
        index_workers: int = 2,
    ) -> None:
        configured_cache = os.environ.get("WANDERING_LIGHT_CORPUS_CACHE")
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (Path(configured_cache) if configured_cache else None)
        )
        index_cache_root = self.cache_dir or default_index_cache_dir()
        self.payload_dir = index_cache_root / "payloads"
        self.payload_staging_dir = index_cache_root / ".payload-staging"
        configured_roots = tuple(roots) if roots is not None else _configured_roots()
        self.roots = tuple(dict.fromkeys(configured_roots))
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=index_workers,
            thread_name_prefix="corpus-index",
        )
        self._sources: dict[str, CorpusSource] = {}
        self._errors: list[dict[str, str]] = []
        self._indices: dict[str, tuple[str, CorpusIndex]] = {}
        self._jobs: dict[str, IndexJob] = {}
        self._fetch_locks: dict[str, threading.Lock] = {}
        self.refresh_sources()

    def close(self) -> None:
        # A running index owns an SQLite connection and an atomic temporary
        # file.  Let it leave both in a defined state before the application
        # lifespan ends instead of abandoning a non-daemon worker thread.
        self._executor.shutdown(wait=True, cancel_futures=True)

    def refresh_sources(self) -> None:
        errors: list[tuple[Any, Exception]] = []
        catalog_sources = discover_corpus_sources(
            self.roots,
            on_error=lambda path, error: errors.append((path, error)),
        )
        cached_sources = discover_corpus_sources((self.payload_dir,))

        catalog_by_id: dict[str, CorpusSource] = {}
        for source in catalog_sources:
            source_id = _source_id(source)
            current = catalog_by_id.get(source_id)
            if current is None or (source.ready and not current.ready):
                catalog_by_id[source_id] = source

        cache_by_id: dict[str, CorpusSource] = {}
        for source in cached_sources:
            source_id = _source_id(source)
            marker = source.path / _VERIFIED_CACHE_MARKER
            try:
                if marker.read_text(encoding="utf-8").strip() != source_id:
                    continue
            except OSError:
                # Only a fully verified, atomically published download may
                # overlay a configured catalog manifest. In-progress and old
                # pre-marker cache directories stay invisible.
                continue
            current = cache_by_id.get(source_id)
            if current is None or (source.ready and not current.ready):
                cache_by_id[source_id] = source

        # Cache entries are overlays for configured catalog manifests, not an
        # independent corpus catalog. This prevents superseded downloads from
        # reappearing as duplicate sources after a package upgrade.
        preferred = {
            source_id: (
                cached
                if not catalog.ready
                and (cached := cache_by_id.get(source_id)) is not None
                and cached.ready
                else catalog
            )
            for source_id, catalog in catalog_by_id.items()
        }
        with self._lock:
            self._sources = preferred
            self._errors = [
                {"location": str(path), "message": str(error)} for path, error in errors
            ]

    @property
    def errors(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            return tuple(self._errors)

    def sources(self) -> tuple[tuple[str, CorpusSource], ...]:
        self.refresh_sources()
        with self._lock:
            return tuple(
                sorted(
                    self._sources.items(),
                    key=lambda item: (item[1].name, item[0]),
                )
            )

    def source(self, source_id: str) -> CorpusSource:
        with self._lock:
            source = self._sources.get(source_id)
        if source is None:
            self.refresh_sources()
            with self._lock:
                source = self._sources.get(source_id)
        if source is None:
            raise KeyError(source_id)
        return source

    def source_status(self, source_id: str) -> dict[str, Any]:
        source = self.source(source_id)
        signature = source_signature(source)
        with self._lock:
            cached = self._indices.get(source_id)
            if cached is not None and cached[0] == signature:
                if cached[1].is_valid(signature):
                    return {"status": "ready", "records_indexed": None}
                # The database can disappear or become invalid after it was
                # opened (for example, when a cache cleaner runs). Atomically
                # forget both the handle and its completed job so clients see
                # an idle source and can start a replacement build.
                self._indices.pop(source_id, None)
            elif cached is not None:
                self._indices.pop(source_id, None)
            job = self._jobs.get(source_id)
            if job is not None and job.status == "indexing":
                payload = _job_payload(job)
                if job.signature != signature:
                    payload["message"] = "Finishing previous index before refresh"
                return payload
            if job is not None and job.signature == signature:
                if job.status == "ready":
                    self._jobs.pop(source_id, None)
                    return {"status": "idle", "records_indexed": 0}
                return _job_payload(job)
        return {"status": "idle", "records_indexed": 0}

    def start_index(self, source_id: str) -> dict[str, Any]:
        source = self.source(source_id)
        if not source.ready:
            raise FileNotFoundError("corpus payload is not available locally")
        signature = source_signature(source)
        with self._lock:
            cached = self._indices.get(source_id)
            if (
                cached is not None
                and cached[0] == signature
                and cached[1].is_valid(signature)
            ):
                return {"status": "ready", "records_indexed": None}
            current = self._jobs.get(source_id)
            if current is not None and current.status == "indexing":
                # The source can change while a large index is still being
                # built. Never let two workers race to replace the same
                # path-keyed SQLite cache. Once this worker finishes, the
                # changed signature is observed as idle and a fresh build can
                # start.
                return _job_payload(current)
            job = IndexJob(signature=signature)
            self._jobs[source_id] = job
            job.future = self._executor.submit(
                self._build_index,
                source_id,
                source,
                signature,
                job,
            )
            return _job_payload(job)

    def _build_index(
        self,
        source_id: str,
        source: CorpusSource,
        signature: str,
        job: IndexJob,
    ) -> CorpusIndex:
        def progress(update) -> None:
            with self._lock:
                if self._jobs.get(source_id) is not job:
                    return
                job.records_indexed = update.records_indexed
                job.file_index = update.file_index
                job.file_count = update.file_count
                job.message = f"Indexing file {update.file_index}/{update.file_count}"

        try:
            index = ensure_corpus_index(
                source,
                cache_dir=self.cache_dir,
                progress=progress,
            )
        except BaseException as error:
            with self._lock:
                if self._jobs.get(source_id) is job:
                    job.status = "error"
                    job.error = str(error)
                    job.message = "Indexing failed"
            raise
        with self._lock:
            if self._jobs.get(source_id) is job:
                self._indices[source_id] = (signature, index)
                job.status = "ready"
                job.message = "Index ready"
        return index

    def index(self, source_id: str) -> CorpusIndex:
        source = self.source(source_id)
        signature = source_signature(source)
        with self._lock:
            cached = self._indices.get(source_id)
            if (
                cached is not None
                and cached[0] == signature
                and cached[1].is_valid(signature)
            ):
                return cached[1]
            if cached is not None:
                self._indices.pop(source_id, None)
            job = self._jobs.get(source_id)
            if job is not None and job.signature == signature:
                if job.status == "error":
                    raise RuntimeError(job.error or "corpus indexing failed")
                if job.status == "indexing":
                    raise LookupError("corpus index is still being built")
                # A ready job without its corresponding valid index is stale.
                # Clear it so the next start request can rebuild the cache.
                self._jobs.pop(source_id, None)
        raise LookupError("corpus has not been indexed")

    def fetch(self, source_id: str) -> CorpusSource:
        source = self.source(source_id)
        if source.ready:
            return source
        if source.manifest_path is None or not source.hub_repo_id:
            raise ValueError("corpus has no pinned Hub payload")
        with self._lock:
            fetch_lock = self._fetch_locks.setdefault(source_id, threading.Lock())
        with fetch_lock:
            from wandering_light.corpus_hub import fetch_corpus

            try:
                self.payload_staging_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    dir=self.payload_staging_dir,
                    prefix=f"{source_id}.",
                ) as temporary:
                    work_root = Path(temporary)
                    staged_payload = work_root / "payload"
                    fetch_corpus(
                        source.manifest_path,
                        destination=staged_payload,
                    )
                    (staged_payload / _VERIFIED_CACHE_MARKER).write_text(
                        f"{source_id}\n",
                        encoding="utf-8",
                    )

                    # Publish only the complete verified directory. A prior
                    # cache is moved out of discovery first and restored if
                    # the final rename unexpectedly fails.
                    self.payload_dir.mkdir(parents=True, exist_ok=True)
                    destination = self.payload_dir / source_id
                    previous = work_root / "previous"
                    if destination.exists():
                        os.replace(destination, previous)
                    try:
                        os.replace(staged_payload, destination)
                    except BaseException:
                        if previous.exists() and not destination.exists():
                            os.replace(previous, destination)
                        raise
            except Exception as error:
                raise CorpusFetchError(
                    "corpus download or cache publication failed; check network "
                    "access, cache permissions, and the pinned Hub "
                    f"repository/revision: {error}"
                ) from error
            with self._lock:
                self._indices.pop(source_id, None)
                self._jobs.pop(source_id, None)
            self.refresh_sources()
        return self.source(source_id)


def _configured_roots() -> tuple[str | Path, ...]:
    configured = os.environ.get("WANDERING_LIGHT_CORPUS_PATHS")
    if configured:
        return tuple(path for path in configured.split(os.pathsep) if path)
    return DEFAULT_CORPUS_ROOTS


def _source_id(source: CorpusSource) -> str:
    if source.manifest_path is not None:
        # A packaged catalog manifest and its downloaded cache copy describe
        # the same logical corpus. Keep the browser selection stable as the
        # source moves from "fetchable" to "ready".
        identity = source.manifest_path.read_bytes()
    else:
        identity = str(source.path.resolve()).encode()
    return hashlib.sha256(identity).hexdigest()[:20]


def _job_payload(job: IndexJob) -> dict[str, Any]:
    return {
        "status": job.status,
        "records_indexed": job.records_indexed,
        "file_index": job.file_index,
        "file_count": job.file_count,
        "message": job.message,
        "error": job.error,
    }


def _source_payload(
    source_id: str, source: CorpusSource, index_status: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": source_id,
        "name": source.name,
        "ready": source.ready,
        "missing_files": [path.name for path in source.missing_files],
        "expected_records": source.expected_records,
        "basis_set_id": source.basis_set_id,
        "basis_set_digest": source.basis_set_digest,
        "fetchable": bool(source.hub_repo_id and source.manifest_path),
        "hub_repo_id": source.hub_repo_id,
        "hub_revision": source.hub_revision,
        "index": index_status,
    }


def _filters(
    *,
    splits: Sequence[str],
    min_distance: int | None,
    max_distance: int | None,
    input_types: Sequence[str],
    output_types: Sequence[str],
    root_indices: Sequence[int],
    function_keys: Sequence[str],
    function_match: Literal["any", "all"],
    function_roles: Sequence[str],
) -> CorpusFilters:
    try:
        return CorpusFilters(
            splits=tuple(splits),
            min_distance=min_distance,
            max_distance=max_distance,
            input_types=tuple(input_types),
            output_types=tuple(output_types),
            root_indices=tuple(root_indices),
            function_keys=tuple(function_keys),
            function_match=function_match,
            function_roles=tuple(function_roles),
        )
    except ValueError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
        ) from error


def _cursor_encode(summary: RecordSummary) -> str:
    payload = [
        summary.distance if summary.distance is not None else -1,
        summary.split,
        summary.row_id,
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(encoded).decode().rstrip("=")


def _cursor_decode(value: str | None) -> tuple[int, str, int] | None:
    if value is None:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        if (
            not isinstance(payload, list)
            or len(payload) != 3
            or not isinstance(payload[0], int)
            or not isinstance(payload[1], str)
            or not isinstance(payload[2], int)
        ):
            raise ValueError
        return payload[0], payload[1], payload[2]
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid task cursor",
        ) from error


def _summary_payload(summary: RecordSummary) -> dict[str, Any]:
    return {
        "row_id": summary.row_id,
        "task_id": summary.task_id,
        "split": summary.split,
        "distance": summary.distance,
        "input_type": summary.input_type,
        "output_type": summary.output_type,
        "certified": summary.certified,
        "witness_function_names": list(summary.witness_function_names),
        "input_preview": summary.input_preview,
        "output_preview": summary.output_preview,
        "root_index": summary.root_index,
    }


def _basis_for_record(record: RecordDetail):
    assumed = record.basis_set_id is None
    basis = load_basis_set(record.basis_set_id or "default")
    if record.basis_set_digest is not None and basis.digest != record.basis_set_digest:
        raise BasisSetDigestMismatchError(
            f"record says {record.basis_set_digest}, installed basis is {basis.digest}"
        )
    return basis, basis.as_function_set(), assumed


def _basis_payload(record: RecordDetail) -> tuple[dict[str, Any], FunctionDefSet]:
    basis, functions, assumed = _basis_for_record(record)
    return (
        {
            "id": basis.basis_set_id,
            "digest": basis.digest,
            "assumed": assumed,
            "functions": [
                {
                    "id": function.metadata.get("basis_function_id"),
                    "name": function.name,
                    "input_type": function.input_type,
                    "output_type": function.output_type,
                }
                for function in functions
            ],
        },
        functions,
    )


def _record_payload(record: RecordDetail) -> dict[str, Any]:
    basis, _functions = _basis_payload(record)
    return {
        "row_id": record.row_id,
        "task_id": record.task_id,
        "schema_kind": record.schema_kind,
        "split": record.split,
        "input": record.input,
        "output": record.output,
        "input_type": record.input_type,
        "output_type": record.output_type,
        "distance": record.distance,
        "certified": record.certified,
        "witness_function_names": list(record.witness_function_names),
        "witness_function_ids": list(record.witness_function_ids),
        "root_index": record.root_index,
        "certification": record.certification,
        "metadata": _json_safe(record.metadata),
        "functions_by_role": {
            role: list(values) for role, values in record.functions_by_role.items()
        },
        "raw_json": json.dumps(record.raw, ensure_ascii=False, sort_keys=True),
        "basis": basis,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _graph_payload(view: GraphView) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.node_id,
                "x": node.x,
                "y": node.y,
                "depth": int(node.x),
                "label": node.label,
                "value": node.value_repr,
                "role": node.role,
            }
            for node in view.nodes
        ],
        "edges": [
            {
                "id": f"{edge.source_id}:{edge.target_id}",
                "source": edge.source_id,
                "target": edge.target_id,
                "function_names": list(edge.function_names),
                "highlighted": edge.highlighted,
            }
            for edge in view.edges
        ],
        "root_ids": list(view.root_ids),
        "total_nodes": view.total_nodes,
        "total_edges": view.total_edges,
        "rendered_nodes": len(view.nodes),
        "rendered_edge_groups": len(view.edges),
        "truncated": view.truncated,
        "diagnostics": asdict(view.diagnostics),
    }


async def _run_blocking(
    app: FastAPI,
    operation: Callable[[], Any],
    *,
    finish_on_cancel: bool = False,
) -> Any:
    """Run blocking work without touching asyncio's process-global executor.

    ``asyncio.to_thread`` lazily creates the event loop's default executor.
    Some Python launch environments install shutdown hooks which can then
    deadlock ``asyncio.run`` while that executor is being joined.  In those
    environments, the selector wake-up posted by a worker can also be lost.
    Polling the owned concurrent future from an asyncio timer avoids both
    process-global lifecycle hooks and reliance on that cross-thread wake-up.
    """
    executor: ThreadPoolExecutor = app.state.blocking_executor
    future = executor.submit(operation)
    try:
        while not future.done():
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        # Pending work can be discarded. A running filesystem/index/graph
        # operation must be allowed to leave its resources consistent; consume
        # any eventual error so it cannot escape through an orphaned future.
        if future.cancel():
            raise
        if finish_on_cancel:
            while not future.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.01)
            _consume_background_result(future)
        else:
            future.add_done_callback(_consume_background_result)
        raise
    return future.result()


def _consume_background_result(future: Future[Any]) -> None:
    with suppress(BaseException):
        future.result()


async def _bounded_graph_call(app: FastAPI, operation: Callable[[], Any]) -> Any:
    semaphore: asyncio.Semaphore = app.state.graph_semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
    except TimeoutError as error:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "graph workers are busy; retry shortly",
        ) from error
    try:
        return await _run_blocking(app, operation, finish_on_cancel=True)
    finally:
        semaphore.release()


def create_app(
    *,
    roots: Sequence[str | Path] | None = None,
    cache_dir: str | Path | None = None,
    bind_host: str | None = None,
    auth_token: str | None = None,
) -> FastAPI:
    graph_host = bind_host if bind_host is not None else _configured_graph_host()
    graph_token = _validated_graph_token(
        auth_token if auth_token is not None else os.environ.get(GRAPH_TOKEN_ENV)
    )
    remote_mode = not _is_loopback_hostname(graph_host)
    if remote_mode and graph_token is None:
        raise RuntimeError(
            f"Refusing non-loopback {GRAPH_HOST_ENV}={graph_host!r} without "
            f"{GRAPH_TOKEN_ENV}; configure a random token of at least "
            f"{MIN_GRAPH_TOKEN_LENGTH} characters"
        )
    service = CorpusGraphService(roots=roots, cache_dir=cache_dir)
    blocking_executor = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="trajectory-graph-api",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            # Starlette stops accepting requests before lifespan teardown, so
            # no new work can be submitted while these owned pools are joined.
            blocking_executor.shutdown(wait=True, cancel_futures=True)
            service.close()

    app = FastAPI(
        title="Trajectory Graph",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.corpus_graph_service = service
    app.state.blocking_executor = blocking_executor
    app.state.graph_semaphore = asyncio.Semaphore(2)
    app.state.graph_bind_host = graph_host
    app.add_middleware(GZipMiddleware, minimum_size=1_024)

    @app.middleware("http")
    async def local_api_security(request: Request, call_next):
        def secured(response: Response) -> Response:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            # Cytoscape positions its internal canvases through element.style;
            # scripts remain restricted to same-origin bundled assets.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; font-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            )
            if request.url.path.startswith("/api/"):
                response.headers["Cache-Control"] = "no-store"
            return response

        request_hostname = _normalized_hostname(request.url.hostname)
        if not remote_mode and not _is_loopback_hostname(
            request_hostname, allow_test=True
        ):
            return secured(
                JSONResponse(
                    {"detail": "invalid Host header"},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            )

        authorization_kind = _authorization_kind(
            request.headers.get("authorization"), graph_token
        )
        if remote_mode and authorization_kind is None:
            return secured(
                JSONResponse(
                    {"detail": "authentication required"},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={
                        "WWW-Authenticate": (
                            'Basic realm="Wandering Light", charset="UTF-8"'
                        )
                    },
                )
            )

        if (
            request.url.path.startswith("/api/")
            and request.method in _STATE_CHANGING_METHODS
        ):
            origin = request.headers.get("origin")
            # A bearer token is an explicit, non-ambient CSRF credential for
            # scripts and command-line clients. Browser Basic auth is ambient,
            # so it still requires a matching Origin just like loopback mode.
            if origin is None and authorization_kind != "bearer":
                return secured(
                    JSONResponse(
                        {"detail": "a same-origin Origin header is required"},
                        status_code=status.HTTP_403_FORBIDDEN,
                    )
                )
            if origin is not None and _origin_tuple(origin) != _request_origin(request):
                return secured(
                    JSONResponse(
                        {"detail": "cross-origin API mutation rejected"},
                        status_code=status.HTTP_403_FORBIDDEN,
                    )
                )

        response = await call_next(request)
        return secured(response)

    def api_service() -> CorpusGraphService:
        return app.state.corpus_graph_service

    def require_source(source_id: str) -> CorpusSource:
        try:
            return api_service().source(source_id)
        except KeyError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown corpus") from error

    def require_index(source_id: str) -> CorpusIndex:
        require_source(source_id)
        try:
            return api_service().index(source_id)
        except LookupError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, str(error)
            ) from error

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/sources")
    async def sources() -> dict[str, Any]:
        def collect() -> dict[str, Any]:
            entries = []
            for source_id, source in api_service().sources():
                entries.append(
                    _source_payload(
                        source_id,
                        source,
                        api_service().source_status(source_id),
                    )
                )
            return {"sources": entries, "errors": list(api_service().errors)}

        return await _run_blocking(app, collect)

    @app.post("/api/v1/sources/{source_id}/index")
    async def start_index(source_id: str, response: Response) -> dict[str, Any]:
        require_source(source_id)
        try:
            result = await _run_blocking(
                app,
                lambda: api_service().start_index(source_id),
            )
        except FileNotFoundError as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        response.status_code = (
            status.HTTP_200_OK
            if result["status"] == "ready"
            else status.HTTP_202_ACCEPTED
        )
        return result

    @app.post("/api/v1/sources/{source_id}/fetch")
    async def fetch_source(source_id: str) -> dict[str, Any]:
        require_source(source_id)
        try:
            source = await _run_blocking(
                app,
                lambda: api_service().fetch(source_id),
            )
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
        except CorpusFetchError as error:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
        return _source_payload(
            source_id,
            source,
            api_service().source_status(source_id),
        )

    @app.get("/api/v1/sources/{source_id}/facets")
    async def facets(
        source_id: str,
        function_role: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        index = require_index(source_id)

        def collect() -> dict[str, Any]:
            try:
                functions = index.function_totals(
                    roles=function_role or FUNCTION_ROLES,
                    limit=500,
                )
            except ValueError as error:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
                ) from error
            return {
                "stats": asdict(index.stats()),
                "splits": list(index.values("split")),
                "distance_counts": [
                    {"value": value, "records": records}
                    for value, records in index.counts("distance")
                ],
                "input_types": list(index.values("input_type")),
                "output_types": list(index.values("output_type")),
                "certifications": [
                    {"value": value, "records": records}
                    for value, records in index.counts("certification")
                ],
                "functions": [asdict(function) for function in functions],
            }

        return await _run_blocking(app, collect)

    @app.get("/api/v1/sources/{source_id}/tasks")
    async def tasks(
        source_id: str,
        split: Annotated[list[str] | None, Query()] = None,
        min_distance: Annotated[int | None, Query(ge=0)] = None,
        max_distance: Annotated[int | None, Query(ge=0)] = None,
        input_type: Annotated[list[str] | None, Query()] = None,
        output_type: Annotated[list[str] | None, Query()] = None,
        root_index: Annotated[list[int] | None, Query()] = None,
        function_key: Annotated[list[str] | None, Query()] = None,
        function_match: Literal["any", "all"] = "any",
        function_role: Annotated[list[str] | None, Query()] = None,
        task_prefix: Annotated[str | None, Query(max_length=128)] = None,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        index = require_index(source_id)
        filters = _filters(
            splits=split or (),
            min_distance=min_distance,
            max_distance=max_distance,
            input_types=input_type or (),
            output_types=output_type or (),
            root_indices=root_index or (),
            function_keys=function_key or (),
            function_match=function_match,
            function_roles=function_role or ("witness",),
        )
        after = _cursor_decode(cursor)

        def collect() -> dict[str, Any]:
            if task_prefix:
                rows = index.find_task(task_prefix, filters=filters)
                return {
                    "items": [_summary_payload(row) for row in rows],
                    "next_cursor": None,
                    "total": len(rows),
                }
            rows = index.records_after(filters, limit=limit + 1, after=after)
            has_more = len(rows) > limit
            visible = rows[:limit]
            return {
                "items": [_summary_payload(row) for row in visible],
                "next_cursor": (
                    _cursor_encode(visible[-1]) if has_more and visible else None
                ),
                "total": index.count_records(filters),
            }

        return await _run_blocking(app, collect)

    @app.get("/api/v1/sources/{source_id}/tasks/{row_id}")
    async def task_detail(source_id: str, row_id: int) -> dict[str, Any]:
        index = require_index(source_id)

        def collect() -> dict[str, Any]:
            try:
                record = index.get_record(row_id)
            except KeyError as error:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, "unknown task"
                ) from error
            return _record_payload(record)

        return await _run_blocking(app, collect)

    @app.post("/api/v1/sources/{source_id}/graphs/witnesses")
    async def witness_graph(
        source_id: str,
        body: WitnessGraphRequest,
    ) -> dict[str, Any]:
        index = require_index(source_id)

        def build() -> dict[str, Any]:
            try:
                record = index.get_record(body.row_id)
            except KeyError as error:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, "unknown task"
                ) from error
            _basis, functions, _assumed = _basis_for_record(record)
            try:
                require_reproducible_basis_runtime(_basis)
            except RuntimeError as error:
                raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
            records = [record]
            if body.scope == "root":
                if record.root_index is None:
                    raise ValueError("selected task has no root grouping")
                summaries = index.records(
                    CorpusFilters(root_indices=(record.root_index,)),
                    limit=body.max_records,
                )
                records.extend(
                    index.get_record(summary.row_id)
                    for summary in summaries
                    if summary.row_id != record.row_id
                )
            projection = build_witness_projection(
                records,
                functions,
                selected_task_id=record.task_id,
                max_records=body.max_records,
                max_nodes=body.max_nodes,
                max_edges=body.max_edges,
            )
            return {
                "mode": "witnesses",
                "graph": _graph_payload(projection.view),
                "processed_records": projection.processed_records,
                "skipped_records": projection.skipped_records,
                "errors": list(projection.errors),
            }

        try:
            return await _bounded_graph_call(app, build)
        except ValueError as error:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.post("/api/v1/sources/{source_id}/graphs/expand")
    async def expansion_graph(
        source_id: str,
        body: ExpansionGraphRequest,
    ) -> dict[str, Any]:
        index = require_index(source_id)

        def build() -> dict[str, Any]:
            try:
                record = index.get_record(body.row_id)
            except KeyError as error:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, "unknown task"
                ) from error
            basis, functions, _assumed = _basis_for_record(record)
            try:
                require_reproducible_basis_runtime(basis)
            except RuntimeError as error:
                raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error
            by_id = {
                function.metadata.get("basis_function_id"): function
                for function in functions
            }
            selected = []
            for function_id in body.function_ids:
                function = by_id.get(function_id)
                if function is None:
                    raise ValueError(
                        f"basis does not contain function ID {function_id!r}"
                    )
                selected.append(function)
            input_serialized = body.input_serialized or record.input
            input_value = typed_list_from_builtin_str(input_serialized)
            validate_typed_list_workload(
                input_value,
                input_serialized,
                label="local expansion input",
            )
            projection = build_local_expansion(
                input_value,
                FunctionDefSet(selected),
                max_depth=body.max_depth,
                max_states=body.max_states,
                max_transitions=body.max_transitions,
                skip_self_loops=not body.include_self_loops,
                max_nodes=body.max_nodes,
                max_edges=body.max_edges,
            )
            return {
                "mode": "expansion",
                "graph": _graph_payload(projection.view),
                "tasks": [asdict(task) for task in projection.tasks],
                "attempted_transitions": projection.attempted_transitions,
                "failed_transitions": projection.failed_transitions,
                "skipped_self_loops": projection.skipped_self_loops,
                "certified_depth": projection.certified_depth,
                "stop_reason": projection.stop_reason,
            }

        try:
            return await _bounded_graph_call(app, build)
        except ValueError as error:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.get("/assets/{asset_path:path}", include_in_schema=False)
    async def frontend_asset(asset_path: str):
        assets_root = (WEB_ROOT / "assets").resolve()
        candidate = (assets_root / asset_path).resolve()
        if not candidate.is_relative_to(assets_root) or not candidate.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "asset not found")
        content = await _run_blocking(app, candidate.read_bytes)
        media_type = (
            mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        )
        return Response(
            content=content,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        if path.startswith("api/"):
            return JSONResponse(
                {"detail": "not found"}, status_code=status.HTTP_404_NOT_FOUND
            )
        index_path = WEB_ROOT / "index.html"
        if not index_path.is_file():
            return JSONResponse(
                {
                    "detail": "frontend assets are missing; run npm run build in "
                    "web/trajectory-graph"
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        content = await _run_blocking(app, index_path.read_bytes)
        return Response(
            content=content,
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=app.state.graph_bind_host,
        port=int(os.environ.get("WANDERING_LIGHT_GRAPH_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    main()
