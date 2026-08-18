"""FastAPI app for the explorer.

The browser holds the session -- the current trajectory, what is selected, what
is drawn -- and calls here only to execute basis functions, expand a graph, run
a solver or read a corpus. Nothing is stored server-side, so any tab can be
reloaded or duplicated without losing or sharing state.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wandering_light.basis_set import (
    available_basis_set_aliases,
    available_basis_sets,
    load_basis_set,
    require_reproducible_basis_runtime,
)
from wandering_light.evals import corpus_view
from wandering_light.webapp import core

if TYPE_CHECKING:
    from wandering_light.function_def import FunctionDefSet

DEFAULT_BASIS = "wl-core-v1"
STATIC_ROOT = Path(__file__).parent / "static"
MAX_CORPUS_RECORDS = 5000


@lru_cache(maxsize=8)
def _functions(basis_set_id: str) -> FunctionDefSet:
    basis_set = load_basis_set(basis_set_id)
    # Refuse a randomized-hash palette rather than serve states that will not
    # reproduce in the next process.
    require_reproducible_basis_runtime(basis_set)
    return basis_set.as_function_set()


def _resolve(basis_set_id: str | None, function_names: list[str] | None):
    try:
        functions = _functions(basis_set_id or DEFAULT_BASIS)
    except Exception as error:
        raise HTTPException(400, f"basis: {error}") from error
    if function_names is None:
        return functions
    unknown = [n for n in function_names if n not in functions.name_to_function]
    if unknown:
        raise HTTPException(400, f"unknown functions: {', '.join(unknown)}")
    return core.palette(functions, function_names)


def _state(text: str):
    try:
        return core.parse_state(text)
    except Exception as error:
        raise HTTPException(400, f"could not parse state: {error}") from error


class BasisRequest(BaseModel):
    basis_set_id: str | None = None
    functions: list[str] | None = None


class StateRequest(BasisRequest):
    state: str


class TrajectoryRequest(StateRequest):
    steps: list[str] = Field(default_factory=list)


class ExpandRequest(StateRequest):
    max_depth: int = Field(2, ge=1, le=6)
    max_states: int = Field(200, ge=2, le=5000)
    max_transitions: int = Field(20_000, ge=10, le=2_000_000)
    include_self_loops: bool = True


class SolveRequest(StateRequest):
    target: str
    solver: str = "bfs"
    budget: int = Field(2000, ge=1, le=200_000)
    max_depth: int = Field(3, ge=1, le=8)


def create_app() -> FastAPI:
    app = FastAPI(title="Wandering Light explorer", docs_url="/api/docs")

    @app.get("/api/basis")
    def basis(basis_set_id: str | None = None) -> dict[str, Any]:
        resolved = basis_set_id or DEFAULT_BASIS
        try:
            basis_set = load_basis_set(resolved)
        except Exception as error:
            raise HTTPException(400, f"basis: {error}") from error
        aliases = {v: k for k, v in available_basis_set_aliases().items()}
        return {
            "available": [
                {"id": name, "alias": aliases.get(name)}
                for name in available_basis_sets()
            ],
            "basis_set_id": basis_set.basis_set_id,
            "digest": basis_set.digest,
            "description": basis_set.description,
            "functions": [
                {
                    "name": function.name,
                    "input_type": function.input_type.removeprefix("builtins."),
                    "output_type": function.output_type.removeprefix("builtins."),
                    "code": function.code,
                    "function_id": function.function_id,
                }
                for function in basis_set.functions
            ],
        }

    @app.post("/api/state")
    def parse_state(request: StateRequest) -> dict[str, Any]:
        return core.state_view(_state(request.state)).dict()

    @app.post("/api/trajectory")
    def trajectory(request: TrajectoryRequest) -> dict[str, Any]:
        functions = _resolve(request.basis_set_id, None)
        start = _state(request.state)
        steps = core.run_trajectory(start, request.steps, functions)
        return {
            "root": core.state_view(start).dict(),
            "steps": [step.dict() for step in steps],
        }

    @app.post("/api/successors")
    def successors(request: StateRequest) -> dict[str, Any]:
        functions = _resolve(request.basis_set_id, request.functions)
        value = _state(request.state)
        results = core.successors(value, functions)
        return {
            "state": core.state_view(value).dict(),
            "successors": [item.dict() for item in results],
            "applicable": len(results),
            "dead_end": not any(item.ok and not item.self_loop for item in results),
        }

    @app.post("/api/expand")
    def expand(request: ExpandRequest) -> dict[str, Any]:
        functions = _resolve(request.basis_set_id, request.functions)
        return core.expand(
            _state(request.state),
            functions,
            max_depth=request.max_depth,
            max_states=request.max_states,
            max_transitions=request.max_transitions,
            include_self_loops=request.include_self_loops,
        ).dict()

    @app.post("/api/solve")
    def solve(request: SolveRequest) -> dict[str, Any]:
        functions = _resolve(request.basis_set_id, request.functions)
        try:
            attempt = core.solve(
                _state(request.state),
                _state(request.target),
                functions,
                solver=request.solver,
                budget=request.budget,
                max_depth=request.max_depth,
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return attempt.dict()

    @app.get("/api/corpora")
    def corpora() -> dict[str, Any]:
        entries = []
        for ref in corpus_view.discover_corpora():
            manifest = corpus_view.load_manifest(ref)
            headline = corpus_view.corpus_headline(manifest)
            entries.append(
                {
                    "name": ref.name,
                    "tasks": headline["tasks"],
                    "basis_set_id": headline["basis_set_id"],
                    "splits": list(manifest["splits"]),
                    "missing_splits": corpus_view.missing_splits(ref, manifest),
                    "distances": corpus_view.corpus_distance_profile(
                        manifest, name=ref.name
                    ).counts,
                }
            )
        return {"corpora": entries}

    @app.get("/api/corpora/{name}/tasks")
    def corpus_tasks(
        name: str, split: str, limit: int = 200, distance: int | None = None
    ) -> dict[str, Any]:
        limit = max(1, min(limit, MAX_CORPUS_RECORDS))
        ref = next((r for r in corpus_view.discover_corpora() if r.name == name), None)
        if ref is None:
            raise HTTPException(404, f"no corpus named {name!r}")
        manifest = corpus_view.load_manifest(ref)
        if split not in manifest["splits"]:
            raise HTTPException(404, f"no split {split!r} in {name!r}")
        if split in corpus_view.missing_splits(ref, manifest):
            raise HTTPException(
                409, f"payload for {split!r} is not downloaded; run fetch_corpus"
            )
        # A distance filter has to read past its own limit, since records are
        # not ordered by distance.
        scan = limit if distance is None else MAX_CORPUS_RECORDS
        records = corpus_view.load_records(ref, manifest, split, limit=scan)
        if distance is not None:
            records = corpus_view.filter_records(records, distances=[distance])
        records = records[:limit]
        return {
            "tasks": [
                {
                    "task_id": record.task_id,
                    "input": record.input,
                    "output": record.output,
                    "input_label": repr(record.input_value),
                    "output_label": repr(record.output_value),
                    "witness": list(record.witness_function_names),
                    "distance": record.metadata["certified_distance"],
                    "certification": record.metadata["certification"],
                    "optimal_first": record.metadata["optimal_first_action_names"],
                }
                for record in records
            ]
        }

    # The build output is not committed, so the app has to be useful without it:
    # the API still serves, and the root explains the one command that is missing.
    index_html = STATIC_ROOT / "index.html"
    assets = STATIC_ROOT / "assets"
    if index_html.is_file():
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(index_html)

    else:

        @app.get("/")
        def missing_build() -> dict[str, str]:
            return {
                "detail": "Frontend is not built. Run: "
                "cd wandering_light/webapp/frontend && npm install && npm run build"
            }

    return app


app = create_app()
