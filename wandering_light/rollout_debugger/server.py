"""Serve the interactive rollout debugger over localhost.

Only three things need the server: building a session (which runs the search),
inspecting one node on demand (raw logits, model input), and taking a step in
drive mode. Everything else -- stepping through expansions, changing layers,
selecting nodes, replaying -- is navigation of a trace the client already
holds, so it costs no round-trip and stays instant under the hand.

State is recomputed from a path rather than stored per node. A path plus the
task is a complete description of a node, so a request carries one, the server
replays it through the same ``Executor`` the search used, and there is no
session cache to go stale or to grow.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from wandering_light.debugger_http import (
    is_loopback_host,
    local_debugger_requests,
    parse_op_path,
)
from wandering_light.executor import Executor
from wandering_light.rollout_debugger.session import (
    SearchSettings,
    build_session,
    encoding_view,
    select_task,
)

if TYPE_CHECKING:
    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.rollout_debugger.context import DebugContext
    from wandering_light.typed_list import TypedList

STATIC_DIR = Path(__file__).parent / "static"


def _record_for(context: DebugContext, task_id: str) -> BasisTaskRecord:
    for record in context.records:
        if record.task_id == task_id:
            return record
    raise HTTPException(status_code=404, detail=f"No task {task_id} in this split")


def _replay(
    context: DebugContext, record: BasisTaskRecord, path: list[str]
) -> TypedList:
    """Re-execute a path from the task input, through the production executor."""

    by_name = {function.name: function for function in context.available_functions}
    executor = Executor(context.available_functions)
    state = record.input_value
    for name in path:
        function = by_name.get(name)
        if function is None:
            raise HTTPException(status_code=400, detail=f"Unknown function {name}")
        if state.item_type is not function.input_type_cls():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} does not accept {state.item_type.__name__}; "
                    "this transition is not in the search space"
                ),
            )
        try:
            state = executor.execute(function, state)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"{name} raised: {type(error).__name__}: {error}",
            ) from error
    return state


def _ranked_actions(
    context: DebugContext, state: TypedList, target: TypedList
) -> dict[str, Any]:
    """Score one state and rank the type-compatible actions.

    This is the same scorer call, mask and per-node normalization the rollout
    performs, so a hand-driven step and a searched step rank actions alike.
    """

    compatible = [
        function
        for function in context.available_functions
        if state.item_type is function.input_type_cls()
    ]
    row = context.policy.score_actions_batch(
        [(state, target)], context.available_functions
    )[0]
    if not compatible:
        return {"actions": [], "log_normalizer": None, "mask_size": 0}

    top = max(row[function.name] for function in compatible)
    log_normalizer = top + math.log(
        sum(math.exp(row[function.name] - top) for function in compatible)
    )
    actions = sorted(
        (
            {
                "function": function.name,
                "logit": round(row[function.name], 6),
                "log_probability": round(row[function.name] - log_normalizer, 6),
                "probability": round(math.exp(row[function.name] - log_normalizer), 6),
                "output_type": function.output_type,
            }
            for function in compatible
        ),
        key=lambda entry: (-entry["logit"], entry["function"]),
    )
    for rank, entry in enumerate(actions, start=1):
        entry["rank"] = rank
    return {
        "actions": actions,
        "log_normalizer": round(log_normalizer, 6),
        "mask_size": len(compatible),
        "basis_size": len(context.available_functions.functions),
    }


def create_app(context: DebugContext) -> FastAPI:
    app = FastAPI(title="Rollout debugger")
    # The largest thing served is one session's trace: a few megabytes of very
    # repetitive JSON, which compresses about fourteen-fold.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.middleware("http")(local_debugger_requests)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/context")
    def get_context() -> dict[str, Any]:
        distances = sorted(
            {
                record.metadata.get("certified_distance")
                for record in context.records
                if isinstance(record.metadata.get("certified_distance"), int)
            }
        )
        input_types = sorted(
            {record.input_value.item_type.__name__ for record in context.records}
        )
        return {
            "provenance": context.provenance,
            "distances": distances,
            "input_types": input_types,
            "tasks": len(context.records),
        }

    @app.get("/api/session")
    def get_session(
        task_id: str | None = None,
        distance: int | None = None,
        input_type: str | None = None,
        seed: int | None = None,
        beam_width: int = Query(default=64, ge=1, le=128),
        max_depth: int = Query(default=8, ge=0, le=32),
        budget: int = Query(default=100_000, ge=0, le=1_000_000),
        capture_logits: bool = False,
        include_map: bool = True,
    ) -> dict[str, Any]:
        try:
            record, selection = select_task(
                context.records,
                task_id=task_id,
                certified_distance=distance,
                input_type=input_type,
                seed=seed,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        session = build_session(
            context.policy,
            record,
            context.available_functions,
            settings=SearchSettings(
                beam_width=beam_width, max_depth=max_depth, budget=budget
            ),
            provenance={**context.provenance, "selection": selection},
            capture_logits=capture_logits,
        )
        payload = session.to_dict()
        payload["map"] = _session_map(context, record, payload) if include_map else None
        return payload

    @app.get("/api/logits")
    def get_logits(
        task_id: str, path: str = Query(default="", max_length=4096)
    ) -> dict[str, Any]:
        """The full raw row for one node, fetched only when the panel opens."""

        record = _record_for(context, task_id)
        steps = parse_op_path(path)
        state = _replay(context, record, steps)
        row = context.policy.score_actions_batch(
            [(state, record.output_value)], context.available_functions
        )[0]
        compatible = {
            function.name
            for function in context.available_functions
            if state.item_type is function.input_type_cls()
        }
        ranked = sorted(row.items(), key=lambda item: (-item[1], item[0]))
        return {
            "path": steps,
            "logits": [
                {
                    "function": name,
                    "logit": round(score, 6),
                    "compatible": name in compatible,
                }
                for name, score in ranked
            ],
        }

    @app.get("/api/encoding")
    def get_encoding(
        task_id: str, path: str = Query(default="", max_length=4096)
    ) -> dict[str, Any]:
        record = _record_for(context, task_id)
        steps = parse_op_path(path)
        state = _replay(context, record, steps)
        return encoding_view(state, record.output_value)

    @app.get("/api/drive")
    def get_drive(
        task_id: str, path: str = Query(default="", max_length=4096)
    ) -> dict[str, Any]:
        """One hand-driven step: where the path leads, and what to press next."""

        record = _record_for(context, task_id)
        steps = parse_op_path(path)
        state = _replay(context, record, steps)
        target = record.output_value
        ranked = _ranked_actions(context, state, target)
        # Every state the walk has passed through, so the map can draw a trail
        # rather than a single dot with no history.
        trail = [
            _replay(context, record, steps[:index]) for index in range(len(steps) + 1)
        ]
        return {
            "path": steps,
            "state": {
                "item_type": state.item_type.__name__,
                "value": repr(state.items),
                "length": len(state.items),
            },
            "solved": state == target,
            "progress": _progress(context, record, trail),
            **ranked,
        }

    return app


def _has_encoder(context: DebugContext) -> bool:
    return hasattr(getattr(context.policy, "model", None), "pooled_features")


def _progress(
    context: DebugContext, record: BasisTaskRecord, trail: list[TypedList]
) -> dict[str, Any] | None:
    """Where the driven walk sits on the start-to-target axis."""

    if not _has_encoder(context):
        return None
    from wandering_light.rollout_debugger.feature_map import progress_frame

    return progress_frame(
        context.policy, record.input_value, record.output_value, trail
    )


def _session_map(
    context: DebugContext, record: BasisTaskRecord, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Lay out the session's retained nodes in the policy's own space.

    The map is the one thing here that needs more than the ``ActionScorer``
    interface: it reads the encoder underneath. A search driven by a scorer
    without one -- a lookup table, a stub -- is still fully debuggable, so this
    returns ``None`` rather than refusing the session.
    """

    from wandering_light.rollout_debugger.feature_map import session_map

    if not _has_encoder(context):
        return None

    layers = payload["trace"]["layers"]
    states_by_depth = [
        [_replay(context, record, entry["path"]) for entry in layer["retained"]]
        for layer in layers
    ]
    return session_map(
        context.policy,
        record.output_value,
        layers,
        states_by_depth,
        start=record.input_value,
    )


def serve_debugger(context: DebugContext, *, host: str, port: int) -> None:
    import uvicorn

    if not is_loopback_host(host):
        raise ValueError("Debugger must bind to a loopback address")

    uvicorn.run(create_app(context), host=host, port=port, log_level="info")
