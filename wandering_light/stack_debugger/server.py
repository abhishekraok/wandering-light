"""Serve the stack debugger over localhost.

Four things need the server: building a session (which runs the search),
inspecting one state (the full product, the policy's raw row, the tokens the
model saw), taking a drive step, and playing a program. Everything else --
stepping ops, changing layers, selecting nodes -- is navigation of a trace
the page already holds.

A state is recomputed from its path through ``step_all_explained``, the same
call the search makes, so there is no per-node store to grow or go stale.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from wandering_light.quotation_guard import blocked
from wandering_light.stack_beam import step_all_explained
from wandering_light.stack_debugger.session import (
    SearchSettings,
    bucket_of,
    build_session,
    cohort_indices,
    encoding_view,
    parse_product,
    select_task,
    witness_programs,
)
from wandering_light.stack_debugger.trace import (
    OP_NAMES,
    SUCCESSOR_DISPLAY_LIMIT,
    product_text,
    stack_text,
)
from wandering_light.stack_debugger.walk import play_program
from wandering_light.stack_machine import Sym, value_key
from wandering_light.stack_pbe import OPS
from wandering_light.stack_policy import encode_item, encode_targets, item_array

if TYPE_CHECKING:
    from wandering_light.stack_beam import Product
    from wandering_light.stack_debugger.context import DebugContext
    from wandering_light.stack_pbe import Task

STATIC_DIR = Path(__file__).parent / "static"


def _task(context: DebugContext, index: int) -> Task:
    if not 0 <= index < len(context.tasks):
        raise HTTPException(status_code=404, detail=f"No eval task {index}")
    return context.tasks[index]


def _replay(task: Task, path: list[str]) -> list[Product]:
    """Every product along a path, from the inputs, through the search's step."""
    by_name = {op.name: op for op in OPS}
    product = parse_product(task.inputs)
    trail = [product]
    for name in path:
        op = by_name.get(name)
        if op is None:
            raise HTTPException(status_code=400, detail=f"Unknown op {name!r}")
        moved, _fuel, _stepped, reason = step_all_explained(product, op)
        if moved is None:
            raise HTTPException(status_code=422, detail=f"{name} refused: {reason}")
        product = moved
        trail.append(product)
    return trail


def _op_table(
    context: DebugContext,
    product: Product,
    outputs: Product,
    remaining: int,
    guard: str,
) -> dict[str, Any]:
    """Every op from one state: machine verdict, policy row, guard on the successor.

    The same mask and per-node normalization the beam uses, so a hand-driven
    step and a searched step rank ops alike.
    """
    targets = encode_targets(outputs)
    block = item_array(
        [encode_item(s, t) for s, t in zip(product, targets, strict=True)]
    )
    logits = context.scorer.logits([block])[0]
    rows = []
    legal = np.zeros(len(OPS), dtype=bool)
    for index, op in enumerate(OPS):
        moved, fuel, stepped, reason = step_all_explained(product, op)
        legal[index] = moved is not None
        captured = (
            moved is not None
            and len(moved[0]) == len(product[0]) + 1
            and moved[0][-1] == Sym(op.name)
        )
        rows.append(
            {
                "op": op.name,
                "logit": round(float(logits[index]), 4),
                "outcome": "refused"
                if moved is None
                else ("captured" if captured else "executed"),
                "reason": reason,
                "fuel": fuel,
                "stepped": stepped,
                "successor": None
                if moved is None
                else stack_text(moved[0], SUCCESSOR_DISPLAY_LIMIT),
                "guard": None
                if moved is None or guard == "off"
                else blocked(moved[0], remaining, guard),
            }
        )
    log_probs = context.scorer.log_probs([block], legal[None, :])[0]
    for index, row in enumerate(rows):
        value = float(log_probs[index])
        row["log_probability"] = None if math.isinf(value) else round(value, 4)
        row["probability"] = None if math.isinf(value) else round(math.exp(value), 6)
    ranked = sorted(
        (row for row in rows if row["log_probability"] is not None),
        key=lambda row: (-row["logit"], row["op"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", None)
    return {"ops": rows, "legal": int(legal.sum())}


def create_app(context: DebugContext) -> FastAPI:
    app = FastAPI(title="Stack debugger")
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    cohort = set(cohort_indices(context.tasks))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/context")
    def get_context() -> dict[str, Any]:
        behaviours: dict[str, dict[str, Any]] = {}
        for i, task in enumerate(context.tasks):
            entry = behaviours.setdefault(
                task.behaviour,
                {
                    "name": task.behaviour,
                    "summary": task.summary,
                    "length": task.length,
                    "tasks": 0,
                    "cohort": 0,
                },
            )
            entry["tasks"] += 1
            entry["cohort"] += i in cohort
        return {
            "provenance": context.provenance,
            "tasks": len(context.tasks),
            "cohort": len(cohort),
            "buckets": [
                bucket_of(low)
                for low, _ in ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10**9))
            ],
            "behaviours": sorted(
                behaviours.values(), key=lambda b: (b["length"], b["name"])
            ),
            "ops": list(OP_NAMES),
        }

    @app.get("/api/session")
    def get_session(
        task: int | None = None,
        behaviour: str | None = None,
        bucket: str | None = None,
        cohort_only: bool = False,
        seed: int | None = None,
        beam_width: int = Query(default=64, ge=1),
        max_depth: int = Query(default=48, ge=1),
        guard: str = Query(default="both"),
        budget: int | None = Query(default=None, ge=0),
        include_map: bool = True,
    ) -> dict[str, Any]:
        try:
            index, selection = select_task(
                context.tasks,
                task=task,
                behaviour=behaviour,
                bucket=bucket,
                cohort=cohort_only,
                seed=seed,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        settings = SearchSettings(
            beam_width=beam_width, max_depth=max_depth, guard=guard, budget=budget
        )
        session = build_session(
            context.scorer,
            context.tasks,
            index,
            settings=settings,
            provenance={**context.provenance, "selection": selection},
        )
        payload = session.to_dict()
        payload["task"]["in_cohort"] = index in cohort
        payload["map"] = None
        if include_map:
            from wandering_light.stack_debugger.feature_map import session_map

            payload["map"] = session_map(
                context.scorer,
                parse_product(session.task.inputs),
                parse_product(session.task.outputs),
                payload["trace"]["layers"],
                [layer.retained_products for layer in session.recorder.layers],
                [layer.features for layer in session.recorder.layers],
            )
        return payload

    @app.get("/api/state")
    def get_state(
        task: int,
        path: str = "",
        max_depth: int = Query(default=48, ge=1),
        guard: str = Query(default="both"),
    ) -> dict[str, Any]:
        """One state in full: every example, every op, the tokens, the map frame."""
        record = _task(context, task)
        steps = [name for name in path.split(",") if name]
        trail = _replay(record, steps)
        product = trail[-1]
        outputs = parse_product(record.outputs)
        target = [tuple(map(value_key, s)) for s in outputs]
        remaining = max(0, max_depth - len(steps) - 1)
        table = _op_table(context, product, outputs, remaining, guard)
        from wandering_light.stack_debugger.feature_map import progress_frame

        return {
            "path": steps,
            "states": product_text(product),
            "matches": [
                tuple(map(value_key, s)) == t
                for s, t in zip(product, target, strict=True)
            ],
            "solved": all(
                tuple(map(value_key, s)) == t
                for s, t in zip(product, target, strict=True)
            ),
            "remaining": remaining,
            "encoding": encoding_view(product, outputs),
            "progress": progress_frame(
                context.scorer, parse_product(record.inputs), outputs, trail
            ),
            **table,
        }

    @app.get("/api/play")
    def get_play(task: int, program: str = "witness") -> dict[str, Any]:
        """Step a program through the machine: ``witness``, ``witness:N``,
        ``solution`` or a comma-separated op list."""
        record = _task(context, task)
        programs = witness_programs(record)
        if program.startswith("witness"):
            which = int(program.split(":")[1]) if ":" in program else 0
            if not 0 <= which < len(programs):
                raise HTTPException(status_code=404, detail=f"No witness {which}")
            ops = programs[which]
            label = f"witness {which + 1} of {len(programs)}"
        elif program == "solution":
            raise HTTPException(
                status_code=400,
                detail="pass the found program as an op list; the server keeps no session",
            )
        else:
            ops = [name for name in program.split(",") if name]
            label = "program"
        return {"label": label, **play_program(record, ops)}

    return app


def serve_debugger(context: DebugContext, *, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(context), host=host, port=port, log_level="info")
