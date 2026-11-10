"""Signal House over localhost: one JSON POST route per workshop action.

``api(route, data)`` is the whole service and is what the tests call; the
HTTP handler only decodes, guards the origin and encodes. Requests are served
on 127.0.0.1 and a browser page from any other origin is refused, so a page
on the open web cannot drive the workshop.

    uv run python -m wandering_light.stack_workshop.server --port 8018
"""

from __future__ import annotations

import argparse
import gzip
import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wandering_light.stack_machine import MARK_OPS
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    MACROS,
    execute,
    expansion,
    inspect_op,
    stack_from,
)
from wandering_light.stack_workshop.model import ADVISER

HOST = "127.0.0.1"
DEFAULT_PORT = 8018
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
MAX_BODY = 1 << 20
SANDBOX = "sandbox"
FAILURES_SHOWN = 3

WORLD_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "data"
    / "stack_pbe_world_v2"
    / "manifest.json"
)
WORLD_EXAMPLES = 6


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def public_lesson(lesson: dict[str, Any]) -> dict[str, Any]:
    """A lesson as the page may see it: no solution, extra tests or hints."""
    hidden = {"solution", "heldout", "hints"}
    view = {key: value for key, value in lesson.items() if key not in hidden}
    view["extra_tests"] = len(lesson["heldout"])
    view["hint_count"] = len(lesson["hints"])
    return view


def _macro_entries() -> list[dict[str, Any]]:
    entries = []
    for name, macro in MACROS.items():
        try:
            expanded, error = " ".join(expansion(name)), None
        except ValueError as failure:
            expanded, error = None, str(failure)
        entries.append(
            {
                "name": name,
                "body": macro["body"],
                "group": macro["group"],
                "expansion": expanded,
                "error": error,
            }
        )
    return entries


@lru_cache(maxsize=1)
def world_shelf(manifest_path: Path = WORLD_MANIFEST) -> dict[str, Any]:
    """World-authored tasks to try in the sandbox, when they are on disk.

    Read only from the local corpus directory, never downloaded: the shelf
    is a convenience and must not make the workshop depend on the network.
    Any problem yields an empty shelf with a reason.
    """
    shelf: dict[str, Any] = {"available": False, "corpus": None, "tasks": []}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shelf["corpus"] = manifest.get("corpus")
        tasks = []
        for split in sorted(manifest.get("splits", {})):
            payload = manifest_path.parent / manifest["splits"][split]["path"]
            if not payload.exists():
                continue
            with gzip.open(payload, "rt", encoding="utf-8") as handle:
                for line in handle:
                    task = _shelf_task(json.loads(line), split)
                    if task is not None:
                        tasks.append(task)
        shelf["tasks"] = tasks
        shelf["available"] = bool(tasks)
        if not tasks:
            shelf["reason"] = (
                "The corpus payload is not on disk; fetch it with "
                "wandering_light.corpus_hub.fetch_corpus to fill the shelf."
            )
    except Exception as error:  # the shelf is optional; never fail the context
        shelf["reason"] = f"{type(error).__name__}: {error}"
    return shelf


def _shelf_task(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    examples = []
    for given, wanted in zip(
        row.get("inputs", ()), row.get("outputs", ()), strict=False
    ):
        try:
            stack_from(given)
            stack_from(wanted, expected=True)
        except ValueError:
            continue
        examples.append({"input": given, "output": wanted})
        if len(examples) == WORLD_EXAMPLES:
            break
    if not examples:
        return None
    return {
        "behaviour": row.get("behaviour"),
        "summary": row.get("summary"),
        "split": split,
        "examples": examples,
    }


def context() -> dict[str, Any]:
    return {
        "lessons": [public_lesson(lesson) for lesson in LESSONS],
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect} for op in MARK_OPS
        ],
        "macros": _macro_entries(),
        "model": ADVISER.status(),
        "world": world_shelf(),
    }


# ---------------------------------------------------------------------------
# Lesson routes
# ---------------------------------------------------------------------------


def _lesson(data: dict[str, Any]) -> dict[str, Any] | None:
    """The lesson a request names; ``None`` for the sandbox."""
    name = data.get("lesson")
    if name == SANDBOX:
        return None
    if not isinstance(name, str) or name not in BY_ID:
        raise ValueError(f"Unknown lesson {name!r}.")
    return BY_ID[name]


def _shown(data: dict[str, Any], lesson: dict[str, Any] | None) -> list:
    return data.get("examples") if lesson is None else lesson["examples"]


def _program(data: dict[str, Any]) -> str:
    program = data.get("program", "")
    if not isinstance(program, str):
        raise ValueError("program is text.")
    return program


def _failures(result: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        {**row, "kind": kind, "index": index}
        for index, row in enumerate(result["examples"])
        if not row["matched"]
    ]


def verify(data: dict[str, Any]) -> dict[str, Any]:
    lesson = _lesson(data)
    program = _program(data)
    shown = execute(program, _shown(data, lesson), capture=False)
    extra_tests = [] if lesson is None else lesson["heldout"]
    extra = execute(program, extra_tests, capture=False) if extra_tests else None
    failures = _failures(shown, "shown")
    if extra is not None:
        failures += _failures(extra, "extra")
    extra_matches = extra["matches"] if extra else 0
    return {
        "passed": shown["matches"] == shown["count"]
        and extra_matches == len(extra_tests),
        "shown": shown["matches"],
        "shown_count": shown["count"],
        "extra": extra_matches,
        "extra_count": len(extra_tests),
        "failures": failures[:FAILURES_SHOWN],
        "evidence": "shown and extra tests" if extra_tests else "examples",
    }


def _needs_lesson(data: dict[str, Any], what: str) -> dict[str, Any]:
    lesson = _lesson(data)
    if lesson is None:
        raise ValueError(f"The sandbox has no {what}.")
    return lesson


def hint(data: dict[str, Any]) -> dict[str, Any]:
    lesson = _needs_lesson(data, "hints")
    index = data.get("index", 0)
    hints = lesson["hints"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("index is a hint number.")
    if not 0 <= index < len(hints):
        raise ValueError(f"No hint {index}; this lesson has {len(hints)}.")
    return {"hint": hints[index], "index": index, "hint_count": len(hints)}


def reveal(data: dict[str, Any]) -> dict[str, Any]:
    lesson = _needs_lesson(data, "solution")
    return {"solution": lesson["solution"], "discovery": lesson["discovery"]}


def model(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("load"):
        checkpoint = data.get("checkpoint")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("checkpoint is a path.")
        return ADVISER.load(checkpoint, download=bool(data.get("download")))
    return ADVISER.status()


def api(route: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one request. Raises ``ValueError`` for anything malformed."""
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("The request body is a JSON object.")
    if route == "/api/context":
        return context()
    if route == "/api/model":
        return model(data)
    if route == "/api/verify":
        return verify(data)
    if route == "/api/hint":
        return hint(data)
    if route == "/api/reveal":
        return reveal(data)
    if route in ("/api/evaluate", "/api/inspect", "/api/suggest"):
        lesson = _lesson(data)
        program, shown = _program(data), _shown(data, lesson)
        if route == "/api/evaluate":
            return execute(program, shown)
        if route == "/api/inspect":
            return inspect_op(program, shown, data.get("step"), data.get("example", 0))
        return ADVISER.rank(program, shown)
    raise ValueError(f"Unknown route {route!r}.")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "SignalHouse/1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return urlparse(origin).hostname in ALLOWED_HOSTS

    def do_POST(self) -> None:
        if not self._origin_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "Origin not allowed."})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0 or length > MAX_BODY:
                raise ValueError("Request body too large.")
            raw = self.rfile.read(length) if length else b""
            data = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            result = api(urlparse(self.path).path, data)
        except Exception as error:  # every failure is the client's 400
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send(HTTPStatus.OK, result)

    def do_GET(self) -> None:
        self._send(
            HTTPStatus.NOT_FOUND,
            {"error": "Signal House serves POST /api/* only; the page comes later."},
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- quiet
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.stack_workshop.server",
        description="Serve Signal House, the mark-28 workshop, on 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--checkpoint", default=None, help="stack_policy checkpoint for the adviser"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="let the adviser fetch the archived policy when none is local",
    )
    args = parser.parse_args(argv)
    if args.checkpoint or args.download:
        status = ADVISER.load(args.checkpoint, download=args.download)
        print(f"adviser: {'ready' if status['ready'] else status['error']}")
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"Signal House on http://{HOST}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
