"""Signal House's JSON service, on localhost only.

Every route is a POST of a JSON object to ``api``; the browser frontend that
calls it comes later. Run with ``python -m wandering_light.stack_workshop.server``.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wandering_light.stack_machine import MARK_OPS
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    MACROS,
    compile_program,
    execute,
    inspect_op,
)
from wandering_light.stack_workshop.model import ADVISER

DEFAULT_PORT = 8018
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
PRIVATE_FIELDS = ("solution", "heldout", "hints")
WORLD_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "data"
    / "stack_pbe_world_v2"
    / "manifest.json"
)
MAX_BODY = 1 << 20

__all__ = ["Handler", "ThreadingHTTPServer", "api", "main"]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _public(lesson: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in lesson.items() if key not in PRIVATE_FIELDS}
    public["extra_tests"] = len(lesson["heldout"])
    public["hint_count"] = len(lesson["hints"])
    return public


def _macros() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "body": entry["body"],
            "group": entry["group"],
            "expansion": " ".join(token["op"] for token in compile_program(name)),
        }
        for name, entry in MACROS.items()
    ]


def world_shelf(path: Path = WORLD_MANIFEST) -> dict[str, Any]:
    """What the local world-corpus manifest says; never touches the network."""
    try:
        manifest = json.loads(path.read_text())
        return {
            "available": True,
            "corpus": manifest.get("corpus"),
            "basis": manifest.get("basis"),
            "counts": manifest.get("counts", {}),
            "sources": manifest.get("config", {}).get("sources", []),
            "hub": manifest.get("hub", {}),
        }
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def context() -> dict[str, Any]:
    return {
        "lessons": [_public(lesson) for lesson in LESSONS],
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect} for op in MARK_OPS
        ],
        "macros": _macros(),
        "model": ADVISER.status(),
        "world": world_shelf(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _tests(data: dict[str, Any]) -> tuple[dict[str, Any] | None, list, list]:
    """The lesson (None for the sandbox), its shown examples and extra tests."""
    name = data.get("lesson")
    if name == "sandbox":
        return None, data.get("examples"), []
    lesson = BY_ID.get(name)
    if lesson is None:
        raise ValueError(f"Unknown lesson {name!r}")
    return lesson, lesson["examples"], lesson["heldout"]


def _program(data: dict[str, Any]) -> str:
    program = data.get("program", "")
    if not isinstance(program, str):
        raise ValueError("program must be text")
    return program


def _lesson_only(data: dict[str, Any]) -> dict[str, Any]:
    lesson, _shown, _extra = _tests(data)
    if lesson is None:
        raise ValueError("The sandbox has no hints or solution")
    return lesson


def _verify(data: dict[str, Any]) -> dict[str, Any]:
    _lesson, shown, extra = _tests(data)
    program = _program(data)
    shown_run = execute(program, shown, capture=False)
    rows = [
        {"set": "shown", "index": index, **row}
        for index, row in enumerate(shown_run["examples"])
    ]
    extra_matches = 0
    if extra:
        extra_run = execute(program, extra, capture=False)
        rows += [
            {"set": "extra", "index": index, **row}
            for index, row in enumerate(extra_run["examples"])
        ]
        extra_matches = extra_run["matches"]
    passed = shown_run["matches"] == len(shown) and extra_matches == len(extra)
    return {
        "passed": passed,
        "shown": shown_run["matches"],
        "shown_count": len(shown),
        "extra": extra_matches,
        "extra_count": len(extra),
        "failures": [row for row in rows if not row["matched"]][:3],
        "evidence": "shown and extra tests" if extra else "examples",
    }


def _hint(data: dict[str, Any]) -> dict[str, Any]:
    hints = _lesson_only(data)["hints"]
    index = data.get("index", 0)
    if not isinstance(index, int) or not 0 <= index < len(hints):
        raise ValueError(f"Hint index must lie in 0..{len(hints) - 1}")
    return {"hint": hints[index], "index": index, "hint_count": len(hints)}


def _reveal(data: dict[str, Any]) -> dict[str, Any]:
    lesson = _lesson_only(data)
    return {
        "solution": lesson["solution"],
        "discovery": lesson["discovery"],
        "review": lesson["review"],
    }


def _model(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("checkpoint") is not None or data.get("download"):
        return ADVISER.load(data.get("checkpoint"), download=bool(data.get("download")))
    return ADVISER.status()


def api(route: str, data: dict[str, Any]) -> dict[str, Any]:
    """Answer one route; a ``ValueError`` is the caller's mistake."""
    if not isinstance(data, dict):
        raise ValueError("The request body must be a JSON object")
    if route == "/api/context":
        return context()
    if route == "/api/evaluate":
        _lesson, shown, _extra = _tests(data)
        return execute(_program(data), shown)
    if route == "/api/verify":
        return _verify(data)
    if route == "/api/inspect":
        _lesson, shown, _extra = _tests(data)
        return inspect_op(
            _program(data), shown, data.get("step"), data.get("example", 0)
        )
    if route == "/api/hint":
        return _hint(data)
    if route == "/api/reveal":
        return _reveal(data)
    if route == "/api/suggest":
        _lesson, shown, _extra = _tests(data)
        return ADVISER.rank(_program(data), shown)
    if route == "/api/model":
        return _model(data)
    raise ValueError(f"Unknown route {route!r}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "SignalHouse/1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and urlparse(origin).hostname not in ALLOWED_HOSTS:
            self._send(403, {"error": "Origin not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("Request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            payload = api(urlparse(self.path).path, json.loads(raw))
        except Exception as error:
            self._send(400, {"error": str(error) or type(error).__name__})
            return
        self._send(200, payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.stack_workshop.server",
        description="Serve the Signal House workshop API on localhost.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", default=None, help="stack_policy checkpoint")
    args = parser.parse_args(argv)
    if args.checkpoint:
        ADVISER.load(args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Signal House on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
