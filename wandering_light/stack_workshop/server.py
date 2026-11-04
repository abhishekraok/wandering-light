"""The workshop's HTTP service: JSON over POST on loopback only.

``api(route, data)`` is the whole interface; ``Handler`` is the thin HTTP
wrapper a browser frontend will talk to. Routes other than ``/api/context``
take ``data["lesson"]`` -- a lesson id, or ``sandbox`` with the learner's own
``data["examples"]`` and no extra tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from wandering_light.stack_machine import MARK_OPS, MachineError
from wandering_light.stack_workshop import curriculum, engine
from wandering_light.stack_workshop.model import ADVISER

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_PORT = 8018
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SANDBOX = "sandbox"
HIDDEN = ("solution", "heldout", "hints")
WORLD_MANIFEST = (
    engine.REPO / "wandering_light/evals/data/stack_pbe_world_v2/manifest.json"
)
WORLD_SHELF_SIZE = 24


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def public_lesson(lesson: dict) -> dict:
    shown = {key: value for key, value in lesson.items() if key not in HIDDEN}
    shown["extra_tests"] = len(lesson["heldout"])
    shown["hint_count"] = len(lesson["hints"])
    return shown


def macro_shelf() -> list[dict]:
    shelf = []
    for name, entry in engine.MACROS.items():
        try:
            expanded = engine.expansion(name)
        except ValueError as error:
            expanded = f"(does not expand: {error})"
        shelf.append(
            {
                "name": name,
                "group": entry["group"],
                "body": entry["body"],
                "expansion": expanded,
            }
        )
    return shelf


def world_shelf(manifest_path: Path = WORLD_MANIFEST) -> dict:
    """A shelf of world-authored tasks, when the corpus payload is on disk."""
    try:
        if not manifest_path.exists():
            return {"available": False, "reason": "no world corpus manifest"}
        manifest = json.loads(manifest_path.read_text())
        shelf: dict[str, Any] = {
            "available": False,
            "corpus": manifest.get("corpus"),
            "tasks": manifest.get("counts", {}).get("tasks"),
            "hub": manifest.get("hub", {}).get("repo_id"),
            "behaviours": [],
        }
        payloads = [
            manifest_path.parent / split["path"]
            for split in manifest.get("splits", {}).values()
        ]
        if not all(path.exists() for path in payloads):
            shelf["reason"] = "corpus payload not downloaded"
            return shelf
        from wandering_light.stack_pbe import load

        tasks = load(manifest_path.parent)
        shelf["available"] = True
        shelf["behaviours"] = [
            {
                "behaviour": task.behaviour,
                "summary": task.summary,
                "macro_program": task.macro_program,
                "length": task.length,
                "examples": [
                    {"input": given, "output": wanted}
                    for given, wanted in zip(task.inputs, task.outputs, strict=True)
                ],
            }
            for task in sorted(tasks, key=lambda task: task.length)[:WORLD_SHELF_SIZE]
        ]
        return shelf
    except Exception as error:  # the shelf is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def context() -> dict:
    return {
        "lessons": [public_lesson(lesson) for lesson in curriculum.LESSONS],
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect} for op in MARK_OPS
        ],
        "macros": macro_shelf(),
        "model": ADVISER.status(),
        "world": world_shelf(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _lesson(data: dict) -> tuple[dict | None, list[dict], list[dict]]:
    """``(lesson, shown examples, extra tests)``; the sandbox has no lesson."""
    name = data.get("lesson", SANDBOX)
    if name == SANDBOX:
        examples = data.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("The sandbox needs at least one example")
        return None, examples, []
    lesson = curriculum.BY_ID.get(name)
    if lesson is None:
        raise ValueError(f"Unknown lesson '{name}'")
    return lesson, lesson["examples"], lesson["heldout"]


def _program(data: dict) -> str:
    program = data.get("program", "")
    if not isinstance(program, str):
        raise ValueError("The program is text")
    return program


def _failure(source: str, index: int, row: dict, example: dict) -> dict:
    return {
        "source": source,
        "index": index,
        "input": example["input"],
        "expected": row["output"],
        "got": row["frames"][-1]["stack"],
        "error": row["error"],
    }


def verify(data: dict) -> dict:
    _, shown, extra = _lesson(data)
    program = _program(data)
    shown_result = engine.execute(program, shown)
    extra_result = engine.execute(program, extra, capture=False) if extra else None
    failures = [
        _failure("shown", index, row, example)
        for index, (row, example) in enumerate(
            zip(shown_result["examples"], shown, strict=True)
        )
        if not row["matched"]
    ]
    if extra_result is not None:
        failures.extend(
            _failure("extra", index, row, example)
            for index, (row, example) in enumerate(
                zip(extra_result["examples"], extra, strict=True)
            )
            if not row["matched"]
        )
    return {
        "passed": not failures,
        "shown": shown_result,
        "shown_count": len(shown),
        "extra": extra_result,
        "extra_count": len(extra),
        "failures": failures[:3],
        "evidence": "shown and extra tests" if extra else "examples",
    }


def hint(data: dict) -> dict:
    lesson, _, _ = _lesson(data)
    if lesson is None:
        raise ValueError("The sandbox has no hints")
    hints = lesson["hints"]
    index = int(data.get("hint", 0))
    if not 0 <= index < len(hints):
        raise ValueError(f"The lesson has {len(hints)} hints")
    return {"hint": hints[index], "index": index, "remaining": len(hints) - index - 1}


def reveal(data: dict) -> dict:
    lesson, _, _ = _lesson(data)
    if lesson is None:
        raise ValueError("The sandbox has nothing to reveal")
    return {"solution": lesson["solution"], "review": lesson["review"]}


def model(data: dict) -> dict:
    if data.get("checkpoint") is not None or data.get("load") or data.get("download"):
        return ADVISER.load(data.get("checkpoint"), download=bool(data.get("download")))
    return ADVISER.status()


def api(route: str, data: dict | None = None) -> dict:
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError("The request body is a JSON object")
    if route == "/api/context":
        return context()
    if route == "/api/evaluate":
        _, shown, _ = _lesson(data)
        return engine.execute(_program(data), shown, capture=data.get("capture", True))
    if route == "/api/verify":
        return verify(data)
    if route == "/api/inspect":
        _, shown, _ = _lesson(data)
        return engine.inspect_op(
            _program(data), shown, int(data.get("step", 0)), int(data.get("example", 0))
        )
    if route == "/api/hint":
        return hint(data)
    if route == "/api/reveal":
        return reveal(data)
    if route == "/api/suggest":
        _, shown, _ = _lesson(data)
        return ADVISER.rank(_program(data), shown)
    if route == "/api/model":
        return model(data)
    raise ValueError(f"Unknown route {route}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "SignalHouse/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return urlsplit(origin).hostname in LOCAL_HOSTS

    def do_GET(self) -> None:
        if not self._origin_allowed():
            self._send(403, {"error": "Origin not allowed"})
            return
        if urlsplit(self.path).path in {"/", "/api", "/api/"}:
            self._send(200, {"service": "Signal House", "routes": ROUTES})
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._origin_allowed():
            self._send(403, {"error": "Origin not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            data = json.loads(raw.decode() or "{}")
            payload = api(urlsplit(self.path).path, data)
        except (ValueError, KeyError, TypeError, MachineError) as error:
            self._send(400, {"error": str(error)})
            return
        self._send(200, payload)


ROUTES = [
    "/api/context",
    "/api/evaluate",
    "/api/verify",
    "/api/inspect",
    "/api/hint",
    "/api/reveal",
    "/api/suggest",
    "/api/model",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wandering_light.stack_workshop.server",
        description="Serve the Signal House workshop on loopback.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", default=None, help="a stack_policy checkpoint")
    parser.add_argument(
        "--download", action="store_true", help="fetch the archived policy if needed"
    )
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)
    if args.checkpoint or args.download:
        status = ADVISER.load(args.checkpoint, download=args.download)
        print(f"adviser: {'ready' if status['ready'] else status['error']}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.verbose = args.verbose
    print(f"Signal House at http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
