"""Signal House over localhost: one JSON object in, one JSON object out.

    uv run python -m wandering_light.stack_workshop.server --port 8018

``api`` is the whole service and needs no socket, so everything it does can
be called directly. ``Handler`` only carries JSON to it and back. The server
binds 127.0.0.1 and refuses a request that names an ``Origin`` other than
this machine, so a page on another site cannot drive it through a browser.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    MACROS,
    OPS,
    compile_program,
    execute,
    inspect_op,
    read_examples,
)
from wandering_light.stack_workshop.model import ADVISER

SANDBOX = "sandbox"
DEFAULT_PORT = 8018
LOCAL_HOSTS = ("127.0.0.1", "localhost")
MAX_BODY = 1 << 20
FAILURES_SHOWN = 3
WITHHELD = ("solution", "heldout", "hints")

WORLD_MANIFEST = (
    Path(__file__).resolve().parents[1] / "evals/data/stack_pbe_world_v2/manifest.json"
)
WORLD_SHELF_SIZE = 24


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _public(lesson: dict[str, Any]) -> dict[str, Any]:
    """A lesson as the page may see it: nothing that gives the answer away."""
    shown = {key: value for key, value in lesson.items() if key not in WITHHELD}
    shown["extra_tests"] = len(lesson["heldout"])
    shown["hint_count"] = len(lesson["hints"])
    return shown


def _macro_shelf() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "body": macro["body"],
            "group": macro["group"],
            "expansion": " ".join(token["op"] for token in compile_program(name)),
        }
        for name, macro in MACROS.items()
    ]


def _world_payload(manifest: dict[str, Any], split: str) -> Path | None:
    """A split's payload if it is already on this machine and is the right file."""
    entry = manifest["splits"][split]
    name = entry["path"]
    candidates = [WORLD_MANIFEST.parent / name]
    # The Hub cache is a second place to look, never a reason to fail: no
    # huggingface_hub, no hub block and a malformed repo id all mean "not there".
    with contextlib.suppress(Exception):
        from huggingface_hub import try_to_load_from_cache

        hub = manifest["hub"]
        cached = try_to_load_from_cache(
            hub["repo_id"], name, revision=hub["revision"], repo_type="dataset"
        )
        if isinstance(cached, str):
            candidates.append(Path(cached))
    for path in candidates:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if f"sha256:{digest}" == entry["sha256"]:
                return path
    return None


def world_shelf() -> dict[str, Any]:
    """Problems from the world corpus to try in the sandbox, if it is on disk.

    Only local files are read: the payload beside the committed manifest, or
    a copy the Hub cache already holds. Nothing is fetched, and any failure
    is an empty shelf with a note. Tasks whose examples the workshop would
    not accept as typed input are left off, shortest witness first.
    """
    shelf: dict[str, Any] = {
        "available": False,
        "corpus": None,
        "tasks": [],
        "note": None,
    }
    try:
        manifest = json.loads(WORLD_MANIFEST.read_text(encoding="utf-8"))
        shelf["corpus"] = manifest["corpus"]
        rows = []
        for split in sorted(manifest["splits"]):
            path = _world_payload(manifest, split)
            if path is None:
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        tasks = []
        for row in sorted(rows, key=lambda row: (row["length"], row["behaviour"])):
            examples = [
                {"input": given, "output": wanted}
                for given, wanted in zip(row["inputs"], row["outputs"], strict=True)
            ]
            try:
                read_examples(examples)
            except ValueError:
                continue
            tasks.append(
                {
                    "id": row["behaviour"],
                    "summary": row["summary"],
                    "length": row["length"],
                    "examples": examples,
                }
            )
            if len(tasks) == WORLD_SHELF_SIZE:
                break
        shelf["tasks"] = tasks
        shelf["available"] = bool(tasks)
        if not tasks:
            shelf["note"] = "The world corpus payload is not on this machine."
    except Exception as error:
        # The shelf is a nicety. A missing manifest, a corrupt payload or a
        # row from a newer schema must not take the lessons down with it.
        shelf.update(available=False, tasks=[], note=f"{type(error).__name__}: {error}")
    return shelf


def context() -> dict[str, Any]:
    return {
        "lessons": [_public(lesson) for lesson in LESSONS],
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect}
            for op in OPS.values()
        ],
        "macros": _macro_shelf(),
        "model": ADVISER.status(),
        "world": world_shelf(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _lesson(data: dict[str, Any]) -> dict[str, Any] | None:
    """The lesson a request names, or None for the sandbox."""
    name = data.get("lesson")
    if name == SANDBOX:
        return None
    if not isinstance(name, str) or name not in BY_ID:
        raise ValueError(f"Unknown lesson {name!r}.")
    return BY_ID[name]


def _examples(data: dict[str, Any]) -> tuple[Any, list]:
    """``(shown examples, extra tests)``; the sandbox brings its own and no extras."""
    lesson = _lesson(data)
    if lesson is None:
        return data.get("examples"), []
    return lesson["examples"], lesson["heldout"]


def _program(data: dict[str, Any]) -> str:
    program = data.get("program")
    if not isinstance(program, str):
        raise ValueError("The request needs a 'program' as text.")
    return program


def _authored(data: dict[str, Any]) -> dict[str, Any]:
    lesson = _lesson(data)
    if lesson is None:
        raise ValueError("The sandbox has no authored hints or solution.")
    return lesson


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    shown, _ = _examples(data)
    return execute(_program(data), shown)


def verify(data: dict[str, Any]) -> dict[str, Any]:
    """Run the shown examples and the extra tests, and say which the claim rests on."""
    shown, extra = _examples(data)
    program = _program(data)
    results = [("shown", execute(program, shown, capture=False))]
    if extra:
        results.append(("extra", execute(program, extra, capture=False)))
    counts = dict.fromkeys(("shown", "extra"), (0, 0))
    failures = []
    for source, result in results:
        counts[source] = (result["matches"], result["count"])
        failures.extend(
            {"source": source, "index": index, **row}
            for index, row in enumerate(result["examples"])
            if not row["matched"]
        )
    return {
        "passed": not failures,
        "shown": counts["shown"][0],
        "shown_count": counts["shown"][1],
        "extra": counts["extra"][0],
        "extra_count": counts["extra"][1],
        "failures": failures[:FAILURES_SHOWN],
        "evidence": "shown and extra tests" if extra else "examples",
    }


def inspect(data: dict[str, Any]) -> dict[str, Any]:
    shown, _ = _examples(data)
    return inspect_op(_program(data), shown, data.get("step"), data.get("example", 0))


def hint(data: dict[str, Any]) -> dict[str, Any]:
    hints = _authored(data)["hints"]
    index = data.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("'index' must be a whole number.")
    if not 0 <= index < len(hints):
        raise ValueError(f"No hint {index}; this lesson has {len(hints)}.")
    return {"hint": hints[index], "index": index, "count": len(hints)}


def reveal(data: dict[str, Any]) -> dict[str, Any]:
    return {"solution": _authored(data)["solution"]}


def suggest(data: dict[str, Any]) -> dict[str, Any]:
    shown, _ = _examples(data)
    return ADVISER.rank(_program(data), shown)


def load_model(data: dict[str, Any]) -> dict[str, Any]:
    checkpoint = data.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError("'checkpoint' must be a path as text.")
    return ADVISER.load(checkpoint or None, download=data.get("download") is True)


ROUTES = {
    "/api/context": lambda data: context(),
    "/api/evaluate": evaluate,
    "/api/verify": verify,
    "/api/inspect": inspect,
    "/api/hint": hint,
    "/api/reveal": reveal,
    "/api/suggest": suggest,
    "/api/model": load_model,
}


def api(route: str, data: dict[str, Any]) -> dict[str, Any]:
    """Answer one request. A request that cannot be answered is a ``ValueError``."""
    handler = ROUTES.get(route)
    if handler is None:
        raise ValueError(f"Unknown route {route!r}.")
    if not isinstance(data, dict):
        raise ValueError("The request body must be a JSON object.")
    return handler(data)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _local_origin(origin: str | None) -> bool:
    """No ``Origin`` (curl, a script) is local; a named one must be this machine."""
    if origin is None:
        return True
    try:
        return urlsplit(origin).hostname in LOCAL_HOSTS
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    """POST a JSON object to an ``/api/...`` route; the reply is JSON."""

    server_version = "SignalHouse"
    timeout = 30

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        if status == 405:
            self.send_header("Allow", "POST")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Content-Length is not a number.") from None
        if not 0 <= length <= MAX_BODY:
            raise ValueError(f"The request body is over {MAX_BODY} bytes.")
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except (ValueError, RecursionError):
            raise ValueError("The request body is not JSON.") from None

    def do_POST(self) -> None:
        if not _local_origin(self.headers.get("Origin")):
            self._reply(403, {"error": "Signal House answers this machine only."})
            return
        try:
            payload = api(urlsplit(self.path).path, self._body())
        except ValueError as error:
            self._reply(400, {"error": str(error)})
            return
        except Exception as error:
            # Not the request's fault. Say so, rather than dropping the socket.
            self._reply(500, {"error": f"{type(error).__name__}: {error}"})
            return
        self._reply(200, payload)

    def do_GET(self) -> None:
        self._reply(405, {"error": "POST a JSON object to an /api/ route."})

    def log_message(self, *args: Any) -> None:
        """Quiet: a request per keystroke is not worth a line each."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve Signal House on localhost.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"Signal House is listening on http://127.0.0.1:{server.server_port}/api/",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
