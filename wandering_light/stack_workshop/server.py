"""Serve Signal House's JSON API on localhost."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    MACROS,
    OPS,
    REPO,
    compile_program,
    execute,
    inspect_op,
    stack_from,
)
from wandering_light.stack_workshop.model import ADVISER

_WORLD_MANIFEST = REPO / "wandering_light/evals/data/stack_pbe_world_v2/manifest.json"


def _world_shelf() -> dict:
    """Sample the checked-out world reports without fetching any payloads."""
    try:
        manifest = json.loads(_WORLD_MANIFEST.read_text(encoding="utf-8"))
        probes = manifest["config"]["probe"]
        if not isinstance(probes, list):
            raise ValueError("World corpus probes must be a list")
        shelf = {
            "corpus": manifest["corpus"],
            "counts": manifest["counts"],
            "tasks": [],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {"corpus": None, "counts": {}, "tasks": []}

    for probe in probes:
        try:
            with (REPO / probe / "behaviours.jsonl").open(encoding="utf-8") as rows:
                added = 0
                for line in rows:
                    try:
                        row = json.loads(line)
                        examples = []
                        for index in row["shown"][:3]:
                            test = row["tests"][index]
                            stack_from(test["args"])
                            stack_from([test["out"]], expected=True)
                            examples.append(
                                {"input": test["args"], "output": [test["out"]]}
                            )
                        if len(examples) < 3:
                            continue
                        shelf["tasks"].append(
                            {
                                "id": row["id"],
                                "title": row["name"],
                                "goal": row["description"],
                                "source": row["source"],
                                "examples": examples,
                            }
                        )
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    added += 1
                    if added == 6:
                        break
        except (OSError, UnicodeError, TypeError):
            continue
    return shelf


def _context() -> dict:
    private = {"solution", "heldout", "hints"}
    return {
        "lessons": [
            {
                **{
                    key: deepcopy(value)
                    for key, value in lesson.items()
                    if key not in private
                },
                "extra_tests": len(lesson["heldout"]),
                "hint_count": len(lesson["hints"]),
            }
            for lesson in LESSONS
        ],
        "ops": [
            {"name": op.name, "group": op.group, "effect": op.effect}
            for op in OPS.values()
        ],
        "macros": [
            {
                "name": name,
                **macro,
                "expansion": [token["op"] for token in compile_program(name)],
            }
            for name, macro in MACROS.items()
        ],
        "model": ADVISER.status(),
        "world": _world_shelf(),
    }


def api(route: str, data: dict) -> dict:
    """Dispatch a JSON request. Step, example, and hint indices start at zero."""
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    if route == "/api/context":
        return _context()
    if route == "/api/model":
        download = data.get("download", False)
        if type(download) is not bool:
            raise ValueError("download must be a boolean")
        if "checkpoint" in data or "download" in data or data.get("load") is True:
            return ADVISER.load(checkpoint=data.get("checkpoint"), download=download)
        return ADVISER.status()
    if route not in {
        "/api/evaluate",
        "/api/verify",
        "/api/inspect",
        "/api/hint",
        "/api/reveal",
        "/api/suggest",
    }:
        raise ValueError(f"Unknown route {route!r}")

    lesson_id = data.get("lesson")
    if not isinstance(lesson_id, str):
        raise ValueError("A lesson id is required")
    lesson = BY_ID.get(lesson_id)
    if lesson is not None:
        examples, heldout = lesson["examples"], lesson["heldout"]
    elif lesson_id == "sandbox":
        examples, heldout = data.get("examples"), []
    else:
        raise ValueError(f"Unknown lesson {lesson_id!r}")

    program = data.get("program", "")
    if route == "/api/evaluate":
        return execute(program, examples)
    if route == "/api/inspect":
        return inspect_op(program, examples, data.get("step"), data.get("example", 0))
    if route == "/api/suggest":
        return ADVISER.rank(program, examples)
    if route == "/api/verify":
        shown = execute(program, examples, capture=False)
        extra = (
            execute(program, heldout, capture=False)
            if heldout
            else {"matches": 0, "count": 0, "examples": []}
        )
        failures = [
            row
            for row in [*shown["examples"], *extra["examples"]]
            if not row["matched"]
        ]
        return {
            "passed": not failures,
            "shown": shown["matches"],
            "shown_count": shown["count"],
            "extra": extra["matches"],
            "extra_count": extra["count"],
            "failures": failures[:3],
            "evidence": "shown and extra tests" if heldout else "examples",
        }
    if lesson is None:
        raise ValueError("The sandbox has no authored hints or solution")
    if route == "/api/reveal":
        return {"solution": lesson["solution"]}

    index = data.get("index", 0)
    if type(index) is not int or not 0 <= index < len(lesson["hints"]):
        raise ValueError("Hint index out of range")
    return {"hint": lesson["hints"][index]}


class Handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        for origin in self.headers.get_all("Origin", []):
            try:
                host = urlsplit(origin).hostname
            except ValueError:
                host = None
            if host not in {"127.0.0.1", "localhost"}:
                self._write_json(403, {"error": "Only localhost origins are allowed"})
                return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("A JSON request body is required")
            data = json.loads(self.rfile.read(length))
            route = urlsplit(self.path).path
            payload = api(route, data)
        except Exception as error:
            self._write_json(400, {"error": str(error)})
        else:
            self._write_json(200, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
