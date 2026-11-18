"""Signal House JSON API. Run with python -m wandering_light.stack_workshop.server."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    MACROS,
    OPS,
    compile_program,
    execute,
    inspect_op,
)
from wandering_light.stack_workshop.model import ADVISER

WORLD_DIR = Path(__file__).resolve().parents[1] / "evals/data/stack_pbe_world_v2"


def _world():
    """Use the checked-out manifest and any locally installed eval split."""
    from wandering_light.stack_pbe import load

    try:
        manifest = json.loads((WORLD_DIR / "manifest.json").read_text(encoding="utf-8"))
        shelf = {"corpus": manifest["corpus"], "tasks": []}
        split = WORLD_DIR / manifest["splits"]["eval"]["path"]
        if split.is_file():
            shelf["tasks"] = [
                {
                    "id": task.behaviour,
                    "title": task.summary,
                    "examples": [
                        {"input": i, "output": o}
                        for i, o in zip(task.inputs, task.outputs, strict=True)
                    ],
                }
                for task in load(split)
            ]
        return shelf
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        return {"corpus": WORLD_DIR.name, "tasks": []}


def _context():
    return {
        "lessons": [
            {
                **{
                    k: v
                    for k, v in lesson.items()
                    if k not in {"solution", "heldout", "hints"}
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
                "expansion": [t["op"] for t in compile_program(name)],
            }
            for name, macro in MACROS.items()
        ],
        "model": ADVISER.status(),
        "world": _world(),
    }


def _verify(program, shown, extra):
    report = execute(program, shown + extra if extra else shown, capture=False)
    rows = report["examples"]
    shown_matches = sum(row["matched"] for row in rows[: len(shown)])
    return {
        "passed": report["matches"] == report["count"],
        "shown": shown_matches,
        "shown_count": len(shown),
        "extra": report["matches"] - shown_matches,
        "extra_count": len(extra),
        "failures": [row for row in rows if not row["matched"]][:3],
        "evidence": "shown and extra tests" if extra else "examples",
    }


def api(route, data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Supply a JSON object.")
    if route == "/api/context":
        return _context()
    lesson_id = data.get("lesson")
    if lesson_id == "sandbox":
        lesson = {"examples": data.get("examples"), "heldout": [], "hints": []}
    elif isinstance(lesson_id, str) and lesson_id in BY_ID:
        lesson = BY_ID[lesson_id]
    else:
        raise ValueError("Unknown lesson.")
    shown = lesson["examples"]
    program = data.get("program", "")
    if route == "/api/evaluate":
        return execute(program, shown)
    if route == "/api/verify":
        return _verify(program, shown, lesson["heldout"])
    if route == "/api/inspect":
        return inspect_op(program, shown, data.get("step"), data.get("example", 0))
    if route == "/api/hint":
        index = data.get("index", 0)
        if type(index) is not int or not 0 <= index < len(lesson["hints"]):
            raise ValueError("Hint index is out of range.")
        return {"hint": lesson["hints"][index]}
    if route == "/api/reveal":
        if "solution" not in lesson:
            raise ValueError("The sandbox has no authored solution.")
        return {"solution": lesson["solution"]}
    if route == "/api/suggest":
        return ADVISER.rank(program, shown)
    if route == "/api/model":
        download = data.get("download", False)
        if not isinstance(download, bool):
            raise ValueError("download must be a boolean.")
        return ADVISER.load(data.get("checkpoint"), download)
    raise ValueError("Unknown API route.")


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        for origin in self.headers.get_all("Origin", []):
            try:
                host = urlsplit(origin).hostname
            except ValueError:
                host = None
            if host not in {"127.0.0.1", "localhost"}:
                self._json(403, {"error": "Origin must be local."})
                return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                raise ValueError("Supply a JSON request body.")
            data = json.loads(self.rfile.read(length))
            result = api(self.path, data)
        except (ValueError, RecursionError, OverflowError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, result)


def main():
    parser = argparse.ArgumentParser(description="Serve the Signal House workshop")
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
