"""Local-only HTTP service; core teaching requires only Python's standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
import os
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .curriculum import BY_ID, LESSONS
from .engine import MACROS, ROOT, compile_program, execute, inspect_op, op_catalog
from .model import ADVISER

STATIC = Path(__file__).parent / "static"
GET_ROUTES = {"/api/context", "/api/model"}
WORLD_CACHE = ROOT / ".workshop-cache" / "world"
WORLD = {}
WORLD_LOCK = threading.Lock()


def load_world(download=False):
    manifest_path = ROOT / "wandering_light/evals/data/stack_pbe_world_v2/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with WORLD_LOCK:
        for split, entry in manifest["splits"].items():
            local = WORLD_CACHE / entry["path"]
            if not local.exists() and download:
                url = (
                    f"https://huggingface.co/datasets/{manifest['hub']['repo_id']}/resolve/"
                    f"{manifest['hub']['revision']}/{entry['path']}"
                )
                payload = urllib.request.urlopen(url, timeout=30).read()
                if "sha256:" + hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError(
                        "The downloaded corpus does not match its published manifest."
                    )
                WORLD_CACHE.mkdir(parents=True, exist_ok=True)
                local.write_bytes(payload)
            if local.exists():
                payload = local.read_bytes()
                if "sha256:" + hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError(
                        "The cached corpus does not match its published manifest."
                    )
                for line in gzip.decompress(payload).decode().splitlines():
                    task = json.loads(line)
                    task_id = "world:" + task["behaviour"]
                    title = (
                        task["behaviour"]
                        .split("-", 1)[-1]
                        .replace("-", " ")
                        .capitalize()
                    )
                    WORLD[task_id] = {
                        "id": task_id,
                        "title": title,
                        "chapter": "06 · The world outside",
                        "concept": task.get("summary", "World-authored behaviour"),
                        "story": "A task from the world-authored corpus. Use the macro shelf, inspect expansions, and investigate the extra tests.",
                        "goal": task.get("summary", "Match all examples."),
                        "starter": "",
                        "solution": " ".join(task["program"]),
                        "examples": [
                            {"input": i, "output": o}
                            for i, o in zip(
                                task["inputs"], task["outputs"], strict=True
                            )
                        ],
                        "heldout": [
                            {"input": i, "output": o}
                            for i, o in zip(
                                task.get("held_out_inputs", []),
                                task.get("held_out_outputs", []),
                                strict=True,
                            )
                        ],
                        "hints": [
                            "Break the transformation into smaller stages. Look for the vocabulary each stage needs."
                        ],
                        "discovery": "These tests provide evidence about a behaviour, not a proof for every possible input.",
                        "review": [],
                        "prediction": None,
                        "macros": True,
                        "source": f"stack_pbe_world_v2 · {split} · {task['behaviour']}",
                        "macro_program": task.get("macro_program"),
                        "primitive_count": len(task["program"]),
                    }
    return {
        "loaded": len(WORLD),
        "available": manifest["counts"]["tasks"],
        "tasks": [public(x) for x in WORLD.values()],
    }


def public(lesson):
    return {
        k: v
        for k, v in lesson.items()
        if k not in {"solution", "heldout", "hints", "macro_program"}
    } | {"extra_tests": len(lesson["heldout"]), "hint_count": len(lesson["hints"])}


def task_for(data):
    lesson_id = data.get("lesson")
    if lesson_id == "sandbox":
        return {
            "examples": data.get("examples", []),
            "heldout": [],
            "source": "Your examples",
        }
    task = BY_ID.get(lesson_id) or WORLD.get(lesson_id)
    if task is None:
        raise ValueError("Choose a lesson or load a world task first.")
    return task


def api(route, data):
    if route == "/api/context":
        try:
            world = load_world()
        except Exception as exc:
            world = {"loaded": 0, "available": 183, "tasks": [], "error": str(exc)}
        return {
            "lessons": [public(x) for x in LESSONS],
            "ops": op_catalog(),
            "macros": [
                dict(
                    name=k, **v, expanded=" ".join(t["op"] for t in compile_program(k))
                )
                for k, v in MACROS.items()
            ],
            "model": ADVISER.status(),
            "world": world,
        }
    if route == "/api/model":
        return ADVISER.status()
    if route == "/api/model/load":
        return ADVISER.load(data.get("checkpoint"), data.get("download", False))
    if route == "/api/world":
        return load_world(download=True)
    task = task_for(data)
    program = data.get("program", "")
    if route == "/api/evaluate":
        return execute(program, task["examples"])
    if route == "/api/verify":
        shown = execute(program, task["examples"], capture=False)
        extra = task["heldout"]
        # Bound each batch while preserving all corpus test cases.
        batches = [
            execute(program, extra[i : i + 16], capture=False)
            for i in range(0, len(extra), 16)
        ]
        extra_matches = sum(b["matches"] for b in batches)
        failures = [r for b in batches for r in b["examples"] if not r["matched"]]
        return {
            "passed": shown["matches"] == shown["count"]
            and extra_matches == len(extra),
            "shown": shown["matches"],
            "shown_count": shown["count"],
            "extra": extra_matches,
            "extra_count": len(extra),
            "failures": failures[:3],
            "evidence": "examples" if not extra else "shown and extra tests",
        }
    if route == "/api/inspect":
        return inspect_op(
            program,
            task["examples"],
            int(data.get("step", 0)),
            int(data.get("example", 0)),
        )
    if route == "/api/preview":
        candidate = data.get("candidate", "")
        if candidate not in MACROS and candidate not in {
            x["name"] for x in op_catalog()
        }:
            raise ValueError("Unknown completion.")
        return execute(program + " " + candidate, task["examples"])
    if route == "/api/hint":
        if data.get("lesson") == "sandbox":
            return {
                "text": "Try an empty list, a negative number, and a boundary case. You supply the expected outputs for custom tasks."
            }
        level = max(0, min(int(data.get("level", 0)), len(task["hints"]) - 1))
        return {"text": task["hints"][level], "level": level}
    if route == "/api/reveal":
        if "solution" not in task:
            raise ValueError(
                "A custom task has no reference solution. Ask the adviser for next moves."
            )
        macro = task.get("macro_program")
        return {
            "program": " ".join(macro)
            if isinstance(macro, list)
            else macro or task["solution"]
        }
    if route == "/api/suggest":
        return ADVISER.rank(program, task["examples"])
    raise ValueError("Unknown workshop action.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def json(self, value, status=200):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            if path not in GET_ROUTES:
                self.json(
                    {
                        "error": "This workshop action requires a POST from the local page."
                    },
                    405,
                )
                return
            try:
                self.json(api(path, {}))
            except Exception as exc:
                self.json({"error": str(exc)}, 400)
            return
        file = (STATIC / (path.lstrip("/") or "index.html")).resolve()
        if not file.is_relative_to(STATIC.resolve()) or not file.is_file():
            self.send_error(404, "Build the workshop frontend first (see README).")
            return
        payload = file.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(file.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost"}:
            self.json({"error": "Only the local workshop can call this service."}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 150_000:
                raise ValueError("Request is empty or too large.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            self.json(api(urlparse(self.path).path, data))
        except Exception as exc:
            self.json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="Signal House stack workshop")
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    checkpoint = args.checkpoint or os.environ.get("STACK_POLICY_CHECKPOINT")
    cached = ROOT / ".workshop-cache" / "policy"
    if not checkpoint and (cached / "policy_config.json").exists():
        checkpoint = str(cached)
    if checkpoint:
        threading.Thread(target=ADVISER.load, args=(checkpoint,), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Signal House is ready: http://127.0.0.1:{args.port}", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
