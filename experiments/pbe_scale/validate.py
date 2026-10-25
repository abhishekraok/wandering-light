"""Bounded reference checks, shared-probe deduplication, and Task export.

Run under the campaign's systemd MemoryMax=12G / MemorySwapMax=0 scope.
Accepted/rejected snapshots are atomically replaced every 512 candidates or 60 seconds; atomic
per-candidate journal records make interrupted work resumable between snapshots.
The 21-value scalar domain is exhaustive (6 shown / 15 hidden).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gzip
import hashlib
import io
import json
import multiprocessing
import random
import resource
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from experiments.world_behaviours.verify import load_macros
from wandering_light.stack_machine import Config, format_stack, value_key
from wandering_light.stack_pbe import (
    OP_BY_NAME,
    Task,
    assign_task_hash_split,
    run_program,
    skeleton,
    to_stack,
    to_value,
)

SHARED_PROBES = 192
PROBE_POLICY = "v3-edge-cartesian"
LEGACY_PROBE_POLICY = "v2-shared64"

KINDS = {"list", "list_int", "two_lists", "int", "two_ints"}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "divmod",
        "enumerate",
        "filter",
        "int",
        "dict",
        "pow",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "str",
        "sorted",
        "sum",
        "tuple",
        "zip",
    )
}
SAFE_METHODS = {
    "append",
    "extend",
    "count",
    "index",
    "copy",
    "reverse",
    "sort",
    "pop",
    "join",
    "intersection",
    "union",
    "difference",
    "symmetric_difference",
}
FORBIDDEN = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Delete,
    ast.Raise,
    ast.Try,
    ast.NamedExpr,
)
MAX_PROGRAM = 4096
MAX_OUTPUT = 256


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def check_reference(source: str) -> None:
    """Reject capability access; subprocess limits bound otherwise legal code."""
    if len(source) > 20000:
        raise ValueError("reference exceeds 20000 characters")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("reference must contain only a function definition f")
    function = tree.body[0]
    if function.name != "f" or function.decorator_list:
        raise ValueError("reference must define undecorated f")
    if function.args.defaults or any(
        value is not None for value in function.args.kw_defaults
    ):
        raise ValueError("reference function defaults forbidden (persistent state)")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN):
            raise ValueError(f"forbidden reference syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (
            node.id.startswith("__")
            or node.id
            in {
                "open",
                "eval",
                "exec",
                "compile",
                "getattr",
                "setattr",
                "delattr",
                "globals",
                "locals",
                "vars",
                "input",
                "breakpoint",
                "help",
            }
        ):
            raise ValueError(f"forbidden reference name: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr not in SAFE_METHODS or not isinstance(node.ctx, ast.Load)
        ):
            raise ValueError(f"forbidden reference attribute: {node.attr}")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.returns):
            raise ValueError("reference decorators and return annotations forbidden")
        if isinstance(node, ast.arg) and node.annotation:
            raise ValueError("reference argument annotations forbidden")
        if isinstance(node, ast.Call) and not isinstance(
            node.func, (ast.Name, ast.Attribute)
        ):
            raise ValueError("indirect reference calls forbidden")


def _list(rng: random.Random, length: int | None = None) -> list[int]:
    return [
        rng.randint(-10, 10)
        for _ in range(rng.randrange(13) if length is None else length)
    ]


def probe_bank(
    kind: str,
    seed: int,
    count: int | None = None,
    exclude: list | None = None,
    policy: str = PROBE_POLICY,
) -> list:
    """Shared deterministic inputs, with empty and varied-length edge cases first."""
    if kind not in KINDS:
        raise ValueError(f"unknown input_kind: {kind}")
    rng = random.Random(f"pbe-scale:{kind}:{seed}")
    seen = {canonical(args) for args in exclude or []}
    rows: list = []
    edges = [
        [],
        [0],
        [-10, 10],
        [1, -1, 1],
        [0] * 8,
        list(range(-6, 6)),
        [-10] * 12,
        [10] * 12,
        list(range(10, -3, -1))[:12],
        [0, -1, 0, 1],
    ]
    if kind == "int":
        proposals = [[n] for n in [0, -10, 10, -1, 1, -2, *range(-9, 10)]]
    elif kind == "two_ints":
        proposals = [[a, b] for a in [0, -10, 10, -1, 1] for b in [0, -10, 10, -1, 1]]
    elif kind == "list":
        proposals = [[xs] for xs in edges]
    elif kind == "list_int":
        proposals = [
            [xs, n]
            for xs, n in zip(edges, [0, -10, 10, -1, 1, -2, 2, 3, -3, 9], strict=True)
        ]
    else:
        proposals = [[xs, edges[(i + 3) % len(edges)]] for i, xs in enumerate(edges)]
    if policy == PROBE_POLICY:
        edges += [
            [1],
            [-1],
            [10],
            [-10],
            [1, 1],
            [-1, -1],
            [0, 0],
            [1, -1],
            [2, 2, 2],
            [-2, -2, -2],
        ]
        paired = [[], [0], [1], [-1], [10], [-10], [1, 1], [-1, -1], [1, -1], [0, 0]]
        if kind == "two_ints":
            proposals = [[a, b] for a in range(-10, 11) for b in range(-10, 11)]
        elif kind == "two_lists":
            proposals += [[a, b] for a in paired for b in paired]
            proposals += [
                [xs, edges[(i + 3) % len(edges)]] for i, xs in enumerate(edges)
            ]
        elif kind == "list_int":
            proposals += [[xs, n] for xs in paired[:8] for n in range(-10, 11)]
        elif kind == "list":
            proposals = [[xs] for xs in edges]
    if count is None:
        count = (
            (441 if kind == "two_ints" else SHARED_PROBES)
            if policy == PROBE_POLICY
            else 64
        )
    capacity = 21 if kind == "int" else 441 if kind == "two_ints" else 10**9
    target = min(count, capacity - len(seen))
    attempts = 0
    while len(rows) < target:
        if proposals:
            args = proposals.pop(0)
        elif kind == "int":
            args = [rng.randint(-10, 10)]
        elif kind == "two_ints":
            args = [rng.randint(-10, 10), rng.randint(-10, 10)]
        elif kind == "list":
            args = [_list(rng)]
        elif kind == "list_int":
            args = [_list(rng), rng.randint(-10, 10)]
        else:
            args = [_list(rng), _list(rng)]
        key = canonical(args)
        if key not in seen:
            seen.add(key)
            rows.append(args)
        attempts += 1
        if attempts > 10000:
            raise ValueError("could not draw requested distinct probes")
    return rows


def _valid_result(result: object) -> bool:
    values = result if type(result) is list else [result]
    return len(values) <= MAX_OUTPUT and all(
        type(x) is int and abs(x) <= 10**12 for x in values
    )


def _reference(source: str, inputs: list) -> list:
    namespace = {"__builtins__": SAFE_BUILTINS}
    exec(compile(source, "<candidate-reference>", "exec"), namespace)
    function = namespace["f"]
    outputs = []
    for args in inputs:
        # Give references private data: mutation cannot affect later witness replay.
        copied = json.loads(canonical(args))
        output = function(*copied)
        if not _valid_result(output):
            raise ValueError("reference result must be int or flat int list (bounded)")
        outputs.append(output)
    return outputs


def _replay(program: list[str], inputs: list, outputs: list) -> dict:
    encoded_inputs, encoded_outputs, fuels = [], [], []
    for index, (args, expected) in enumerate(zip(inputs, outputs, strict=True)):
        initial = to_stack(tuple(args))
        outcome = run_program(program, initial)
        if isinstance(outcome, str):
            raise ValueError(
                f"witness refusal at probe {index}, args={args}: {outcome}"
            )
        result, fuel = outcome
        if len(result) != 1 or value_key(result[0]) != value_key(to_value(expected)):
            raise ValueError(
                f"witness mismatch at probe {index}, args={args}, expected={expected}"
            )
        encoded_inputs.append(format_stack(Config(initial)))
        encoded_outputs.append(format_stack(Config(result)))
        fuels.append(fuel)
    return {"inputs": encoded_inputs, "outputs": encoded_outputs, "fuels": fuels}


def _worker(connection, operation: str, arguments: tuple) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        # Fork may inherit a large virtual mapping from a test runner; bound
        # new allocation while the enclosing systemd scope bounds physical RAM.
        virtual_bytes = (
            int(Path("/proc/self/statm").read_text().split()[0])
            * resource.getpagesize()
        )
        memory_limit = virtual_bytes + 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        value = (
            _reference(*arguments) if operation == "reference" else _replay(*arguments)
        )
        connection.send({"ok": value})
    except BaseException as error:
        connection.send({"error": f"{type(error).__name__}: {error}"[:2000]})
    finally:
        connection.close()


def bounded(operation: str, *arguments, timeout: float = 5.0):
    """A child per candidate, terminated by its own PID on timeout."""
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, operation, arguments))
    process.start()
    writer.close()
    try:
        if not reader.poll(timeout):
            raise ValueError(f"{operation} timeout or resource limit ({timeout:g}s)")
        try:
            result = reader.recv()
        except EOFError as error:
            raise ValueError(
                f"{operation} worker exited (resource limit or crash)"
            ) from error
        if "error" in result:
            raise ValueError(result["error"])
        return result["ok"]
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()


def expanded_program(candidate: dict) -> list[str]:
    macros = load_macros()
    private = candidate.get("macros", {})
    if not isinstance(private, dict) or len(private) > 128:
        raise ValueError("macros must be an object with at most 128 definitions")
    for name, body in private.items():
        tokens = body.split() if isinstance(body, str) else body
        if (
            not isinstance(name, str)
            or not isinstance(tokens, list)
            or not all(isinstance(x, str) for x in tokens)
        ):
            raise ValueError(
                "private macros need string names and string/token-list bodies"
            )
        if name in OP_BY_NAME or (name in macros and macros[name] != tokens):
            raise ValueError(f"private macro shadows canonical name: {name}")
        macros[name] = tokens

    # Bound expansion while constructing it, rather than after exponential growth.
    def walk(tokens, depth=0):
        if depth > 32:
            raise ValueError("macro recursion too deep")
        result = []
        for token in tokens:
            result.extend(
                walk(macros[token], depth + 1) if token in macros else [token]
            )
            if len(result) > MAX_PROGRAM:
                raise ValueError("expanded program too long")
        return result

    program = walk(candidate["program"].split())
    if not program or any(token not in OP_BY_NAME for token in program):
        raise ValueError("program is empty or has unknown operations")
    return program


def validate_candidate(
    candidate: dict,
    seed: int,
    signatures: dict[str, str],
    dev_share: float = 0.0,
    probe_policy: str = PROBE_POLICY,
) -> dict:
    """Return an accepted/rejected journal record; dedup occurs before replay."""
    record = {
        "candidate_digest": digest(candidate),
        "candidate": candidate,
        "probe_policy": probe_policy,
        "status": "rejected",
        "stage": "schema",
    }
    try:
        for key in (
            "id",
            "family",
            "template_id",
            "description",
            "input_kind",
            "program",
            "reference",
        ):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ValueError(f"missing/non-string field: {key}")
        if not isinstance(candidate.get("parameters", {}), dict):
            raise ValueError("parameters must be an object")
        kind = candidate["input_kind"]
        record["stage"] = "reference"
        check_reference(candidate["reference"])
        shared = probe_bank(kind, seed, policy=probe_policy)
        fresh = probe_bank(
            kind,
            seed + int(record["candidate_digest"][:12], 16) + 1,
            16,
            exclude=shared,
            policy=probe_policy,
        )
        inputs = shared + fresh
        outputs = bounded("reference", candidate["reference"], inputs)
        shared_outputs = outputs[: len(shared)]
        record["reference_count"] = len(inputs)
        signature = digest([kind, shared_outputs])
        record["signature"] = signature
        record["stage"] = "dedup"
        if len({canonical(out) for out in shared_outputs}) == 1:
            raise ValueError("constant behavior on shared probes")
        arity = len(shared[0])
        if any(
            all(
                out == args[i] for args, out in zip(shared, shared_outputs, strict=True)
            )
            for i in range(arity)
        ):
            raise ValueError("identity behavior on shared probes")
        if signature in signatures:
            record["duplicate_of"] = signatures[signature]
            raise ValueError("behavioral duplicate on shared probes")
        record["stage"] = "witness"
        program = expanded_program(candidate)
        replay = bounded("witness", program, inputs, outputs)
        shown = 6
        # Fingerprints share their probes; the solver prompt must not. Prefer
        # candidate-specific fresh inputs and preserve diverse list lengths.
        rng = random.Random(record["candidate_digest"] + ":shown-v2")
        pool = list(range(len(shared), len(inputs)))
        rng.shuffle(pool)
        fallback = list(range(len(shared)))
        rng.shuffle(fallback)
        selected: list[int] = []
        lengths: set[int] = set()
        for index in pool + fallback:
            if "list" in kind:
                length = len(inputs[index][0])
                if length in lengths:
                    continue
                lengths.add(length)
            selected.append(index)
            if len(selected) == shown:
                break
        if len(selected) < shown:
            for index in pool + fallback:
                if index not in selected:
                    selected.append(index)
                if len(selected) == shown:
                    break
        if len({canonical(outputs[i]) for i in selected}) == 1:
            shown_value = canonical(outputs[selected[0]])
            alternatives = [
                i
                for i in range(len(inputs))
                if i not in selected and canonical(outputs[i]) != shown_value
            ]
            rng.shuffle(alternatives)
            if alternatives:
                replacement = alternatives[0]
                slot = rng.randrange(shown)
                if "list" in kind:
                    same_length = [
                        j
                        for j, i in enumerate(selected)
                        if len(inputs[i][0]) == len(inputs[replacement][0])
                    ]
                    if same_length:
                        slot = same_length[0]
                selected[slot] = replacement
        order = selected + [i for i in range(len(inputs)) if i not in selected]
        replay = {key: [values[i] for i in order] for key, values in replay.items()}
        split_value = int(digest([seed, candidate["template_id"]])[:16], 16) / 2**64
        split = "eval" if split_value < dev_share else "train"
        task = Task(
            behaviour=candidate["id"],
            summary=candidate["description"][:200],
            split=split,
            task_hash_split=assign_task_hash_split(
                tuple(replay["inputs"][:shown]), tuple(replay["outputs"][:shown])
            ),
            inputs=tuple(replay["inputs"][:shown]),
            outputs=tuple(replay["outputs"][:shown]),
            held_out_inputs=tuple(replay["inputs"][shown:]),
            held_out_outputs=tuple(replay["outputs"][shown:]),
            program=tuple(program),
            length=len(program),
            examples=shown,
            held_out=len(inputs) - shown,
            fuel=sum(replay["fuels"][:shown]),
            tags=(candidate["family"], candidate["template_id"]),
            skeleton=skeleton(program),
            witness_source="authored",
            macro_program=candidate["program"],
        )
        record.update(
            status="accepted",
            stage="verified",
            task=asdict(task),
            verification_count=len(inputs),
            shared_count=len(shared),
            fresh_count=len(fresh),
            shown_count=shown,
            shown_output_values=len({canonical(outputs[i]) for i in selected}),
            hidden_count=len(inputs) - shown,
            scalar_exhaustive=(kind == "int"),
        )
        signatures[signature] = candidate["id"]
    except (ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
        record["reason"] = str(error)[:2000]
    return record


def provenance(
    paths: list[Path], seed: int, dev_share: float, probe_policy: str = PROBE_POLICY
) -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    macro_files = sorted(
        (Path(__file__).parents[1] / "world_behaviours").glob("macros*.txt")
    )
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "config": {
            "seed": seed,
            "dev_share": dev_share,
            "probe_policy": probe_policy,
            "shared_probes": SHARED_PROBES if probe_policy == PROBE_POLICY else 64,
            "shared_counts_by_kind": {
                kind: (
                    21
                    if kind == "int"
                    else 441
                    if kind == "two_ints" and probe_policy == PROBE_POLICY
                    else SHARED_PROBES
                    if probe_policy == PROBE_POLICY
                    else 64
                )
                for kind in sorted(KINDS)
            },
            "fresh_hidden": 16,
            "shown": 6,
            "shown_policy": "v3-output-variation",
            "reference_timeout_seconds": 5,
            "scalar_domain": "-10..10 exhaustive; 6 shown + 15 hidden",
        },
        "seed": seed,
        "source_data_digests": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths + macro_files
        },
        "wandb_run.url": None,
    }


def snapshot(
    output_dir: Path, records: list[dict], origin: dict, origins: dict | None = None
) -> dict:
    origins = origins or {digest(origin): origin}
    combined = dict(origin)
    combined["source_data_digests"] = {
        path: value
        for item in [*origins.values(), origin]
        for path, value in item.get("source_data_digests", {}).items()
    }
    accepted = [r for r in records if r["status"] == "accepted"]
    rejected = [r for r in records if r["status"] != "accepted"]
    for name, rows in (("accepted", accepted), ("rejected", rejected)):
        atomic_text(
            output_dir / f"{name}.jsonl", "".join(canonical(r) + "\n" for r in rows)
        )
    tasks = [r["task"] for r in accepted]
    for split in ("train", "eval"):
        path = output_dir / f"tasks_{split}.jsonl.gz"
        temporary = path.with_suffix(".tmp")
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed,
            io.TextIOWrapper(packed, encoding="utf-8") as handle,
        ):
            for task in tasks:
                if task["split"] == split:
                    handle.write(canonical(task) + "\n")
        temporary.replace(path)
    summary = {
        "provenance": combined,
        "resume_caveats": origin.get("resume_caveats", []),
        "candidates": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejections_by_stage": dict(Counter(r["stage"] for r in rejected)),
        "rejections_by_reason": dict(
            Counter(r["reason"].split(":", 1)[0] for r in rejected)
        ),
        "families": dict(Counter(r["candidate"]["family"] for r in accepted)),
        "templates": dict(Counter(r["candidate"]["template_id"] for r in accepted)),
        "skeletons": dict(Counter(t["skeleton"] for t in tasks)),
        "lengths": dict(Counter(str(t["length"]) for t in tasks)),
        "splits": dict(Counter(t["split"] for t in tasks)),
        "verification_counts": dict(
            Counter(str(r["verification_count"]) for r in accepted)
        ),
        "exact_verified_pairs": sum(r["verification_count"] for r in accepted),
        "split_policy": (
            "train-only; requires separately authored dev families"
            if not origin["config"]["dev_share"]
            else "template-id hash; internal dev only, skeleton overlap possible"
        ),
        "provenance_history": {key: f"provenance/{key}.json" for key in origins},
        "scalar_exhaustive_tasks": sum(r["scalar_exhaustive"] for r in accepted),
        "unique_families": len({r["candidate"]["family"] for r in accepted}),
        "unique_templates": len({r["candidate"]["template_id"] for r in accepted}),
        "unique_skeletons": len({t["skeleton"] for t in tasks}),
        "artifacts": {
            name: {
                "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
                "provenance": "summary.json:provenance and provenance_history",
            }
            for name in (
                "accepted.jsonl",
                "rejected.jsonl",
                "tasks_train.jsonl.gz",
                "tasks_eval.jsonl.gz",
            )
        },
    }
    atomic_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def run(
    paths: list[Path],
    output_dir: Path,
    seed: int = 0,
    limit: int | None = None,
    dev_share: float = 0.0,
) -> dict:
    if not 0 <= dev_share < 1:
        raise ValueError("dev_share must be in [0, 1)")
    output_dir.mkdir(parents=True, exist_ok=True)
    journal = output_dir / "records"
    journal.mkdir(exist_ok=True)
    # The journal is authoritative; snapshots may lag an interrupted process.
    records = [json.loads(p.read_text()) for p in sorted(journal.glob("*.json"))]
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    origins = {
        p.stem: json.loads(p.read_text()) for p in sorted(provenance_dir.glob("*.json"))
    }
    # Older journals embedded the full source map per row. Export compact views
    # and preserve their original manifests without rewriting old journal files.
    for record in records:
        if "provenance" in record:
            old_origin = record.pop("provenance")
            key = digest(old_origin)
            if key not in origins:
                atomic_text(
                    provenance_dir / f"{key}.json", canonical(old_origin) + "\n"
                )
                origins[key] = old_origin
            record["provenance_id"] = key
    requested = {"seed": seed, "dev_share": dev_share, "probe_policy": PROBE_POLICY}
    config_path = output_dir / "campaign_config.json"
    if config_path.exists():
        effective = json.loads(config_path.read_text())
        effective.setdefault("probe_policy", LEGACY_PROBE_POLICY)
    elif records:
        first = origins[records[0]["provenance_id"]]["config"]
        effective = {
            "seed": first["seed"],
            "dev_share": first["dev_share"],
            "probe_policy": first.get("probe_policy", LEGACY_PROBE_POLICY),
        }
        atomic_text(config_path, canonical(effective) + "\n")
    else:
        effective = requested
        atomic_text(config_path, canonical(effective) + "\n")
    seed, dev_share = effective["seed"], effective["dev_share"]
    probe_policy = effective["probe_policy"]
    origin = provenance(paths, seed, dev_share, probe_policy)
    origin["requested_config"] = requested
    if requested != effective:
        origin["resume_caveats"] = [
            f"Requested config {canonical(requested)} differs from campaign config; retaining {canonical(effective)} for consistent fingerprints and splits."
        ]
    if probe_policy != PROBE_POLICY:
        origin.setdefault("resume_caveats", []).append(
            "This directory retains its older shared64 probe policy; use a fresh output directory for full v3-edge-cartesian coverage. Existing task records are unchanged."
        )
    origin_id = digest(origin)
    if origin_id not in origins:
        atomic_text(provenance_dir / f"{origin_id}.json", canonical(origin) + "\n")
        origins[origin_id] = origin
    seen = {r["candidate_digest"] for r in records}
    signatures = {
        r["signature"]: r["candidate"]["id"]
        for r in records
        if r["status"] == "accepted"
    }
    accepted_ids = {r["candidate"]["id"] for r in records if r["status"] == "accepted"}
    processed = 0
    started = time.monotonic()
    last_snapshot = started
    try:
        for path in paths:
            with path.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    if limit is not None and processed >= limit:
                        break
                    try:
                        candidate = json.loads(line)
                    except json.JSONDecodeError as error:
                        candidate = {
                            "invalid_json": line[:20000],
                            "source": str(path),
                            "line": line_number,
                            "error": str(error),
                        }
                    if not isinstance(candidate, dict):
                        candidate = {"invalid_row": candidate}
                    content_digest = digest(candidate)
                    if content_digest in seen:
                        continue
                    if (
                        isinstance(candidate.get("id"), str)
                        and candidate["id"] in accepted_ids
                    ):
                        record = {
                            "candidate_digest": content_digest,
                            "candidate": candidate,
                            "status": "rejected",
                            "stage": "schema",
                            "reason": "candidate id already accepted with different content",
                        }
                    else:
                        record = validate_candidate(
                            candidate, seed, signatures, dev_share, probe_policy
                        )
                    record["provenance_id"] = origin_id
                    record["source"] = {
                        "path": str(path),
                        "line": line_number,
                        "sha256": origin["source_data_digests"][str(path)],
                    }
                    atomic_text(
                        journal / f"{len(records):08d}-{content_digest}.json",
                        canonical(record) + "\n",
                    )
                    records.append(record)
                    seen.add(content_digest)
                    if record["status"] == "accepted":
                        accepted_ids.add(candidate["id"])
                    processed += 1
                    if (
                        processed == 1
                        or processed % 512 == 0
                        or time.monotonic() - last_snapshot >= 60
                    ):
                        report = snapshot(output_dir, records, origin, origins)
                        last_snapshot = time.monotonic()
                        print(
                            canonical(
                                {
                                    "processed": processed,
                                    "accepted": report["accepted"],
                                    "rejected": report["rejected"],
                                    "seconds": round(time.monotonic() - started, 2),
                                }
                            ),
                            flush=True,
                        )
            if limit is not None and processed >= limit:
                break
    finally:
        report = snapshot(output_dir, records, origin, origins)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dev-share", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = run(args.input, args.output_dir, args.seed, args.limit, args.dev_share)
    print(
        canonical(
            {
                key: report[key]
                for key in (
                    "candidates",
                    "accepted",
                    "rejected",
                    "unique_families",
                    "unique_templates",
                    "unique_skeletons",
                    "exact_verified_pairs",
                    "resume_caveats",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
