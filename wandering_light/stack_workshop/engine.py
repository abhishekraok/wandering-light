"""Compile workshop words and observe transitions on the canonical machine."""

import re
import sys
import sysconfig
from collections import deque
from pathlib import Path
from threading import RLock

from wandering_light import stack_machine as machine

OPS = {op.name: op for op in machine.MARK_OPS}
FUEL = 100_000
MACHINE_LOCK = RLock()
_WORDS = re.compile(r"#[^\r\n]*|[\[\]]|[^\s\[\]#]+")
_MACRO_FILE = (
    Path(__file__).resolve().parents[2] / "experiments/world_behaviours/macros.txt"
)
if not _MACRO_FILE.is_file():
    _MACRO_FILE = (
        Path(sysconfig.get_path("data"))
        / "share/wandering-light/world_behaviours/macros.txt"
    )


def _read_macros():
    macros = {}
    group = ""
    for line in _MACRO_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ----"):
            group = line.removeprefix("# ----").strip(" -")
        definition = line.partition("#")[0].strip()
        if definition:
            name, body = definition.split(":", 1)
            macros[name.strip()] = {"body": body.strip(), "group": group}
    return macros


MACROS = _read_macros()


def _words(text):
    return (m for m in _WORDS.finditer(text) if not m[0].startswith("#"))


def compile_program(text: str) -> list[dict]:
    if not isinstance(text, str):
        raise ValueError("Program must be text.")
    tokens = []

    def expand(name, start, end, path):
        if name in OPS:
            tokens.append(
                {
                    "op": name,
                    "start": start,
                    "end": end,
                    "macro": path[0] if path else None,
                }
            )
        elif name in path:
            raise ValueError(f"Recursive macro: {name}")
        elif name in MACROS:
            for word in _words(MACROS[name]["body"]):
                expand(word[0], start, end, (*path, name))
        else:
            raise ValueError(
                f"Unknown word '{name}'. Use a mark-28 operation or a library macro."
            )

    for match in _words(text):
        expand(match[0], *match.span(), ())
    return tokens


def stack_from(value, *, expected=False) -> tuple[machine.Value, ...]:
    limit = 10**512 - 1 if expected else 10**12

    from_text = isinstance(value, str)

    def convert(item, depth):
        if from_text:
            if isinstance(item, (machine.Int, machine.Bool)):
                item = item.value
            elif isinstance(item, machine.Lst):
                item = item.items
        if isinstance(item, bool):
            return machine.Bool(item)
        if isinstance(item, int):
            if abs(item) > limit:
                raise ValueError("Integer exceeds the example limit.")
            return machine.Int(item)
        if isinstance(item, tuple if from_text else list):
            if depth > 12 or len(item) > 64:
                raise ValueError(
                    "Lists allow at most 64 items and 12 levels of nesting."
                )
            return machine.Lst(tuple(convert(child, depth + 1) for child in item))
        raise ValueError("Examples accept only integers, booleans and lists.")

    if from_text:
        try:
            values = machine.parse_stack(value, basis="mark-28")
        except (machine.MachineError, RecursionError) as exc:
            raise ValueError("Invalid stack notation.") from exc
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("A stack must be a JSON array or stack-notation text.")
    if len(values) > 16:
        raise ValueError("A stack may contain at most 16 values.")
    return tuple(convert(item, 1) for item in values)


def _display(value):
    if isinstance(value, machine.Lst):
        items = [_display(item) for item in value.items]
        text = "(" + " ".join(item["text"] for item in items) + ")" if items else "nil"
        return {"type": "list", "text": text, "items": items}
    kind = {
        machine.Int: "int",
        machine.Bool: "bool",
        machine.Sym: "symbol",
        machine.Mark: "mark",
    }[type(value)]
    try:
        text = machine.format_value(value)
    except ValueError:
        # Python can refuse decimal rendering of a large computed integer.
        text = f"<{value.value.bit_length()}-bit integer>"
    return {"type": kind, "text": text}


def snapshot(stack) -> list[dict]:
    if len(stack) > 48:
        return [
            _display(stack[0]),
            {"type": "elision", "text": f"… {len(stack) - 48} omitted"},
            *(_display(v) for v in stack[-47:]),
        ]
    return [_display(v) for v in stack]


def example_stacks(examples):
    if not isinstance(examples, list) or not 1 <= len(examples) <= 16:
        raise ValueError("Supply between 1 and 16 examples.")
    pairs = []
    for row in examples:
        if not isinstance(row, dict) or not {"input", "output"} <= row.keys():
            raise ValueError("Each example needs an input and an output.")
        pairs.append(
            (stack_from(row["input"]), stack_from(row["output"], expected=True))
        )
    return pairs


def _replay(tokens, stack):
    """Yield the input, then each attempted transition. Caller holds MACHINE_LOCK."""
    spent = 0
    yield None, stack, spent, None
    for token in tokens:
        name = token["op"]
        outcome = machine.step_raw(OPS[name], stack, FUEL - spent, 0)
        if isinstance(outcome, str):
            yield name, stack, spent, f"{name}: {outcome}"
            return
        stack, cost = outcome
        spent += cost
        yield name, stack, spent, None


def replay_final(tokens, stack):
    """Consume the replay, retaining only its final state or refusal."""
    return deque(_replay(tokens, stack), maxlen=1).pop()


def _frame(name, stack, fuel, error):
    return {
        "stack": snapshot(stack),
        "fuel": fuel,
        "op": name,
        "error": error,
        "quoting": machine.quotation_open(stack),
    }


def execute(program, examples, *, capture=True) -> dict:
    tokens = compile_program(program)
    rows = []
    with MACHINE_LOCK:
        for initial, target in example_stacks(examples):
            frames = []
            max_depth = 0
            for name, stack, fuel, error in _replay(tokens, initial):
                max_depth = max(max_depth, len(stack))
                if capture or name is None:
                    frames.append(_frame(name, stack, fuel, error))
            if not capture and name is not None:
                frames.append(_frame(name, stack, fuel, error))
            matched = (
                error is None
                and len(stack) == len(target)
                and all(
                    machine.value_key(a) == machine.value_key(b)
                    for a, b in zip(stack, target, strict=True)
                )
            )
            rows.append(
                {
                    "frames": frames,
                    "matched": matched,
                    "error": error,
                    "fuel": fuel,
                    "max_depth": max_depth,
                    "output": snapshot(target),
                }
            )
    return {
        "tokens": tokens,
        "examples": rows,
        "matches": sum(r["matched"] for r in rows),
        "count": len(rows),
        "primitive_count": len(tokens),
        "fuel": sum(r["fuel"] for r in rows),
        "max_depth": max(r["max_depth"] for r in rows),
    }


def inspect_op(program, examples, step, example=0) -> dict:
    """Observe one zero-based expanded token, including its outermost return."""
    tokens = compile_program(program)
    pairs = example_stacks(examples)
    if type(example) is not int or not 0 <= example < len(pairs):
        raise ValueError("Example index is out of range.")
    if type(step) is not int or not 0 <= step < len(tokens):
        raise ValueError("Step index is out of range.")
    frames = []
    step_code = machine.step_raw.__code__

    def observe(frame, event, result):
        if event == "return" and frame.f_code is step_code and len(frames) < 350:
            args = frame.f_locals
            error = result if isinstance(result, str) else None
            before = args["stack"]
            frames.append(
                {
                    "op": args["op"].name,
                    "depth": args["depth"],
                    "before": snapshot(before),
                    "after": snapshot(before if error else result[0]),
                    "error": error,
                }
            )

    with MACHINE_LOCK:
        _, stack, fuel, error = replay_final(tokens[:step], pairs[example][0])
        if error:
            return {"frames": [], "truncated": False, "error": error}
        machine.run_body.cache_clear()
        previous = sys.getprofile()
        name = tokens[step]["op"]
        try:
            sys.setprofile(observe)
            outcome = machine.step_raw(OPS[name], stack, FUEL - fuel, 0)
        finally:
            sys.setprofile(previous)
    error = f"{name}: {outcome}" if isinstance(outcome, str) else None
    return {"frames": frames, "truncated": len(frames) == 350, "error": error}
