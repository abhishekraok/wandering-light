"""Signal House: engine, curriculum, adviser and service."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from wandering_light.stack_machine import MARK_OPS, Int, Lst, Sym, run_body
from wandering_light.stack_workshop import engine
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACROS,
    OPS,
    compile_program,
    execute,
    inspect_op,
    snapshot,
    stack_from,
)
from wandering_light.stack_workshop.model import Adviser
from wandering_light.stack_workshop.server import (
    Handler,
    ThreadingHTTPServer,
    api,
)


def ex(inp, out):
    return {"input": inp, "output": out}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_ops_fuel_and_macros():
    assert list(OPS) == [op.name for op in MARK_OPS]
    assert FUEL == 100_000
    assert MACROS["inc"] == {"body": "1 add", "group": "scalars"}
    assert MACROS["2"]["group"] == "constants"
    assert MACROS["sum"]["group"] == "lists"
    assert MACROS["gcd"]["group"] == "scalars"  # comment lines do not reset groups
    assert all(":" not in name for name in MACROS)


def test_compile_spans_and_outermost_macro():
    text = "sq [inc] # a comment\nadd"
    tokens = compile_program(text)
    assert [t["op"] for t in tokens] == ["dup", "mul", "[", "1", "add", "]", "add"]
    assert tokens[0] == {"op": "dup", "start": 0, "end": 2, "macro": "sq"}
    assert tokens[2] == {"op": "[", "start": 3, "end": 4, "macro": None}
    assert tokens[3]["macro"] == "inc" and text[4:7] == "inc"
    assert (tokens[3]["start"], tokens[3]["end"]) == (4, 7)
    assert tokens[-1]["start"] == text.index("add") and tokens[-1]["macro"] is None
    # Nested macros report the outermost name the learner wrote.
    nested = compile_program("abs")
    assert {t["macro"] for t in nested} == {"abs"}
    assert "sub" in [t["op"] for t in nested]  # `neg` expanded inside `abs`
    assert compile_program("  # only a comment") == []


def test_every_macro_expands_to_primitives():
    for name in MACROS:
        assert all(t["op"] in OPS for t in compile_program(name))


def test_compile_errors(monkeypatch):
    with pytest.raises(ValueError, match=r"^Unknown word 'frob'\. "):
        compile_program("1 frob")
    monkeypatch.setitem(MACROS, "loop-a", {"body": "1 loop-b", "group": None})
    monkeypatch.setitem(MACROS, "loop-b", {"body": "loop-a", "group": None})
    with pytest.raises(ValueError, match=r"^Recursive macro: loop-a$"):
        compile_program("loop-a")
    monkeypatch.setitem(MACROS, "bad", {"body": "dup nope", "group": None})
    with pytest.raises(ValueError, match="Unknown word 'nope'"):
        compile_program("bad")


def test_stack_from_accepts_json_and_text():
    assert stack_from([1, True, [2, [3]]]) == stack_from("1 true (2 (3))")
    assert stack_from([]) == () == stack_from("")
    assert stack_from("[ 1 nil ]") == (Int(1), Lst(()))
    assert stack_from([10**12, -(10**12)]) == (Int(10**12), Int(-(10**12)))
    assert stack_from([10**100], expected=True) == (Int(10**100),)


@pytest.mark.parametrize(
    "value",
    [
        [1.5],
        ["1"],
        [None],
        [{"a": 1}],
        "1 dup",
        "MARK",
        "(1 add)",
        [10**12 + 1],
        "1000000000001",
        list(range(17)),
        [list(range(65))],
        [[[[[[[[[[[[[[1]]]]]]]]]]]]]],  # thirteen levels
        "((((((((((((( 1 )))))))))))))",
        "( 1",
        5,
        None,
        {"input": 1},
    ],
)
def test_stack_from_rejects(value):
    with pytest.raises(ValueError):
        stack_from(value)


def test_stack_from_limits_are_inclusive():
    assert len(stack_from(list(range(16)))) == 16
    stack_from([list(range(64))])
    deep: list = [1]
    for _ in range(11):
        deep = [deep]
    stack_from([deep])  # twelve levels of nesting
    with pytest.raises(ValueError):
        stack_from([[deep]])
    with pytest.raises(ValueError):
        stack_from([10**512], expected=True)
    stack_from([10**512 - 1], expected=True)


def test_snapshot_types_and_elision():
    stack = (*stack_from([1, True, [2]]), Sym("add"), engine.stack_machine.MARK)
    shown = snapshot(stack)
    assert [v["type"] for v in shown] == ["int", "bool", "list", "symbol", "mark"]
    assert shown[2]["items"] == [{"type": "int", "text": "2"}]
    assert shown[1]["text"] == "true" and shown[3]["text"] == "add"
    deep = tuple(Int(i) for i in range(60))
    view = snapshot(deep)
    assert len(view) == 49
    assert view[0]["text"] == "0"
    assert view[1]["type"] == "elision"
    assert [v["text"] for v in view[2:]] == [str(i) for i in range(13, 60)]
    assert len(snapshot(deep[:48])) == 48


def test_execute_frames_and_matching():
    result = execute("dup add", [ex([3], [6]), ex([0], [1])])
    assert result["matches"] == 1 and result["count"] == 2
    assert result["primitive_count"] == 2
    row = result["examples"][0]
    assert row["matched"] and row["error"] is None
    assert [f["op"] for f in row["frames"]] == [None, "dup", "add"]
    assert [f["fuel"] for f in row["frames"]] == [0, 1, 2]
    assert row["frames"][1]["stack"] == [{"type": "int", "text": "3"}] * 2
    assert row["max_depth"] == 2 and row["fuel"] == 2
    assert row["output"] == [{"type": "int", "text": "6"}]
    assert not result["examples"][1]["matched"]
    assert result["fuel"] == 4 and result["max_depth"] == 2


def test_execute_marks_quoting_and_counts_combinator_fuel():
    row = execute("[ 1 add ] i", [ex([5], [6])])["examples"][0]
    assert [f["quoting"] for f in row["frames"]] == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert row["matched"]
    assert row["frames"][-1]["fuel"] == row["fuel"] > 5


def test_refusal_halts_only_its_example():
    result = execute("add 1 add", [ex([1, 2], [4]), ex([1], [2])])
    good, bad = result["examples"]
    assert good["matched"] and len(good["frames"]) == 4
    assert bad["error"] == "add: needs 2 value(s), stack has 1"
    assert not bad["matched"]
    assert len(bad["frames"]) == 2
    assert bad["frames"][-1]["error"] == bad["error"]
    assert bad["frames"][-1]["stack"] == bad["frames"][0]["stack"]
    assert bad["frames"][-1]["op"] == "add"


def test_out_of_fuel_is_a_refusal():
    row = execute("[ dup i ] dup i", [ex([], [])])["examples"][0]
    assert row["error"].startswith("i: ")


def test_capture_false_keeps_first_and_final():
    full = execute("1 add 1 add", [ex([1], [3])])
    lean = execute("1 add 1 add", [ex([1], [3])], capture=False)
    assert len(full["examples"][0]["frames"]) == 5
    frames = lean["examples"][0]["frames"]
    assert frames == [full["examples"][0]["frames"][i] for i in (0, -1)]
    assert lean["examples"][0]["matched"]
    refused = execute("add add", [ex([1, 2], [3])], capture=False)["examples"][0]
    assert len(refused["frames"]) == 2 and refused["frames"][-1]["error"]
    assert len(execute("", [ex([1], [1])], capture=False)["examples"][0]["frames"]) == 1


def test_execute_example_count():
    with pytest.raises(ValueError):
        execute("1", [])
    with pytest.raises(ValueError):
        execute("1", [ex([], [1])] * 17)
    with pytest.raises(ValueError):
        execute("1", [{"input": []}])
    assert execute("1", [ex([], [1])] * 16)["matches"] == 16


def test_inspect_records_nested_calls_in_return_order():
    examples = [ex([[1, 2, 2]], [[2, 3, 3]])]
    first = inspect_op("[ 1 add ] map", examples, 4)
    assert first["error"] is None and not first["truncated"]
    # One `add` per item, including the repeated 2 a cache would have answered.
    assert [f["op"] for f in first["frames"]] == ["add", "add", "add"]
    assert [f["after"][0]["text"] for f in first["frames"]] == ["2", "3", "3"]
    assert first["frames"][0]["before"] == snapshot((Int(1), Int(1)))
    assert all(f["depth"] == 1 for f in first["frames"])
    # A warm cache must not hide nested execution on a second look.
    assert inspect_op("[ 1 add ] map", examples, 4) == first
    nested = inspect_op("[ [ 1 add ] map ] i", [ex([[1, 2]], [[2, 3]])], 7)
    assert [(f["op"], f["depth"]) for f in nested["frames"]] == [
        ("add", 2),
        ("add", 2),
        ("map", 1),
    ]


def test_inspect_reports_refusals_and_truncation():
    refused = inspect_op("[ add ] i", [ex([1], [1])], 3)
    assert refused["error"] == "i: add: needs 2 value(s), stack has 1"
    assert refused["frames"][-1]["error"] == "needs 2 value(s), stack has 1"
    assert refused["frames"][-1]["after"] is None
    early = inspect_op("add 1", [ex([1], [1])], 1)
    assert early == {"frames": [], "truncated": False, "error": early["error"]}
    assert early["error"].startswith("add:")
    tokens = compile_program("zeros1024 [ 1 add ] map")
    big = inspect_op(tokens, [ex([], [])], len(tokens) - 1)
    assert big["truncated"] and len(big["frames"]) == 350
    assert big["error"] is None
    with pytest.raises(ValueError):
        inspect_op("1", [ex([], [])], 1)
    with pytest.raises(ValueError):
        inspect_op("1", [ex([], [])], 0, example=1)
    assert sys.getprofile() is None


def test_inspect_clears_run_body_cache():
    examples = [ex([[1, 2]], [[2, 3]])]
    execute("[ 1 add ] map", examples)  # warm the cache
    assert run_body.cache_info().currsize > 0
    assert len(inspect_op("[ 1 add ] map", examples, 4)["frames"]) == 2


def test_core_does_not_import_torch():
    code = (
        "import sys, wandering_light.stack_workshop.server as s;"
        "s.api('/api/context', {});"
        "print('torch' in sys.modules, 'numpy' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False"]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

FIELDS = {
    "id",
    "chapter",
    "title",
    "concept",
    "story",
    "goal",
    "starter",
    "solution",
    "examples",
    "heldout",
    "hints",
    "discovery",
    "review",
    "prediction",
    "macros",
}


def test_lessons_shape():
    assert len(LESSONS) == 14
    assert {lesson["id"]: lesson for lesson in LESSONS} == BY_ID
    assert len(BY_ID) == 14
    for lesson in LESSONS:
        assert set(lesson) == FIELDS
        assert len(lesson["examples"]) == 3 and len(lesson["heldout"]) == 3
        assert lesson["hints"]
        for name in lesson["macros"]:
            assert name in MACROS


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_solution_passes_all_six(lesson):
    result = execute(lesson["solution"], lesson["examples"] + lesson["heldout"])
    assert result["matches"] == 6, [r["error"] for r in result["examples"]]
    prediction = lesson["prediction"]
    run = execute(prediction["program"], [ex(prediction["input"], [])])
    assert run["examples"][0]["error"] is None


def test_first_lessons_and_coverage():
    assert LESSONS[0]["id"] == "first-light" and LESSONS[0]["solution"] == "1 add"
    assert LESSONS[1]["id"] == "echo" and LESSONS[1]["solution"] == "dup add"
    assert LESSONS[0]["examples"][0] == {"input": [3], "output": [4]}
    words = " ".join(lesson["solution"] for lesson in LESSONS).split()
    for op in ("swap", "over", "drop", "i", "map", "filter", "fold", "dip", "ifte"):
        assert op in words or any(
            op in MACROS[w]["body"].split() for w in words if w in MACROS
        ), op
    assert any(lesson["macros"] for lesson in LESSONS)


def test_fit_is_not_an_explanation():
    lesson = BY_ID["false-dawn"]
    starter = execute(lesson["starter"], lesson["examples"])
    assert starter["matches"] == 3
    verdict = api("/api/verify", {"lesson": "false-dawn", "program": lesson["starter"]})
    assert not verdict["passed"] and verdict["extra"] < verdict["extra_count"]


# ---------------------------------------------------------------------------
# Adviser
# ---------------------------------------------------------------------------


class FakeScorer:
    """Prefers ops by their index: later ops in mark-28 score higher."""

    def __init__(self):
        self.calls = []

    def log_probs(self, blocks, legal):
        import numpy as np

        self.calls.append((blocks, legal))
        logits = np.where(legal, np.arange(legal.shape[1], dtype=float), -np.inf)
        shifted = logits - logits[np.isfinite(logits)].max()
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def test_adviser_not_loaded():
    adviser = Adviser()
    assert adviser.status()["ready"] is False
    out = adviser.rank("1 add", [ex([1], [2])])
    assert out["suggestions"] == [] and out["ready"] is False
    with pytest.raises(ValueError, match="add"):
        adviser.rank("add", [ex([1], [2])])


def test_adviser_load_records_error(tmp_path, monkeypatch):
    monkeypatch.delenv("STACK_POLICY_CHECKPOINT", raising=False)
    adviser = Adviser()
    status = adviser.load()
    assert status["ready"] is False and "checkpoint" in status["error"].lower()
    status = adviser.load(tmp_path)
    assert status["ready"] is False and status["error"]


def test_adviser_ranks_legal_ops():
    pytest.importorskip("torch")
    adviser = Adviser()
    adviser.scorer = FakeScorer()
    examples = [ex([1, 2], [3]), ex([5], [6])]
    out = adviser.rank("1", examples)
    suggestions = out["suggestions"]
    assert out["ready"] is True
    assert 0 < len(suggestions) <= 6
    probs = [s["probability"] for s in suggestions]
    assert probs == sorted(probs, reverse=True)
    names = [s["op"] for s in suggestions]
    assert "]" not in names  # no open quotation on either example
    assert "ifte" not in names  # needs three quotations
    for s in suggestions:
        assert len(s["previews"]) == 2
    add = execute("1 add", examples)["examples"]
    by_op = {s["op"]: s for s in suggestions}
    if "add" in by_op:
        assert by_op["add"]["previews"][1] == add[1]["frames"][-1]["stack"]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def test_context_hides_answers():
    ctx = api("/api/context", {})
    assert len(ctx["lessons"]) == 14
    for public in ctx["lessons"]:
        assert not {"solution", "heldout", "hints"} & set(public)
        assert public["extra_tests"] == 3 and public["hint_count"] >= 1
    assert [op["name"] for op in ctx["ops"]] == list(OPS)
    assert len(ctx["macros"]) == len(MACROS)
    sq = next(m for m in ctx["macros"] if m["name"] == "sq")
    assert sq["expansion"] == "dup mul" and sq["group"] == "scalars"
    assert "ready" in ctx["model"]
    assert isinstance(ctx["world"], dict) and "available" in ctx["world"]
    json.dumps(ctx)


def test_world_shelf_never_fails(tmp_path):
    from wandering_light.stack_workshop.server import world_shelf

    assert world_shelf(tmp_path / "missing.json")["available"] is False


def test_evaluate_and_verify():
    out = api("/api/evaluate", {"lesson": "echo", "program": "dup add"})
    assert out["matches"] == 3 and out["count"] == 3
    verdict = api("/api/verify", {"lesson": "echo", "program": "dup add"})
    assert verdict == {
        "passed": True,
        "shown": 3,
        "shown_count": 3,
        "extra": 3,
        "extra_count": 3,
        "failures": [],
        "evidence": "shown and extra tests",
    }
    wrong = api("/api/verify", {"lesson": "echo", "program": "1 add"})
    assert not wrong["passed"] and len(wrong["failures"]) == 3
    assert wrong["failures"][0]["set"] == "shown"


def test_sandbox():
    data = {"lesson": "sandbox", "program": "sq", "examples": [ex([3], [9])]}
    verdict = api("/api/verify", data)
    assert verdict["passed"] and verdict["evidence"] == "examples"
    assert verdict["extra_count"] == 0
    assert api("/api/evaluate", data)["matches"] == 1
    assert api("/api/inspect", {**data, "program": "[ sq ] i", "step": 4})["frames"]
    with pytest.raises(ValueError):
        api("/api/hint", data)
    with pytest.raises(ValueError):
        api("/api/verify", {"lesson": "sandbox", "program": "1"})


def test_hint_reveal_suggest_model_routes():
    hint = api("/api/hint", {"lesson": "first-light", "index": 1})
    assert hint["hint"] == BY_ID["first-light"]["hints"][1]
    with pytest.raises(ValueError):
        api("/api/hint", {"lesson": "first-light", "index": 9})
    assert api("/api/reveal", {"lesson": "echo"})["solution"] == "dup add"
    assert (
        api("/api/suggest", {"lesson": "echo", "program": "dup"})["suggestions"] == []
    )
    assert "ready" in api("/api/model", {})
    with pytest.raises(ValueError):
        api("/api/evaluate", {"lesson": "nope", "program": ""})
    with pytest.raises(ValueError):
        api("/api/nowhere", {})


@pytest.fixture
def served():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(url, payload, headers=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_handler(served):
    status, body = post(
        f"{served}/api/evaluate", {"lesson": "echo", "program": "dup add"}
    )
    assert status == 200 and body["matches"] == 3
    status, body = post(
        f"{served}/api/context", {}, {"Origin": "http://localhost:5173"}
    )
    assert status == 200 and len(body["lessons"]) == 14
    status, body = post(f"{served}/api/context", {}, {"Origin": "https://evil.example"})
    assert status == 403 and "error" in body
    status, body = post(
        f"{served}/api/context", {}, {"Origin": "http://127.0.0.1.evil.example"}
    )
    assert status == 403
    status, body = post(f"{served}/api/evaluate", {"lesson": "echo", "program": "zap"})
    assert status == 400 and body["error"].startswith("Unknown word 'zap'")
    status, body = post(f"{served}/api/evaluate", b"{not json")
    assert status == 400 and "error" in body
    status, body = post(f"{served}/api/evaluate", [1, 2])
    assert status == 400
