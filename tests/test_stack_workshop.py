"""Signal House's machine traces, lesson evidence, and localhost API."""

import json
import sys
from http.client import HTTPConnection
from threading import Thread

import pytest

from wandering_light import stack_machine as machine
from wandering_light.stack_workshop import engine, server
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
from wandering_light.stack_workshop.server import Handler, ThreadingHTTPServer, api


def test_compile_preserves_source_spans_and_outer_macro():
    assert {op.name: op for op in machine.MARK_OPS} == OPS
    assert FUEL == 100_000
    assert MACROS["16"] == {"body": "8 dup add", "group": "constants"}
    assert MACROS["rot"]["group"] == "stack shuffles"
    source = "[16]# notAnOp [\r\ni"
    tokens = compile_program(source)
    assert [token["op"] for token in tokens] == [
        "[",
        "1",
        "1",
        "add",
        "dup",
        "add",
        "dup",
        "add",
        "dup",
        "add",
        "]",
        "i",
    ]
    assert tokens[0] == {"op": "[", "start": 0, "end": 1, "macro": None}
    assert all(
        (token["start"], token["end"], token["macro"]) == (1, 3, "16")
        for token in tokens[1:-2]
    )
    assert tokens[-2] == {"op": "]", "start": 3, "end": 4, "macro": None}
    assert tokens[-1]["start"] == source.index("\ni") + 1


def test_compile_rejects_unknown_words_and_indirect_recursion(monkeypatch):
    with pytest.raises(ValueError, match=r"^Unknown word 'twice'\. "):
        compile_program("1 twice")
    monkeypatch.setitem(MACROS, "outer", {"body": "1 inner", "group": "test"})
    monkeypatch.setitem(MACROS, "inner", {"body": "outer", "group": "test"})
    with pytest.raises(ValueError, match=r"^Recursive macro: outer$"):
        compile_program("outer")


def test_stack_input_forms_and_limits():
    values = [-(10**12), True, [False, [3], []]]
    stack = stack_from(values)
    assert stack == stack_from("[ -1000000000000 true (false (3) nil) ]")
    assert stack[1] == machine.Bool(True)
    assert stack[1] != machine.Int(1)
    assert len(stack_from([0] * 16)) == 16
    assert len(stack_from([[1] * 64])[0].items) == 64
    assert stack_from("(" * 12 + "0" + ")" * 12)
    large = -(10**512 - 1)
    assert stack_from([large], expected=True) == stack_from(str(large), expected=True)


@pytest.mark.parametrize(
    "value",
    [
        True,
        5,
        None,
        {},
        [1.5],
        ["1"],
        [[None]],
        [machine.Int(1)],
        "[ true MARK ]",
        "[ (add) ]",
        "[ 2.0 ]",
        "((1)",
        ")",
        [10**12 + 1],
        [0] * 17,
        [[0] * 65],
        "(" * 13 + "0" + ")" * 13,
    ],
)
def test_stack_rejects_invalid_data(value):
    with pytest.raises(ValueError):
        stack_from(value)


@pytest.mark.parametrize("value", [[10**512], str(-(10**512))])
def test_stack_rejects_513_digit_expected_values(value):
    with pytest.raises(ValueError, match="512 digits"):
        stack_from(value, expected=True)


def test_snapshot_renders_machine_values_and_elides_middle_of_stack():
    rendered = snapshot(
        (machine.Lst((machine.Sym("add"), machine.TRUE, machine.NIL)), machine.MARK)
    )
    assert rendered == [
        {
            "type": "list",
            "text": "(add true nil)",
            "items": [
                {"type": "symbol", "text": "add"},
                {"type": "bool", "text": "true"},
                {"type": "list", "text": "nil", "items": []},
            ],
        },
        {"type": "mark", "text": "MARK"},
    ]
    values = tuple(map(machine.Int, range(51)))
    assert len(snapshot(values[:48])) == 48
    display = snapshot(values)
    assert len(display) == 49
    assert display[0]["text"] == "0"
    assert display[1]["type"] == "elision"
    assert [item["text"] for item in display[2:]] == list(map(str, range(4, 51)))


def test_execute_uses_canonical_step_and_records_quote_state(monkeypatch):
    calls = []
    original = machine.step_raw

    def observe(op, stack, budget, depth):
        calls.append((op.name, budget, depth))
        return original(op, stack, budget, depth)

    monkeypatch.setattr(machine, "step_raw", observe)
    examples = [{"input": [2], "output": [3]}]
    result = execute("[1 add]i", examples)
    assert result["matches"] == result["count"] == 1
    assert result["fuel"] == 7
    assert result["primitive_count"] == 5
    assert result["max_depth"] == 4
    frames = result["examples"][0]["frames"]
    assert frames[0] == {
        "stack": [{"type": "int", "text": "2"}],
        "fuel": 0,
        "op": None,
        "error": None,
        "quoting": False,
    }
    assert [frame["quoting"] for frame in frames] == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert frames[3]["stack"][-1] == {"type": "symbol", "text": "add"}
    assert frames[-1]["stack"] == result["examples"][0]["output"]
    assert [call for call in calls if call[2] == 0] == [
        ("[", FUEL, 0),
        ("1", FUEL - 1, 0),
        ("add", FUEL - 2, 0),
        ("]", FUEL - 3, 0),
        ("i", FUEL - 4, 0),
    ]
    assert execute("[1 add]i", examples, capture=False)["examples"][0]["frames"] == [
        frames[0],
        frames[-1],
    ]


def test_execute_continues_other_examples_after_a_refusal():
    result = execute(
        "1 add dup",
        [
            {"input": [False], "output": [False, 1]},
            {"input": [2], "output": [3, 3]},
        ],
    )
    bad, good = result["examples"]
    assert result["matches"] == 1
    assert result["fuel"] == bad["fuel"] + good["fuel"] == 4
    assert bad["error"].startswith("add:")
    assert not bad["matched"]
    assert good["matched"]
    assert len(bad["frames"]) == 3
    assert bad["frames"][-1]["stack"] == bad["frames"][-2]["stack"]
    assert bad["frames"][-1]["fuel"] == bad["frames"][-2]["fuel"] == 1
    assert bad["frames"][-1]["error"] == bad["error"]
    assert execute("", [{"input": [True], "output": [1]}])["matches"] == 0


def test_execute_fuel_refusal_and_empty_program(monkeypatch):
    monkeypatch.setattr(engine, "FUEL", 3)
    example = [{"input": [], "output": [2]}]
    result = execute("1 1 add dup", example, capture=False)["examples"][0]
    assert result["fuel"] == 3
    assert result["error"] == "dup: out of fuel"
    assert result["frames"][-1]["stack"] == result["output"]
    assert not result["matched"]
    assert len(result["frames"]) == 2
    assert len(execute("", example, capture=False)["examples"][0]["frames"]) == 1


@pytest.mark.parametrize("examples", [None, [], [{}], [{"input": []}] * 17])
def test_execute_requires_complete_examples(examples):
    with pytest.raises(ValueError):
        execute("", examples)


def test_inspect_observes_real_nested_calls_in_return_order_even_when_warm():
    program = "[[dup]i add]map"
    examples = [{"input": [[3, -1]], "output": [[6, -2]]}]
    run = execute(program, examples)
    assert run["matches"] == 1
    step = run["primitive_count"] - 1
    previous = sys.getprofile()
    result = inspect_op(program, examples, step)
    assert sys.getprofile() is previous
    assert result["error"] is None
    assert not result["truncated"]
    assert [(frame["op"], frame["depth"]) for frame in result["frames"]] == [
        ("dup", 4),
        ("i", 2),
        ("add", 2),
        ("dup", 4),
        ("i", 2),
        ("add", 2),
    ]
    assert result["frames"][0] == {
        "op": "dup",
        "depth": 4,
        "before": snapshot(stack_from([3])),
        "after": snapshot(stack_from([3, 3])),
        "error": None,
    }
    assert inspect_op(program, examples, step) == result
    assert inspect_op(program, examples, 0) == {
        "frames": [],
        "truncated": False,
        "error": None,
    }


def test_inspect_reports_refusals_and_caps_trace():
    example = [{"input": [[1]], "output": [[2]]}]
    refused = inspect_op("[ add ] map", example, 3)
    assert refused["error"].startswith("map: add:")
    assert refused["frames"][0]["error"].startswith("needs 2")
    assert refused["frames"][0]["before"] == refused["frames"][0]["after"]
    prefix = inspect_op("add 1", example, 1)
    assert prefix["frames"] == []
    assert prefix["error"].startswith("add:")
    with pytest.raises(ValueError, match="index"):
        inspect_op("add", example, -1)
    with pytest.raises(ValueError, match="index"):
        inspect_op("add", example, 0, example=1)

    many_items = [{"input": [list(range(64))], "output": [list(range(64))]}]
    trace = inspect_op("[ dup drop dup drop dup drop ] map", many_items, 8)
    assert trace["error"] is None
    assert trace["truncated"]
    assert len(trace["frames"]) == 350


def test_curriculum_has_14_distinct_lessons():
    assert len(LESSONS) == len(BY_ID) == 14
    assert BY_ID["first-light"]["solution"] == "1 add"
    assert BY_ID["echo"]["solution"] == "dup add"


@pytest.mark.parametrize("lesson", LESSONS, ids=[lesson["id"] for lesson in LESSONS])
def test_every_lesson_solution_passes_all_six_examples(lesson):
    assert len(lesson["examples"]) == len(lesson["heldout"]) == 3
    assert all(macro in MACROS for macro in lesson["macros"])
    result = api("/api/verify", {"lesson": lesson["id"], "program": lesson["solution"]})
    assert result == {
        "passed": True,
        "shown": 3,
        "shown_count": 3,
        "extra": 3,
        "extra_count": 3,
        "failures": [],
        "evidence": "shown and extra tests",
    }


def test_api_context_and_authored_hints(monkeypatch, tmp_path):
    context = api("/api/context", {})
    for lesson in context["lessons"]:
        assert not {"solution", "heldout", "hints"} & lesson.keys()
        assert len(lesson["examples"]) == lesson["extra_tests"] == 3
        assert lesson["hint_count"] == len(BY_ID[lesson["id"]]["hints"])
    assert [row["name"] for row in context["ops"]] == list(OPS)
    assert {row["name"] for row in context["macros"]} == MACROS.keys()
    inc = next(row for row in context["macros"] if row["name"] == "inc")
    assert inc["expansion"] == ["1", "add"]
    assert context["world"]["tasks"]
    assert "solution" not in context["world"]["tasks"][0]
    monkeypatch.setattr(server, "_WORLD_MANIFEST", tmp_path / "missing.json")
    assert api("/api/context", {})["world"]["tasks"] == []

    lesson = BY_ID["first-light"]
    data = {"lesson": "first-light", "index": 1, "examples": []}
    assert api("/api/hint", data) == {"hint": lesson["hints"][1]}
    assert api("/api/reveal", data) == {"solution": lesson["solution"]}
    with pytest.raises(ValueError, match="index"):
        api("/api/hint", {**data, "index": 99})


def test_api_distinguishes_a_fit_from_verified_behaviour():
    data = {"lesson": "false-fit", "program": "drop"}
    assert api("/api/evaluate", data)["matches"] == 3
    result = api("/api/verify", data)
    assert not result["passed"]
    assert result["shown"] == result["shown_count"] == 3
    assert result["extra"] == 1
    assert result["extra_count"] == 3
    assert len(result["failures"]) == 2
    assert result["failures"][0]["frames"][0]["stack"] == snapshot(stack_from([6, 2]))


def test_api_sandbox_and_adviser_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ADVISER", Adviser())
    data = {
        "lesson": "sandbox",
        "program": "[ 1 add ] i",
        "step": 4,
        "examples": [{"input": [7], "output": [8]}],
    }
    assert api("/api/evaluate", data)["matches"] == 1
    assert api("/api/verify", data) == {
        "passed": True,
        "shown": 1,
        "shown_count": 1,
        "extra": 0,
        "extra_count": 0,
        "failures": [],
        "evidence": "examples",
    }
    assert [frame["op"] for frame in api("/api/inspect", data)["frames"]] == ["add"]
    assert api("/api/suggest", data)["suggestions"] == []
    with pytest.raises(ValueError, match=r"^add:"):
        api("/api/suggest", {**data, "program": "add"})
    assert api("/api/model", {})["ready"] is False
    status = api("/api/model", {"checkpoint": str(tmp_path / "missing")})
    assert not status["ready"]
    assert status["error"]
    assert api("/api/model", {}) == status


def test_adviser_ranks_legal_product_ops_from_a_local_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from wandering_light.stack_policy import (
        OP_INDEX,
        PolicyArchitecture,
        PolicyScorer,
        StackPairPolicy,
        save_policy,
    )

    policy = StackPairPolicy(PolicyArchitecture(8, 16, 32))
    run_url = "https://wandb.ai/abhishekraok-na/workshop/runs/test"
    path = save_policy(
        tmp_path / "policy", policy, trainer_state={}, manifest={}, wandb_url=run_url
    )

    def logits(_scorer, blocks):
        import numpy as np

        assert blocks[0].shape[0] == 2
        row = np.zeros((1, len(OPS)))
        row[0, OP_INDEX["div"]] = 12
        row[0, OP_INDEX["add"]] = 6
        row[0, OP_INDEX["sub"]] = 5
        return row

    monkeypatch.setattr(PolicyScorer, "logits", logits)
    adviser = Adviser()
    with torch.inference_mode():
        status = adviser.load(path)
    assert status["ready"] and status["error"] is None
    assert status["source"]["wandb_run_url"] == run_url
    examples = [{"input": [8, 2], "output": [10]}, {"input": [5, 0], "output": [5]}]
    suggestions = adviser.rank("", examples)["suggestions"]
    assert len(suggestions) == 6
    assert [row["op"] for row in suggestions[:2]] == ["add", "sub"]
    assert all(row["op"] != "div" for row in suggestions)
    assert suggestions[0]["previews"] == [
        snapshot(stack_from([10])),
        snapshot(stack_from([5])),
    ]
    assert 0 < suggestions[1]["probability"] < suggestions[0]["probability"] < 1


def test_http_handler_origin_validation_and_json_errors():
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection(*httpd.server_address, timeout=5)

            def post(body, origin=None):
                headers = {"Content-Type": "application/json"}
                if origin is not None:
                    headers["Origin"] = origin
                connection.request("POST", "/api/evaluate", body, headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read())

            body = json.dumps({"lesson": "first-light", "program": "1 add"})
            for origin in (None, "http://localhost:8018", "http://127.0.0.1:8018"):
                status, result = post(body, origin)
                assert status == 200
                assert result["matches"] == 3
            for origin in (
                "https://example.com",
                "http://localhost.example.com",
                "null",
            ):
                status, result = post(body, origin)
                assert status == 403
                assert "error" in result
            for bad in ("[", "[]", '{"lesson": "missing"}'):
                status, result = post(bad)
                assert status == 400
                assert set(result) == {"error"}
            connection.close()
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
