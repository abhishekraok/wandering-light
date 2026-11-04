"""Signal House: the workshop observes the canonical `mark-28` machine."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from wandering_light.stack_machine import MARK_OPS, Int, Lst, run_body, value_key
from wandering_light.stack_pbe import run_program, to_stack
from wandering_light.stack_workshop import curriculum, engine, server
from wandering_light.stack_workshop.model import ADVISER, Adviser

# ---------------------------------------------------------------------------
# engine: macros and compiling
# ---------------------------------------------------------------------------


def test_ops_are_the_mark_28_basis():
    assert set(engine.OPS) == {op.name for op in MARK_OPS}
    assert len(engine.OPS) == 28
    assert engine.FUEL == 100_000


def test_macros_come_from_the_library_with_groups():
    assert engine.MACROS["inc"] == {"body": "1 add", "group": "scalars"}
    assert engine.MACROS["2"]["group"] == "constants"
    assert engine.MACROS["sum"]["group"] == "lists"
    assert engine.MACROS["range"]["group"] == "ranges"
    assert len(engine.MACROS) >= 100


def test_compile_keeps_spans_and_outermost_macro():
    tokens = engine.compile_program("inc [dup]  # a comment\n add")
    assert [t["op"] for t in tokens] == ["1", "add", "[", "dup", "]", "add"]
    assert tokens[0] == {"op": "1", "start": 0, "end": 3, "macro": "inc"}
    assert tokens[1]["macro"] == "inc"
    assert tokens[2] == {"op": "[", "start": 4, "end": 5, "macro": None}
    assert tokens[3] == {"op": "dup", "start": 5, "end": 8, "macro": None}
    assert tokens[5] == {"op": "add", "start": 24, "end": 27, "macro": None}


def test_compile_expands_nested_macros_to_the_outer_name():
    tokens = engine.compile_program("sum")
    assert [t["op"] for t in tokens] == ["0", "[", "add", "]", "fold"]
    assert {t["macro"] for t in tokens} == {"sum"}
    # `sq` is used inside `10`'s cousins; `9` is `1 1 add 1 add dup mul`.
    assert {t["macro"] for t in engine.compile_program("9")} == {"9"}


def test_compile_matches_the_library_verifier():
    for name in ("gcd", "range", "digits", "band"):
        expanded = engine.expansion(name).split()
        assert all(op in engine.OPS for op in expanded)
    assert engine.expansion("10") == "1 1 add dup add 1 add dup add"


def test_compile_errors():
    with pytest.raises(ValueError, match="Unknown word 'foo'"):
        engine.compile_program("1 foo")
    with pytest.raises(ValueError, match="Unknown word"):
        engine.compile_program("(")


def test_recursive_macro_is_refused(monkeypatch):
    monkeypatch.setitem(engine.MACROS, "loop", {"body": "1 loop", "group": "test"})
    with pytest.raises(ValueError, match="Recursive macro: loop"):
        engine.compile_program("loop")
    monkeypatch.setitem(engine.MACROS, "ping", {"body": "pong", "group": "test"})
    monkeypatch.setitem(engine.MACROS, "pong", {"body": "ping", "group": "test"})
    with pytest.raises(ValueError, match="Recursive macro: ping"):
        engine.compile_program("ping")


# ---------------------------------------------------------------------------
# engine: values typed by a learner
# ---------------------------------------------------------------------------


def test_stack_from_json_and_text_agree():
    from_json = engine.stack_from([1, [2, True], []])
    from_text = engine.stack_from("[ 1 ( 2 true ) nil ]")
    assert [value_key(v) for v in from_json] == [value_key(v) for v in from_text]
    assert engine.stack_from("[1, [2, 3]]") == engine.stack_from("1 ( 2 3 )")
    assert engine.stack_from("") == ()
    assert engine.stack_from([]) == ()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([1.5], "not an integer"),
        ("[1.5]", "not an integer"),
        (["a"], "string"),
        ("add", "symbol"),
        ("( 1 add )", "symbol"),
        ("MARK", "MARK"),
        ("( 1", "unterminated"),
        ([10**12 + 1], "beyond"),
        (str(10**13), "beyond"),
        (list(range(17)), "at most 16"),
        ([list(range(65))], "longer than 64"),
        (json.loads("[" * 15 + "]" * 15), "nested deeper"),
        (3, "JSON array"),
        (None, "JSON array"),
    ],
)
def test_stack_from_rejects(value, message):
    with pytest.raises(ValueError, match=message):
        engine.stack_from(value)


def test_expected_outputs_allow_big_integers_but_not_huge_ones():
    big = 10**100
    assert engine.stack_from([big], expected=True) == (Int(big),)
    with pytest.raises(ValueError, match="512 digits"):
        engine.stack_from([10**512], expected=True)
    with pytest.raises(ValueError, match="beyond"):
        engine.stack_from([big])


# ---------------------------------------------------------------------------
# engine: display
# ---------------------------------------------------------------------------


def test_snapshot_types():
    shot = engine.snapshot(engine.stack_from("1 true ( 2 nil )"))
    assert shot[0] == {"type": "int", "text": "1"}
    assert shot[1] == {"type": "bool", "text": "true"}
    assert shot[2]["type"] == "list" and shot[2]["text"] == "(2 nil)"
    assert shot[2]["items"][1] == {"type": "list", "text": "nil", "items": []}
    opened, _ = run_program(["[", "add"], ())
    shot = engine.snapshot(opened)
    assert [v["type"] for v in shot] == ["mark", "symbol"]
    assert shot[1]["text"] == "add"


def test_snapshot_elides_a_deep_stack():
    stack = tuple(Int(n) for n in range(60))
    shot = engine.snapshot(stack)
    assert len(shot) == 49
    assert shot[0]["text"] == "0"
    assert shot[1]["type"] == "elision" and "12" in shot[1]["text"]
    assert [v["text"] for v in shot[2:]] == [str(n) for n in range(13, 60)]
    assert len(engine.snapshot(stack[:48])) == 48


# ---------------------------------------------------------------------------
# engine: execute
# ---------------------------------------------------------------------------


def test_execute_traces_every_token_against_the_machine():
    result = engine.execute("[ 1 add ] map", [{"input": [[1, 2]], "output": [[2, 3]]}])
    assert result["matches"] == 1
    assert result["count"] == 5 and result["primitive_count"] == 5
    row = result["examples"][0]
    frames = row["frames"]
    assert [f["op"] for f in frames] == [None, "[", "1", "add", "]", "map"]
    assert frames[0]["fuel"] == 0 and frames[0]["quoting"] is False
    assert frames[1]["quoting"] is True and frames[4]["quoting"] is False
    # `map` costs 1 plus, per item, the push of `1` and the `add`.
    assert [f["fuel"] for f in frames] == [0, 1, 2, 3, 4, 9]
    expected, _ = run_program(["[", "1", "add", "]", "map"], to_stack([[1, 2]]))
    assert frames[-1]["stack"] == engine.snapshot(expected)
    assert row["matched"] and row["error"] is None
    assert row["fuel"] == 9 == result["fuel"]
    assert row["max_depth"] == 4 == result["max_depth"]
    assert row["output"] == engine.snapshot(to_stack([[2, 3]]))


def test_execute_macro_count_is_written_words():
    result = engine.execute("10 add", [{"input": [1], "output": [11]}])
    assert result["count"] == 2
    assert result["primitive_count"] == 10
    assert result["matches"] == 1


def test_execute_refusal_halts_only_that_example():
    result = engine.execute(
        "add 1",
        [{"input": [1], "output": [1]}, {"input": [1, 2], "output": [3, 1]}],
    )
    first, second = result["examples"]
    assert first["error"] == "add: needs 2 value(s), stack has 1"
    assert [f["op"] for f in first["frames"]] == [None, "add"]
    assert first["frames"][-1]["stack"] == first["frames"][0]["stack"]
    assert first["frames"][-1]["error"] == first["error"]
    assert first["matched"] is False and first["fuel"] == 0
    assert second["matched"] and second["error"] is None
    assert result["matches"] == 1


def test_execute_without_capture_keeps_first_and_final_frames():
    result = engine.execute(
        "1 add 1 add", [{"input": [1], "output": [3]}], capture=False
    )
    frames = result["examples"][0]["frames"]
    assert len(frames) == 2
    assert frames[0]["op"] is None and frames[1]["op"] == "add"
    assert frames[1]["fuel"] == 4


def test_execute_takes_one_to_sixteen_examples():
    with pytest.raises(ValueError, match="between 1 and 16"):
        engine.execute("1", [])
    with pytest.raises(ValueError, match="between 1 and 16"):
        engine.execute("1", [{"input": [], "output": [1]}] * 17)
    with pytest.raises(ValueError, match="needs an input and an output"):
        engine.execute("1", [{"input": []}])


def test_execute_runs_out_of_fuel_through_the_machine():
    # `range` is bounded by the fuel budget; the refusal is the machine's.
    result = engine.execute("1000 1000 mul range", [{"input": [], "output": [[]]}])
    row = result["examples"][0]
    assert row["error"] is not None and "out of fuel" in row["error"]


# ---------------------------------------------------------------------------
# engine: inspect_op
# ---------------------------------------------------------------------------


def test_inspect_records_nested_steps_in_return_order():
    examples = [{"input": [[1, 2]], "output": [[2, 3]]}]
    view = engine.inspect_op("[ 1 add ] map", examples, 4)
    assert view["error"] is None and view["truncated"] is False
    # `1` is a literal, pushed by `run_body` itself; only `add` is a step.
    assert [(f["op"], f["depth"]) for f in view["frames"]] == [("add", 2), ("add", 2)]
    assert view["frames"][0]["before"] == engine.snapshot((Int(1), Int(1)))
    assert view["frames"][0]["after"] == engine.snapshot((Int(2),))
    assert view["frames"][1]["after"] == engine.snapshot((Int(3),))
    # A primitive makes no nested calls.
    assert engine.inspect_op("[ 1 add ] map", examples, 0)["frames"] == []


def test_inspect_warm_cache_does_not_hide_nested_execution():
    examples = [{"input": [[1, 2, 3]], "output": [6]}]
    engine.execute("0 [ add ] fold", examples)
    view = engine.inspect_op("0 [ add ] fold", examples, 4)
    assert [f["op"] for f in view["frames"]] == ["add", "add", "add"]
    assert view["frames"][-1]["after"] == engine.snapshot((Int(6),))
    # And the same again, with the cache warm from the inspection itself.
    again = engine.inspect_op("0 [ add ] fold", examples, 4)
    assert again["frames"] == view["frames"]
    assert run_body.cache_info().currsize > 0


def test_inspect_reports_a_nested_refusal():
    view = engine.inspect_op("[ add ] map", [{"input": [[1]], "output": [[]]}], 3)
    assert view["error"] == "map: add: needs 2 value(s), stack has 1"
    assert view["frames"] == [
        {
            "op": "add",
            "depth": 2,
            "before": engine.snapshot((Int(1),)),
            "after": None,
            "error": "needs 2 value(s), stack has 1",
        }
    ]


def test_inspect_truncates_a_long_combinator():
    program = "1000 range sum"
    step = len(engine.compile_program(program)) - 1
    view = engine.inspect_op(program, [{"input": [], "output": [499500]}], step)
    assert view["truncated"] is True
    assert len(view["frames"]) == 350
    assert view["error"] is None


def test_inspect_bounds():
    examples = [{"input": [1], "output": [2]}]
    with pytest.raises(ValueError, match="No step 5"):
        engine.inspect_op("1 add", examples, 5)
    with pytest.raises(ValueError, match="No example 1"):
        engine.inspect_op("1 add", examples, 0, example=1)
    with pytest.raises(ValueError, match="never reached"):
        engine.inspect_op("add add", examples, 1)


def test_machine_operations_share_one_lock():
    seen = []

    def worker(n):
        result = engine.execute("dup add", [{"input": [n], "output": [2 * n]}])
        seen.append(result["matches"])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == [1] * 8
    assert isinstance(engine.MACHINE_LOCK, type(threading.RLock()))


# ---------------------------------------------------------------------------
# curriculum
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


def test_fourteen_lessons_with_every_field():
    assert len(curriculum.LESSONS) == 14
    assert list(curriculum.BY_ID) == [lesson["id"] for lesson in curriculum.LESSONS]
    for lesson in curriculum.LESSONS:
        assert set(lesson) == FIELDS, lesson["id"]
        assert len(lesson["examples"]) == 3 and len(lesson["heldout"]) == 3
        assert lesson["hints"]
        assert set(lesson["prediction"]) == {"input", "prompt", "answer"}
        for name in lesson["macros"]:
            assert name in engine.MACROS


def test_first_lessons():
    first = curriculum.BY_ID["first-light"]
    assert first["solution"] == "1 add"
    assert first["examples"][0] == {"input": [3], "output": [4]}
    assert curriculum.BY_ID["echo"]["solution"] == "dup add"


@pytest.mark.parametrize("lesson", curriculum.LESSONS, ids=lambda l: l["id"])
def test_every_solution_passes_all_six_examples(lesson):
    result = engine.execute(lesson["solution"], lesson["examples"] + lesson["heldout"])
    assert result["matches"] == 6, [row["error"] for row in result["examples"]]
    prediction = lesson["prediction"]
    guess = engine.execute(
        lesson["solution"],
        [{"input": prediction["input"], "output": prediction["answer"]}],
    )
    assert guess["matches"] == 1


def test_the_fit_lesson_separates_a_fit_from_an_explanation():
    lesson = curriculum.BY_ID["the-fit"]
    shown = engine.execute(lesson["starter"], lesson["examples"])
    assert shown["matches"] == 3
    extra = engine.execute(lesson["starter"], lesson["heldout"])
    assert extra["matches"] < 3


def test_expected_outputs_are_independent_of_the_solution():
    # The reference for `tally` is Python's sum, not the fold.
    assert curriculum.BY_ID["tally"]["examples"][1] == {"input": [[]], "output": [0]}
    assert curriculum.BY_ID["fork"]["heldout"][0] == {"input": [-10], "output": [10]}


def test_lessons_cover_the_announced_concepts():
    solutions = " ".join(lesson["solution"] for lesson in curriculum.LESSONS)
    for word in ("swap", "over", "i", "map", "filter", "fold", "dip", "ifte"):
        assert f" {word} " in f" {solutions} ", word
    assert any(lesson["macros"] for lesson in curriculum.LESSONS)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_adviser_status_and_rank_without_a_policy():
    adviser = Adviser()
    assert adviser.status()["ready"] is False
    ranked = adviser.rank("1", [{"input": [1], "output": [2]}])
    assert ranked["ready"] is False and ranked["suggestions"] == []
    with pytest.raises(ValueError, match="add: needs 2"):
        adviser.rank("add", [{"input": [1], "output": [2]}])


def test_adviser_load_records_an_error(tmp_path):
    adviser = Adviser()
    status = adviser.load(tmp_path)
    assert status["ready"] is False
    assert (
        status["error"].startswith("FileNotFoundError")
        and "checkpoint" in status["error"]
    )
    status = adviser.load(None, download=False)
    assert status["ready"] is False and status["error"]
    assert ADVISER.status()["ready"] is False


def test_adviser_ranks_legal_ops_with_a_fake_policy():
    pytest.importorskip("torch")
    import numpy as np

    from wandering_light.stack_policy import OP_NAMES

    class Scorer:
        def log_probs(self, blocks, legal):
            assert len(blocks) == 1 and blocks[0].shape[0] == 2
            logits = np.arange(len(OP_NAMES), dtype=float)[None, :]
            logits = np.where(legal, logits, -np.inf)
            return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    adviser = Adviser()
    adviser.scorer = Scorer()
    examples = [{"input": [1, 2], "output": [3]}, {"input": [[1]], "output": [1]}]
    ranked = adviser.rank("", examples)
    names = [s["op"] for s in ranked["suggestions"]]
    assert len(names) <= 6
    # `add` is legal on the first example only, so it is never suggested.
    assert "add" not in names and "dup" in names
    probabilities = [s["probability"] for s in ranked["suggestions"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(len(s["previews"]) == 2 for s in ranked["suggestions"])


# ---------------------------------------------------------------------------
# server: api
# ---------------------------------------------------------------------------


def test_context_hides_answers_and_lists_ops_and_macros():
    context = server.api("/api/context")
    assert len(context["lessons"]) == 14
    for lesson in context["lessons"]:
        assert not {"solution", "heldout", "hints"} & set(lesson)
        assert lesson["extra_tests"] == 3 and lesson["hint_count"] >= 1
    assert [op["name"] for op in context["ops"]] == [op.name for op in MARK_OPS]
    assert len(context["macros"]) == len(engine.MACROS)
    inc = next(m for m in context["macros"] if m["name"] == "inc")
    assert inc == {
        "name": "inc",
        "group": "scalars",
        "body": "1 add",
        "expansion": "1 add",
    }
    assert context["model"]["ready"] is False
    assert "available" in context["world"]


def test_world_shelf_never_fails(tmp_path):
    assert server.world_shelf(tmp_path / "missing.json")["available"] is False
    (tmp_path / "manifest.json").write_text("not json")
    shelf = server.world_shelf(tmp_path / "manifest.json")
    assert shelf["available"] is False and shelf["reason"]


def test_evaluate_and_verify_a_lesson():
    evaluated = server.api("/api/evaluate", {"lesson": "echo", "program": "dup add"})
    assert evaluated["matches"] == 3
    verified = server.api("/api/verify", {"lesson": "echo", "program": "dup add"})
    assert verified["passed"] is True
    assert verified["shown_count"] == 3 and verified["extra_count"] == 3
    assert verified["evidence"] == "shown and extra tests"
    assert verified["failures"] == []
    assert len(verified["extra"]["examples"][0]["frames"]) == 2


def test_verify_reports_the_first_three_failures():
    verified = server.api("/api/verify", {"lesson": "the-fit", "program": "swap drop"})
    assert verified["passed"] is False
    assert verified["shown"]["matches"] == 3
    assert [f["source"] for f in verified["failures"]] == ["extra", "extra"]
    assert verified["failures"][0]["input"] == [7, 2]
    assert verified["failures"][0]["got"] == [{"type": "int", "text": "2"}]
    verified = server.api("/api/verify", {"lesson": "echo", "program": "add"})
    assert len(verified["failures"]) == 3
    assert verified["failures"][0]["error"].startswith("add:")


def test_sandbox_uses_the_learners_examples_only():
    data = {
        "lesson": "sandbox",
        "program": "1 add",
        "examples": [{"input": [1], "output": [2]}],
    }
    verified = server.api("/api/verify", data)
    assert verified["passed"] is True
    assert verified["extra"] is None and verified["extra_count"] == 0
    assert verified["evidence"] == "examples"
    with pytest.raises(ValueError, match="sandbox"):
        server.api("/api/verify", {"lesson": "sandbox", "program": "1"})
    with pytest.raises(ValueError, match="no hints"):
        server.api("/api/hint", {"lesson": "sandbox", "examples": data["examples"]})
    with pytest.raises(ValueError, match="nothing to reveal"):
        server.api("/api/reveal", {"lesson": "sandbox", "examples": data["examples"]})


def test_inspect_hint_reveal_suggest_model_routes():
    view = server.api(
        "/api/inspect", {"lesson": "each-one", "program": "[ 1 add ] map", "step": 4}
    )
    assert [f["op"] for f in view["frames"]] == ["add", "add", "add"]
    hint = server.api("/api/hint", {"lesson": "first-light"})
    assert hint["index"] == 0 and hint["remaining"] == 1
    assert (
        server.api("/api/hint", {"lesson": "first-light", "hint": 1})["remaining"] == 0
    )
    with pytest.raises(ValueError, match="hints"):
        server.api("/api/hint", {"lesson": "first-light", "hint": 2})
    assert server.api("/api/reveal", {"lesson": "echo"})["solution"] == "dup add"
    suggested = server.api("/api/suggest", {"lesson": "echo", "program": "dup"})
    assert suggested["suggestions"] == [] and suggested["ready"] is False
    assert server.api("/api/model", {})["ready"] is False
    with pytest.raises(ValueError, match="Unknown lesson"):
        server.api("/api/evaluate", {"lesson": "nope", "program": "1"})
    with pytest.raises(ValueError, match="Unknown route"):
        server.api("/api/nothing", {})


# ---------------------------------------------------------------------------
# server: HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def workshop():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(base: str, route: str, data: dict, **headers: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + route,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_posts_json_to_api(workshop):
    status, payload = _post(
        workshop, "/api/evaluate", {"lesson": "echo", "program": "dup add"}
    )
    assert status == 200 and payload["matches"] == 3
    status, payload = _post(workshop, "/api/context", {})
    assert status == 200 and len(payload["lessons"]) == 14


def test_http_errors_are_400_with_a_message(workshop):
    status, payload = _post(
        workshop, "/api/evaluate", {"lesson": "echo", "program": "zap"}
    )
    assert status == 400 and payload["error"].startswith("Unknown word 'zap'")
    status, payload = _post(workshop, "/api/unknown", {})
    assert status == 400 and "Unknown route" in payload["error"]


def test_http_rejects_foreign_origins(workshop):
    data = {"lesson": "echo", "program": "dup add"}
    for origin in ("http://127.0.0.1:5173", "http://localhost:8018"):
        status, _ = _post(workshop, "/api/evaluate", data, Origin=origin)
        assert status == 200, origin
    status, payload = _post(
        workshop, "/api/evaluate", data, Origin="https://example.com"
    )
    assert status == 403 and "Origin" in payload["error"]


def test_http_handles_concurrent_requests(workshop):
    results = []

    def worker(n):
        results.append(
            _post(workshop, "/api/evaluate", {"lesson": "echo", "program": "dup add"})[
                0
            ]
        )

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [200] * 6


def test_main_defaults(monkeypatch):
    seen = {}

    class Server:
        def __init__(self, address, handler):
            seen["address"] = address
            seen["handler"] = handler
            self.server_address = address

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "ThreadingHTTPServer", Server)
    assert server.main([]) == 0
    assert seen == {
        "address": ("127.0.0.1", 8018),
        "handler": server.Handler,
        "closed": True,
    }
    server.main(["--port", "9000"])
    assert seen["address"] == ("127.0.0.1", 9000)


def test_engine_never_reimplements_the_machine():
    """The workshop's only way to change a stack is the machine's `step_raw`."""
    source = Path(engine.__file__).read_text()
    assert "step_raw(" in source
    for forbidden in ("def _apply", "def _guard", "items[0]", "stack[:-2]"):
        assert forbidden not in source


def test_fit_example_type_round_trip():
    # A nested list survives JSON, the machine, and the display form.
    stack = engine.stack_from([[1, [2]], True])
    assert isinstance(stack[0], Lst)
    assert engine.snapshot(stack)[0]["items"][1]["items"][0]["text"] == "2"
