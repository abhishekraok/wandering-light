"""The Signal House API observes the same transitions a learner can replay."""

import json
import subprocess
import sys
import threading
from http.client import HTTPConnection

import pytest

from wandering_light import stack_machine as m
from wandering_light.stack_workshop import engine, server
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.model import Adviser


def case(initial, target):
    return {"input": initial, "output": target}


def test_compile_keeps_written_spans_through_nested_macros_and_comments():
    text = "  [inc] # [unknown]\r16 dup"
    tokens = engine.compile_program(text)
    assert [(t["op"], t["macro"], t["start"], t["end"]) for t in tokens[:4]] == [
        ("[", None, 2, 3),
        ("1", "inc", 3, 6),
        ("add", "inc", 3, 6),
        ("]", None, 6, 7),
    ]
    sixteen = tokens[4:-1]
    assert [t["op"] for t in sixteen] == [
        "1",
        "1",
        "add",
        "dup",
        "add",
        "dup",
        "add",
        "dup",
        "add",
    ]
    assert all(text[t["start"] : t["end"]] == t["macro"] == "16" for t in sixteen)
    assert tokens[-1]["op"] == "dup"
    assert engine.MACROS["rot"]["group"] == "stack shuffles"
    assert {op.name: op for op in m.MARK_OPS} == engine.OPS


def test_compile_errors(monkeypatch):
    with pytest.raises(ValueError, match=r"Unknown word 'blip'\."):
        engine.compile_program("1 blip")
    for name, body in [("once", "twice"), ("twice", "once")]:
        monkeypatch.setitem(engine.MACROS, name, {"body": body, "group": "test"})
    with pytest.raises(ValueError, match=r"^Recursive macro: once$"):
        engine.compile_program("once")


@pytest.mark.parametrize(
    "value",
    [
        1,
        True,
        {},
        ["1"],
        [3.0],
        [[None]],
        [(1,)],
        [m.Int(1)],
        "[(1 MARK)]",
        "[(add)]",
        "[(1]",
        "0.5",
        [10**12 + 1],
        str(-(10**12) - 1),
        [1] * 17,
        [[1] * 65],
    ],
)
def test_stack_rejects_invalid_examples(value):
    with pytest.raises(ValueError):
        engine.stack_from(value)


def test_stack_limits_and_types_apply_to_both_input_forms():
    value = [True, -(10**12), [1, False, []]]
    assert engine.stack_from(value) == engine.stack_from(
        "[true -1000000000000 (1 false nil)]"
    )
    assert engine.stack_from(value)[0] == m.TRUE
    nested = "nil"
    for _ in range(11):
        nested = f"({nested})"
    assert engine.stack_from(nested)
    with pytest.raises(ValueError):
        engine.stack_from(f"({nested})")
    assert engine.stack_from([10**512 - 1], expected=True) == (m.Int(10**512 - 1),)
    with pytest.raises(ValueError):
        engine.stack_from(str(-(10**512)), expected=True)


def test_snapshot_has_typed_children_and_keeps_both_ends_of_deep_stack():
    stack = (m.Lst((m.TRUE, m.Sym("add"), m.MARK)), *(m.Int(i) for i in range(49)))
    display = engine.snapshot(stack)
    assert display[0] == {
        "type": "list",
        "text": "(true add MARK)",
        "items": [
            {"type": "bool", "text": "true"},
            {"type": "symbol", "text": "add"},
            {"type": "mark", "text": "MARK"},
        ],
    }
    assert display[1]["type"] == "elision"
    assert [v["text"] for v in display[2:]] == [str(i) for i in range(2, 49)]


@pytest.mark.parametrize(
    "examples", [None, [], [case([], [])] * 17, [{}], [case([], "MARK")]]
)
def test_execute_checks_examples(examples):
    with pytest.raises(ValueError):
        engine.execute("", examples)


def test_execute_tracks_refusal_quoting_and_real_cost_per_example():
    program = "[dup mul] map 1"
    examples = [case([3], [3]), case([[2, 3]], [[4, 9], 1])]
    report = engine.execute(program, examples)
    assert report["count"] == 2 and report["matches"] == 1
    assert report["primitive_count"] == 6
    refused, ok = report["examples"]
    assert refused["error"] == "map: second value is not a list"
    assert refused["frames"][-1]["stack"] == refused["frames"][-2]["stack"]
    assert refused["frames"][-1]["error"] == refused["error"]
    assert refused["fuel"] == 4 and not refused["matched"]
    assert ok["matched"] and not ok["error"]
    assert [f["fuel"] for f in ok["frames"]] == [0, 1, 2, 3, 4, 9, 10]
    assert [f["quoting"] for f in ok["frames"]] == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert report["fuel"] == 14
    assert report["max_depth"] == ok["max_depth"] == 4
    short = engine.execute(program, examples, capture=False)
    for full, small in zip(report["examples"], short["examples"], strict=True):
        assert small == {**full, "frames": [full["frames"][0], full["frames"][-1]]}
    empty = engine.execute("", [case([True], [1])], capture=False)
    assert empty["matches"] == 0 and len(empty["examples"][0]["frames"]) == 1


def test_execution_and_inspection_use_remaining_fuel(monkeypatch):
    monkeypatch.setattr(engine, "FUEL", 4)
    examples = [case([5], [6])]
    row = engine.execute("[1 add] i", examples)["examples"][0]
    inspection = engine.inspect_op("[1 add] i", examples, 4)
    assert row["fuel"] == 4
    assert inspection["error"] == row["error"] == "i: out of fuel"
    assert inspection["frames"][0]["before"] == inspection["frames"][0]["after"]


def test_inspect_records_real_returns_even_after_warm_replay():
    program = "[[dup] i add] map"
    examples = [case([[4]], [[8]]), case([[1, 3]], [[2, 6]])]
    report = engine.execute(program, examples)
    profile = sys.getprofile()
    trace = engine.inspect_op(program, examples, len(report["tokens"]) - 1, example=1)
    assert sys.getprofile() is profile
    assert trace["error"] is None and not trace["truncated"]
    assert [f["op"] for f in trace["frames"]] == ["dup", "i", "add"] * 2 + ["map"]
    depths = [f["depth"] for f in trace["frames"]]
    assert depths[0] > depths[1] == depths[2] > depths[-1] == 0
    assert trace["frames"][-1]["after"] == engine.snapshot(engine.stack_from([[2, 6]]))
    refusal = engine.inspect_op("[drop drop] i", [case([1], [])], 4)
    assert refusal["error"].startswith("i: drop:")
    assert refusal["frames"][-2]["error"] and refusal["frames"][-2]["before"] == []


def test_inspection_limit_and_unreachable_step():
    program = "[dup drop dup drop dup drop] map"
    trace = engine.inspect_op(program, [case([list(range(64))], [])], 8)
    assert trace["truncated"] and len(trace["frames"]) == 350
    prior = engine.inspect_op("drop dup", [case([], [])], 1)
    assert prior["error"].startswith("drop:") and prior["frames"] == []
    with pytest.raises(ValueError):
        engine.inspect_op("dup", [case([], [])], -1)


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda lesson: lesson["id"])
def test_lesson_solution_on_independent_examples(lesson):
    assert len(lesson["examples"]) == len(lesson["heldout"]) == 3
    result = server.api(
        "/api/verify", {"lesson": lesson["id"], "program": lesson["solution"]}
    )
    assert result == {
        "passed": True,
        "shown": 3,
        "shown_count": 3,
        "extra": 3,
        "extra_count": 3,
        "evidence": "shown and extra tests",
        "failures": [],
    }


def test_context_discovery_routes_and_offline_shelf(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORLD_DIR", tmp_path)
    context = server.api("/api/context", {})
    assert len(context["lessons"]) == len(BY_ID) == 14
    for lesson in context["lessons"]:
        assert {"heldout", "solution", "hints"}.isdisjoint(lesson)
        assert lesson["extra_tests"] == 3 and lesson["hint_count"] > 0
    assert len(context["ops"]) == 28
    macros = {m["name"]: m["expansion"] for m in context["macros"]}
    assert macros.keys() == engine.MACROS.keys()
    assert macros["inc"] == ["1", "add"]
    assert context["world"]["tasks"] == []
    data = {"lesson": "echo", "index": 1}
    assert server.api("/api/hint", data)["hint"] == BY_ID["echo"]["hints"][1]
    assert server.api("/api/reveal", data)["solution"] == "dup add"


def test_verify_distinguishes_shown_fit_from_extra_evidence():
    data = {"lesson": "day-shift", "program": ""}
    assert server.api("/api/evaluate", data)["matches"] == 3
    result = server.api("/api/verify", data)
    assert result["shown"] == 3 and result["extra"] == 1 and not result["passed"]
    assert len(result["failures"]) == 2
    data = {"lesson": "sandbox", "examples": [case([i], [i]) for i in range(4)]}
    result = server.api("/api/verify", data)
    assert (
        result["passed"]
        and result["extra_count"] == 0
        and result["evidence"] == "examples"
    )
    assert len(server.api("/api/verify", {**data, "program": "drop"})["failures"]) == 3
    trace = server.api(
        "/api/inspect", {**data, "program": "dup", "step": 0, "example": 2}
    )
    assert trace["frames"][0]["before"] == [{"type": "int", "text": "2"}]


def test_core_and_unloaded_suggestions_work_without_site_packages():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
from wandering_light.stack_workshop.server import api
assert len(api("/api/context", {})["lessons"]) == 14
data = {"lesson": "echo", "program": "dup add"}
assert api("/api/verify", data)["passed"]
assert api("/api/suggest", data)["suggestions"] == []
""",
        ],
        check=True,
    )


def test_adviser_checks_refusals_without_weights_and_records_load_errors(
    tmp_path, monkeypatch
):
    adviser = Adviser()
    monkeypatch.setattr(server, "ADVISER", adviser)
    data = {"lesson": "echo"}
    status = server.api("/api/model", {**data, "checkpoint": str(tmp_path)})
    assert not status["ready"] and status["error"]
    with pytest.raises(ValueError, match="add: "):
        server.api("/api/suggest", {**data, "program": "add"})


def test_adviser_masks_on_every_example_and_previews_results(tmp_path):
    torch = pytest.importorskip("torch")
    from wandering_light import stack_policy as policy

    model = policy.StackPairPolicy(policy.PolicyArchitecture(8, 16, 32))
    with torch.no_grad():
        model.head[-1].weight.zero_()
        bias = model.head[-1].bias
        bias.zero_()
        bias[policy.OP_INDEX["add"]] = 10
        bias[policy.OP_INDEX["swap"]] = 8
    path = policy.save_policy(
        tmp_path / "model", model, trainer_state={}, manifest={}, wandb_url=None
    )
    adviser = Adviser()
    assert adviser.load(path)["ready"]
    examples = [case([7], [8]), case([True], [True])]
    suggestions = adviser.rank("1", examples)["suggestions"]
    assert len(suggestions) == 6
    assert "add" not in [s["op"] for s in suggestions]
    assert suggestions[0]["op"] == "swap"
    probs = [s["probability"] for s in suggestions]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1
    assert suggestions[0]["previews"] == [
        engine.snapshot(engine.stack_from(stack)) for stack in ([1, 7], [1, True])
    ]


def test_http_requests_and_origin_checks():
    with server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler) as httpd:
        worker = threading.Thread(target=httpd.serve_forever)
        worker.start()
        try:
            for origin, body, status in [
                ("http://127.0.0.1:4567", '{"lesson":"echo","program":"dup add"}', 200),
                ("http://localhost", "{}", 400),
                ("http://localhost.example", "{}", 403),
                ("null", "{}", 403),
                (None, "bad json", 400),
            ]:
                conn = HTTPConnection(*httpd.server_address, timeout=5)
                try:
                    conn.request(
                        "POST",
                        "/api/verify",
                        body,
                        {"Origin": origin} if origin else {},
                    )
                    response = conn.getresponse()
                    assert response.status == status
                    payload = json.load(response)
                    if status == 200:
                        assert payload["passed"]
                    else:
                        assert set(payload) == {"error"}
                finally:
                    conn.close()
        finally:
            httpd.shutdown()
            worker.join()
